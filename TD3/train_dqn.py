# train_dqn.py
import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

# --------------------
# 动作离散化表（你可按需要调整刻度）
# --------------------
V_SET = [0.0, 0.3, 0.6, 0.9]               # 线速度 ∈ [0,1]
W_SET = [-1.0, -0.5, 0.0, 0.5, 1.0]        # 角速度 ∈ [-1,1]
ACTION_TABLE = [(v, w) for v in V_SET for w in W_SET]
NUM_ACTIONS = len(ACTION_TABLE)

def idx_to_continuous(idx: int):
    """离散动作 -> 按你主循环的接口返回 “类似TD3的动作向量” in [-1,1]^2。
       主循环会把 a_in = [(a0+1)/2, a1] 再转换给 env.step。"""
    v, w = ACTION_TABLE[idx]
    return np.array([2.0 * v - 1.0, w], dtype=np.float32)  # 让 evaluate()/主循环保持不变

def batch_action_to_idx(a_batch: np.ndarray):
    """把回放缓冲里存的 a ∈ [-1,1]^2 反推为最邻近的离散动作索引（用于 DQN 训练）。
       a_batch: [B,2] (第一维经 [(a0+1)/2] 还原成 v)"""
    v = (a_batch[:, 0] + 1.0) / 2.0
    w = a_batch[:, 1]
    # 就地量化成最近网格
    vq = np.array([min(V_SET, key=lambda x: abs(x - vi)) for vi in v], dtype=np.float32)
    wq = np.array([min(W_SET, key=lambda x: abs(x - wi)) for wi in w], dtype=np.float32)
    # 找索引
    v_idx = np.array([V_SET.index(vi) for vi in vq], dtype=np.int64)
    w_idx = np.array([W_SET.index(wi) for wi in wq], dtype=np.int64)
    return v_idx * len(W_SET) + w_idx

# --------------------
# Q 网络（Double DQN）
# --------------------
class QNet(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # [B, A]

class DQNAgent:
    def __init__(self, state_dim, num_actions, gamma=0.999, lr=1e-3, tau=0.005):
        self.q = QNet(state_dim, num_actions).to(device)
        self.q_target = QNet(state_dim, num_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions
        self.writer = SummaryWriter(log_dir="./runs_dqn")  # 单独目录，不和 TD3 混
        self.iter_count = 0

    @torch.no_grad()
    def get_action(self, state_np):
        """为了复用 evaluate()/主循环，这里返回 [-1,1]^2 的动作向量（通过离散动作映射）。"""
        s = torch.tensor(state_np.reshape(1, -1), dtype=torch.float32, device=device)
        q = self.q(s)                      # [1, A]
        idx = int(q.argmax(dim=1).item())  # 贪心；探索在主循环里加
        return idx_to_continuous(idx)

    def select_action_idx_eps(self, state_np, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        s = torch.tensor(state_np.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            q = self.q(s)
        return int(q.argmax(dim=1).item())

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def train_step(self, replay: ReplayBuffer, batch_size=64):
        (s, a, r, d, s2) = replay.sample_batch(batch_size)
        # numpy -> torch
        s   = torch.tensor(s, dtype=torch.float32, device=device)
        a   = torch.tensor(a, dtype=torch.float32, device=device)        # [B,2] in [-1,1]
        r   = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        d   = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        s2  = torch.tensor(s2, dtype=torch.float32, device=device)

        # 从 a 反推离散动作索引（不改你的回放结构）
        a_idx_np = batch_action_to_idx(a.cpu().numpy()).reshape(-1, 1)
        a_idx = torch.tensor(a_idx_np, dtype=torch.int64, device=device)

        # 当前 Q(s, a_idx)
        q_all = self.q(s)                             # [B, A]
        q_sa  = q_all.gather(1, a_idx)                # [B, 1]

        with torch.no_grad():
            # Double DQN 目标：在线网选 a*，目标网评估
            next_q_online = self.q(s2)                                    # [B,A]
            next_a_idx = next_q_online.argmax(dim=1, keepdim=True)        # [B,1]
            next_q_target = self.q_target(s2).gather(1, next_a_idx)       # [B,1]
            target = r + (1 - d) * self.gamma * next_q_target

        loss = F.mse_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.soft_update()

        # 统一你的日志风格（和 TD3 一样的 key 方便对比）
        with torch.no_grad():
            self.writer.add_scalar("loss", loss.item(), self.iter_count)
            self.writer.add_scalar("Av. Q", q_all.mean().item(), self.iter_count)
            self.writer.add_scalar("Max. Q", q_all.max().item(), self.iter_count)
        self.iter_count += 1

# --------------------
# 训练主循环（复用你的 evaluate / GazeboEnv / ReplayBuffer）
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 与 TD3 保持一致
    environment_dim = 20
    robot_dim = 4
    state_dim = environment_dim + robot_dim

    # 创建环境（你的 world/模型设置都在对应 launch 和脚本里，不变）
    env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
    time.sleep(5)

    agent = DQNAgent(state_dim, NUM_ACTIONS, gamma=0.999, lr=1e-3, tau=0.005)
    replay = ReplayBuffer(1e6, random_seed=0)

    max_ep = 500
    max_timesteps = int(5e6)
    batch_size = 64

    # ε-greedy 与你探索衰减风格一致
    epsilon = 1.0
    eps_min = 0.1
    eps_decay_steps = 500_000
    def eps_decay(eps):
        if eps > eps_min:
            eps -= (1.0 - eps_min) / eps_decay_steps
            if eps < eps_min: eps = eps_min
        return eps

    # 复用你的 evaluate：它会调用网络的 get_action()，我们已返回 [-1,1]^2 兼容格式
    from train_velodyne_td3 import evaluate  # 直接复用

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    while timestep < max_timesteps:
        if done:
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # 选 “离散动作索引”，再映射成连续格式（保持你主循环/评估不变）
        a_idx = agent.select_action_idx_eps(np.array(state), epsilon)
        action_vec = idx_to_continuous(a_idx)  # [-1,1]^2
        # 主循环仍会把 a_in = [(a0+1)/2, a1] 送入 env
        a_in = [(action_vec[0] + 1) / 2.0, action_vec[1]]
        next_state, reward, done, _ = env.step(a_in)

        done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
        # 回放里继续存 “连续形式动作”，不改你的数据结构
        replay.add(state, action_vec, reward, done_bool, next_state)

        # 学习（预热后再学，避免一开始目标过噪）
        if timestep > 10_000:
            agent.train_step(replay, batch_size=batch_size)

        # 评估（沿用你的频率与写法）
        timesteps_since_eval += 1
        if timesteps_since_eval >= 5e3:
            timesteps_since_eval = 0
            avgR = evaluate(network=agent, epoch=epoch, eval_episodes=10)  # 直接调用
            epoch += 1

        # 更新计数
        state = next_state
        episode_timesteps += 1
        timestep += 1
        epsilon = eps_decay(epsilon)
        episode_reward += reward

        if episode_timesteps >= max_ep:
            done = True

if __name__ == "__main__":
    main()
