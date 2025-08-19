import os
import numpy as np
import torch
import gymnasium as gym
from typing import List, Tuple
from datetime import datetime

from stable_baselines3 import PPO, DQN, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_wrapper import VelodyneGymWrapper


class EvaluationCallback(BaseCallback):
    """每5000步打印一次评估提示（与现有脚本风格保持一致）。"""

    def __init__(self, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            if self.verbose > 0:
                print(f"\n=== 评估触发（第{self.num_timesteps}步）===")
        return True


class HierarchicalEnv(gym.Env):
    """
    层级控制环境：
    - 高层（DQN）选择一组参数：线速度缩放系数、角速度缩放系数。
    - 低层（PPO）根据观测输出基础连续动作，再由高层参数进行缩放后发送给真实环境。

    这样可在不改动底层 PPO 结构的情况下，实现“DQN 决策 + PPO 精细控制”的管线。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        launchfile: str,
        environment_dim: int,
        ppo_model: PPO,
        device: str = "cpu",
        speed_scales: List[float] = None,
        angular_scales: List[float] = None,
    ):
        super().__init__()

        self.device = device
        self.base_env = VelodyneGymWrapper(
            launchfile=launchfile,
            environment_dim=environment_dim,
            action_type="continuous",
            device=device,
        )

        # 固定的低层控制器（也可以换成在线微调）
        self.ppo_model = ppo_model

        # 定义高层参数集合
        if speed_scales is None:
            speed_scales = [0.5, 0.8, 1.0]
        if angular_scales is None:
            angular_scales = [0.7, 1.0, 1.3]
        self.param_pairs: List[Tuple[float, float]] = [
            (s, a) for s in speed_scales for a in angular_scales
        ]

        # 动作空间：高层离散选择
        self.action_space = gym.spaces.Discrete(len(self.param_pairs))
        # 观测空间：与底层环境一致
        self.observation_space = self.base_env.observation_space

        self._last_obs = None

    def step(self, action: int):
        # 取出高层参数
        speed_scale, angular_scale = self.param_pairs[int(action)]

        # 基于最近观测由 PPO 生成基础动作
        if self._last_obs is None:
            obs, _ = self.base_env.reset()
            self._last_obs = obs
        base_action, _ = self.ppo_model.predict(self._last_obs, deterministic=False)

        # 将高层参数注入到低层动作
        v = float(np.clip(base_action[0] * speed_scale, 0.0, 1.0))
        w = float(np.clip(base_action[1] * angular_scale, -1.0, 1.0))
        final_action = np.array([v, w], dtype=np.float32)

        # 与真实环境交互
        obs, reward, terminated, truncated, info = self.base_env.step(final_action)
        self._last_obs = obs

        # 将高层决策回传到 info 便于调试/分析
        info = dict(info)
        info.update(
            {
                "speed_scale": speed_scale,
                "angular_scale": angular_scale,
                "ppo_action": base_action,
                "final_action": final_action,
            }
        )

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


class ScaledActionEnv(gym.Env):
    """
    低层 PPO 训练用包装器：
    - 接收 PPO 连续动作 a_ppo。
    - 使用“固定的”高层 DQN 基于当前观测选择参数对 (s, a)。
    - 执行 a_final = (s * v_ppo, a * w_ppo) 并推进真实环境。

    这样 PPO 在训练时看到的就是最终动作的效果，避免策略失配。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        launchfile: str,
        environment_dim: int,
        dqn_model: DQN,
        device: str = "cpu",
        speed_scales: List[float] = None,
        angular_scales: List[float] = None,
    ):
        super().__init__()

        self.device = device
        self.base_env = VelodyneGymWrapper(
            launchfile=launchfile,
            environment_dim=environment_dim,
            action_type="continuous",
            device=device,
        )

        self.dqn_model = dqn_model

        if speed_scales is None:
            speed_scales = [0.5, 0.8, 1.0]
        if angular_scales is None:
            angular_scales = [0.7, 1.0, 1.3]
        self.param_pairs: List[Tuple[float, float]] = [
            (s, a) for s in speed_scales for a in angular_scales
        ]

        # PPO 的动作空间与底层环境一致（连续动作）
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

        self._last_obs = None

    def _select_scale(self, obs: np.ndarray) -> Tuple[float, float, int]:
        idx, _ = self.dqn_model.predict(obs, deterministic=True)
        s, a = self.param_pairs[int(idx)]
        return s, a, int(idx)

    def step(self, ppo_action: np.ndarray):
        if self._last_obs is None:
            obs, _ = self.base_env.reset()
            self._last_obs = obs

        s, a, idx = self._select_scale(self._last_obs)
        v = float(np.clip(ppo_action[0] * s, 0.0, 1.0))
        w = float(np.clip(ppo_action[1] * a, -1.0, 1.0))
        final_action = np.array([v, w], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.base_env.step(final_action)
        self._last_obs = obs
        info = dict(info)
        info.update({
            "speed_scale": s,
            "angular_scale": a,
            "dqn_choice": idx,
            "final_action": final_action,
        })
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


def main():
    # 设备信息
    print("=== CUDA检测 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  CUDA不可用，将使用CPU")
        device = "cpu"
    print("=" * 20)

    # 路径准备
    os.makedirs("./pytorch_models", exist_ok=True)

    # 1) 预训练/加载 低层控制器（支持 PPO/SAC/TD3）
    algo = os.environ.get("LOW_LEVEL_ALGO", "ppo").lower()  # 通过环境变量切换，也可改为 argparse
    print(f"低层算法: {algo.upper()} (设置环境变量 LOW_LEVEL_ALGO=ppo|sac|td3 可切换)")
    print("正在创建低层训练环境...")
    ppo_env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous",
        device=device,
    )

    if algo == "ppo":
        print("构建 PPO 模型...")
        low_model = PPO(
            "MlpPolicy",
            ppo_env,
            verbose=1,
            tensorboard_log="./logs/ppo_low_level",
            learning_rate=7e-4,
            n_steps=1024,
            batch_size=512,
            n_epochs=5,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            device=device,
        )
    elif algo == "sac":
        print("构建 SAC 模型...")
        low_model = SAC(
            "MlpPolicy",
            ppo_env,
            verbose=1,
            tensorboard_log="./logs/sac_low_level",
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            device=device,
        )
    elif algo == "td3":
        print("构建 TD3 模型...")
        low_model = TD3(
            "MlpPolicy",
            ppo_env,
            verbose=1,
            tensorboard_log="./logs/td3_low_level",
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            device=device,
        )
    else:
        raise ValueError("LOW_LEVEL_ALGO 仅支持 ppo/sac/td3")

    # 为了快速联调，先进行一小段训练；实际项目中可改为加载已训好的模型
    print("=== 低层快速预热（10000步，加快联调）===")
    low_model.learn(total_timesteps=10_000, progress_bar=True)
    low_model.save(f"./pytorch_models/{algo}_low_level")
    ppo_env.close()

    # 如需从磁盘加载，可以使用：
    # if algo == "ppo": low_model = PPO.load("./pytorch_models/ppo_low_level", device=device)

    # 2) 在层级环境中训练 DQN 高层策略
    print("正在创建层级环境（DQN->PPO）...")
    hier_env = HierarchicalEnv(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        ppo_model=low_model,  # 复用相同接口
        device=device,
        # 可按需调整高层参数集合：
        speed_scales=[0.5, 0.8, 1.0],
        angular_scales=[0.7, 1.0, 1.3],
    )

    print("构建 DQN 高层模型...")
    eval_freq = 5_000
    dqn_model = DQN(
        "MlpPolicy",
        hier_env,
        verbose=1,
        tensorboard_log="./logs/dqn_high_level",
        learning_rate=2e-3,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=5_000,
        learning_starts=1_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="hier_dqn"
    )
    eval_callback = EvaluationCallback(eval_freq=eval_freq, verbose=1)

    # 3) 交替微调：先训练 DQN，再训练 PPO（ScaledActionEnv 内部调用 DQN 选择缩放）
    print("=== 交替微调开始 ===")
    # 在交替前保存一份只读的低层模型快照（基线）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = f"./pytorch_models/{algo}_baseline_{ts}"
    low_model.save(baseline_path)
    print(f"已保存低层基线模型: {baseline_path}.zip")
    ALT_CYCLES = 5           # 交替轮数（可根据算力加大）
    DQN_STEPS = 10_000       # 每轮高层训练步数
    LOW_STEPS = 5_000        # 每轮低层训练步数

    # PPO 训练用环境（引用 dqn_model，自动使用最新高层策略）
    scaled_env = ScaledActionEnv(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        dqn_model=dqn_model,
        device=device,
        speed_scales=[0.5, 0.8, 1.0],
        angular_scales=[0.7, 1.0, 1.3],
    )

    for i in range(ALT_CYCLES):
        print(f"\n--- Cycle {i+1}/{ALT_CYCLES}: 训练 DQN {DQN_STEPS} 步 ---")
        dqn_model.learn(
            total_timesteps=DQN_STEPS,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=[checkpoint_callback, eval_callback],
        )

        print(f"--- Cycle {i+1}/{ALT_CYCLES}: 训练 {algo.upper()} {LOW_STEPS} 步 ---")
        low_model.set_env(scaled_env)
        low_model.learn(
            total_timesteps=LOW_STEPS,
            reset_num_timesteps=False,
            progress_bar=True,
        )

    print("保存模型...")
    dqn_model.save("./pytorch_models/hier_dqn_model")
    low_model.save(f"./pytorch_models/hier_{algo}_model")

    scaled_env.close()
    hier_env.close()
    print("交替微调完成！")


if __name__ == "__main__":
    main()


