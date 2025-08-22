# 导入必要的库
import os  # 用于文件和目录操作
import numpy as np  
import torch  
import optuna  # 用于超参数优化
from stable_baselines3 import DQN, TD3  
from stable_baselines3.common.callbacks import CheckpointCallback  # 从sb3导入的模型检查点callback
# 从sb3导入的经验回放缓冲区
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm  # 用于显示进度条

# 移除了PrioritizedReplayBuffer的导入，因为该类不可用
PRIORITIZED_REPLAY_AVAILABLE = False
from gym_wrapper import VelodyneGymWrapper  # 自定义的Gym环境包装器
from velodyne_env import GazeboEnv  # 自定义的Gazebo环境
from typing import Dict, Any, Tuple, List, Optional  # 类型注解

# 导入gym或gymnasium库(修不明白！)
try:
    import gymnasium
    from gymnasium.spaces import Box, Discrete
    gym_lib = 'gymnasium'
except ImportError:
    import gym
    from gym.spaces import Box, Discrete
    gym_lib = 'gym'

import sys, signal, atexit  # 用于信号处理与进程退出清理

# Optuna配置
OPTUNA_STUDY_NAME = "hierarchical_rl_study"  # 超参数优化研究的名称
OPTUNA_N_TRIALS = 50  # 超参数搜索的试验次数，设置为50次以平衡搜索效率和效果


class HierarchicalRL:
    """层级强化学习(HRL)架构

    该类实现了一个两层的强化学习架构，用于机器人导航任务：
    - 高层(High-level): 使用DQN决定导航方向和距离
    - 低层(Low-level): 使用TD3执行具体的控制动作

    这种分层架构的优势在于可以将复杂的导航任务分解为更简单的子任务，提高学习效率和泛化能力。

    """
    def __init__(self, environment_dim=20, max_timesteps=5e6, eval_freq=5e3, device=None, batch_train_size=100,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1e5,
                 noise_start=0.2, noise_end=0.01, noise_decay=1e5):
        # 移除了与PrioritizedReplayBuffer相关的参数，因为该类不可用
        use_per = False
        """初始化HRL Agent"""
        # CUDA使用设置
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 环境参数
        self.environment_dim = environment_dim  # 方向数量。将方向切分成默认20个environment_dim，每个environment_dim表示一个方向
        self.max_timesteps = max_timesteps  # 最大训练时间步。默认5e6
        self.eval_freq = eval_freq  # 评估频率，默认5e3
        self.max_ep = 500  # 每个回合的最大步数，默认500
        self.batch_train_size = batch_train_size  # 批量训练大小，默认100

        # 经验回放参数
        self.use_per = False  # 已禁用优先经验回放(PER)，因为PrioritizedReplayBuffer不可用

        # 探索策略参数
        self.epsilon_start = epsilon_start  # DQN初始探索率，默认1.0
        self.epsilon_end = epsilon_end  # DQN最终探索率，默认0.05
        self.epsilon_decay = epsilon_decay  # DQN探索率衰减速率，默认1e5
        self.noise_start = noise_start  # TD3初始噪声水平，默认0.2
        self.noise_end = noise_end  # TD3最终噪声水平，默认0.01
        self.noise_decay = noise_decay  # TD3噪声衰减速率
        self.current_epsilon = epsilon_start  # 当前epsilon值
        self.current_noise = noise_start  # 当前噪声值

        # 训练开始前收集的经验数量
        self.learn_starts = 1000  # 默认为1000步

        # 高层智能体训练频率（每多少步训练一次）
        self.train_freq = 100  # 默认为每100步训练一次

        # 高层和低层智能体的批量训练大小
        self.H_BS = 100  # 高层智能体批量训练大小
        self.L_BS = 100  # 低层智能体批量训练大小

        # 初始化环境
        self.env = self._init_environment()

        # 创建经验缓冲区
        self._init_replay_buffers()

        # 初始化高层DQN和低层TD3模型
        self.high_level_agent = self._init_high_level_agent()  # 高层决策智能体
        self.low_level_agent = self._init_low_level_agent()  # 低层执行智能体

        # 创建存储目录
        self._create_directories()

        # —— 注册进程退出清理（无论正常结束还是异常退出/按 Ctrl+C）
        atexit.register(self._cleanup_gazebo_ros)

    def _init_environment(self):
        """初始化机器人导航环境。

        该方法创建并返回一个基于Gazebo的Velodyne激光雷达机器人导航环境。
        使用VelodyneGymWrapper(在gym_wrapper.py中)包装器将Gazebo环境转换为符合OpenAI Gym接口的环境, 
        以便于强化学习算法的使用。

        """
        print("正在初始化环境...")
        env = VelodyneGymWrapper(
            launchfile="multi_robot_scenario.launch",  # Gazebo启动文件
            environment_dim=self.environment_dim,  # 默认20
            action_type="continuous",  # 动作类型为连续，适应低层TD3智能体
            device=self.device  # 计算设备
        )
        print("环境初始化完成!")
        return env

    def _init_replay_buffers(self):
        """初始化经验回放缓冲区

        经验回放缓冲区用于存储智能体与环境交互产生的经验，以便后续训练使用。
        该方法为高层和低层智能体分别创建普通经验回放缓冲区。

        """
        print("正在初始化经验回放缓冲区...")
        # 获取状态和动作空间维度
        # 获取状态和动作空间维度
        # 高层状态维度与环境的观察空间维度相同
        high_level_state_dim = self.env.observation_space.shape[0]
        num_directions = self.environment_dim
        num_distances = 10
        high_level_action_dim = num_directions * num_distances  # 高层动作数(方向数量×距离级别)
        low_level_state_dim = high_level_state_dim + 2  # 低层状态维度=高层状态+方向+距离
        low_level_action_dim = self.env.action_space.shape[0]  # 低层动作维度

        # 使用普通经验回放缓冲区
        # 为高层和低层智能体创建观察空间和动作空间
        import numpy as np

        # 高层智能体的观察空间和动作空间
        # 根据gym_wrapper.py中的定义设置观察空间
        laser_low = 0.0
        laser_high = 10.0
        distance_low = 0.0
        distance_high = 7.0
        angle_low = -np.pi
        angle_high = np.pi
        vel_low = 0.0
        vel_high = 1.0
        ang_vel_low = -1.0
        ang_vel_high = 1.0

        high_level_observation_low = np.array([laser_low] * self.environment_dim + [distance_low, angle_low, vel_low, ang_vel_low], dtype=np.float32)
        high_level_observation_high = np.array([laser_high] * self.environment_dim + [distance_high, angle_high, vel_high, ang_vel_high], dtype=np.float32)

        high_level_observation_space = Box(low=high_level_observation_low, high=high_level_observation_high, dtype=np.float32)
        high_level_action_space = Discrete(high_level_action_dim)

        # 低层智能体的观察空间和动作空间
        # 低层智能体的观察空间和动作空间
        low_level_observation_low = np.concatenate([high_level_observation_low, [-np.pi, 0.0]]).astype(np.float32)
        low_level_observation_high = np.concatenate([high_level_observation_high, [np.pi, 10.0]]).astype(np.float32)
        low_level_observation_space = Box(low=low_level_observation_low, high=low_level_observation_high, dtype=np.float32)
        low_level_action_space = self.env.action_space  # 直接使用环境的动作空间

        print(f"使用{gym_lib}库定义观察空间，dtype={np.float32}")

        # self.high_level_buffer = ReplayBuffer(  # 高层（DQN）的经验回放缓冲区
        #     buffer_size=1_000_000,
        #     observation_space=high_level_observation_space,
        #     action_space=high_level_action_space,
        #     device=self.device,
        #     # n_envs=1  # 明确指定环境数量为1
        #     n_envs=getattr(self.env, "num_envs", 1)   # 自动适配环境数
        # )
        # self.low_level_buffer = ReplayBuffer(  # 低层（TD3）的经验回放缓冲区
        #     buffer_size=1_000_000,
        #     observation_space=low_level_observation_space,
        #     action_space=low_level_action_space,
        #     device=self.device,
        #     # n_envs=1  # 明确指定环境数量为1
        #     n_envs=getattr(self.env, "num_envs", 1)   # 自动适配环境数
        # )
        # print("使用普通经验回放缓冲区")

    def _init_high_level_agent(self):
        """初始化高层智能体

        高层智能体使用DQN算法, 负责决定机器人的导航方向和移动距离。DQN适合处理离散动作空间的问题。
        高层智能体的动作空间是离散的, 对应20个不同的方向。

        返回:
            初始化后的DQN Agent
        """
        print("正在初始化高层DQN智能体...")
        # 为高层DQN创建一个具有离散动作空间的环境
        import gymnasium
        from gymnasium import spaces
        from stable_baselines3.common.env_util import make_vec_env
        
        # 高层动作空间：20个方向 × 10个距离级别 = 200个离散动作
        num_directions = 20
        num_distances = 10
        high_level_action_space = spaces.Discrete(num_directions * num_distances)
        
        # 创建一个环境包装器，使用离散动作空间
        class DiscreteActionEnvWrapper(gymnasium.Env):
            """
            仅用于给 DQN 提供一个“合法”的环境接口，避免 SB3 内部调用 reset/step 时
            真的去驱动 Gazebo。我们完全不会用它来 rollouts，只会用 high_level_agent.predict()。
            """
            def __init__(self, env):
                super().__init__()
                self.env = env
                self.observation_space = env.observation_space
                self.action_space = high_level_action_space
                self._last_obs = None

            def reset(self, seed=None, options=None):
                reset_result = self.env.reset(seed=seed, options=options)
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                self._last_obs = obs
                # gymnasium 需要返回 (obs, info)
                return obs, {}

            def step(self, action):
                # 不要在 step() 里 reset 环境！返回一个 no-op 的一步即可。
                assert self._last_obs is not None, "Call reset() before step()."
                return self._last_obs, 0.0, False, False, {}
            
            def render(self, mode='human'):
                return self.env.render(mode=mode)
            
            def close(self):
                return self.env.close()
        
        discrete_env = DiscreteActionEnvWrapper(self.env)
        
        model = DQN(
            "MlpPolicy", 
            discrete_env,
            verbose=0,  
            tensorboard_log="./logs/high_level_dqn",  # 日志保存路径
            learning_rate=1e-4,  # 学习率
            buffer_size=1_000_000,  # 经验回放缓冲区大小
            batch_size=64,  # 批量大小
            gamma=0.99,  # 折扣因子
            device=self.device,  # 计算设备
            exploration_initial_eps=self.epsilon_start,  # 初始探索率
            exploration_final_eps=self.epsilon_end,  # 最终探索率
            exploration_fraction=self.epsilon_decay / self.max_timesteps  # 探索率衰减比例
        )
        print("高层DQN智能体初始化完成!")
        return model

    def _init_low_level_agent(self):
        """初始化低层TD3智能体

        低层智能体使用双延迟深度确定性策略梯度(TD3)算法, 负责执行高层智能体制定的子目标。
        TD3适合处理连续动作空间的问题。低层智能体需要控制机器人的具体运动, 动作空间是连续的。

        返回:
            初始化后的TD3 Agent
        """
        print("正在初始化低层TD3智能体...")
        # 低层TD3处理连续动作
        # 创建一个新的观察空间，包含原始状态(24维)加上方向和距离(2维)
        low = np.append(self.env.observation_space.low, [-np.pi, 0.0]).astype(np.float32)  # 方向范围[-pi, pi]，距离范围[0.0, 5.0]
        high = np.append(self.env.observation_space.high, [np.pi, 5.0]).astype(np.float32)
        extended_observation_space = Box(low=low, high=high, dtype=np.float32)

        # 使用扩展后的观察空间创建一个包装环境
        if gym_lib == 'gymnasium':
            class ExtendedObservationEnvWrapper(gymnasium.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.observation_space = extended_observation_space
        else:  # gym_lib == 'gym'
            class ExtendedObservationEnvWrapper(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.observation_space = extended_observation_space

        extended_env = ExtendedObservationEnvWrapper(self.env)

        model = TD3(
            "MlpPolicy", 
            extended_env,  
            verbose=0, 
            tensorboard_log="./logs/low_level_td3",  # 日志保存路径
            learning_rate=1e-4,  # 学习率
            buffer_size=1_000_000,  # 经验回放缓冲区大小
            batch_size=40,  # 批量大小
            tau=0.005,  # 目标网络更新系数
            gamma=0.99999,  # 折扣因子，接近1表示重视未来奖励
            train_freq=1,  # 训练频率
            gradient_steps=1,  # 每次训练的梯度步数
            policy_delay=2,  # 策略更新延迟
            target_policy_noise=self.noise_start,  # 目标策略噪声
            target_noise_clip=0.5,  # 目标噪声裁剪
            device=self.device  # 计算设备
        )
        print("低层TD3智能体初始化完成!")
        return model

    def _create_directories(self):
        """创建模型和结果存储目录

        创建用于存储训练结果、模型权重的目录结构，确保训练过程中生成的文件有合适的存放位置。如果目录已存在，则不会重复创建。
        """
        directories = ["./results", "./pytorch_models/high_level", "./pytorch_models/low_level"]
        for dir_path in directories:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

    def _prepare_sb3_train(self, agent):
        """
        让 SB3 智能体在不走 learn() 的情况下，也能安全调用 train()。
        - 创建 logger
        - 设置进度变量
        - 补齐统计计数器
        """
        # 1) logger（SB3 的 train() 会用到）
        if getattr(agent, "_logger", None) is None:
            from stable_baselines3.common.logger import configure
            agent._logger = configure(folder=None, format_strings=["stdout"])

        # 2) 进度变量（用于学习率调度）
        if not hasattr(agent, "_current_progress_remaining"):
            agent._current_progress_remaining = 1.0  # 相当于训练刚开始

        # 3) 计数器（有些算法在日志里会用到）
        if not hasattr(agent, "_n_updates"):
            agent._n_updates = 0
        if not hasattr(agent, "num_timesteps"):
            agent.num_timesteps = 0

        # 4) 兜底：有些版本把 lr_schedule 挂在 agent 上；通常已存在，这里防御性补一下
        if not hasattr(agent, "lr_schedule"):
            # 尽量用 agent.learning_rate，缺省 3e-4
            base_lr = float(getattr(agent, "learning_rate", 3e-4))
            agent.lr_schedule = lambda progress: base_lr * 1.0  # 固定学习率

    def load_high_level_model(self, model_path):
        """加载高层DQN模型

        该方法尝试从指定路径加载预训练的高层DQN模型。高层模型负责决策机器人的导航方向和移动距离。
        如果加载成功, 将使用预训练模型进行后续操作; 如果加载失败, 将打印错误信息但不会中断程序。

        参数:
            model_path: 预训练模型的文件路径
        """
        try:
            self.high_level_agent = DQN.load(model_path, env=self.env, device=self.device)
            print(f"成功加载高层模型: {model_path}")
        except Exception as e:
            print(f"加载高层模型失败: {e}")

    def load_low_level_model(self, model_path):
        """加载低层TD3模型

        该方法尝试从指定路径加载预训练的低层TD3模型。低层模型负责执行高层模型制定的子目标, 控制机器人的具体运动。
        如果加载成功, 将使用预训练模型进行后续操作;如果加载失败, 将打印错误信息但不会中断程序。

        参数:
            model_path: 预训练模型的文件路径
        """
        try:
            self.low_level_agent = TD3.load(model_path, env=self.env, device=self.device)
            print(f"成功加载低层模型: {model_path}")
        except Exception as e:
            print(f"加载低层模型失败: {e}")

    def _update_exploration_parameters(self, timestep: int):
        """更新探索策略参数

        允许智能体在未知环境中尝试不同的动作探索以发现最优策略。该方法根据训练步数动态调整探索参数：
        - 对于DQN(高层智能体), 使用epsilon-greedy策略, 随着训练进行, 探索率逐渐降低；
        - 对于TD3(低层智能体), 使用噪声探索策略, 随着训练进行, 噪声强度逐渐降低。

        参数:
            timestep: 当前训练步数
        """
        # 更新epsilon (DQN) - 指数衰减
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * timestep / self.epsilon_decay)

        # 更新噪声参数 (TD3) - 指数衰减
        self.current_noise = self.noise_end + (self.noise_start - self.noise_end) * \
            np.exp(-1. * timestep / self.noise_decay)

        # 应用到DQN模型
        if hasattr(self.high_level_agent, 'exploration_rate'):
            self.high_level_agent.exploration_rate = self.current_epsilon
        elif hasattr(self.high_level_agent, 'policy') and hasattr(self.high_level_agent.policy, 'exploration_rate'):
            self.high_level_agent.policy.exploration_rate = self.current_epsilon
        elif hasattr(self.high_level_agent, 'policy') and hasattr(self.high_level_agent.policy, 'epsilon'):
            self.high_level_agent.policy.epsilon = self.current_epsilon

        # 应用到TD3模型
        if hasattr(self.low_level_agent, 'policy') and hasattr(self.low_level_agent.policy, 'target_policy_noise'): # 如果有这个参数的话
            self.low_level_agent.policy.target_policy_noise = self.current_noise # 更新为current_noise

    def _calculate_rewards(self, state: np.ndarray, next_state: np.ndarray, action: int, distance: float,
                          done: bool, target: bool, episode_timesteps: int, reward: float, info: dict) -> Tuple[float, float]:
        # 确保prev_direction已定义
        if not hasattr(self, 'prev_direction'):
            self.prev_direction = 0.0
        """计算高层和低层智能体的奖励

        奖励函数是强化学习中的关键组件，它定义了智能体行为的好坏。该方法计算
        多种奖励成分，包括方向奖励、距离奖励、障碍物规避奖励和路径平滑性奖励，
        并将它们组合成总奖励。高层和低层智能体有不同的奖励计算方式。

        参数:
            state: 当前状态
            next_state: 下一个状态
            action: 高层智能体选择的动作
            distance: 高层智能体选择的距离
            done: 是否结束回合
            target: 是否到达目标
            episode_timesteps: 当前回合步数
            reward: 环境原始奖励

        返回:
            Tuple[float, float]: 高层智能体奖励和低层智能体奖励
        """
        # 提取位置信息
        current_position = state[:2]
        next_position = next_state[:2]
        direction = action % self.environment_dim

        # 计算实际移动
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        actual_distance = np.sqrt(dx**2 + dy**2)
        actual_direction = np.arctan2(dy, dx) * 180 / np.pi % 360

        # 目标方向
        target_direction = (direction * 360 / self.environment_dim) % 360

        # 方向和距离差异
        direction_diff = min(abs(actual_direction - target_direction), 360 - abs(actual_direction - target_direction))
        distance_diff = abs(actual_distance - distance)

        # 高层奖励组成
        direction_reward = 1.0 - (direction_diff / 180.0)
        distance_reward = 1.0 - min(distance_diff / distance, 1.0)
        collision_penalty = -1.0 if done and not target else 0.0
        time_penalty = -0.01

        # 新增: 子目标合理性奖励
        # 检查目标方向是否有障碍物
        obstacle_avoidance_reward = 0.0
        # 检查目标方向是否有障碍物
        obstacle_avoidance_reward = 0.2 if 'obstacle_ahead' in info and not info['obstacle_ahead'] else 0.0

        # 新增: 路径质量奖励 (平滑性)
        if episode_timesteps > 0 and hasattr(self, 'prev_direction'):
            direction_change = min(abs(actual_direction - self.prev_direction), 360 - abs(actual_direction - self.prev_direction))
            smoothness_reward = 0.1 * (1.0 - min(direction_change / 90.0, 1.0))
        else:
            smoothness_reward = 0.0

        # 更新前一方向
        self.prev_direction = actual_direction

        # 综合高层奖励 = 方向奖励 * 0.4 + 距离奖励 * 0.4 + 障碍物奖励 * 0.1 + 平滑奖励 * 0.1 + 碰撞奖励 + 时间奖励
        high_level_reward = (direction_reward * 0.4 + distance_reward * 0.4 + 
                            obstacle_avoidance_reward * 0.1 + smoothness_reward * 0.1 + 
                            collision_penalty + time_penalty)
        high_level_reward = max(high_level_reward, -2.0)

        # 低层奖励: 结合环境反馈和子目标完成度
        low_level_reward = reward  # 原始环境奖励
        low_level_reward += 0.5 * (direction_reward + distance_reward)  # 子目标完成度奖励
        low_level_reward += collision_penalty * 0.5  # 碰撞惩罚

        return high_level_reward, low_level_reward

    def train(self, log_interval=1000):
        """分层强化学习训练（外部采样 → 内部 buffer → 仅梯度更新）"""
        print("开始层级强化学习训练...")

        self._prepare_sb3_train(self.high_level_agent)
        self._prepare_sb3_train(self.low_level_agent)

        timestep = 0
        evaluations = []
        self.prev_direction = 0.0  # 用于平滑奖励
        episode_count = 0  # 回合计数器，从1开始

        # 方便书写
        H_BUF = self.high_level_agent.replay_buffer
        L_BUF = self.low_level_agent.replay_buffer
        H_BS  = self.high_level_agent.batch_size
        L_BS  = self.low_level_agent.batch_size

        # 计算总回合数
        total_episodes = int(self.max_timesteps / self.max_ep) + 1
        print(f"总回合数: {total_episodes}")

        # 创建进度条
        with tqdm(total=self.max_timesteps, desc="训练进度", position=1, leave=True, dynamic_ncols=True) as pbar:
            while timestep < self.max_timesteps:
                # 新回合开始
                episode_count += 1
                reset_result = self.env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                done = False
                episode_reward = 0.0
                episode_timesteps = 0
                pbar.set_postfix({"回合": episode_count, "当前奖励": f"{episode_reward:.2f}"})

                while not done and episode_timesteps < self.max_ep:
                    # 探索参数更新（epsilon / 噪声）
                    self._update_exploration_parameters(timestep)

                    # ===== 高层：离散 action -> (direction, distance) =====
                    # 这里只用 policy 的 predict，不让 SB3 自己 rollouts
                    high_level_action = self.high_level_agent.predict(state, deterministic=False)[0]
                    direction = high_level_action % 20
                    distance  = (high_level_action // 20) * 0.5 + 0.5  # 0.5 ~ 5.0

                    # ===== 低层：连续动作（把子目标拼进观测）=====
                    sub_goal_state = np.append(state, [direction, distance])
                    low_level_action = self.low_level_agent.predict(sub_goal_state, deterministic=False)[0]

                    # 与 Gazebo 真正交互的一步
                    next_state, env_reward, terminated, truncated, info = self.env.step(low_level_action)
                    done = terminated or truncated
                    target = info.get('target_reached', False) if info else False

                    # 计算自定义奖励
                    high_level_reward, low_level_reward = self._calculate_rewards(
                        state=state, 
                        next_state=next_state, 
                        action=high_level_action, 
                        distance=distance, 
                        done=done, 
                        target=target, 
                        episode_timesteps=episode_timesteps, 
                        reward=env_reward, 
                        info=info
                    )

                    # 更新回合奖励（使用高层奖励）
                    episode_reward += high_level_reward

                    # 高层经验：(s, a_high, R)，R是回合总奖励（延迟更新）
                    # 低层经验：(s', a_low, r)，r是单步奖励
                    # 注意：高层奖励将在回合结束时更新
                    # Add experience to replay buffers with correct parameter order for custom ReplayBuffer
                    # Format experience for SB3 ReplayBuffer
                    obs_h = state.reshape(1, -1)
                    next_obs_h = next_state.reshape(1, -1)
                    act_h = np.array([[high_level_action]], dtype=np.int64)
                    rew_h = np.array([0.0], dtype=np.float32)
                    done_h = np.array([bool(done)], dtype=np.bool_)
                    H_BUF.add(obs_h, next_obs_h, act_h, rew_h, done_h, infos=[{}])

                    obs_l = sub_goal_state.reshape(1, -1)
                    next_obs_l = np.append(next_state, [direction, distance]).reshape(1, -1)
                    act_l = np.array(low_level_action, dtype=np.float32).reshape(1, -1)
                    rew_l = np.array([low_level_reward], dtype=np.float32)
                    done_l = np.array([bool(done)], dtype=np.bool_)
                    L_BUF.add(obs_l, next_obs_l, act_l, rew_l, done_l, infos=[{}])

                    # 高级智能体的更新条件：每 N 步 AND 缓冲区足够大
                    if H_BUF.size() > self.learn_starts and timestep % self.train_freq == 0:
                        self.high_level_agent.train(batch_size=self.H_BS, gradient_steps=1)

                    # 低级智能体的更新条件：只要缓冲区足够大
                    if L_BUF.size() > self.learn_starts:
                        self.low_level_agent.train(batch_size=self.L_BS, gradient_steps=1)

                    # 状态转移
                    state = next_state
                    timestep += 1
                    episode_timesteps += 1

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({"回合": episode_count, "当前奖励": f"{episode_reward:.2f}"})

                # 回合结束时更新高层经验的奖励值
                if done:
                    # 这里简化处理，实际应该找到该回合的所有经验并更新
                    # 为了简单，我们只更新最后一条经验的奖励
                    if H_BUF.size() > 0:
                        # 确保last_idx是有效的
                        last_idx = H_BUF.size() - 1
                        # 更新奖励为回合总奖励
                        H_BUF.rewards[last_idx] = episode_reward

                # ===== 低层：连续动作（把子目标拼进观测）=====
                sub_goal_state = np.append(state, [direction, distance])
                low_level_action = self.low_level_agent.predict(sub_goal_state, deterministic=False)[0]

                # 与 Gazebo 真正交互的一步
                next_state, reward, terminated, truncated, info = self.env.step(low_level_action)

                # 规整 done（terminated 或 truncated）
                if isinstance(terminated, np.ndarray):
                    terminated = bool(np.any(terminated))
                else:
                    terminated = bool(terminated)
                if isinstance(truncated, np.ndarray):
                    truncated = bool(np.any(truncated))
                else:
                    truncated = bool(truncated)
                done = terminated or truncated
                target = info.get('target_reached', False)

                # 奖励（沿用你已有的函数）
                high_level_reward, low_level_reward = self._calculate_rewards(
                    state, next_state, high_level_action, distance,
                    done, target, episode_timesteps, reward, info
                )

                # ====== 往“各自模型的内部 buffer”里 add ======
                # 高层（obs_dim）
                obs_h      = state.reshape(1, -1)
                next_obs_h = next_state.reshape(1, -1)
                act_h      = np.array([[high_level_action]], dtype=np.int64)
                rew_h      = np.array([high_level_reward], dtype=np.float32)
                done_h     = np.array([bool(done)], dtype=np.bool_)
                H_BUF.add(obs_h, next_obs_h, act_h, rew_h, done_h, infos=[{}])

                # 低层（obs_dim + 2）
                obs_l      = sub_goal_state.reshape(1, -1)
                next_obs_l = np.append(next_state, [direction, distance]).reshape(1, -1)
                act_l      = np.array(low_level_action, dtype=np.float32).reshape(1, -1)
                rew_l      = np.array([low_level_reward], dtype=np.float32)
                done_l     = np.array([bool(done)], dtype=np.bool_)
                L_BUF.add(obs_l, next_obs_l, act_l, rew_l, done_l, infos=[{}])

                # ====== 触发“纯梯度更新”而不采样环境 ======
                # 注意：train() 只会从 replay_buffer 采样，不会 reset/step 环境
                if H_BUF.size() >= max(self.H_BS, self.batch_train_size) and L_BUF.size() >= max(self.L_BS, self.batch_train_size):
                    # print(f"进行批量训练 - 高层经验: {H_BUF.size()}, 低层经验: {L_BUF.size()}")
                    # 先准备好智能体（补 logger、进度变量等）
                    self._prepare_sb3_train(self.high_level_agent)
                    self._prepare_sb3_train(self.low_level_agent)
                    # 再做纯梯度更新（不触发环境交互）
                    self.high_level_agent.train(gradient_steps=self.batch_train_size, batch_size=self.H_BS)
                    self.low_level_agent.train(gradient_steps=self.batch_train_size, batch_size=self.L_BS)

                # 滚动
                state = next_state
                episode_reward += reward
                episode_timesteps += 1
                timestep += 1
                pbar.update(1)
                pbar.set_postfix({"回合": episode_count, "当前奖励": f"{episode_reward:.2f}"})

                # 评估 + 定期手动保存（原先的 CheckpointCallback 依赖 learn）
                if timestep % self.eval_freq == 0:
                    eval_reward = self.evaluate()
                    evaluations.append(eval_reward)
                    np.save("./results/hierarchical_rl_evaluations.npy", evaluations)
                    self.high_level_agent.save(f"./pytorch_models/high_level/ckpt_{timestep}")
                    self.low_level_agent.save(f"./pytorch_models/low_level/ckpt_{timestep}")

            # ------- 回合结束后，用“剩余经验”再多做一些梯度步 -------
            if H_BUF.size() > 0 or L_BUF.size() > 0:
                print(f"回合结束，训练剩余经验 - 高层: {H_BUF.size()}, 低层: {L_BUF.size()}")
                
                # 仅当样本 >= batch_size 才训练，避免 SB3 采样报错
                if H_BUF.size() >= self.H_BS:
                    steps_h = max(1, H_BUF.size() // 2)
                    self._prepare_sb3_train(self.high_level_agent)
                    self.high_level_agent.train(gradient_steps=steps_h, batch_size=self.H_BS)

                if L_BUF.size() >= self.L_BS:
                    steps_l = max(1, L_BUF.size() // 2)
                    self._prepare_sb3_train(self.low_level_agent)
                    self.low_level_agent.train(gradient_steps=steps_l, batch_size=self.L_BS)

            # 日志
            if episode_count % log_interval == 0:
                print(f"回合 {episode_count} 完成，总步数: {timestep}, 奖励: {episode_reward:.2f}")

        # 最终保存
        self.high_level_agent.save("./pytorch_models/high_level/final_model")
        self.low_level_agent.save("./pytorch_models/low_level/final_model")
        print("训练完成!")

    @staticmethod
    def objective(trial: optuna.Trial) -> float:
        """Optuna超参数优化的目标函数

        该方法定义了超参数搜索空间，并返回模型在验证集上的性能指标，供Optuna
        优化器寻找最优超参数组合。超参数优化是提高强化学习性能的关键步骤，
        通过尝试不同的参数组合，可以找到最适合当前任务的设置。

        参数:
            trial: Optuna Trial对象，用于采样超参数

        返回:
            float: 模型在验证集上的平均奖励，作为优化目标
        """
        # 定义超参数搜索空间
        environment_dim = 20
        max_timesteps = 1e6  # 为了快速优化，使用较小的训练步数
        eval_freq = 1e4

        # 超参数搜索空间
        batch_train_size = trial.suggest_int("batch_train_size", 32, 256, step=32)
        high_level_lr = trial.suggest_float("high_level_lr", 1e-5, 1e-3, log=True)
        low_level_lr = trial.suggest_float("low_level_lr", 1e-5, 1e-3, log=True)
        gamma_high = trial.suggest_float("gamma_high", 0.9, 0.999)
        gamma_low = trial.suggest_float("gamma_low", 0.99, 0.99999)
        epsilon_decay = trial.suggest_int("epsilon_decay", 5e4, 2e5)
        noise_decay = trial.suggest_int("noise_decay", 5e4, 2e5)

        # 创建层级RL实例
        agent = HierarchicalRL(
            environment_dim=environment_dim,
            max_timesteps=max_timesteps,
            eval_freq=eval_freq,
            batch_train_size=batch_train_size,
            epsilon_decay=epsilon_decay,
            noise_decay=noise_decay
        )

        # 注意: 直接修改已初始化模型的参数不是最佳实践
        # 更好的方式是在创建模型时就设置这些参数
        # 这里为了兼容Optuna优化流程而保持当前实现
        agent.high_level_agent.learning_rate = high_level_lr
        agent.high_level_agent.gamma = gamma_high
        agent.low_level_agent.learning_rate = low_level_lr
        agent.low_level_agent.gamma = gamma_low

        # 训练一小部分步骤后评估
        agent.train()
        avg_reward = agent.evaluate(eval_episodes=5)

        return avg_reward

    @staticmethod
    def optimize_hyperparameters():
        """使用Optuna优化超参数

        该方法负责创建或加载Optuna研究，并执行超参数优化过程。超参数优化是
        强化学习中的重要环节，通过系统地搜索超参数空间，可以显著提高模型性能。
        该方法会保存最优参数供后续训练使用。

        原理:
            Optuna是一个自动超参数优化框架，它使用贝叶斯优化算法高效地探索
            超参数空间，找到最优参数组合。
        """
        print("开始超参数优化...")

        # 创建或加载研究
        try:
            study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage="sqlite:///optuna.db")
            print("加载已有研究")
        except:
            study = optuna.create_study(
                study_name=OPTUNA_STUDY_NAME,
                storage="sqlite:///optuna.db",
                direction="maximize"
            )
            print("创建新研究")

        # 优化目标函数
        study.optimize(HierarchicalRL.objective, n_trials=OPTUNA_N_TRIALS, n_jobs=1)

        # 打印最佳参数
        print("最佳参数:")
        print(study.best_params)
        print(f"最佳平均奖励: {study.best_value:.2f}")

        # 保存最佳参数
        os.makedirs("./configs", exist_ok=True)
        with open("./configs/best_hyperparams.txt", "w") as f:
            f.write(str(study.best_params))

        return study.best_params

    def evaluate(self, eval_episodes=10):
        """评估训练好的模型性能

        该方法用于评估训练好的层级强化学习模型在测试环境中的表现。评估指标
        包括平均奖励、成功率和碰撞率，这些指标反映了模型的导航能力和安全性。

        参数:
            eval_episodes: 评估的回合数

        返回:
            float: 平均奖励
        """
        print(f"开始评估，共 {eval_episodes} 个回合...")
        total_reward = 0
        total_steps = 0
        success_count = 0
        collision_count = 0
        # 确保prev_direction已定义
        self.prev_direction = 0.0

        for episode in range(eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            self.prev_direction = 0.0  # 重置前一方向

            while not done and episode_steps < self.max_ep:
                # 高层决策 (使用确定性策略)
                high_level_action = self.high_level_agent.predict(state, deterministic=True)[0]
                direction = high_level_action % 20
                distance = (high_level_action // 20) * 0.5 + 0.5

                # 低层执行 (使用确定性策略)
                sub_goal_state = np.append(state, [direction, distance])
                low_level_action = self.low_level_agent.predict(sub_goal_state, deterministic=True)[0]

                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(low_level_action)
                done = terminated or truncated

                # 更新状态
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

            total_reward += episode_reward

            # 统计成功和碰撞次数
            if 'success' in info and info['success']:
                success_count += 1
            if terminated:
                collision_count += 1

            print(f"评估回合 {episode+1}/{eval_episodes} - 奖励: {episode_reward:.2f}, 步数: {episode_steps}")

        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes
        success_rate = success_count / eval_episodes
        collision_rate = collision_count / eval_episodes

        print(f"评估完成 - 平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.1f}")
        print(f"成功率: {success_rate:.2%}, 碰撞率: {collision_rate:.2%}")

        return avg_reward
    
    def _cleanup_gazebo_ros(self):
        """
        自动清理：关闭 gym/env，杀掉残留的 ros/gazebo 进程，并清理共享内存文件。
        只在本脚本内部调用，不改外部工程。
        """
        # 1) 先尝试优雅关闭 wrapper/env（如果实现了）
        try:
            if hasattr(self, "env") and hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass
        # 也尝试关掉 agent 里可能持有的 env
        try:
            if hasattr(self, "high_level_agent") and hasattr(self.high_level_agent, "env") and self.high_level_agent.env is not None:
                self.high_level_agent.env.close()
        except Exception:
            pass
        try:
            if hasattr(self, "low_level_agent") and hasattr(self.low_level_agent, "env") and self.low_level_agent.env is not None:
                self.low_level_agent.env.close()
        except Exception:
            pass

        # 2) 兜底：干掉常见的 ros/gazebo 进程（只针对本机用户，-f 以命令行匹配）
        os.system("pkill -9 -f 'gazebo_ros/gzserver|gzserver -e ode|roslaunch|rosmaster|rosout|gzclient' 2>/dev/null || true")

        # 3) 清理 Gazebo 共享内存/锁文件，避免下次启动 255
        os.system("rm -f /dev/shm/gazebo-* /tmp/gazebo* 2>/dev/null || true")



def main(optimize=False):
    """程序主入口，支持训练和超参数优化

    该函数是程序的主入口，负责根据参数决定执行训练或超参数优化任务。
    如果启用超参数优化，它会先调用Optuna寻找最佳参数，然后使用这些参数
    初始化模型并开始训练；否则，直接使用默认参数初始化模型并训练。

    参数:
        optimize (bool): 是否进行超参数优化

    功能流程:
        1. 根据optimize参数决定是否进行超参数优化
        2. 如果进行优化，调用HierarchicalRL.optimize_hyperparameters()获取最佳参数
        3. 过滤掉与HierarchicalRL构造函数不兼容的参数
        4. 使用参数初始化HierarchicalRL实例
        5. 调用train()方法开始训练
    """
    if optimize:
        # 进行超参数优化
        best_params = HierarchicalRL.optimize_hyperparameters()
        print("超参数优化完成，开始使用最佳参数训练...")
        # 过滤掉与HierarchicalRL构造函数不兼容的参数
        valid_params = {k: v for k, v in best_params.items()
                        if k in HierarchicalRL.__init__.__code__.co_varnames}
        # 使用最佳参数创建实例并训练
        hierarchical_agent = HierarchicalRL(
            environment_dim=20,
            max_timesteps=5e6,
            **valid_params
        )
    else:
        # 直接训练
        hierarchical_agent = HierarchicalRL(environment_dim=20, max_timesteps=5e6)

    # 开始训练
    # —— 安装信号处理：Ctrl+C / kill 都会先清理再退出
    def _sig_handler(sig, frame):
        try:
            hierarchical_agent._cleanup_gazebo_ros()
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # —— 开始训练（确保无论如何都会清理一次）
    try:
        hierarchical_agent.train()
    finally:
        hierarchical_agent._cleanup_gazebo_ros()


if __name__ == "__main__":
    """程序入口点

    当脚本直接运行时，这部分代码会被执行。它负责解析命令行参数，并调用main
    函数开始执行相应的任务。
    """
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="层级强化学习训练")
    # 添加--optimize参数，用于控制是否进行超参数优化
    parser.add_argument("--optimize", action="store_true", help="是否进行超参数优化")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用main函数，传入optimize参数
    main(optimize=args.optimize)