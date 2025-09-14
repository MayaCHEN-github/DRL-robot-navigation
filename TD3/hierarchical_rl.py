# 导入必要的库
import os  # 用于文件和目录操作
import time  # 用于时间戳
import numpy as np  
import torch  
import optuna  # 用于超参数优化
from stable_baselines3 import DQN, TD3  
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback  # 从sb3导入的回调

# 自定义日志回调类
class LoggingCallback(BaseCallback):
    def __init__(self, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"步数: {self.n_calls}, 完成评估并记录日志")
        return True
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
    def __init__(self, environment_dim=20, max_timesteps=2e6, eval_freq=5000, device=None, batch_train_size=100,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1e5,
                 noise_start=0.2, noise_end=0.01, noise_decay=1e5):
        # 移除了与PrioritizedReplayBuffer相关的参数，因为该类不可用
        use_per = False
        """初始化HRL Agent"""
        # CUDA使用设置
        self.device = torch.device("cpu")  # 强制使用CPU，不使用GPU
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

        # 添加模型保存回调，每eval_freq步保存一次
        self.high_level_checkpoint_callback = CheckpointCallback(
            save_freq=self.eval_freq,
            save_path="./pytorch_models/high_level/",
            name_prefix="high_level_dqn",
        )

        # 添加日志记录回调，每eval_freq步记录一次详细日志
        self.high_level_log_callback = LoggingCallback(eval_freq=self.eval_freq, verbose=1)

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
            batch_size=40,  # 与velodyne版本一致
            tau=0.005,  # 目标网络更新系数
            gamma=0.99999,  # 与velodyne版本一致
            train_freq=1,  # 训练频率
            gradient_steps=1,  # 每次训练的梯度步数
            policy_delay=2,  # 策略更新延迟
            target_policy_noise=0.2,  # 目标策略噪声
            target_noise_clip=0.5,  # 噪声裁剪
            device=self.device  # 计算设备
        )

        # 添加模型保存回调，每eval_freq步保存一次
        self.low_level_checkpoint_callback = CheckpointCallback(
            save_freq=self.eval_freq,
            save_path="./pytorch_models/low_level/",
            name_prefix="low_level_td3",
        )

        # 添加日志记录回调，每eval_freq步记录一次详细日志
        self.low_level_log_callback = LoggingCallback(eval_freq=self.eval_freq, verbose=1)

        print("低层TD3智能体初始化完成!")
        return model

    def _create_directories(self):
        """创建模型和结果存储目录

        创建用于存储训练结果、模型权重的目录结构，确保训练过程中生成的文件有合适的存放位置。如果目录已存在，则不会重复创建。
        """
        directories = [
            "./results", 
            "./pytorch_models/high_level", 
            "./pytorch_models/low_level",
            "./logs/high_level_dqn",
            "./logs/low_level_td3"
        ]
        for dir_path in directories:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录: {dir_path}")

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
            # 确定TensorBoard日志路径
            if hasattr(agent, 'tensorboard_log') and agent.tensorboard_log:
                log_folder = agent.tensorboard_log
            else:
                log_folder = "./logs/default"
            
            # 确保日志目录存在
            os.makedirs(log_folder, exist_ok=True)
            agent._logger = configure(folder=log_folder, format_strings=["stdout", "tensorboard"])

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
        
        # 确保所有必要的目录都存在
        self._create_directories()

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
        print(f"训练设置:")
        print(f"  - 总时间步数: {int(self.max_timesteps):,}")
        print(f"  - 总回合数: {total_episodes}")
        print(f"  - 保存频率: 每 {self.eval_freq} 步保存一次")
        print(f"  - 预计保存次数: {int(self.max_timesteps / self.eval_freq)} 次")
        print(f"  - 调试信息: max_timesteps={self.max_timesteps}, eval_freq={self.eval_freq}")

        # 创建进度条
        with tqdm(total=self.max_timesteps, desc="训练进度", position=1, leave=True, dynamic_ncols=True) as pbar:
            while timestep < self.max_timesteps:
                # 新回合开始
                episode_count += 1
                reset_result = self.env.reset()
                state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                # 确保 state 是 numpy array
                if isinstance(state, tuple):
                    state = state[0]
                state = np.array(state, dtype=np.float32)
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
                    # 确保 next_state 是 numpy array
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
                    next_state = np.array(next_state, dtype=np.float32)
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
                        # 记录训练日志到TensorBoard
                        self._log_training_metrics_to_tensorboard("high_level", timestep, episode_reward)

                    # 低级智能体的更新条件：只要缓冲区足够大
                    if L_BUF.size() > self.learn_starts:
                        self.low_level_agent.train(batch_size=self.L_BS, gradient_steps=1)
                        # 记录训练日志到TensorBoard
                        self._log_training_metrics_to_tensorboard("low_level", timestep, episode_reward)

                    # 状态转移
                    state = next_state
                    timestep += 1
                    episode_timesteps += 1

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({"回合": episode_count, "当前奖励": f"{episode_reward:.2f}"})

                    # 定期记录训练指标到TensorBoard（每100步记录一次）
                    if timestep % 100 == 0:
                        self._log_training_metrics_to_tensorboard("high_level", timestep, episode_reward)
                        self._log_training_metrics_to_tensorboard("low_level", timestep, episode_reward)

                    # 评估 + 定期手动保存
                    if timestep % self.eval_freq == 0:
                        print(f"\n=== 触发评估和保存 (timestep={timestep}, eval_freq={self.eval_freq}) ===")
                        try:
                            # 根据训练进度动态调整评估回合数
                            progress = timestep / self.max_timesteps
                            if progress < 0.1:  # 前10%的训练
                                eval_episodes = 3
                            elif progress < 0.5:  # 10%-50%的训练
                                eval_episodes = 5
                            else:  # 50%以后的训练
                                eval_episodes = 10
                            
                            print(f"评估回合数: {eval_episodes} (训练进度: {progress:.1%})")
                            try:
                                # 使用详细评估
                                detailed_summary = self.evaluate_detailed(eval_episodes=eval_episodes)
                                # 也进行基础评估以保持向后兼容
                                eval_reward = self.evaluate(eval_episodes=eval_episodes)
                                evaluations.append(eval_reward)
                                np.save("./results/hierarchical_rl_evaluations.npy", evaluations)
                            except Exception as e:
                                print(f"评估过程中出现错误: {e}")
                                print("跳过本次评估，继续训练...")
                                # 使用默认值
                                eval_reward = 0.0
                                evaluations.append(eval_reward)
                            # 确保目录存在
                            os.makedirs("./pytorch_models/high_level", exist_ok=True)
                            os.makedirs("./pytorch_models/low_level", exist_ok=True)
                            # 保存模型（需要完整的文件路径）
                            self.high_level_agent.save(f"./pytorch_models/high_level/ckpt_{timestep}")
                            self.low_level_agent.save(f"./pytorch_models/low_level/ckpt_{timestep}")
                            print(f"已保存模型检查点: timestep={timestep}")
                        except Exception as e:
                            print(f"评估或保存过程中出现错误: {e}")
                            # 即使评估失败，也继续训练

                # 回合结束时更新高层经验的奖励值
                if done:
                    # 这里简化处理，实际应该找到该回合的所有经验并更新
                    # 为了简单，我们只更新最后一条经验的奖励
                    if H_BUF.size() > 0:
                        # 确保last_idx是有效的
                        last_idx = H_BUF.size() - 1
                        # 更新奖励为回合总奖励
                        H_BUF.rewards[last_idx] = episode_reward

                    # ------- 回合结束后，用"剩余经验"再多做一些梯度步 -------
                    if H_BUF.size() > 0 or L_BUF.size() > 0:
                        # 仅当样本 >= batch_size 才训练，避免 SB3 采样报错
                        if H_BUF.size() >= self.H_BS:
                            steps_h = max(1, H_BUF.size() // 2)
                            self._prepare_sb3_train(self.high_level_agent)
                            self.high_level_agent.train(gradient_steps=steps_h, batch_size=self.H_BS)

                        if L_BUF.size() >= self.L_BS:
                            steps_l = max(1, L_BUF.size() // 2)
                            self._prepare_sb3_train(self.low_level_agent)
                            self.low_level_agent.train(gradient_steps=steps_l, batch_size=self.L_BS)


        # 最终保存
        os.makedirs("./pytorch_models/high_level", exist_ok=True)
        os.makedirs("./pytorch_models/low_level", exist_ok=True)
        self.high_level_agent.save("./pytorch_models/high_level/final_model")
        self.low_level_agent.save("./pytorch_models/low_level/final_model")
        print("训练完成! 最终模型已保存。")

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
        eval_freq = 1000

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
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            # 确保 state 是 numpy array
            if isinstance(state, tuple):
                state = state[0]
            state = np.array(state, dtype=np.float32)
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

                # 更新状态 - 确保 next_state 是 numpy array
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state = np.array(next_state, dtype=np.float32)
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

        # 保存详细评估结果到文件
        os.makedirs("./results", exist_ok=True)
        evaluation_data = {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'episodes': eval_episodes
        }
        np.save("./results/evaluation_metrics.npy", evaluation_data)

        return avg_reward
    
    def evaluate_detailed(self, eval_episodes=10):
        """使用现有环境进行详细评估（避免环境冲突）
        
        该方法使用当前训练环境进行详细评估，避免启动新的Gazebo环境。
        包括：
        - 成功率 (success_rate)
        - 路径效率 (path_efficiency) 
        - 轨迹平滑度 (trajectory_smoothness)
        - 时间成本 (time_cost)
        - 碰撞率 (collision_rate)
        
        参数:
            eval_episodes: 评估的回合数
            
        返回:
            dict: 详细评估结果
        """
        print(f"开始详细评估，共 {eval_episodes} 个回合...")
        
        # 使用现有环境进行评估，避免环境冲突
        episode_metrics = []
        success_episodes = 0
        collision_episodes = 0
        total_time_cost = 0.0
        total_path_efficiency = 0.0
        total_trajectory_smoothness = 0.0
        
        for episode in range(eval_episodes):
            print(f"评估回合 {episode+1}/{eval_episodes}")
            
            # 重置环境
            reset_result = self.env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            if isinstance(state, tuple):
                state = state[0]
            state = np.array(state, dtype=np.float32)
            
            # 初始化评估指标
            start_time = time.time()
            trajectory = []
            total_distance = 0.0
            episode_collision = False
            
            # 获取起始位置和目标位置
            if hasattr(self.env, 'gazebo_env'):
                start_x = self.env.gazebo_env.odom_x
                start_y = self.env.gazebo_env.odom_y
                goal_x = self.env.gazebo_env.goal_x
                goal_y = self.env.gazebo_env.goal_y
                trajectory.append((start_x, start_y))
                
                straight_line_distance = np.linalg.norm([goal_x - start_x, goal_y - start_y])
            else:
                # 如果无法获取位置信息，使用默认值
                start_x = start_y = goal_x = goal_y = 0.0
                trajectory.append((start_x, start_y))
                straight_line_distance = 1.0
            
            done = False
            episode_steps = 0
            self.prev_direction = 0.0
            
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
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state = np.array(next_state, dtype=np.float32)
                
                # 记录轨迹
                if hasattr(self.env, 'gazebo_env'):
                    current_x = self.env.gazebo_env.odom_x
                    current_y = self.env.gazebo_env.odom_y
                    trajectory.append((current_x, current_y))
                    
                    # 计算步长
                    if len(trajectory) > 1:
                        prev_x, prev_y = trajectory[-2]
                        step_distance = np.linalg.norm([current_x - prev_x, current_y - prev_y])
                        total_distance += step_distance
                
                # 检查碰撞
                if reward < -90 and not episode_collision:
                    episode_collision = True
                
                episode_steps += 1
            
            # 计算评估指标
            time_cost = time.time() - start_time
            total_time_cost += time_cost
            
            # 计算成功率
            if hasattr(self.env, 'gazebo_env'):
                final_x = self.env.gazebo_env.odom_x
                final_y = self.env.gazebo_env.odom_y
                final_distance = np.linalg.norm([goal_x - final_x, goal_y - final_y])
                success = 1.0 if final_distance < 0.3 else 0.0
            else:
                success = 0.0
            
            if success >= 0.5:
                success_episodes += 1
            
            if episode_collision:
                collision_episodes += 1
            
            # 计算路径效率（只有成功到达目标时才计算，失败时设为0）
            if success >= 0.5:  # 成功到达目标
                if total_distance > 0.1:  # 确保有足够的移动距离
                    path_efficiency = min(1.0, straight_line_distance / total_distance)
                else:
                    path_efficiency = 1.0  # 成功但移动距离很小，认为效率很高
            else:  # 未成功到达目标
                path_efficiency = 0.0  # 失败时路径效率为0
            
            total_path_efficiency += path_efficiency
            
            # 计算轨迹平滑度
            smoothness = self._calculate_trajectory_smoothness(trajectory)
            total_trajectory_smoothness += smoothness
            
            # 保存回合指标（包含调试数据）
            episode_metrics.append({
                'success_rate': success,
                'time_cost': time_cost,
                'path_efficiency': path_efficiency,
                'trajectory_smoothness': smoothness,
                # 调试数据
                'debug_straight_line_distance': straight_line_distance,
                'debug_total_distance': total_distance,
                'debug_actual_distance': np.linalg.norm([final_x - start_x, final_y - start_y]) if hasattr(self.env, 'gazebo_env') else 0.0,
                'debug_start_pos': (start_x, start_y),
                'debug_final_pos': (final_x, final_y),
                'debug_goal_pos': (goal_x, goal_y),
                'debug_trajectory_length': len(trajectory),
                'debug_episode_steps': episode_steps
            })
        
        # 计算平均指标
        n = len(episode_metrics) if episode_metrics else 1
        summary = {
            'episodes': n,
            'success_rate': success_episodes / n,
            'collision_rate': collision_episodes / n,
            'avg_time_cost': total_time_cost / n,
            'avg_path_efficiency': total_path_efficiency / n,
            'avg_trajectory_smoothness': total_trajectory_smoothness / n,
            # 调试数据
            'avg_straight_line_distance': np.mean([m['debug_straight_line_distance'] for m in episode_metrics]),
            'avg_total_distance': np.mean([m['debug_total_distance'] for m in episode_metrics]),
            'avg_actual_distance': np.mean([m['debug_actual_distance'] for m in episode_metrics]),
            'avg_trajectory_length': np.mean([m['debug_trajectory_length'] for m in episode_metrics]),
            'avg_episode_steps': np.mean([m['debug_episode_steps'] for m in episode_metrics]),
        }
        
        # 打印详细评估结果
        print(f"详细评估完成:")
        print(f"  成功率: {summary['success_rate']:.2%}")
        print(f"  碰撞率: {summary['collision_rate']:.2%}")
        print(f"  平均时间成本: {summary['avg_time_cost']:.2f}秒")
        print(f"  平均路径效率: {summary['avg_path_efficiency']:.2%}")
        print(f"  平均轨迹平滑度: {summary['avg_trajectory_smoothness']:.2%}")
        print(f"📊 调试数据:")
        print(f"  平均直线距离: {summary['avg_straight_line_distance']:.2f}m")
        print(f"  平均总距离: {summary['avg_total_distance']:.2f}m")
        print(f"  平均实际距离: {summary['avg_actual_distance']:.2f}m")
        print(f"  平均轨迹点数: {summary['avg_trajectory_length']:.1f}")
        print(f"  平均回合步数: {summary['avg_episode_steps']:.1f}")
        print(f"💡 说明:")
        print(f"  - 路径效率: 成功回合的路径效率，失败回合为0%")
        print(f"  - 效率比值: {summary['avg_straight_line_distance']:.2f} / {summary['avg_total_distance']:.2f} = {summary['avg_straight_line_distance']/summary['avg_total_distance']:.3f}")
        
        # 保存详细评估结果
        os.makedirs("./results", exist_ok=True)
        detailed_evaluation_data = {
            'summary': summary,
            'episode_metrics': episode_metrics,
            'timestamp': time.time()
        }
        np.save("./results/detailed_evaluation_metrics.npy", detailed_evaluation_data)
        
        # 记录到TensorBoard
        self._log_evaluation_to_tensorboard(summary)
        
        return summary
    
    def _calculate_trajectory_smoothness(self, trajectory):
        """计算轨迹平滑度（基于轨迹点的曲率变化）"""
        if len(trajectory) < 3:
            return 1.0  # 轨迹点太少，认为平滑
        
        # 计算轨迹的曲率变化
        curvature_changes = []
        for i in range(1, len(trajectory) - 1):
            p1 = np.array(trajectory[i-1])
            p2 = np.array(trajectory[i])
            p3 = np.array(trajectory[i+1])
            
            # 计算两个向量的夹角
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
                angle = np.arccos(cos_angle)
                curvature_changes.append(angle)
        
        if not curvature_changes:
            return 1.0
        
        # 平滑度 = 1 / (1 + 平均曲率变化)
        avg_curvature = np.mean(curvature_changes)
        smoothness = 1.0 / (1.0 + avg_curvature)
        return smoothness
    
    def _cleanup_environment(self):
        """清理环境，解决冲突问题"""
        try:
            print("正在清理环境...")
            
            # 1. 关闭当前环境
            if hasattr(self, 'env') and hasattr(self.env, 'close'):
                self.env.close()
            
            # 2. 清理ROS进程
            os.system("pkill -9 -f 'gazebo_ros/gzserver|gzserver|roslaunch|rosmaster|rosout|gzclient' 2>/dev/null || true")
            
            # 3. 清理Gazebo共享内存
            os.system("rm -f /dev/shm/gazebo-* /tmp/gazebo* 2>/dev/null || true")
            
            # 4. 等待进程完全结束
            import time
            time.sleep(2)
            
            print("环境清理完成")
            
        except Exception as e:
            print(f"环境清理过程中出现错误: {e}")
    
    def _log_training_metrics_to_tensorboard(self, agent_type, timestep, episode_reward=None):
        """记录训练指标到TensorBoard"""
        try:
            if agent_type == "high_level":
                agent = self.high_level_agent
                prefix = "high_level"
            else:
                agent = self.low_level_agent
                prefix = "low_level"
            
            if hasattr(agent, '_logger') and agent._logger is not None:
                # 记录奖励（如果提供）
                if episode_reward is not None:
                    agent._logger.record(f"{prefix}/reward", episode_reward)
                
                # 记录经验缓冲区大小
                if hasattr(agent, 'replay_buffer'):
                    agent._logger.record(f"{prefix}/buffer_size", agent.replay_buffer.size())
                
                # 记录学习率
                if hasattr(agent, 'learning_rate'):
                    agent._logger.record(f"{prefix}/learning_rate", agent.learning_rate)
                
                # 记录探索参数
                if agent_type == "high_level" and hasattr(self, 'current_epsilon'):
                    agent._logger.record(f"{prefix}/epsilon", self.current_epsilon)
                elif agent_type == "low_level" and hasattr(self, 'current_noise'):
                    agent._logger.record(f"{prefix}/noise", self.current_noise)
                
                # 记录更新次数
                if hasattr(agent, '_n_updates'):
                    agent._logger.record(f"{prefix}/n_updates", agent._n_updates)
                
                # 记录到TensorBoard
                agent._logger.dump(step=timestep)
                
        except Exception as e:
            print(f"记录训练指标到TensorBoard时出错: {e}")
    
    def _log_evaluation_to_tensorboard(self, summary):
        """将评估指标记录到TensorBoard"""
        try:
            # 为高层和低层智能体都记录评估指标
            for agent_name, agent in [("high_level", self.high_level_agent), ("low_level", self.low_level_agent)]:
                if hasattr(agent, '_logger') and agent._logger is not None:
                    # 记录评估指标
                    agent._logger.record("evaluation/success_rate", summary['success_rate'])
                    agent._logger.record("evaluation/collision_rate", summary['collision_rate'])
                    agent._logger.record("evaluation/avg_time_cost", summary['avg_time_cost'])
                    agent._logger.record("evaluation/avg_path_efficiency", summary['avg_path_efficiency'])
                    agent._logger.record("evaluation/avg_trajectory_smoothness", summary['avg_trajectory_smoothness'])
                    agent._logger.record("evaluation/episodes", summary['episodes'])
                    
                    # 记录调试数据
                    agent._logger.record("debug/avg_straight_line_distance", summary['avg_straight_line_distance'])
                    agent._logger.record("debug/avg_total_distance", summary['avg_total_distance'])
                    agent._logger.record("debug/avg_actual_distance", summary['avg_actual_distance'])
                    agent._logger.record("debug/avg_trajectory_length", summary['avg_trajectory_length'])
                    agent._logger.record("debug/avg_episode_steps", summary['avg_episode_steps'])
                    agent._logger.record("debug/efficiency_ratio", summary['avg_straight_line_distance']/summary['avg_total_distance'])
                    
                    # 记录到TensorBoard
                    agent._logger.dump(step=agent.num_timesteps)
        except Exception as e:
            print(f"记录评估指标到TensorBoard时出错: {e}")
    
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
            max_timesteps=2e6,
            **valid_params
        )
    else:
        # 直接训练
        hierarchical_agent = HierarchicalRL(environment_dim=20, max_timesteps=2e6)

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