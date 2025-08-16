import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
from velodyne_env import GazeboEnv

class VelodyneGymWrapper(gym.Env):
    """
    将VelodyneEnv包装成Gym环境, 方便使用Gym的API。
    支持CUDA检测，但返回numpy数组以确保与stable-baselines3兼容。
    """
    def __init__(self, launchfile, environment_dim, action_type="continuous", device=None):
        super().__init__()

        # 检测CUDA设备（仅用于信息显示，不强制转换数据）
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✅ 环境检测到CUDA可用，模型将使用GPU加速")
            else:
                device = "cpu"
                print(f"⚠️  CUDA不可用，模型将使用CPU")
        
        self.device = device
        print(f"🎯 环境设备检测: {self.device} (观测数据保持numpy格式)")

        # 创建GazeboEnv实例
        self.gazebo_env = GazeboEnv(launchfile, environment_dim)

        # 定义动作空间
        self.action_type = action_type # 动作类型，连续或离散
        if action_type == "discrete":
            # 离散动作空间（如DQN算法），线速度和角速度的离散值。
            # 参照了train_dqn.py中的常量。
            self.V_SET = [0.0, 0.3, 0.6, 0.9]               # 线速度 ∈ [0,1]
            self.W_SET = [-1.0, -0.5, 0.0, 0.5, 1.0]        # 角速度 ∈ [-1,1]

            # 动作数量：4个线速度 × 5个角速度 = 20个动作
            self.action_space = spaces.Discrete(len(self.V_SET) * len(self.W_SET))
            self.action_mapping = {} # 动作索引到连续值的映射
            action_idx = 0
            for v in self.V_SET:
                for w in self.W_SET:
                    self.action_mapping[action_idx] = [v, w]
                    action_idx += 1
        else:
            # 连续动作空间(如PPO算法)。参考train_velodyne_td3.py的设计，线速度范围改为[0,1]（因为差分驱动机器人不能后退？）
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0]), # 线速度范围
                high=np.array([1.0, 1.0]), # 角速度范围
                dtype=np.float32
            )

        # 定义观测空间，参考velodyne_env.py的实际使用。
        # 激光雷达数据：从check_pos看环境是±4.5米，对角线约6.4米
        laser_low = 0.0      # 最小距离（碰撞距离0.35米）
        laser_high = 10.0    # 最大距离（velodyne_env.py中初始化为10）
        # 距离范围：基于环境实际大小
        distance_low = 0.0      # 最小距离
        distance_high = 7.0     # 最大距离（考虑对角线距离）
        # 角度范围：-π 到 π
        angle_low = -np.pi
        angle_high = np.pi
        # 速度范围
        vel_low = 0.0      # 线速度最小值
        vel_high = 1.0     # 线速度最大值
        ang_vel_low = -1.0 # 角速度最小值
        ang_vel_high = 1.0 # 角速度最大值
        
        self.observation_space = spaces.Box(
            low=np.array([laser_low] * environment_dim + [distance_low, angle_low, vel_low, ang_vel_low]),
            high=np.array([laser_high] * environment_dim + [distance_high, angle_high, vel_high, ang_vel_high]),
            dtype=np.float32
        )

    def _ensure_numpy(self, data):
        """
        确保数据是numpy数组格式，与stable-baselines3兼容
        """
        if isinstance(data, torch.Tensor):
            # 如果是tensor，转换为numpy数组
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            # 如果已经是numpy数组，直接返回
            return data
        else:
            # 其他类型，尝试转换为numpy数组
            return np.array(data, dtype=np.float32)

    def step(self, action):
        if self.action_type == "discrete":
            action = self.action_mapping[action] # 将离散动作索引转换为连续动作值
        # 调用GazeboEnv的step方法。返回state，reward，done，target
        state, reward, done, target = self.gazebo_env.step(action)

        # 确保观测数据是numpy数组格式，与stable-baselines3兼容
        state = self._ensure_numpy(state)

        # 构建info字典，包含额外信息
        info = {
            'target_reached': target,  # 是否到达目标
            'device': self.device,     # 添加设备信息
            # 有必要的话再添加新的调试信息！
        }

        # gymnasium要求step()方法返回5个值：(observation, reward, terminated, truncated, info)
        # terminated: episode自然结束（到达目标或碰撞）
        # truncated: episode被外部因素中断（达到最大步数等）
        terminated = done  # 我们的环境使用done表示episode结束
        truncated = False  # 目前没有外部中断机制
        
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # 如果提供了seed参数，设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        state = self.gazebo_env.reset()
        
        # 确保观测数据是numpy数组格式，与stable-baselines3兼容
        state = self._ensure_numpy(state)
        
        # gymnasium要求reset()方法返回(observation, info)元组
        info = {'device': self.device}
        return state, info

    def render(self, mode='human'):
        if mode == 'human':
            current_action = [0.0, 0.0]  # 默认动作（初始化为停止）
            self.gazebo_env.publish_markers(current_action)
        else:
            print(f"Warning: {mode} mode not supported, only 'human' mode available")

    def close(self):
        try:
            # 停止机器人运动
            stop_action = [0.0, 0.0]  # 停止线速度和角速度
            self.gazebo_env.step(stop_action)
            
            # 关闭ROS节点（如果已初始化）
            import rospy # 在需要时再导入,因为ROS可能在__init__中还没有启动完成。
            if rospy.core.is_initialized():
                rospy.signal_shutdown("Environment closed by user")
                print("ROS node shutdown initiated")
            
            print("Environment cleanup completed")
            
        except Exception as e:
            print(f"Warning: Error during environment cleanup: {e}")
    