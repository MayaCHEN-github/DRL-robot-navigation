import gym
from gym import spaces
import numpy as np
from velodyne_env import GazeboEnv

class VelodyneGymWrapper(gym.Env):
    """
    将VelodyneEnv包装成Gym环境, 方便使用Gym的API。
    """
    def __init__(self, launchfile, environment_dim, action_type="continuous"):
        super().__init__()

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

    
    def step(self, action):
        if self.action_type == "discrete":
            action = self.action_mapping[action] # 将离散动作索引转换为连续动作值
        # 调用GazeboEnv的step方法。返回state，reward，done，target
        state, reward, done, target = self.gazebo_env.step(action)

        # 构建info字典，包含额外信息
        info = {
            'target_reached': target,  # 是否到达目标
            # 有必要的话再添加新的调试信息！
        }

        return state, reward, done, info


    def reset(self):
        state = self.gazebo_env.reset()
        return state

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
    