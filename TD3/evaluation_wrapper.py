import numpy as np
from gym_wrapper import VelodyneGymWrapper
from velodyne_env import GazeboEnv

class EvaluationWrapper(VelodyneGymWrapper):
    """
    继承VelodyneGymWrapper，添加评估功能
    用于计算导航效率和性能指标
    """
    
    def __init__(self, launchfile, environment_dim, action_type="continuous"):
        super().__init__(launchfile, environment_dim, action_type)
        self.reset_evaluation_metrics() # 初始化metrics
    
    def step(self, action):
        state, reward, done, info = super().step(action)
        self._update_evaluation_metrics() # 更新metrics
        
        return state, reward, done, info
    
    def reset(self):
        state = super().reset() # 调用父类的reset方法
        self.reset_evaluation_metrics() # 重置metrics
        self._initialize_evaluation_metrics() # 获取起始位置和目标位置
        return state
    
    def _initialize_evaluation_metrics(self):
        """初始化评估指标，私有方法"""
        # 获取起始位置（机器人当前位置）
        self.start_x = self.gazebo_env.odom_x
        self.start_y = self.gazebo_env.odom_y
        self.current_x = self.start_x
        self.current_y = self.start_y
        # 获取目标位置
        self.goal_x = self.gazebo_env.goal_x
        self.goal_y = self.gazebo_env.goal_y
        # 计算直线距离
        self.straight_line_distance = np.linalg.norm([
            self.goal_x - self.start_x,
            self.goal_y - self.start_y
        ])
        # 初始化轨迹
        self.trajectory = [(self.start_x, self.start_y)]
    
    def _update_evaluation_metrics(self):
        """更新评估指标，私有方法"""
        # 获取当前位置
        self.current_x = self.gazebo_env.odom_x
        self.current_y = self.gazebo_env.odom_y
        # 添加到轨迹
        self.trajectory.append((self.current_x, self.current_y))
        # 计算这一步移动的距离（累计总路程，欧几里得距离）
        if len(self.trajectory) > 1:
            prev_x, prev_y = self.trajectory[-2]
            step_distance = np.linalg.norm([
                self.current_x - prev_x,
                self.current_y - prev_y
            ])
            self.total_distance_traveled += step_distance

    def reset_evaluation_metrics(self):
        """重置评估指标，公有方法（因为可能会在外部调用？）"""
        # 机器人起始位置
        self.start_x = None
        self.start_y = None
        # 目标位置
        self.goal_x = None
        self.goal_y = None
        # 机器人当前位置
        self.current_x = None
        self.current_y = None
        # 运动轨迹
        self.trajectory = []
        # 总路程
        self.total_distance_traveled = 0.0
        # 直线距离（起始点到目标点）
        self.straight_line_distance = 0.0
    
    def get_evaluation_metrics(self):
        """获取评估指标"""
        # 安全检查
        if self.start_x is None or self.goal_x is None or self.current_x is None:
            print("Warning: Evaluation metrics not initialized yet")
            return None
        # 计算当前到目标的距离
        current_to_goal_distance = np.linalg.norm([
            self.goal_x - self.current_x,
            self.goal_y - self.current_y
        ])
        # 计算效率比值（直线距离 / 总路程）
        efficiency_ratio = 0.0
        if self.total_distance_traveled > 0:
            efficiency_ratio = self.straight_line_distance / self.total_distance_traveled
        else:
            efficiency_ratio = 1.0  # 如果还没移动，效率为100%（虽然不一定有意义……）
        
        return {
            'start_position': (self.start_x, self.start_y),
            'goal_position': (self.goal_x, self.goal_y),
            'current_position': (self.current_x, self.current_y),
            'straight_line_distance': self.straight_line_distance,
            'total_distance_traveled': self.total_distance_traveled,
            'current_to_goal_distance': current_to_goal_distance,
            'efficiency_ratio': efficiency_ratio,
            'trajectory_length': len(self.trajectory)
        }
    
    def print_evaluation_summary(self):
        """打印评估摘要"""
        metrics = self.get_evaluation_metrics()
        
        print("=" * 50)
        print("EVALUATION SUMMARY 评估摘要")
        print("=" * 50)
        print(f"📍 起始位置: ({metrics['start_position'][0]:.2f}, {metrics['start_position'][1]:.2f})")
        print(f"🎯 目标位置:  ({metrics['goal_position'][0]:.2f}, {metrics['goal_position'][1]:.2f})")
        print(f"🤖 当前位置: ({metrics['current_position'][0]:.2f}, {metrics['current_position'][1]:.2f})")
        print("-" * 50)
        print(f"📏 直线距离: {metrics['straight_line_distance']:.2f} m")
        print(f"🛤️ 总路程: {metrics['total_distance_traveled']:.2f} m")
        print(f"📐 当前到目标距离: {metrics['current_to_goal_distance']:.2f} m")
        print(f"💡 效率比值: {metrics['efficiency_ratio']:.3f}")
        print(f"📊 轨迹点数: {metrics['trajectory_length']}")
        print("=" * 50)

class GazeboEvaluationWrapper:
    """为GazeboEnv添加评估功能的包装器"""
    
    def __init__(self, gazebo_env):
        self.gazebo_env = gazebo_env
        self.reset_evaluation_metrics()
    
    def step(self, action):
        state, reward, done, info = self.gazebo_env.step(action)
        self._update_evaluation_metrics()
        return state, reward, done, info
    
    def reset(self):
        state = self.gazebo_env.reset()
        self.reset_evaluation_metrics()
        self._initialize_evaluation_metrics()
        return state
    
    def _initialize_evaluation_metrics(self):
        """初始化评估指标"""
        self.start_x = self.gazebo_env.odom_x
        self.start_y = self.gazebo_env.odom_y
        self.current_x = self.start_x
        self.current_y = self.start_y
        self.goal_x = self.gazebo_env.goal_x
        self.goal_y = self.gazebo_env.goal_y
        self.straight_line_distance = np.linalg.norm([
            self.goal_x - self.start_x,
            self.goal_y - self.start_y
        ])
        self.trajectory = [(self.start_x, self.start_y)]
    
    def _update_evaluation_metrics(self):
        """更新评估指标"""
        self.current_x = self.gazebo_env.odom_x
        self.current_y = self.gazebo_env.odom_y
        self.trajectory.append((self.current_x, self.current_y))
        
        if len(self.trajectory) > 1:
            prev_x, prev_y = self.trajectory[-2]
            step_distance = np.linalg.norm([
                self.current_x - prev_x,
                self.current_y - prev_y
            ])
            self.total_distance_traveled += step_distance
    
    def reset_evaluation_metrics(self):
        """重置评估指标"""
        self.start_x = None
        self.start_y = None
        self.goal_x = None
        self.goal_y = None
        self.current_x = None
        self.current_y = None
        self.trajectory = []
        self.total_distance_traveled = 0.0
        self.straight_line_distance = 0.0
    
    def get_evaluation_metrics(self):
        """获取评估指标"""
        if self.start_x is None or self.goal_x is None or self.current_x is None:
            print("Warning: Evaluation metrics not initialized yet")
            return None
        
        current_to_goal_distance = np.linalg.norm([
            self.goal_x - self.current_x,
            self.goal_y - self.current_y
        ])
        
        efficiency_ratio = 0.0
        if self.total_distance_traveled > 0:
            efficiency_ratio = self.straight_line_distance / self.total_distance_traveled
        else:
            efficiency_ratio = 1.0
        
        return {
            'start_position': (self.start_x, self.start_y),
            'goal_position': (self.goal_x, self.goal_y),
            'current_position': (self.current_x, self.current_y),
            'straight_line_distance': self.straight_line_distance,
            'total_distance_traveled': self.total_distance_traveled,
            'current_to_goal_distance': current_to_goal_distance,
            'efficiency_ratio': efficiency_ratio,
            'trajectory_length': len(self.trajectory)
        }
    
    def print_evaluation_summary(self):
        """打印评估摘要"""
        metrics = self.get_evaluation_metrics()
        
        print("=" * 50)
        print("EVALUATION SUMMARY 评估摘要")
        print("=" * 50)
        print(f"📍 起始位置: ({metrics['start_position'][0]:.2f}, {metrics['start_position'][1]:.2f})")
        print(f"🎯 目标位置:  ({metrics['goal_position'][0]:.2f}, {metrics['goal_position'][1]:.2f})")
        print(f"🤖 当前位置: ({metrics['current_position'][0]:.2f}, {metrics['current_position'][1]:.2f})")
        print("-" * 50)
        print(f"📏 直线距离: {metrics['straight_line_distance']:.2f} m")
        print(f"🛤️ 总路程: {metrics['total_distance_traveled']:.2f} m")
        print(f"📐 当前到目标距离: {metrics['current_to_goal_distance']:.2f} m")
        print(f"💡 效率比值: {metrics['efficiency_ratio']:.3f}")
        print(f"📊 轨迹点数: {metrics['trajectory_length']}")

# 统一的评估函数
def evaluate_with_metrics(network, env, epoch, eval_episodes=10, use_evaluation=True):
    """
    统一的评估函数，支持不同类型的环境和网络
    
    Args:
        network: 训练好的网络（需要有get_action方法）
        env: 环境实例（GazeboEnv或VelodyneGymWrapper）
        epoch: 当前训练轮次
        eval_episodes: 评估回合数
        use_evaluation: 是否使用评估指标
    """
    if use_evaluation:
        # 根据环境类型选择合适的评估包装器
        if isinstance(env, GazeboEnv):
            eval_env = GazeboEvaluationWrapper(env)
        elif hasattr(env, 'gazebo_env'):  # VelodyneGymWrapper
            eval_env = EvaluationWrapper(env.launchfile, env.environment_dim, env.action_type)
        else:
            eval_env = None
            print("Warning: Unknown environment type, evaluation metrics disabled")
    else:
        eval_env = None
    
    avg_reward = 0.0
    col = 0
    
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        if eval_env:
            eval_env.reset()
        
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            
            # 处理不同算法的动作格式
            if hasattr(network, 'num_actions'):  # DQN算法
                a_in = [(action[0] + 1) / 2, action[1]]
            else:  # TD3等连续动作算法
                a_in = action
            
            state, reward, done, _ = env.step(a_in)
            if eval_env:
                eval_env.step(a_in)
            
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    
    # 打印评估指标
    if eval_env:
        eval_env.print_evaluation_summary()
    
    return avg_reward


