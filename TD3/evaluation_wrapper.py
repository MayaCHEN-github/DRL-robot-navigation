import numpy as np
from gym_wrapper import VelodyneGymWrapper

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