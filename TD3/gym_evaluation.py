import numpy as np
import time
from gym_wrapper import VelodyneGymWrapper

class EvaluationWrapper(VelodyneGymWrapper):
    """
    继承VelodyneGymWrapper, 计算单回合评估指标（供多回合评估聚合使用）。
    输出的核心metrics:
      - success_rate: 是否到达目标（0/1）
      - time_cost: 到达目标所用时间（或至回合结束）
      - path_efficiency: 路径效率（直线距离/累计路径）
      - trajectory_smoothness: 轨迹平滑度
    """
    
    def __init__(self, launchfile, environment_dim, action_type="continuous"):
        super().__init__(launchfile, environment_dim, action_type)
        self.reset_evaluation_metrics() # 初始化metrics
    
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        self._update_evaluation_metrics(reward) # 基于奖励阈值更新碰撞等metrics
        
        return state, reward, terminated, truncated, info
    
    def reset(self):
        state, info = super().reset() # 调用父类的reset方法（返回observation, info）
        self.reset_evaluation_metrics() # 重置metrics
        self._initialize_evaluation_metrics() # 获取起始位置和目标位置
        return state, info
    
    def _initialize_evaluation_metrics(self):
        """初始化评估指标"""
        # A.位置相关
        self.start_x = self.gazebo_env.odom_x # 获取起始位置（机器人当前位置）
        self.start_y = self.gazebo_env.odom_y
        self.current_x = self.start_x
        self.current_y = self.start_y
        
        self.goal_x = self.gazebo_env.goal_x # 获取目标位置
        self.goal_y = self.gazebo_env.goal_y
        
        self.straight_line_distance = np.linalg.norm([ # 计算直线距离
            self.goal_x - self.start_x,
            self.goal_y - self.start_y
        ])
        # 初始化轨迹
        self.trajectory = [(self.start_x, self.start_y)]
        
        # 初始化新增指标
        self.start_time = time.time()  # 记录开始时间
        
    
    def _update_evaluation_metrics(self, reward=None):
        """更新评估指标。"""

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
        
        # 这里不在类内统计碰撞率，碰撞统计在多回合评估中通过 reward<-90 聚合

    def reset_evaluation_metrics(self):
        """重置评估指标"""
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
        
        # 新增评估指标
        self.start_time = None  # 开始时间
        
        
    
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
        
        # 计算新增指标（供多回合聚合）
        time_cost = time.time() - self.start_time if self.start_time else 0
        success_rate = 1.0 if current_to_goal_distance < 0.3 else 0.0
        
        # 计算轨迹平滑度（通过轨迹点的曲率变化）
        smoothness = self._calculate_trajectory_smoothness()
        
        return {
            'success_rate': success_rate,
            'time_cost': time_cost,
            'path_efficiency': efficiency_ratio,
            'trajectory_smoothness': smoothness
        }
    
    # 不再在类内打印摘要，聚合由 evaluate_over_episodes 完成
    
    def _calculate_trajectory_smoothness(self):
        """计算轨迹平滑度（基于轨迹点的曲率变化）"""
        if len(self.trajectory) < 3:
            return 1.0  # 轨迹点太少，认为平滑
        
        # 计算轨迹的曲率变化
        curvature_changes = []
        for i in range(1, len(self.trajectory) - 1):
            p1 = np.array(self.trajectory[i-1])
            p2 = np.array(self.trajectory[i])
            p3 = np.array(self.trajectory[i+1])
            
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
    
    
def evaluate_over_episodes(
    agent,
    launchfile,
    environment_dim,
    num_episodes=10,
    max_steps=500,
    action_type="continuous",
    deterministic=True,
):
    """
    多次运行评估回合并聚合指标（不依赖 SB3 的 VecEnv）。
    - agent: 支持 SB3 的 model（带 predict）或可调用函数 f(obs)->action
    - 返回: (summary, episode_metrics)
    """
    env = EvaluationWrapper(launchfile, environment_dim, action_type)
    episode_metrics = []
    success_episodes = 0
    collision_episodes = 0
    try:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_collision = False
            for _ in range(max_steps):
                if hasattr(agent, "predict"):
                    action, _states = agent.predict(obs, deterministic=deterministic)
                else:
                    action = agent(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if reward < -90 and not episode_collision:
                    episode_collision = True
                if terminated or truncated:
                    break
            m = env.get_evaluation_metrics()
            if m is None:
                continue
            episode_metrics.append(m)
            if m.get("success_rate", 0.0) >= 0.5:
                success_episodes += 1
            if episode_collision:
                collision_episodes += 1
    finally:
        env.close()

    n = len(episode_metrics) if episode_metrics else 1
    avg_time_cost = float(np.mean([m["time_cost"] for m in episode_metrics])) if episode_metrics else 0.0
    avg_path_efficiency = float(np.mean([m["path_efficiency"] for m in episode_metrics])) if episode_metrics else 0.0
    avg_trajectory_smoothness = float(np.mean([m["trajectory_smoothness"] for m in episode_metrics])) if episode_metrics else 0.0

    summary = {
        "episodes": len(episode_metrics),
        "success_rate": success_episodes / n,
        "collision_rate": collision_episodes / n,
        "avg_time_cost": avg_time_cost,
        "avg_path_efficiency": avg_path_efficiency,
        "avg_trajectory_smoothness": avg_trajectory_smoothness,
    }

    return summary, episode_metrics

