import numpy as np
import time
from velodyne_env import GazeboEnv


def _calculate_trajectory_smoothness(trajectory):
    """计算轨迹平滑度（基于轨迹点的曲率变化）。"""
    if len(trajectory) < 3:
        return 1.0
    curvature_changes = []
    for i in range(1, len(trajectory) - 1):
        p1 = np.array(trajectory[i - 1])
        p2 = np.array(trajectory[i])
        p3 = np.array(trajectory[i + 1])
        v1 = p2 - p1
        v2 = p3 - p2
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvature_changes.append(angle)
    if not curvature_changes:
        return 1.0
    avg_curvature = np.mean(curvature_changes)
    smoothness = 1.0 / (1.0 + avg_curvature)
    return smoothness


def run_episode_native(env, agent_fn, max_steps=500, action_transform=None):
    """
    运行单个原生GazeboEnv回合，返回单回合指标（不依赖gym/sb3）。
    - agent_fn: 可调用 (obs)->action
    - action_transform: 可选，对动作进行转换，例如 lambda a: [(a[0]+1)/2, a[1]]
    """
    obs = env.reset()
    start_x = env.odom_x
    start_y = env.odom_y
    goal_x = env.goal_x
    goal_y = env.goal_y
    straight_line_distance = np.linalg.norm([goal_x - start_x, goal_y - start_y])
    trajectory = [(start_x, start_y)]
    total_distance_traveled = 0.0
    start_time = time.time()

    episode_collision = False
    episode_success = False

    for _ in range(max_steps):
        action = agent_fn(np.array(obs))
        if action_transform is not None:
            action = action_transform(action)
        obs, reward, done, target = env.step(action)

        # 轨迹更新
        current_x = env.odom_x
        current_y = env.odom_y
        trajectory.append((current_x, current_y))
        if len(trajectory) > 1:
            prev_x, prev_y = trajectory[-2]
            step_distance = np.linalg.norm([current_x - prev_x, current_y - prev_y])
            total_distance_traveled += step_distance

        # 碰撞判定（与train_velodyne_td3一致）
        if (reward < -90) and (not episode_collision):
            episode_collision = True

        if target:
            episode_success = True
        if done:
            break

    time_cost = time.time() - start_time
    efficiency_ratio = (
        straight_line_distance / total_distance_traveled
        if total_distance_traveled > 0
        else 1.0
    )
    smoothness = _calculate_trajectory_smoothness(trajectory)

    return {
        "success": 1.0 if episode_success else 0.0,
        "collision": 1.0 if episode_collision else 0.0,
        "time_cost": time_cost,
        "path_efficiency": efficiency_ratio,
        "trajectory_smoothness": smoothness,
    }


def evaluate_over_episodes_native(
    agent_fn,
    launchfile,
    environment_dim,
    num_episodes=10,
    max_steps=500,
    action_transform=None,
):
    """
    完全原生（不依赖gym/sb3）的多回合评估：
    - agent_fn: 可调用 (obs)->action
    - action_transform: 可选，对动作进行转换（如将tanh输出[-1,1]映射到[0,1]×[-1,1]）
    返回(summary, episode_metrics)
    """
    env = GazeboEnv(launchfile, environment_dim)
    episode_metrics = []
    try:
        for _ in range(num_episodes):
            m = run_episode_native(env, agent_fn, max_steps=max_steps, action_transform=action_transform)
            episode_metrics.append(m)
    finally:
        try:
            import rospy
            if rospy.core.is_initialized():
                rospy.signal_shutdown("Native evaluation done")
        except Exception:
            pass

    n = len(episode_metrics) if episode_metrics else 1
    success_rate = float(np.mean([m["success"] for m in episode_metrics])) if episode_metrics else 0.0
    collision_rate = float(np.mean([m["collision"] for m in episode_metrics])) if episode_metrics else 0.0
    avg_time_cost = float(np.mean([m["time_cost"] for m in episode_metrics])) if episode_metrics else 0.0
    avg_path_efficiency = float(np.mean([m["path_efficiency"] for m in episode_metrics])) if episode_metrics else 0.0
    avg_trajectory_smoothness = float(np.mean([m["trajectory_smoothness"] for m in episode_metrics])) if episode_metrics else 0.0

    summary = {
        "episodes": len(episode_metrics),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_time_cost": avg_time_cost,
        "avg_path_efficiency": avg_path_efficiency,
        "avg_trajectory_smoothness": avg_trajectory_smoothness,
    }

    return summary, episode_metrics
