import os
import torch
import time
import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_wrapper import VelodyneGymWrapper


class DetailedEvaluationCallback(BaseCallback):
    """详细评估回调: 每5000步进行详细评估并保存结果"""

    def __init__(self, env, eval_freq: int = 5000, eval_episodes: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.env = env  # 传入环境以便在回调中使用
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            if self.verbose > 0:
                print(f"\n=== 评估触发（第{self.num_timesteps}步）===")
                
            # 进行详细评估
            summary = self._evaluate_detailed(self.env, self.model, self.eval_episodes)
            
            # 保存评估结果到npy文件
            self._save_evaluation_results(summary, self.num_timesteps)
            
            # 记录到TensorBoard
            self._log_evaluation_to_tensorboard(summary, self.num_timesteps)
        return True
    
    def _evaluate_detailed(self, env, model, eval_episodes):
        """执行详细评估并计算各项指标"""
        print(f"开始详细评估，共 {eval_episodes} 个回合...")
        
        # 初始化指标
        episode_metrics = []
        success_episodes = 0
        collision_episodes = 0
        total_time_cost = 0.0
        total_path_efficiency = 0.0
        total_trajectory_smoothness = 0.0
        total_reward = 0.0
        
        for episode in range(eval_episodes):
            print(f"评估回合 {episode+1}/{eval_episodes}")
            
            # 重置环境
            reset_result = env.reset()
            state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            if isinstance(state, tuple):
                state = state[0]
            state = np.array(state, dtype=np.float32)
            
            # 初始化评估指标
            start_time = time.time()
            trajectory = []
            total_distance = 0.0
            episode_collision = False
            episode_reward = 0.0
            
            # 获取起始位置和目标位置
            if hasattr(env, 'gazebo_env'):
                start_x = env.gazebo_env.odom_x
                start_y = env.gazebo_env.odom_y
                goal_x = env.gazebo_env.goal_x
                goal_y = env.gazebo_env.goal_y
                trajectory.append((start_x, start_y))
                
                straight_line_distance = np.linalg.norm([goal_x - start_x, goal_y - start_y])
            else:
                # 如果无法获取位置信息，使用默认值
                start_x = start_y = goal_x = goal_y = 0.0
                trajectory.append((start_x, start_y))
                straight_line_distance = 1.0
            
            done = False
            episode_steps = 0
            
            while not done and episode_steps < 500:  # 设置最大步数
                # 使用确定性策略
                action = model.predict(state, deterministic=True)[0]
                
                # 执行动作
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 更新状态
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                state = np.array(next_state, dtype=np.float32)
                
                # 记录本回合奖励
                episode_reward += reward
                
                # 记录轨迹
                if hasattr(env, 'gazebo_env'):
                    current_x = env.gazebo_env.odom_x
                    current_y = env.gazebo_env.odom_y
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
            
            # 累计总奖励
            total_reward += episode_reward
            
            # 计算评估指标
            time_cost = time.time() - start_time
            total_time_cost += time_cost
            
            # 计算成功率
            if hasattr(env, 'gazebo_env'):
                final_x = env.gazebo_env.odom_x
                final_y = env.gazebo_env.odom_y
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
            
            # 保存回合指标
            episode_metrics.append({
                'success_rate': success,
                'time_cost': time_cost,
                'path_efficiency': path_efficiency,
                'trajectory_smoothness': smoothness,
                'reward': episode_reward,
                # 调试数据
                'debug_straight_line_distance': straight_line_distance,
                'debug_total_distance': total_distance,
                'debug_actual_distance': np.linalg.norm([final_x - start_x, final_y - start_y]) if hasattr(env, 'gazebo_env') else 0.0,
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
            'avg_reward': total_reward / n,
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
        print(f"  平均奖励: {summary['avg_reward']:.2f}")
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
    
    def _save_evaluation_results(self, summary, timestep):
        """保存评估结果到npy文件"""
        os.makedirs("./logs", exist_ok=True)
        detailed_evaluation_data = {
            'summary': summary,
            'timestamp': time.time(),
            'timestep': timestep
        }
        np.save("./logs/evaluation_metrics.npy", detailed_evaluation_data)
        
    def _log_evaluation_to_tensorboard(self, summary, timestep):
        """记录评估结果到TensorBoard"""
        try:
            # 记录评估指标到TensorBoard
            self.model.logger.record("evaluation/success_rate", summary['success_rate'])
            self.model.logger.record("evaluation/collision_rate", summary['collision_rate'])
            self.model.logger.record("evaluation/avg_time_cost", summary['avg_time_cost'])
            self.model.logger.record("evaluation/avg_path_efficiency", summary['avg_path_efficiency'])
            self.model.logger.record("evaluation/avg_trajectory_smoothness", summary['avg_trajectory_smoothness'])
            
            # 新增：记录平均奖励
            if 'avg_reward' in summary:
                self.model.logger.record("evaluation/avg_reward", summary['avg_reward'])
            
            # 记录调试数据
            self.model.logger.record("evaluation_debug/avg_straight_line_distance", summary['avg_straight_line_distance'])
            self.model.logger.record("evaluation_debug/avg_total_distance", summary['avg_total_distance'])
            self.model.logger.record("evaluation_debug/avg_actual_distance", summary['avg_actual_distance'])
            self.model.logger.record("evaluation_debug/avg_trajectory_length", summary['avg_trajectory_length'])
            self.model.logger.record("evaluation_debug/avg_episode_steps", summary['avg_episode_steps'])
            
            # 强制写入TensorBoard
            self.model.logger.dump(step=timestep)
            
            print(f"评估结果已记录到TensorBoard和npy文件")
            
        except Exception as e:
            print(f"记录评估结果到TensorBoard时出错: {e}")


def main():
    # 强制使用CPU
    print("=== 设备设置 ===")
    device = "cpu"
    print(f"已设置使用CPU进行计算")
    print("=" * 20)

    # 环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous",
        device=device,
    )

    # TD3 模型（偏向动作平滑与精细控制）
    print("正在创建TD3模型...")
    eval_freq = 5_000  # 与HRL一致，设置为5000步
    # 添加探索噪声衰减参数
    expl_noise = 1.0  # 初始探索噪声
    expl_decay_steps = 500_000  # 噪声衰减步数
    expl_min = 0.1  # 最小探索噪声

    model = TD3(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./logs/td3_velodyne",
        learning_rate=1e-4,  # 降低学习率
        buffer_size=1_000_000,  # 增加缓冲区大小
        batch_size=40,  # 与velodyne版本一致
        tau=0.005,
        gamma=0.99999,  # 与velodyne版本一致
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,  # 与velodyne版本一致
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        device=device,
    )

    print(f"✅ TD3模型已创建在设备: {device}")
    print(f"📊 评估设置:")
    print(f"  - 评估频率: 每{eval_freq}步")

    # 模型保存目录
    os.makedirs("./pytorch_models", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="td3_velodyne",
    )
    eval_callback = DetailedEvaluationCallback(env=env, eval_freq=eval_freq, eval_episodes=10, verbose=1)

    # 快速验证
    print("=== 第一阶段：快速验证（5000步）===")
    model.learn(total_timesteps=5_000, progress_bar=True, callback=[checkpoint_callback, eval_callback])
    print("✅ 第一阶段完成！程序运行正常。")

    # 正式训练（可按需继续）
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == "y":
        print("=== 第二阶段：正式训练（1,995,000步，延续同一模型与时间轴）===")
        model.learn(total_timesteps=1_995_000, progress_bar=True, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    print("正在保存模型...")
    model.save("./pytorch_models/td3_test_model")
    env.close()
    print("训练完成！")


if __name__ == "__main__":
    main()


