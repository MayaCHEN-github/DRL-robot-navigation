import numpy as np
from stable_baselines3 import PPO
from gym_wrapper import VelodyneGymWrapper
from evaluation_wrapper import EvaluationWrapper

def test_model(model, env):
    """简单测试模型，验证训练是否成功"""
    print("简单测试模型...")
    
    obs = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    # 运行一个episode，最多100步
    while not done and step_count < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # 合并terminated和truncated
        episode_reward += reward
        step_count += 1
        
        if info['target_reached']:
            print(f"✅ 测试成功！到达目标，总奖励: {episode_reward:.2f}")
            return True
    
    print(f"⚠️  测试完成，未到达目标，总奖励: {episode_reward:.2f}")
    return False

def main():
    """使用stable-baselines3训练一个PPO模型的示例"""

    # 创建环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous"  # 使用连续动作空间（PPO算法）
    )

    # 创建PPO模型
    print("正在创建PPO模型...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

    # 渐进式训练测试
    print("=== 第一阶段：快速验证（5000步）===")
    model.learn(
        total_timesteps=500,    # 先用100步快速验证
        progress_bar=True
    )
    
    print("✅ 第一阶段完成！程序运行正常。")
    
    # 询问是否继续训练
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == 'y':
        print("=== 第二阶段：正式训练（10万步）===")
        model.learn(
            total_timesteps=100_000,  # 正式训练
            progress_bar=True
        )
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    # 保存模型
    print("正在保存模型...")
    model.save("ppo_test_model")

    # 简单测试
    test_success = test_model(model, env)

    # 关闭环境
    env.close()
    print("训练完成！")

if __name__ == "__main__":
    main()