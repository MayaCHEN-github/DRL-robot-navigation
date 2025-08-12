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
        obs, reward, done, info = env.step(action)
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

    # 训练模型
    print("开始训练...")
    model.learn(
        total_timesteps=1_000_000,  # 总训练步数
        progress_bar=True           # 显示进度条
    )

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