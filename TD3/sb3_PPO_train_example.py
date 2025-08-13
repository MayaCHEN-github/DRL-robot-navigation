import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_wrapper import VelodyneGymWrapper
from evaluation_wrapper import EvaluationWrapper

def test_model(model, env):
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

    # 创建环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous"  # 使用连续动作空间（PPO算法）
    )

    # 创建PPO模型
    print("正在创建PPO模型...")
    # 设置与其他训练文件一致的参数
    eval_freq = 5_000  # 每5000步评估一次，与TD3和DQN保持一致
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        eval_freq=eval_freq, # 每5000步评估一次
        n_eval_episodes=10,  # 评估时运行10个episode
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # 创建保存目录
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    # 设置自动保存回调（每5000步保存一次，与其他文件保持一致）
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="ppo_velodyne"
    )
    
    # 快速验证
    print("=== 第一阶段：快速验证（1000步）===")
    model.learn(
        total_timesteps=1_000,   # 1000步
        progress_bar=True,
        callback=checkpoint_callback
    )
    print("✅ 第一阶段完成！程序运行正常。")
    
    # 正式训练
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == 'y':
        print("=== 第二阶段：正式训练（与其他文件保持一致）===")
        model.learn(
            total_timesteps=5_000_000,  # 500万步，与TD3和DQN两份文件保持一致。
            progress_bar=True,
            callback=checkpoint_callback
        )
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    # 保存模型
    print("正在保存模型...")
    model.save("ppo_test_model")

    # 简单测试
    test_success = test_model(model, env)

    env.close()
    print("训练完成！")

if __name__ == "__main__":
    main()