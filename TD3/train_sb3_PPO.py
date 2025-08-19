import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gym_wrapper import VelodyneGymWrapper

class EvaluationCallback(BaseCallback):
    """评估回调，每5000步触发一次（与TD3和DQN文件保持一致）"""
    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            if self.verbose > 0:
                print(f"\n=== 评估触发（第{self.num_timesteps}步）===")
        return True

def test_model(model, env):
    obs, info = env.reset()  # 正确解包reset()的返回值
    episode_reward = 0
    done = False
    step_count = 0
    
    # 运行一个episode，最多100步
    while not done and step_count < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  
        episode_reward += reward
        step_count += 1
        
        if info['target_reached']:
            print(f"✅ 测试成功！到达目标，总奖励: {episode_reward:.2f}")
            return True
    
    print(f"⚠️  测试完成，未到达目标，总奖励: {episode_reward:.2f}")
    return False

def main():
    # 检查CUDA可用性
    print("=== CUDA检测 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  CUDA不可用，将使用CPU")
        device = "cpu"
    print("=" * 20)

    # 创建环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous",  # 使用连续动作空间（PPO算法）
        device=device  # 传递设备信息给环境
    )

    # 创建PPO模型
    print("正在创建PPO模型...")
    # 训练节奏：每5000步评估/保存一次，目标总步数≈250k
    eval_freq = 5_000  # 每5000步评估一次，与TD3和DQN保持一致
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/ppo_velodyne",
        # 超参数设置（加速早期学习并减少每次迭代耗时）
        learning_rate=7e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 略增探索力度（无ε衰减机制，靠熵项鼓励探索）
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        device=device,  # 明确指定使用CUDA或CPU
    )

    # 验证模型设备
    print(f"✅ PPO模型已创建在设备: {device}")
    if device == "cuda":
        print(f"模型参数设备: {next(model.policy.parameters()).device}")

    # 创建保存目录
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    # 设置自动保存回调（每5000步保存一次，与其他文件保持一致）
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="ppo_velodyne"
    )
    
    # 设置评估回调（每5000步触发一次，与其他文件保持一致）
    eval_callback = EvaluationCallback(
        eval_freq=eval_freq,
        verbose=1
    )
    
    # 组合所有回调函数
    callbacks = [checkpoint_callback, eval_callback]
    
    # 快速验证
    print("=== 第一阶段：快速验证（5000步）===")
    model.learn(
        total_timesteps=5_000,   # 5000步，对齐checkpoint/eval频率
        progress_bar=True,
        callback=callbacks
    )
    print("✅ 第一阶段完成！程序运行正常。")
    
    # 正式训练
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == 'y':
        print("=== 第二阶段：正式训练（245000步，延续同一模型与时间轴）===")
        model.learn(
            total_timesteps=245_000,  # 与第一阶段合计约25万步，≈50个checkpoint
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=False  # 关键：不重置时间步，继续同一模型
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