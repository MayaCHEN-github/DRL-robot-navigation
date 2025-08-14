import numpy as np
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from gym_wrapper import VelodyneGymWrapper

class EvaluationCallback(BaseCallback):
    """评估回调，每5000步触发一次"""
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

def main():
    # 创建环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="discrete"  # 使用离散动作空间（DQN算法）
    )

    # 创建DQN模型
    print("正在创建DQN模型...")
    # 参数设置与train_dqn.py一致
    eval_freq = 5_000  # 每5000步评估一次
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/",
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=64,
        gamma=0.999,  # 与train_dqn.py的discount保持一致
    )

    # 创建保存目录
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    # 设置自动保存回调（每5000步保存一次，与其他文件保持一致）
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="dqn_velodyne"
    )
    
    # 设置评估回调（每5000步触发一次，与其他文件保持一致）
    eval_callback = EvaluationCallback(
        eval_freq=eval_freq,
        verbose=1
    )
    
    # 组合所有回调函数
    callbacks = [checkpoint_callback, eval_callback]
    
    # 快速验证
    print("=== 第一阶段：快速验证（1000步）===")
    model.learn(
        total_timesteps=1_000,   # 1000步，快速验证程序运行
        progress_bar=True,
        callback=callbacks
    )
    print("✅ 第一阶段完成！程序运行正常。")
    
    # 正式训练
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == 'y':
        print("=== 第二阶段：正式训练（500万步）===")
        model.learn(
            total_timesteps=5_000_000,
            progress_bar=True,
            callback=callbacks
        )
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    # 保存模型
    print("正在保存模型...")
    model.save("dqn_test_model")

    env.close()
    print("训练完成！")

if __name__ == "__main__":
    main()
