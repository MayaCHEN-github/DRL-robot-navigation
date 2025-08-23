import os
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_wrapper import VelodyneGymWrapper


class EvaluationCallback(BaseCallback):
    """评估回调: 每5000步打印一次提示"""

    def __init__(self, eval_freq: int = 5000, verbose: int = 1):
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
    eval_freq = 5_000
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

    # 模型保存目录
    os.makedirs("./pytorch_models", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="td3_velodyne",
    )
    eval_callback = EvaluationCallback(eval_freq=eval_freq, verbose=1)

    # 快速验证
    print("=== 第一阶段：快速验证（5000步）===")
    model.learn(total_timesteps=5_000, progress_bar=True, callback=[checkpoint_callback, eval_callback])
    print("✅ 第一阶段完成！程序运行正常。")

    # 正式训练（可按需继续）
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == "y":
        print("=== 第二阶段：正式训练（4,995,000步，延续同一模型与时间轴）===")
        model.learn(total_timesteps=4_995_000, progress_bar=True, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    print("正在保存模型...")
    model.save("./pytorch_models/td3_test_model")
    env.close()
    print("训练完成！")


if __name__ == "__main__":
    main()


