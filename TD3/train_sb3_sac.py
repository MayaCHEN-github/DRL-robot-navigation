import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from gym_wrapper import VelodyneGymWrapper


class EvaluationCallback(BaseCallback):
    """评估回调：每5000步打印一次提示"""

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
    # 设备信息
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

    # 环境
    print("正在创建训练环境...")
    env = VelodyneGymWrapper(
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        action_type="continuous",
        device=device,
    )

    # SAC 模型（偏向快速联调）
    print("正在创建SAC模型...")
    eval_freq = 5_000
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/sac_velodyne",
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        device=device,
    )

    print(f"✅ SAC模型已创建在设备: {device}")

    # 模型保存目录
    os.makedirs("./pytorch_models", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./pytorch_models/",
        name_prefix="sac_velodyne",
    )
    eval_callback = EvaluationCallback(eval_freq=eval_freq, verbose=1)

    # 快速验证
    print("=== 第一阶段：快速验证（5000步）===")
    model.learn(total_timesteps=5_000, progress_bar=True, callback=[checkpoint_callback, eval_callback])
    print("✅ 第一阶段完成！程序运行正常。")

    # 正式训练（可按需继续）
    user_input = input("程序运行正常！是否继续训练更多步数？(y/n): ")
    if user_input.lower() == "y":
        print("=== 第二阶段：正式训练（245000步，延续同一模型与时间轴）===")
        model.learn(total_timesteps=245_000, progress_bar=True, callback=[checkpoint_callback, eval_callback], reset_num_timesteps=False)
        print("✅ 训练完成！")
    else:
        print("训练已停止。")

    print("正在保存模型...")
    model.save("./pytorch_models/sac_test_model")
    env.close()
    print("训练完成！")


if __name__ == "__main__":
    main()


