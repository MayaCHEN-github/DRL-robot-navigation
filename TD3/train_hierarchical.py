import os
import argparse
import torch
from hierarchical_rl import HierarchicalRL


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练层级强化学习算法')
    parser.add_argument('--environment_dim', type=int, default=20, help='环境维度（激光雷达数据维度）')
    parser.add_argument('--max_timesteps', type=int, default=2000000, help='最大训练步数')
    parser.add_argument('--eval_freq', type=int, default=1000, help='评估频率')
    parser.add_argument('--device', type=str, default=None, help='使用的设备（cuda或cpu）')
    parser.add_argument('--load_high_level', type=str, default=None, help='加载高层模型的路径')
    parser.add_argument('--load_low_level', type=str, default=None, help='加载低层模型的路径')
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建存储目录
    if not os.path.exists("./results"):
        os.makedirs("./results", exist_ok=True)
    if not os.path.exists("./pytorch_models/high_level"):
        os.makedirs("./pytorch_models/high_level", exist_ok=True)
    if not os.path.exists("./pytorch_models/low_level"):
        os.makedirs("./pytorch_models/low_level", exist_ok=True)

    # 创建层级强化学习实例
    hierarchical_agent = HierarchicalRL(
        environment_dim=args.environment_dim,
        max_timesteps=args.max_timesteps,
        eval_freq=args.eval_freq,
        device=device
    )

    # 加载已有模型（如果提供）
    if args.load_high_level is not None:
        print(f"加载高层模型: {args.load_high_level}")
        hierarchical_agent.load_high_level_model(args.load_high_level)
    if args.load_low_level is not None:
        print(f"加载低层模型: {args.load_low_level}")
        hierarchical_agent.load_low_level_model(args.load_low_level)

    # 开始训练
    hierarchical_agent.train()


if __name__ == "__main__":
    main()