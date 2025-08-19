#!/usr/bin/env python3
import os
import sys
import argparse
from rl_zoo3 import train, enjoy, optimize_hyperparams
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TD3.gym_wrapper import VelodyneGymWrapper


def register_custom_envs():
    """注册自定义环境"""
    try:
        from gymnasium.envs.registration import register
        # 注册Velodyne环境
        register(
            id='VelodyneEnv-v0',
            entry_point='TD3.gym_wrapper:VelodyneGymWrapper',
            kwargs={
                'launchfile': 'multi_robot_scenario.launch',
                'environment_dim': 20,
                'action_type': 'continuous'
            },
            max_episode_steps=500,
        )
        print("✅ 自定义环境 'VelodyneEnv-v0' 注册成功")
    except Exception as e:
        print(f"⚠️ 注册环境时出错: {e}")


def train_model(config_path):
    """使用RL Baselines3 Zoo训练模型"""
    print(f"🚀 开始训练，配置文件: {config_path}")
    # 调用RL Baselines3 Zoo的train函数
    train.main([
        '--algo', 'td3',
        '--env', 'VelodyneEnv-v0',
        '--config', config_path,
        '--log-folder', './logs/td3_zoo',
        '--save-folder', './pytorch_models/zoo_models',
    ])
    print("✅ 训练完成")


def evaluate_model(model_path):
    """使用RL Baselines3 Zoo评估模型"""
    print(f"📊 开始评估，模型路径: {model_path}")
    # 调用RL Baselines3 Zoo的enjoy函数
    enjoy.main([
        '--algo', 'td3',
        '--env', 'VelodyneEnv-v0',
        '--model', model_path,
        '--n-episodes', '10',
        '--deterministic',
    ])
    print("✅ 评估完成")


def optimize_hyperparams_(config_path):
    """使用RL Baselines3 Zoo优化超参数"""
    print(f"🔍 开始超参数优化，配置文件: {config_path}")
    # 调用RL Baselines3 Zoo的optimize_hyperparams函数
    optimize_hyperparams.main([
        '--algo', 'td3',
        '--env', 'VelodyneEnv-v0',
        '--config', config_path,
        '--n-trials', '50',
        '--n-timesteps', '100000',
        '--log-folder', './logs/td3_zoo_opt',
    ])
    print("✅ 超参数优化完成")


def main():
    parser = argparse.ArgumentParser(description='使用RL Baselines3 Zoo训练、评估和优化TD3模型')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'optimize'], required=True,
                        help='运行模式: train(训练), evaluate(评估), optimize(超参数优化)')
    parser.add_argument('--config', default='./configs/td3_velodyne.yaml',
                        help='配置文件路径 (默认: ./configs/td3_velodyne.yaml)')
    parser.add_argument('--model', default='./pytorch_models/zoo_models/td3/VelodyneEnv-v0_1.zip',
                        help='模型文件路径 (用于评估模式, 默认: ./pytorch_models/zoo_models/td3/VelodyneEnv-v0_1.zip)')
    args = parser.parse_args()

    # 注册自定义环境
    register_custom_envs()

    if args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'evaluate':
        evaluate_model(args.model)
    elif args.mode == 'optimize':
        optimize_hyperparams_(args.config)


if __name__ == '__main__':
    main()