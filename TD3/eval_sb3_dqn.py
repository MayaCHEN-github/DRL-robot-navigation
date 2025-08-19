from stable_baselines3 import DQN
from gym_evaluation import evaluate_over_episodes


def main():
    model_path = "dqn_test_model.zip"  # 你训练保存的权重
    model = DQN.load(model_path)

    summary, episodes = evaluate_over_episodes(
        agent=model,
        launchfile="multi_robot_scenario.launch",
        environment_dim=20,
        num_episodes=10,
        max_steps=500,
        action_type="discrete",
        deterministic=True,
    )

    print("=" * 50)
    print("SB3-DQN 原生评估结果")
    print("-" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()


