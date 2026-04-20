import numpy as np
import sys

sys.path.insert(0, ".")

from environment.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
from training.train_dqn import train_dqn
from analysis.visualization_dqn import generate_all_visualizations
from analysis.metrics import evaluate_dqn, print_metrics
from analysis.refinement_logger import RefinementLogger


def main():
    print("\n" + "=" * 70)
    print("🚀 DQN TRAINING PIPELINE FOR TRAFFIC SIGNAL CONTROL")
    print("=" * 70 + "\n")

    print("📚 Initializing Environment and Agent...")
    env = TrafficEnv(
        max_steps=200, throughput_bonus_weight=2.0, reward_balance_weight=0.1
    )

    agent = DQNAgent(
        state_size=4,
        action_size=3,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.990,
        use_double_dqn=True,
    )

    print("✅ Environment and Agent initialized!")
    print(f"   State Size: {agent.state_size}")
    print(f"   Action Size: {agent.action_size}")
    print(f"   Double DQN: {agent.use_double_dqn}")
    print(f"   Normalized Observations: True")
    print(f"   Throughput Bonus Weight: 2.0")
    print(f"   Reward Balance Weight: 0.1")

    print("\n📖 Starting DQN Training...")
    print("-" * 70)
    rewards, losses = train_dqn(env, agent, episodes=500, normalize_obs=True)
    print("-" * 70)
    print("✅ Training completed!")

    print("\n📊 Generating Visualizations...")
    generate_all_visualizations(agent, env, rewards, losses)

    print("\n📈 Computing Evaluation Metrics...")
    metrics = evaluate_dqn(rewards, threshold=-300.0)
    print_metrics(metrics, "DQN")

    print("\n📝 Refinement Log:")
    logger = RefinementLogger()
    logger.log_change(
        "Initial training run completed",
        "Double DQN + reward normalization (scale=50) + throughput bonus (2.0) + balance penalty (0.1)",
        f"Final avg reward: {np.mean(rewards[-50:]):.2f}",
    )
    logger.print_log()

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Results saved to: results/")
    print(f"🎯 Model saved to: models/dqn_final.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
