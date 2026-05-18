import numpy as np
import sys

sys.path.insert(0, ".")

from environment.traffic_env import TrafficEnv
from agents.qlearning_agent import QLearningAgent
from training.train_qlearning import train
from analysis.qlearning_visualization import generate_all_visualizations
from analysis.qlearning_analysis import random_policy, evaluate_policy


def main():
    print("\n" + "=" * 70)
    print("Q-LEARNING TRAINING PIPELINE FOR TRAFFIC SIGNAL CONTROL")
    print("=" * 70 + "\n")

    print("Initializing Environment and Agent...")
    print("Environment and Agent initialized!")
    print(f"   Max Steps: 200")
    print(f"   Episodes: 500")
    print(f"   Alpha: 0.1 | Gamma: 0.9 | Epsilon decay: 0.99")

    print("\nStarting Q-Learning Training...")
    print("-" * 70)
    agent, rewards, epsilon_history = train(episodes=500)
    print("-" * 70)
    print("Training completed!")

    print("\nGenerating Visualizations...")
    generate_all_visualizations(agent, rewards, epsilon_history)

    print("\nBaseline vs Learned Policy:")
    env = TrafficEnv(max_steps=200)
    random_reward = random_policy(env, episodes=100)
    q_reward = evaluate_policy(env, agent, episodes=100)
    improvement = ((q_reward - random_reward) / abs(random_reward)) * 100

    print(f"   Random Policy Avg Reward : {random_reward:.2f}")
    print(f"   Q-Learning Avg Reward    : {q_reward:.2f}")
    print(f"   Improvement over Random  : {improvement:.2f}%")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Results saved to: results/")
    print(f"Model saved to: models/q_table.pkl")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
