"""
===========================================================
Traffic Signal Control - Q-Learning Entry Point
===========================================================

This script is the main orchestrator of the experiment.

It connects all system components:
- Environment (traffic simulation)
- Agent (Q-Learning)
- Training pipeline
- Evaluation metrics
- Visualization tools
- Model persistence

Flow:
1. Train Q-Learning agent
2. Evaluate against random baseline
3. Compute performance improvement
4. Visualize learning behavior
5. Save trained Q-table

This file acts as the "main experiment runner".
===========================================================
"""

from training.train_qlearning import train
from analysis.metrics import evaluate_random, evaluate_q
from analysis.visualization import (
    plot_rewards,
    plot_epsilon,
    plot_smoothed,
    plot_policy
)
from analysis.logger import save_q_table
from environment.traffic_env import TrafficEnv


def main():
    """
    Main execution pipeline for the RL experiment.
    """

    print("\n====================================")
    print("🚦 Q-Learning Traffic Control System")
    print("====================================\n")

    # -----------------------------------------------------
    # 1. TRAINING PHASE
    # -----------------------------------------------------
    # The agent interacts with the environment and learns
    # an optimal policy using Q-Learning updates.
    agent, rewards, epsilon_history = train(episodes=1500)

    print("\n====================================")
    print("✅ Training Completed Successfully")
    print("====================================\n")

    # -----------------------------------------------------
    # 2. EVALUATION PHASE
    # -----------------------------------------------------
    # Compare learned policy against a random baseline
    env = TrafficEnv()

    print("Evaluating Random Policy (Baseline)...")
    random_score = evaluate_random(env)

    print("Evaluating Q-Learning Policy...")
    q_score = evaluate_q(agent, env)

    # -----------------------------------------------------
    # 3. PERFORMANCE ANALYSIS
    # -----------------------------------------------------
    improvement = ((q_score - random_score) / abs(random_score)) * 100

    print("\n=========== PERFORMANCE RESULTS ===========")
    print(f"Random Policy Reward : {random_score:.2f}")
    print(f"Q-Learning Reward    : {q_score:.2f}")
    print(f"Performance Gain     : {improvement:.2f}%")
    print("==========================================\n")

    # -----------------------------------------------------
    # 4. VISUALIZATION PHASE
    # -----------------------------------------------------
    # Show learning dynamics and policy behavior
    print("Generating training visualizations...")

    plot_rewards(rewards)           # learning curve
    plot_smoothed(rewards)         # trend analysis
    plot_epsilon(epsilon_history)  # exploration decay
    plot_policy(agent.Q)           # learned policy heatmap

    # -----------------------------------------------------
    # 5. MODEL SAVING
    # -----------------------------------------------------
    save_q_table(agent.Q)

    print("\nModel saved successfully -> models/q_table.pkl")
    print(" Experiment finished successfully")


# ---------------------------------------------------------
# Entry Point Guard (Best Practice in Python Projects)
# ---------------------------------------------------------
if __name__ == "__main__":
    main()