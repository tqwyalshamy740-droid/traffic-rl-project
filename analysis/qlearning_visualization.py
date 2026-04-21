"""
Q-Learning Visualization Module

Description:
Contains all visualization tools:
- Reward curves
- Epsilon decay
- Policy heatmap
- Stability analysis
"""

import matplotlib.pyplot as plt
import numpy as np


# ==============================
# Learning Curves
# ==============================
def plot_rewards(avg_rewards):
    plt.figure()
    plt.plot(avg_rewards)
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()


def plot_epsilon(epsilon_history):
    plt.figure()
    plt.plot(epsilon_history)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()


def plot_training_curve(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.title("Q-Learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


# ==============================
# Smoothed Curve
# ==============================
def plot_smoothed(rewards, window=30):
    if len(rewards) > window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')

        plt.figure()
        plt.plot(smooth)
        plt.title("Smoothed Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward")
        plt.show()


# ==============================
# Policy Visualization
# ==============================
def plot_policy(agent):
    policy_grid = np.zeros((5, 5))

    for ns in range(5):
        for ew in range(5):
            policy_grid[ns][ew] = np.argmax(agent.Q[(ns, ew, 0)])

    plt.figure()
    plt.imshow(policy_grid, cmap="coolwarm")
    plt.title("Policy Visualization (Green Phase)")
    plt.xlabel("EW Traffic")
    plt.ylabel("NS Traffic")
    plt.colorbar()
    plt.show()


# ==============================
# Stability Analysis
# ==============================
def plot_stability(rewards, window=50):
    if len(rewards) > window:
        stability = [
            np.std(rewards[i:i+window])
            for i in range(len(rewards) - window)
        ]

        plt.figure()
        plt.plot(stability)
        plt.title("Training Stability")
        plt.xlabel("Episode")
        plt.ylabel("Variance")
        plt.show()