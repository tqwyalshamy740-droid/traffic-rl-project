import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards):
    """
    Visualizes raw reward per episode.
    Shows learning trend over time.
    """
    plt.figure()
    plt.plot(rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


def plot_smoothed(rewards, window=30):
    """
    Applies moving average smoothing to reduce noise
    and highlight learning trend.
    """
    if len(rewards) < window:
        return

    smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.figure()
    plt.plot(smooth)
    plt.title("Smoothed Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()


def plot_epsilon(eps):
    """
    Shows exploration decay over training.
    Confirms transition from exploration → exploitation.
    """
    plt.figure()
    plt.plot(eps)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()


def plot_policy(Q):
    """
    Visualizes learned policy as heatmap.

    Each cell represents best action for a traffic state.
    """
    grid = np.zeros((5, 5))

    for ns in range(5):
        for ew in range(5):
            grid[ns][ew] = np.argmax(Q[(ns, ew, 0)])

    plt.imshow(grid)
    plt.title("Learned Traffic Policy (Green Phase)")
    plt.colorbar()
    plt.show()