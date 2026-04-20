import numpy as np
import matplotlib.pyplot as plt


def plot_training(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.title("Q-Learning Training Curve")
    plt.show()


def plot_smooth(rewards, window=30):
    if len(rewards) > window:
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.figure()
        plt.plot(smooth)
        plt.title("Smoothed Learning Curve")
        plt.show()


def plot_epsilon(eps):
    plt.figure()
    plt.plot(eps)
    plt.title("Epsilon Decay")
    plt.show()


def plot_stability(rewards, window=50):
    if len(rewards) > window:
        var = [
            np.std(rewards[i:i+window])
            for i in range(len(rewards)-window)
        ]
        plt.figure()
        plt.plot(var)
        plt.title("Training Stability")
        plt.show()


def plot_policy(Q):
    grid = np.zeros((5, 5))

    for ns in range(5):
        for ew in range(5):
            grid[ns][ew] = np.argmax(Q[(ns, ew, 0)])

    plt.figure()
    plt.imshow(grid, cmap="coolwarm")
    plt.title("Policy Visualization (Green Phase)")
    plt.colorbar()
    plt.show()