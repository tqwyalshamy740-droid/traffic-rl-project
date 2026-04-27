import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)


def plot_learning_curve(
    rewards, window=50, save_path="results/qlearning_learning_curve.png"
):
    fig, ax = plt.subplots(figsize=(14, 6))

    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    episodes = np.arange(len(moving_avg))

    ax.plot(rewards, alpha=0.3, color="green", label="Raw Reward")
    ax.plot(
        episodes,
        moving_avg,
        linewidth=2.5,
        color="darkgreen",
        label=f"Moving Average (window={window})",
    )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(
        "Q-Learning Learning Curve - Traffic Signal Control",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Learning curve saved to {save_path}")


def plot_epsilon(epsilon_history, save_path="results/qlearning_epsilon.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history, color="orange", linewidth=2)
    plt.title("Epsilon Decay - Q-Learning", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Epsilon", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Epsilon decay saved to {save_path}")


def plot_q_values_heatmap(agent, save_path="results/qlearning_qvalues.png"):
    q_grid = np.zeros((5, 5, 3))

    for ns in range(5):
        for ew in range(5):
            for green in range(3):
                state = (ns, ew, green)
                q_grid[ns, ew, green] = np.max(agent.Q[state])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    green_labels = ["Short (0-9s)", "Medium (10-19s)", "Long (20s+)"]

    for g in range(3):
        im = axes[g].imshow(q_grid[:, :, g], cmap="RdYlGn", interpolation="nearest")
        axes[g].set_title(
            f"Max Q-Value\nGreen Phase: {green_labels[g]}",
            fontsize=11,
            fontweight="bold",
        )
        axes[g].set_xlabel("EW Traffic Bin", fontsize=10)
        axes[g].set_ylabel("NS Traffic Bin", fontsize=10)
        plt.colorbar(im, ax=axes[g])

    plt.suptitle("Q-Learning Q-Values Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Q-values heatmap saved to {save_path}")


def plot_policy(agent, save_path="results/qlearning_policy.png"):
    import matplotlib.patches as mpatches

    policy_grid = np.zeros((5, 5))

    for ns in range(5):
        for ew in range(5):
            policy_grid[ns][ew] = np.argmax(agent.Q[(ns, ew, 0)])

    cmap = plt.cm.get_cmap("Set1", 3)

    plt.figure(figsize=(8, 7))
    img = plt.imshow(
        policy_grid, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest"
    )

    cbar = plt.colorbar(img, shrink=0.8, ticks=[0, 1, 2])
    cbar.set_ticklabels(["0 — Keep", "1 — Increase +5s", "2 — Decrease -5s"])
    cbar.set_label("Action", fontsize=12)

    action_counts = {0: 0, 1: 0, 2: 0}
    for v in policy_grid.flatten():
        action_counts[int(v)] += 1
    total = 25
    patches = [
        mpatches.Patch(
            color=cmap(i), label=f"Action {i}: {action_counts[i]/total*100:.1f}%"
        )
        for i in range(3)
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=10)

    plt.title("Q-Learning Learned Policy Visualization", fontsize=14, fontweight="bold")
    plt.xlabel("EW Traffic Bin (0=low → 4=high)", fontsize=12)
    plt.ylabel("NS Traffic Bin (0=low → 4=high)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Policy visualization saved to {save_path}")


def plot_stability(
    rewards, runs=5, window=50, save_path="results/qlearning_stability.png"
):
    import sys

    sys.path.insert(0, ".")
    from environment.traffic_env import TrafficEnv
    from agents.qlearning_agent import QLearningAgent

    all_rewards = []

    for run in range(runs):
        env = TrafficEnv(max_steps=200)
        agent = QLearningAgent()
        run_rewards = []

        for ep in range(100):
            state = agent.get_state(env.reset())
            total_reward = 0
            done = False

            while not done:
                action = np.argmax(agent.Q[state])
                next_state_raw, reward, done, _ = env.step(action)
                state = agent.get_state(next_state_raw)
                total_reward += reward

            run_rewards.append(total_reward)

        all_rewards.append(run_rewards)

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes_range = np.arange(len(mean_rewards))

    plt.figure(figsize=(14, 7))
    plt.fill_between(
        episodes_range,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color="green",
        label="±1 Std Dev",
    )
    plt.plot(
        episodes_range,
        mean_rewards,
        linewidth=2.5,
        color="darkgreen",
        label="Mean Reward",
    )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(
        f"Q-Learning Stability Plot ({runs} Independent Runs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Stability plot saved to {save_path}")


def generate_all_visualizations(agent, rewards, epsilon_history):
    print("\n📊 Generating Q-Learning Visualizations...")
    print("-" * 50)

    plot_learning_curve(rewards)
    plot_epsilon(epsilon_history)
    plot_q_values_heatmap(agent)
    plot_policy(agent)
    plot_stability(rewards)

    print("-" * 50)
    print("✅ All visualizations completed!")
