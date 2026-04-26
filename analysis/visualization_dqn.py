import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.insert(0, ".")


def plot_learning_curve(
    rewards, losses=None, window=50, save_path="results/dqn_learning_curve.png"
):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    episodes = np.arange(len(moving_avg))

    ax1.plot(rewards, alpha=0.3, color="blue", label="Raw Reward")
    ax1.plot(
        episodes,
        moving_avg,
        linewidth=2.5,
        color="darkblue",
        label=f"Moving Average (window={window})",
    )
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Total Reward", fontsize=12)
    ax1.set_title(
        "DQN Learning Curve - Traffic Signal Control", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    if losses is not None:
        ax2 = ax1.twinx()
        ax2.plot(losses, alpha=0.2, color="red", linestyle="--", linewidth=1.5)
        ax2.set_ylabel("Loss", fontsize=12, color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Learning curve saved to {save_path}")


def _collect_real_states(env, n=400):
    states = []
    env.reset()
    done = False
    step = 0
    while len(states) < n:
        action = np.random.randint(0, 3)
        _, _, done, _ = env.step(action)
        norm = env.get_state_normalized()
        states.append(np.array(norm, dtype=np.float32))
        if done or step > 200:
            env.reset()
            step = 0
        step += 1
    return states[:n]


def plot_q_values_heatmap(
    agent, env, state_samples=400, save_path="results/dqn_qvalues.png"
):
    real_states = _collect_real_states(env, n=state_samples)

    grid_size = 20
    heatmap = np.full((grid_size, grid_size, 3), np.nan)

    for s in real_states:
        ni = min(int(s[0] * grid_size), grid_size - 1)
        ei = min(int(s[1] * grid_size), grid_size - 1)
        state_tensor = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_vals = agent.main_network(state_tensor).cpu().numpy()[0]
        heatmap[ni, ei] = q_vals

    all_vals = heatmap[~np.isnan(heatmap)]
    vmin = all_vals.min() if len(all_vals) > 0 else -400
    vmax = all_vals.max() if len(all_vals) > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    action_labels = ["Action 0: Keep", "Action 1: Increase +5s", "Action 2: Decrease -5s"]

    for a in range(3):
        im = axes[a].imshow(
            heatmap[:, :, a],
            cmap="RdYlGn",
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1, 0, 1],
        )
        axes[a].set_title(action_labels[a], fontsize=11, fontweight="bold")
        axes[a].set_xlabel("EW Cars (normalized)", fontsize=10)
        axes[a].set_ylabel("NS Cars (normalized)", fontsize=10)
        plt.colorbar(im, ax=axes[a], label="Q-Value")

    plt.suptitle(
        "DQN Q-Values Heatmap — Real Environment States",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Q-values heatmap saved to {save_path}")


def plot_policy_visualization(agent, env, grid_size=20, save_path="results/dqn_policy.png"):
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm

    policy_grid = np.zeros((grid_size, grid_size))
    action_counts = {0: 0, 1: 0, 2: 0}

    for ni in range(grid_size):
        for ei in range(grid_size):
            state = np.array(
                [ni / grid_size, ei / grid_size, 0.5, 0.3], dtype=np.float32
            )
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.main_network(state_tensor).cpu().numpy()[0]
            best = int(np.argmax(q_values))
            policy_grid[ni, ei] = best
            action_counts[best] += 1

    cmap = plt.cm.get_cmap("Set1", 3)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], ncolors=3)

    plt.figure(figsize=(10, 8))
    img = plt.imshow(
        policy_grid,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="lower",
        extent=[0, 1, 0, 1],
    )

    cbar = plt.colorbar(img, shrink=0.8, ticks=[0, 1, 2])
    cbar.set_ticklabels(["0 — Keep", "1 — Increase +5s", "2 — Decrease -5s"])
    cbar.set_label("Action", fontsize=12)

    total = grid_size * grid_size
    patches = [
        mpatches.Patch(
            color=cmap(i), label=f"Action {i}: {action_counts[i]/total*100:.1f}%"
        )
        for i in range(3)
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=10)

    plt.xlabel("EW Cars Ratio (0=low → 1=high)", fontsize=12)
    plt.ylabel("NS Cars Ratio (0=low → 1=high)", fontsize=12)
    plt.title("DQN Learned Policy Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Policy visualization saved to {save_path}")


def plot_stability(
    agent, env, runs=5, episodes_per_run=100, save_path="results/dqn_stability.png"
):
    all_rewards = []

    for run in range(runs):
        run_rewards = []

        for episode in range(episodes_per_run):
            env.reset()
            state = np.array(env.get_state_normalized(), dtype=np.float32)
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 1000:
                state_tensor = (
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                )
                with torch.no_grad():
                    q_values = agent.main_network(state_tensor)
                action = q_values.argmax(dim=1).item()

                _, reward, done, _ = env.step(action)
                state = np.array(env.get_state_normalized(), dtype=np.float32)
                episode_reward += reward
                step_count += 1

            run_rewards.append(episode_reward)

        all_rewards.append(run_rewards)

    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    plt.figure(figsize=(14, 7))
    episodes_range = np.arange(len(mean_rewards))

    plt.fill_between(
        episodes_range,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color="blue",
        label="±1 Std Dev",
    )
    plt.plot(
        episodes_range,
        mean_rewards,
        linewidth=2.5,
        color="darkblue",
        label="Mean Reward",
    )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(
        f"DQN Stability Plot ({runs} Independent Runs)", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Stability plot saved to {save_path}")

    return all_rewards


def generate_all_visualizations(agent, env, rewards, losses):
    print("\n📊 Generating DQN Visualizations...")
    print("-" * 50)

    plot_learning_curve(rewards, losses)
    plot_q_values_heatmap(agent, env)
    plot_policy_visualization(agent, env)
    plot_stability(agent, env)

    print("-" * 50)
    print("✅ All visualizations completed!")
