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


def plot_q_values_heatmap(
    agent, state_samples=500, save_path="results/dqn_qvalues.png"
):
    q_values_list = []

    for _ in range(state_samples):
        state = np.random.uniform(0, 1, agent.state_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.main_network(state_tensor).cpu().numpy()[0]
        q_values_list.append(q_values)

    q_values_matrix = np.array(q_values_list)

    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        q_values_matrix.T, aspect="auto", cmap="RdYlGn", interpolation="nearest"
    )
    plt.colorbar(im, label="Q-Value")
    plt.xlabel("State Sample Index", fontsize=12)
    plt.ylabel("Action Index", fontsize=12)
    plt.title(
        "DQN Q-Values Heatmap Across Different States", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Q-values heatmap saved to {save_path}")


def plot_policy_visualization(agent, grid_size=50, save_path="results/dqn_policy.png"):
    policy_grid = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            state = np.array([i / grid_size, j / grid_size, 0.5, 0.5], dtype=np.float32)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.main_network(state_tensor).cpu().numpy()[0]
            policy_grid[i, j] = np.argmax(q_values)

    plt.figure(figsize=(10, 8))
    plt.imshow(policy_grid, cmap="tab10", interpolation="nearest")
    plt.colorbar(label="Action Index", shrink=0.8)
    plt.xlabel("NS Cars Ratio", fontsize=12)
    plt.ylabel("EW Cars Ratio", fontsize=12)
    plt.title("DQN Learned Policy Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Policy visualization saved to {save_path}")


def plot_stability(
    agent, env, runs=10, episodes_per_run=100, save_path="results/dqn_stability.png"
):
    all_rewards = []

    for run in range(runs):
        run_rewards = []

        for episode in range(episodes_per_run):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 1000:
                state_tensor = (
                    torch.FloatTensor(np.array(state, dtype=np.float32))
                    .unsqueeze(0)
                    .to(agent.device)
                )
                with torch.no_grad():
                    q_values = agent.main_network(state_tensor)
                action = q_values.argmax(dim=1).item()

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
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
    plot_q_values_heatmap(agent)
    plot_policy_visualization(agent)
    plot_stability(agent, env)

    print("-" * 50)
    print("✅ All visualizations completed!")
