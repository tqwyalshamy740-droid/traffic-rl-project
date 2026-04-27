import numpy as np

REWARD_THRESHOLD = -1200.0
FINAL_WINDOW = 50
STABILITY_RUNS = 5


def compute_sample_efficiency(rewards, threshold=REWARD_THRESHOLD):
    for i, reward in enumerate(rewards):
        if reward >= threshold:
            return i
    return len(rewards)


def compute_final_performance(rewards, window=FINAL_WINDOW):
    return np.mean(rewards[-window:])


def compute_convergence_speed(rewards, threshold=REWARD_THRESHOLD, window=FINAL_WINDOW):
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window : i]) >= threshold:
            return i * 1000
    return len(rewards) * 1000


def compute_stability(rewards, window=FINAL_WINDOW):
    final_rewards = rewards[-window:]
    return np.std(final_rewards)


def evaluate_dqn(rewards, threshold=REWARD_THRESHOLD):
    metrics = {
        "sample_efficiency": compute_sample_efficiency(rewards, threshold),
        "final_performance": compute_final_performance(rewards),
        "convergence_speed": compute_convergence_speed(rewards, threshold),
        "stability": compute_stability(rewards),
    }
    return metrics


def print_metrics(metrics, algorithm_name="DQN"):
    print(f"\n{'='*50}")
    print(f"📊 {algorithm_name} Evaluation Metrics")
    print(f"{'='*50}")
    print(
        f"Sample Efficiency (episodes to reach {REWARD_THRESHOLD}): {metrics['sample_efficiency']}"
    )
    print(
        f"Final Performance (avg last {FINAL_WINDOW} episodes): {metrics['final_performance']:.2f}"
    )
    print(f"Convergence Speed (timesteps): {metrics['convergence_speed']:.0f}")
    print(
        f"Stability (std dev last {FINAL_WINDOW} episodes): {metrics['stability']:.4f}"
    )
    print(f"{'='*50}\n")
