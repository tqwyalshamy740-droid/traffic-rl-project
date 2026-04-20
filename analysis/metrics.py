import numpy as np


def compute_sample_efficiency(rewards, threshold=50.0):
    for i, reward in enumerate(rewards):
        if reward >= threshold:
            return i
    return len(rewards)


def compute_final_performance(rewards, window=50):
    return np.mean(rewards[-window:])


def compute_convergence_speed(rewards, threshold=50.0, window=50):
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window : i]) >= threshold:
            return i * 1000
    return len(rewards) * 1000


def compute_stability(rewards, window=50):
    final_rewards = rewards[-window:]
    return np.std(final_rewards)


def evaluate_dqn(rewards, threshold=50.0):
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
    print(f"Sample Efficiency (episodes): {metrics['sample_efficiency']}")
    print(f"Final Performance (avg reward): {metrics['final_performance']:.2f}")
    print(f"Convergence Speed (timesteps): {metrics['convergence_speed']:.0f}")
    print(f"Stability (std dev): {metrics['stability']:.4f}")
    print(f"{'='*50}\n")
