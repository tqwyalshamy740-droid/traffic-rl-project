import glob
import os
from typing import Dict, List

import numpy as np

REWARD_THRESHOLD = -1200.0
FINAL_WINDOW = 50
DEFAULT_STEPS_PER_EPISODE = 200
RESULTS_DIR = "results"


def _safe_mean_last_window(rewards: np.ndarray, window: int) -> float:
    if rewards.size == 0:
        return float("nan")
    return float(np.mean(rewards[-min(window, rewards.size) :]))


def compute_sample_efficiency(rewards: np.ndarray, threshold: float) -> int:
    """Episodes needed to reach threshold reward once."""
    for i, reward in enumerate(rewards):
        if reward >= threshold:
            return i + 1
    return int(rewards.size)


def compute_convergence_speed(
    rewards: np.ndarray,
    threshold: float,
    window: int,
    steps_per_episode: int,
) -> int:
    """Timesteps needed for moving average to exceed threshold."""
    if rewards.size == 0:
        return 0

    if rewards.size < window:
        moving_avg = np.array([np.mean(rewards)])
        start_index = 1
    else:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        start_index = window

    for i, avg_reward in enumerate(moving_avg):
        if avg_reward >= threshold:
            episode_idx = start_index + i
            return int(episode_idx * steps_per_episode)

    return int(rewards.size * steps_per_episode)


def compute_stability_variance(all_runs_rewards: List[np.ndarray], window: int) -> float:
    """Variance of final performance across independent runs."""
    if not all_runs_rewards:
        return float("nan")

    final_performances = [_safe_mean_last_window(run, window) for run in all_runs_rewards]
    return float(np.var(final_performances))


def _load_runs(pattern: str) -> List[np.ndarray]:
    files = sorted(glob.glob(pattern))
    runs: List[np.ndarray] = []
    for file_path in files:
        arr = np.load(file_path)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        runs.append(arr.astype(np.float64))
    return runs


def _candidate_results_dirs() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    parent_of_project = os.path.dirname(project_root)

    candidates = [
        os.path.abspath(RESULTS_DIR),
        os.path.join(project_root, RESULTS_DIR),
        os.path.join(parent_of_project, RESULTS_DIR),
    ]

    # Keep order, remove duplicates.
    unique_dirs: List[str] = []
    for path in candidates:
        if path not in unique_dirs:
            unique_dirs.append(path)
    return unique_dirs


def _collect_algorithm_runs(algorithm_key: str) -> List[np.ndarray]:
    """
    Load runs for an algorithm.
    Supports:
    - results/<algorithm_key>_rewards.npy
    - results/<algorithm_key>_rewards_run*.npy
    - results/<algorithm_key>/<algorithm_key>_rewards*.npy
    """
    runs: List[np.ndarray] = []
    for results_dir in _candidate_results_dirs():
        base_pattern = os.path.join(results_dir, f"{algorithm_key}_rewards*.npy")
        nested_pattern = os.path.join(
            results_dir, algorithm_key, f"{algorithm_key}_rewards*.npy"
        )
        runs.extend(_load_runs(base_pattern))
        runs.extend(_load_runs(nested_pattern))

    return runs


def evaluate_algorithm(
    algorithm_name: str,
    algorithm_key: str,
    threshold: float = REWARD_THRESHOLD,
    final_window: int = FINAL_WINDOW,
    steps_per_episode: int = DEFAULT_STEPS_PER_EPISODE,
) -> Dict[str, float]:
    runs = _collect_algorithm_runs(algorithm_key)
    if not runs:
        raise FileNotFoundError(
            f"No reward files found for {algorithm_name}. "
            f"Expected files like: results/{algorithm_key}_rewards.npy"
        )

    main_run = runs[0]
    metrics = {
        "sample_efficiency_episodes": compute_sample_efficiency(main_run, threshold),
        "final_performance_mean_reward": _safe_mean_last_window(main_run, final_window),
        "convergence_speed_timesteps": compute_convergence_speed(
            main_run, threshold, final_window, steps_per_episode
        ),
        "stability_variance_across_runs": compute_stability_variance(runs, final_window),
        "num_runs_used_for_stability": len(runs),
    }
    return metrics


def _print_algorithm_metrics(algorithm_name: str, metrics: Dict[str, float]) -> None:
    print(f"\n{algorithm_name}")
    print("-" * len(algorithm_name))
    print(
        f"Sample efficiency (episodes to threshold): "
        f"{int(metrics['sample_efficiency_episodes'])}"
    )
    print(
        f"Final performance (mean reward last {FINAL_WINDOW} episodes): "
        f"{metrics['final_performance_mean_reward']:.3f}"
    )
    print(
        f"Convergence speed (timesteps to stable policy): "
        f"{int(metrics['convergence_speed_timesteps'])}"
    )
    print(
        "Stability (variance in final performance across independent runs): "
        f"{metrics['stability_variance_across_runs']:.6f}"
    )
    print(f"Independent runs used for stability: {int(metrics['num_runs_used_for_stability'])}")


def compare_models() -> None:
    print("\n===== Q-Learning vs DQN: Final Comparison =====")
    print(f"Threshold reward: {REWARD_THRESHOLD}")
    print(f"Final performance window: {FINAL_WINDOW} episodes")
    print(f"Steps per episode for convergence metric: {DEFAULT_STEPS_PER_EPISODE}")

    try:
        qlearning_metrics = evaluate_algorithm("Q-Learning", "qlearning")
    except FileNotFoundError as exc:
        print(f"\n[Missing Data] {exc}")
        return

    try:
        dqn_metrics = evaluate_algorithm("DQN", "dqn")
    except FileNotFoundError as exc:
        print(f"\n[Missing Data] {exc}")
        return

    _print_algorithm_metrics("Q-Learning", qlearning_metrics)
    _print_algorithm_metrics("DQN", dqn_metrics)

    if (
        qlearning_metrics["num_runs_used_for_stability"] < 2
        or dqn_metrics["num_runs_used_for_stability"] < 2
    ):
        print(
            "\nNote: Stability metric is most meaningful with >=2 independent runs "
            "per algorithm."
        )

    print("\n===== End of Comparison =====\n")


if __name__ == "__main__":
    compare_models()
