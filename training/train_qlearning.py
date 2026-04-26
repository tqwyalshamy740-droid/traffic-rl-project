import numpy as np
import pickle
import sys
import os

sys.path.insert(0, ".")

from environment.traffic_env import TrafficEnv
from agents.qlearning_agent import QLearningAgent

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
            return i * 200
    return len(rewards) * 200


def compute_stability(rewards, window=FINAL_WINDOW):
    return np.std(rewards[-window:])


def print_metrics(metrics, algorithm_name="Q-Learning"):
    print(f"\n{'='*50}")
    print(f"📊 {algorithm_name} Evaluation Metrics")
    print(f"{'='*50}")
    print(f"Sample Efficiency (episodes to reach {REWARD_THRESHOLD}): {metrics['sample_efficiency']}")
    print(f"Final Performance (avg last {FINAL_WINDOW} episodes): {metrics['final_performance']:.2f}")
    print(f"Convergence Speed (timesteps): {metrics['convergence_speed']:.0f}")
    print(f"Stability (std dev last {FINAL_WINDOW} episodes): {metrics['stability']:.4f}")
    print(f"{'='*50}\n")


def train(episodes=500):
    env = TrafficEnv(max_steps=200)
    agent = QLearningAgent()

    rewards_history = []
    epsilon_history = []

    for ep in range(episodes):
        state = agent.get_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state_raw, reward, done, _ = env.step(action)
            next_state = agent.get_state(next_state_raw)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg:.2f} | Epsilon: {agent.epsilon:.4f}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("results/qlearning_rewards.npy", np.array(rewards_history))
    with open("models/q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

    print("Q-table saved successfully")

    metrics = {
        "sample_efficiency": compute_sample_efficiency(rewards_history),
        "final_performance": compute_final_performance(rewards_history),
        "convergence_speed": compute_convergence_speed(rewards_history),
        "stability": compute_stability(rewards_history),
    }
    print_metrics(metrics, "Q-Learning")

    return agent, np.array(rewards_history), np.array(epsilon_history)
