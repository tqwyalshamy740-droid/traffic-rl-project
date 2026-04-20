from training.train_qlearning import get_training_results
from traffic_env import TrafficEnv
import numpy as np
import random
import pickle

from analysis.q_analysis import *

# ==============================
# Train
# ==============================
Q, rewards_history, epsilon_history = get_training_results()

env = TrafficEnv()


# ==============================
# Evaluation
# ==============================

def evaluate_policy(env, Q, episodes=100):
    rewards = []

    def discretize(x):
        return min(x // 5, 4)

    def get_state(s):
        cars_ns, cars_ew, green, _ = s
        return (
            discretize(cars_ns),
            discretize(cars_ew),
            min(green // 10, 5)
        )

    for _ in range(episodes):
        state = get_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            state = get_state(next_state)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)


def random_policy(env, episodes=100):
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = random.randint(0, 2)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)


# ==============================
# Compare
# ==============================
q_reward = evaluate_policy(env, Q)
r_reward = random_policy(env)

print("Random Policy Avg Reward:", r_reward)
print("Q-Learning Avg Reward:", q_reward)

improvement = ((q_reward - r_reward) / abs(r_reward)) * 100
print(f"Improvement over Random Policy: {improvement:.2f}%")


# ==============================
# Save Model
# ==============================
with open("models/q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

print("Q-table saved successfully")


# ==============================
# ANALYSIS CALLS
# ==============================
plot_training(rewards_history)
plot_smooth(rewards_history)
plot_epsilon(epsilon_history)
plot_stability(rewards_history)
plot_policy(Q)