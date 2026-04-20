"""
Q-Learning for Traffic Signal Control
Author: Engineer 2

Description:
Implements Tabular Q-Learning algorithm to optimize traffic signal timing
using a custom reinforcement learning environment.

Features:
- Epsilon-greedy policy
- State discretization
- Q-table update using Bellman equation
- Training analysis and evaluation
- Policy visualization and stability analysis
"""
# ==============================
# Setup & Hyperparameters
# ==============================
from traffic_env import TrafficEnv
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# Traffic environment instance
env = TrafficEnv()

# Q-table initialization (state-action values)
Q = defaultdict(lambda: np.zeros(3))

# Learning hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor for future rewards
epsilon = 1.0  # initial exploration rate
epsilon_decay = 0.99  # decay factor for exploration
min_epsilon = 0.01  # minimum exploration limit

# Reproducibility
random.seed(42)
np.random.seed(42)

episodes = 1500
rewards_history = []
epsilon_history = []

# ==============================
# State Discretization
# ==============================

def discretize(x):
    """
    Reduce continuous state space into discrete bins
    to make tabular Q-Learning feasible
    """
    return min(x // 5, 4)


def get_state(s):
    cars_ns, cars_ew, green, _ = s

    # State representation:
    # (North-South traffic, East-West traffic, green phase level)
    return (
        discretize(cars_ns),
        discretize(cars_ew),
        min(green // 10, 5)
    )

# ==============================
# Q-Learning Training
# ==============================

# Training loop over episodes
for ep in range(episodes):
    state = get_state(env.reset())
    total_reward = 0
    done = False

    while not done:

        # epsilon-greedy policy (exploration vs exploitation)
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Q[state])

        next_state_raw, reward, done, _ = env.step(action)
        next_state = get_state(next_state_raw)

        # Q-value update rule (Bellman equation)
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)
    epsilon_history.append(epsilon)

    # Decay exploration rate over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# ==============================
# Learning Analysis
# ==============================

# Normalize rewards per episode
avg_rewards = [r / env.max_steps for r in rewards_history]

# Learning performance curve
plt.figure()
plt.plot(avg_rewards)
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()

# Exploration decay visualization
plt.figure()
plt.plot(epsilon_history)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.show()

# ==============================
# Evaluation
# ==============================

# Random policy baseline for comparison
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


# Greedy policy evaluation using learned Q-table
def evaluate_policy(env, Q, episodes=100):
    rewards = []

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


test_env = TrafficEnv()

random_reward = random_policy(test_env)
eval_env = TrafficEnv()
q_reward = evaluate_policy(eval_env, Q)

print("Random Policy Avg Reward:", random_reward)
print("Q-Learning Avg Reward:", q_reward)

# ==============================
# Improvement Analysis
# ==============================

# Performance improvement over baseline
improvement = ((q_reward - random_reward) / abs(random_reward)) * 100
print(f"Improvement over Random Policy: {improvement:.2f}%")

# ==============================
# Training Curve
# ==============================

# Raw learning curve
plt.figure()
plt.plot(rewards_history)
plt.title("Q-Learning Training Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# Smoothed curve for trend visualization
window = 30
if len(rewards_history) > window:
    smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')

    plt.figure()
    plt.plot(smooth)
    plt.title("Smoothed Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.show()

# ==============================
# Policy Extraction
# ==============================

# Extract best action per state from Q-table
policy = {state: np.argmax(actions) for state, actions in Q.items()}

print("\nSample Policy:\n")
for i, (state, action) in enumerate(policy.items()):
    if i > 15:
        break
    print(f"State {state} -> Action {action}")

# ==============================
# Policy Visualization
# ==============================

# Heatmap-style visualization of learned policy
policy_grid = np.zeros((5, 5))

for ns in range(5):
    for ew in range(5):
        actions = Q[(ns, ew, 0)]
        policy_grid[ns][ew] = np.argmax(actions)
plt.figure()
plt.imshow(policy_grid, cmap="coolwarm")
plt.title("Policy Visualization (Green Phase)")
plt.xlabel("EW Traffic")
plt.ylabel("NS Traffic")
plt.colorbar()
plt.show()

# ==============================
# Q-values Inspection
# ==============================

# Inspect sample learned Q-values
print("\nSample Q-values:\n")

for i, (state, actions) in enumerate(Q.items()):
    if i > 10:
        break
    print(state, "->", actions)

# ==============================
# Stability Analysis
# ==============================

# Measure training stability using reward variance
window = 50

if len(rewards_history) > window:
    stability = [
        np.std(rewards_history[i:i+window])
        for i in range(len(rewards_history) - window)
    ]

    plt.figure()
    plt.plot(stability)
    plt.title("Training Stability")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.show()

# ==============================
# Save Model
# ==============================

# Save trained Q-table for later use
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

print("Q-table saved successfully")