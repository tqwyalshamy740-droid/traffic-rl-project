"""
Q-Learning Analysis Module

Description:
Provides evaluation metrics:
- Random baseline policy
- Learned policy performance
- Improvement percentage
"""

import numpy as np
import random


# ==============================
# Baseline Policy
# ==============================
def random_policy(env, episodes=100):
    """
    Evaluate random actions as baseline
    """
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
# Learned Policy Evaluation
# ==============================
def evaluate_policy(env, agent, episodes=100):
    """
    Evaluate greedy policy derived from Q-table
    """
    rewards = []

    for _ in range(episodes):
        state = agent.get_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(agent.Q[state])
            next_state, reward, done, _ = env.step(action)
            state = agent.get_state(next_state)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)
