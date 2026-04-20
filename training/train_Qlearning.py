from traffic_env import TrafficEnv
import random
import numpy as np
from collections import defaultdict

# Traffic environment instance
env = TrafficEnv()

# Q-table initialization (state-action values)
Q = defaultdict(lambda: np.zeros(3))

# Learning hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01

random.seed(42)
np.random.seed(42)

episodes = 1500
rewards_history = []
epsilon_history = []


# ==============================
# State Discretization
# ==============================

def discretize(x):
    return min(x // 5, 4)


def get_state(s):
    cars_ns, cars_ew, green, _ = s
    return (
        discretize(cars_ns),
        discretize(cars_ew),
        min(green // 10, 5)
    )


# ==============================
# Q-Learning Training
# ==============================

for ep in range(episodes):

    state = get_state(env.reset())
    total_reward = 0
    done = False

    while not done:

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(Q[state])

        next_state_raw, reward, done, _ = env.step(action)
        next_state = get_state(next_state_raw)

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)
    epsilon_history.append(epsilon)

    epsilon = max(min_epsilon, epsilon * epsilon_decay)


def get_training_results():
    return Q, rewards_history, epsilon_history