"""
Q-Learning Agent for Traffic Signal Control

Description:
Defines a Tabular Q-Learning agent responsible for:
- State discretization
- Action selection using epsilon-greedy policy
- Q-table updates using Bellman equation
"""

import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning Agent.

    Learns optimal policy using Q-table:
    Q(state, action) → expected reward
    """

    def __init__(self, actions=3,
                 alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.99,
                 min_epsilon=0.01):

        # Q-table initialization
        self.Q = defaultdict(lambda: np.zeros(actions))

        # Learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.actions = actions

        # Reproducibility
        random.seed(42)
        np.random.seed(42)

    # ==============================
    # State Discretization
    # ==============================
    def discretize(self, x):
        """
        Convert continuous values into discrete bins
        to enable tabular Q-learning
        """
        return min(x // 5, 4)

    def get_state(self, s):
        """
        Transform raw environment state into discrete representation

        State format:
        (cars_ns, cars_ew, green_phase, _)
        """
        cars_ns, cars_ew, green, _ = s

        return (
            self.discretize(cars_ns),
            self.discretize(cars_ew),
            min(green // 10, 5)
        )

    # ==============================
    # Action Selection
    # ==============================
    def choose_action(self, state):
        """
        Epsilon-greedy policy:
        - Explore (random action)
        - Exploit (best known action)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        return np.argmax(self.Q[state])

    # ==============================
    # Q-Learning Update
    # ==============================
    def update(self, state, action, reward, next_state):
        """
        Bellman Equation:
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
        """
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

    def decay_epsilon(self):
        """
        Gradually reduce exploration over time
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
