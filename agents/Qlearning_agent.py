import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def _init_(self, action_space=3, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):

        self.Q = defaultdict(lambda: np.zeros(action_space))

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.action_space = action_space

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)