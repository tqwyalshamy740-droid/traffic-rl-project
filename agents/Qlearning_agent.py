import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning Agent for Traffic Signal Control.

    This agent learns an optimal policy using a Q-table,
    where each state-action pair is updated using the Bellman equation.

    Suitable for discrete / discretized environments.
    """

    def _init_(self, state_bins=5, actions=3,
                 alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):

        # Q-table: maps state -> action values
        self.Q = defaultdict(lambda: np.zeros(actions))

        # Learning hyperparameters
        self.alpha = alpha          # learning rate (how fast we update)
        self.gamma = gamma          # discount factor (future reward importance)

        # Exploration parameters (epsilon-greedy strategy)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.actions = actions
        self.state_bins = state_bins

    # ----------------------------------------------------
    # State Discretization
    # ----------------------------------------------------
    def discretize(self, x):
        """
        Converts continuous traffic values into discrete bins.

        This is required because Q-learning uses a finite state space.
        """
        return min(x // 5, self.state_bins - 1)

    def get_state(self, obs):
        """
        Transforms raw environment observation into a discrete state.

        State representation:
        (North-South traffic, East-West traffic, green signal phase)
        """
        cars_ns, cars_ew, green, _ = obs

        return (
            self.discretize(cars_ns),
            self.discretize(cars_ew),
            min(green // 10, 5)
        )

    # ----------------------------------------------------
    # Action Selection (Epsilon-Greedy Policy)
    # ----------------------------------------------------
    def act(self, state):
        """
        Selects an action using epsilon-greedy policy:
        - With probability epsilon -> explore (random action)
        - Otherwise -> exploit learned Q-values
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)

        return int(np.argmax(self.Q[state]))

    # ----------------------------------------------------
    # Learning Step (Bellman Update)
    # ----------------------------------------------------
    def update(self, state, action, reward, next_state):
        """
        Core Q-learning update rule:

        Q(s,a) ← Q(s,a) + α [r + γ max Q(s') − Q(s,a)]
        """
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        )

    # ----------------------------------------------------
    # Exploration Decay
    # ----------------------------------------------------
    def decay(self):
        """
        Gradually reduces exploration rate over time
        to shift from exploration → exploitation.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)