import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.990,
        use_double_dqn=True,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_double_dqn = use_double_dqn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.target_update_frequency = 500
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if isinstance(state, tuple):
            state = np.array(state, dtype=np.float32)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = (
            self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.main_network(next_states).argmax(
                    dim=1, keepdim=True
                )
                next_q_values = (
                    self.target_network(next_states).gather(1, next_actions).squeeze(1)
                )
            else:
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        return loss.item()

    def save(self, filepath):
        torch.save(self.main_network.state_dict(), filepath)

    def load(self, filepath):
        self.main_network.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )
        self.target_network.load_state_dict(self.main_network.state_dict())
