"""
Q-Learning Training Script

Description:
Handles training loop for the Q-Learning agent.
Tracks:
- Rewards per episode
- Exploration decay
- Saves trained Q-table
"""

from traffic_env import TrafficEnv
from agents.qlearning_agent import QLearningAgent
import pickle


def train(episodes=1500):
    # Environment setup
    env = TrafficEnv()

    # Agent initialization
    agent = QLearningAgent()

    rewards_history = []
    epsilon_history = []

    # ==============================
    # Training Loop
    # ==============================
    for ep in range(episodes):

        state = agent.get_state(env.reset())
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.choose_action(state)

            # Environment step
            next_state_raw, reward, done, _ = env.step(action)
            next_state = agent.get_state(next_state_raw)

            # Q-table update
            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # Logging
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        # Exploration decay
        agent.decay_epsilon()

    # ==============================
    # Save Model
    # ==============================
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)

    print("Q-table saved successfully")

    return agent, rewards_history, epsilon_history
