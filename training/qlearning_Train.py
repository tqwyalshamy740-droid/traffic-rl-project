from environment.traffic_env import TrafficEnv
from agents.q_learning_agent import QLearningAgent


def train(episodes=1500):
    """
    Main training loop for Q-Learning agent.

    Each episode represents a full traffic simulation run.
    The agent interacts with the environment and updates Q-table.
    """

    env = TrafficEnv()
    agent = QLearningAgent()

    rewards_history = []
    epsilon_history = []

    for ep in range(episodes):

        # Reset environment at start of episode
        state = agent.get_state(env.reset())

        done = False
        total_reward = 0

        while not done:

            # Choose action using epsilon-greedy policy
            action = agent.act(state)

            # Apply action in environment
            next_obs, reward, done, _ = env.step(action)
            next_state = agent.get_state(next_obs)

            # Update Q-table using Bellman equation
            agent.update(state, action, reward, next_state)

            # Move to next state
            state = next_state
            total_reward += reward

        # Decay exploration after each episode
        agent.decay()

        # Store metrics for analysis
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        print(f"[Episode {ep+1}/{episodes}] Reward: {total_reward:.2f}")

    return agent, rewards_history, epsilon_history