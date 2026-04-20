import numpy as np


def evaluate_random(env, episodes=100):
    """
    Baseline evaluation using random policy.
    Used for comparison against learned agent.
    """

    rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0

        while not done:
            action = np.random.randint(0, 3)
            _, reward, done, _ = env.step(action)
            total += reward

        rewards.append(total)

    return np.mean(rewards)


def evaluate_q(agent, env, episodes=100):
    """
    Evaluation of trained Q-Learning agent using greedy policy
    (no exploration).
    """

    rewards = []

    for _ in range(episodes):
        state = agent.get_state(env.reset())
        done = False
        total = 0

        while not done:
            action = int(np.argmax(agent.Q[state]))
            next_obs, reward, done, _ = env.step(action)
            state = agent.get_state(next_obs)
            total += reward

        rewards.append(total)

    return np.mean(rewards)