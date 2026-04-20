import numpy as np
import sys
import os

sys.path.insert(0, ".")

from environment.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

REWARD_SCALE = 50.0


def _maybe_normalize(env, state, normalize_obs: bool):
    if normalize_obs:
        return env.get_state_normalized()
    return state


def train_dqn(env, agent, episodes=500, max_steps=1000, normalize_obs=True):
    episode_rewards = []
    episode_losses = []

    for episode in range(episodes):
        raw_state = env.reset()
        state = _maybe_normalize(env, raw_state, normalize_obs)
        episode_reward = 0
        episode_loss = 0
        loss_count = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_raw, reward, done, _ = env.step(action)
            next_state = _maybe_normalize(env, next_raw, normalize_obs)

            scaled_reward = reward / REWARD_SCALE
            agent.remember(state, action, scaled_reward, next_state, done)

            loss = agent.replay()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}"
            )

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("results/dqn_rewards.npy", np.array(episode_rewards))
    np.save("results/dqn_losses.npy", np.array(episode_losses))
    agent.save("models/dqn_final.pt")

    return np.array(episode_rewards), np.array(episode_losses)


if __name__ == "__main__":
    env = TrafficEnv(
        max_steps=200, throughput_bonus_weight=2.0, reward_balance_weight=0.1
    )
    agent = DQNAgent(
        state_size=4,
        action_size=3,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.990,
        use_double_dqn=True,
    )

    print("🚀 Starting DQN Training...")
    print("=" * 60)
    rewards, losses = train_dqn(env, agent, episodes=500, normalize_obs=True)
    print("=" * 60)
    print("✅ Training completed!")
    print(f"Final average reward: {np.mean(rewards[-50:]):.2f}")
