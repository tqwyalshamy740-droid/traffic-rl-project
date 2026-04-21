"""
Q-Learning Runner Script

Description:
Main execution file:
- Runs training
- Evaluates performance
- Generates visualizations
"""

from training.train_qlearning import train
from traffic_env import TrafficEnv
from analysis.qlearning_analysis import *
from analysis.qlearning_visualization import *

# ==============================
# Training
# ==============================
agent, rewards, epsilon = train()

env = TrafficEnv()

# ==============================
# Evaluation
# ==============================
random_reward = random_policy(env)
q_reward = evaluate_policy(env, agent)

print("Random Policy Avg Reward:", random_reward)
print("Q-Learning Avg Reward:", q_reward)

# Improvement calculation
improvement = ((q_reward - random_reward) / abs(random_reward)) * 100
print(f"Improvement over Random Policy: {improvement:.2f}%")

# ==============================
# Visualization
# ==============================
avg_rewards = [r / env.max_steps for r in rewards]

plot_rewards(avg_rewards)
plot_epsilon(epsilon)
plot_training_curve(rewards)
plot_smoothed(rewards)
plot_policy(agent)
plot_stability(rewards)