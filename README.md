🚦 Traffic Signal Control using Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-DQN-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A reinforcement learning project for optimizing traffic signal control at a single intersection using a two-stage approach:

- 🟡 Tabular Q-Learning (baseline + behavioral analysis)
- 🔵 Deep Q-Network (DQN) (advanced scalable solution)

The goal is to minimize traffic congestion and average waiting time while maximizing throughput at a single intersection.

---

## 🎯 Problem Statement

Urban traffic congestion leads to increased waiting time, inefficiency, and pollution.  
This project explores how reinforcement learning can dynamically control traffic signals to optimize flow without fixed timing rules.

---

## 🧠 Approach

### 1️⃣ Q-Learning (Baseline Phase)

Used as an initial approach to understand environment dynamics.

Key Features:
- Tabular Q-learning
- Discretized state space
- Epsilon-greedy exploration
- Bellman equation updates
- Policy extraction & visualization

---

### 2️⃣ Deep Q-Network (DQN) (Advanced Phase)

Designed to handle continuous state space and improve scalability.

Key Features:
- Deep neural network approximation
- Experience replay buffer
- Target network updates
- Double DQN stability improvements
- Reward normalization

---

### 3️⃣ Final Comparison (Q-Learning vs DQN)
Final comparison was performed using `analysis/compare&evaluate.py` with the required metrics:
- Sample efficiency: episodes needed to reach reward threshold
- Final performance: mean reward over last 50 episodes
- Convergence speed: timesteps to reach stable policy
- Stability: variance in final performance across independent runs
Results:
| Metric | Q-Learning | DQN |
|--------|-----------:|----:|
| Sample efficiency (episodes) | 3 | 1 |
| Final performance (mean reward, last 50 episodes) | -1134.901 | -1111.157 |
| Convergence speed (timesteps) | 38000 | 11400 |
| Stability variance (across runs) | 0.000000 | 0.000000 |
Notes:
- DQN achieved faster learning and better final reward in this experiment.
- Stability variance is currently zero because each algorithm was evaluated from one saved run; multiple independent runs are needed for stronger statistical stability analysis.
---


## 🌍 Environment

| Feature | Range |
|--------|------|
| Cars North-South | 0 – 20 |
| Cars East-West | 0 – 20 |
| Green duration | 5s – 60s |
| Average wait time | Dynamic |

---

## 🎮 Action Space

| Action | Effect |
|--------|--------|
| 0 | Keep current signal |
| 1 | Increase green time (+5s) |
| 2 | Decrease green time (-5s) |

---

## 🎯 Reward Function

The reward is designed to optimize traffic flow:

- Minimize average waiting time
- Maximize throughput
- Maintain balance between queues

---

## 📁 Project Structure

```bash
traffic_signal_rl/

├── run_qlearning.py        # Baseline training
├── run_dqn.py              # DQN training

├── environment/
│   └── traffic_env.py      # Custom traffic simulation

├── agents/
│   ├── q_learning_agent.py
│   └── dqn_agent.py

├── training/
│   ├── train_qlearning.py
│   └── train_dqn.py

├── analysis/
│   ├── metrics.py
│   ├── visualization.py
│   ├── compare&evaluate.py 
│   └── refinement_logger.py
تت

├── models/
│   ├── q_table.pkl
│   └── dqn_final.pt

├── results/
│   ├── q_learning/
│   └── dqn/

└── requirements.txt


---

📊 Results Summary

🟡 Q-Learning (Baseline)

Fast convergence

Works well for small state space

Limited scalability


🔵 DQN (Advanced)

Better long-term reward optimization

Handles larger state space efficiently

More stable policy learning



---

⚙️ Installation

pip install -r requirements.txt


---

🚀 How to Run

▶ Q-Learning

python run_qlearning.py

▶ DQN

python run_dqn.py


---

📈 Outputs

Q-Learning

Learning curve

Epsilon decay

Policy heatmap

Q-table analysis


DQN

Reward curve

Loss curve

Policy visualization

Trained model (.pt)



---

👥 Team Contribution

Role	Responsibility

Engineer 1	Environment design
Engineer 2	Q-Learning + analysis
Engineer 3	DQN implementation
Engineer 4	Evaluation & reporting



---

💡 Key Insight

Q-Learning works as a strong baseline for understanding environment dynamics, while DQN significantly improves performance in more complex and continuous state spaces.
