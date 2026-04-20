# 🚦 Traffic Signal Control using Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-DQN-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A reinforcement learning project for optimizing traffic signal control at a single intersection using a two-stage approach:

1. Tabular Q-Learning (baseline + analysis)
2. Deep Q-Network (DQN) (advanced solution)

Q-Learning was implemented first to build a baseline understanding of the environment and policy behavior, followed by a more advanced DQN model for improved performance and scalability.


---

📁 Project Structure

traffic_signal_rl/
│
├── run_qlearning.py           # Q-Learning entry point (baseline stage)
├── run_dqn.py                 # DQN entry point (advanced stage)
│
├── environment/
│   ├── _init_.py
│   └── traffic_env.py         # Custom traffic simulation environment
│
├── agents/
│   ├── _init_.py
│   ├── q_learning_agent.py    # Tabular Q-Learning implementation
│   └── dqn_agent.py           # DQN + Double DQN agent
│
├── training/
│   ├── _init_.py
│   ├── train_qlearning.py     # Q-Learning training loop
│   └── train_dqn.py           # DQN training loop
│
├── analysis/
│   ├── _init_.py
│   ├── metrics.py
│   ├── visualization.py
│   └── refinement_logger.py
│
├── models/
│   ├── q_table.pkl            # Trained Q-Learning model
│   └── dqn_final.pt           # Trained DQN model
│
├── results/
│   ├── q_learning/            # Q-Learning outputs
│   └── dqn/                   # DQN outputs
│
├── requirements.txt
└── .gitignore


---

🌍 Environment

Feature	Range

Cars North-South	0 – 20
Cars East-West	0 – 20
Green duration	5s – 60s
Average wait time	dynamic


🎮 Actions

Action	Effect

0	Keep green duration
1	Increase +5s
2	Decrease -5s


🎯 Reward Function

reward = -avg_wait + throughput_bonus * cars_served - balance_penalty * queue_imbalance


---

🧠 1. Q-Learning (Baseline Phase - First Implementation)

📌 Overview

First algorithm implemented in the project

Used as a baseline reinforcement learning model

Helps understand environment dynamics and reward behavior

Uses tabular learning with discretized state space


⚙️ Key Features

Epsilon-greedy exploration

Q-table update using Bellman equation

State discretization

Policy extraction and visualization

Stability analysis



---

🧠 2. Deep Q-Network (DQN) (Advanced Phase)

📌 Overview

Built after Q-Learning for performance improvement

Handles continuous state space efficiently

More scalable and stable learning approach


⚙️ Key Features

Double DQN architecture

Experience replay buffer

Target network updates

Reward normalization

Stable convergence techniques



---

⚙️ Installation

pip install -r requirements.txt


---

🚀 Usage

▶ Run Q-Learning (Baseline Stage)

python run_qlearning.py

▶ Run DQN (Advanced Stage)

python run_dqn.py


---

📊 Results Comparison

Metric	Q-Learning (Baseline)	DQN (Advanced)

Learning Type	Tabular	Deep Neural Network
Performance	Moderate	Higher
Stability	Medium	High
Scalability	Limited	Strong
Training Time	Fast	Longer



---

📂 Outputs

🟡 Q-Learning Outputs

Learning curve

Epsilon decay

Policy heatmap

Q-table

Stability analysis


🔵 DQN Outputs

Reward curve

Loss curve

Policy visualization

Stability analysis

Trained model (.pt)



---

👥 Team

Role	Responsibility

Engineer 1	Environment design and simulation
Engineer 2	Q-Learning implementation and analysis (First Phase)
Engineer 3	DQN implementation and training
Engineer 4	Integration, evaluation, and reporting
