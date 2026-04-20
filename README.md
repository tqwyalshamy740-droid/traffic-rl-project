# 🚦 Traffic Signal Control using Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-DQN-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A reinforcement learning project that trains a DQN agent to control traffic signals at a single intersection, minimizing vehicle waiting time and maximizing throughput.

---

## 📁 Project Structure

```
traffic_signal_rl/
│
├── run_dqn.py                  # Main entry point
│
├── environment/
│   ├── __init__.py
│   └── traffic_env.py          # Custom traffic simulation environment
│
├── agents/
│   ├── __init__.py
│   └── dqn_agent.py            # DQN + Double DQN agent
│
├── training/
│   ├── __init__.py
│   └── train_dqn.py            # Training loop
│
├── analysis/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization_dqn.py    # Plots and visualizations
│   └── refinement_logger.py    # CSV refinement logger
│
├── requirements.txt
└── .gitignore
```

---

## 🌍 Environment

**State (4 values):**
| Feature | Range |
|---|---|
| Cars waiting North-South | 0 – 20 |
| Cars waiting East-West | 0 – 20 |
| Current green duration | 5s – 60s |
| Average waiting time | 0 – max |

**Actions (Discrete):**
| Action | Effect |
|---|---|
| 0 | Keep green duration |
| 1 | Increase green by 5s |
| 2 | Decrease green by 5s |

**Reward:**
```
reward = -avg_wait + throughput_bonus * cars_served - balance_penalty * queue_imbalance
```

---

## 🧠 Algorithm

**Deep Q-Network (DQN)** with the following improvements:
- Double DQN to reduce overestimation bias
- Experience replay buffer (20,000 transitions)
- Target network updated every 500 steps
- Observation normalization to [0, 1]
- Reward scaling (÷ 50) for stable training
- Per-episode epsilon decay

**Hyperparameters:**
| Parameter | Value |
|---|---|
| Learning rate | 0.0005 |
| Gamma (discount) | 0.99 |
| Epsilon start | 1.0 |
| Epsilon min | 0.05 |
| Epsilon decay | 0.990 per episode |
| Batch size | 128 |
| Hidden size | 256 |
| Episodes | 500 |

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python run_dqn.py
```

Results are saved to `results/` and the trained model to `models/dqn_final.pt`.

---

## 📊 Results

| Metric | Value |
|---|---|
| Final Avg Reward | ~750 – 950 |
| Convergence Episode | ~50 |
| Stability (std dev) | ~144 |

> Note: Some variance between runs is expected due to environment stochasticity.

---

## 📂 Output Files

| File | Description |
|---|---|
| `results/dqn_learning_curve.png` | Reward per episode with moving average |
| `results/dqn_qvalues.png` | Q-values heatmap across states |
| `results/dqn_policy.png` | Learned policy visualization |
| `results/dqn_stability.png` | Reward variance across 10 runs |
| `results/dqn_refinement_log.csv` | Experiment log |
| `models/dqn_final.pt` | Trained model weights |

---

## 👥 Team

| Role | Responsibility |
|---|---|
| Engineer 1 | Environment design and simulation |
| Engineer 2 | Q-Learning implementation and analysis |
| Engineer 3 | DQN implementation and training |
| Engineer 4 | Integration, evaluation, and report writing |
