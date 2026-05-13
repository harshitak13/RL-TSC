# RL-Based Traffic Signal Control

## Overview

This project focuses on optimizing traffic signal control using Reinforcement Learning (RL) for both single and multi-intersection environments. The goal is to improve traffic flow and reduce congestion under dynamic conditions.

---

## Features

- Supports single and multi-intersection traffic environments
- Implements multiple RL algorithms:
  - Deep Q-Network (DQN)
  - Double DQN (DDQN)
  - Proximal Policy Optimization (PPO)

- Performance metrics:
  - Reward
  - Queue Length
  - Average Travel Time

- CoLight (Graph Attention RL):
  - Single intersection
  - 4×4 grid network

---

## Methodology

- Simulate traffic using SUMO
- Train RL agents to control signals dynamically
- Use Graph Attention Networks (GAT) to enable coordination between intersections

---

## Tech Stack

- Python
- PyTorch / TensorFlow / Stable-Baselines
- SUMO Traffic Simulator
- Graph Neural Networks (GAT)

---

## Future Work

- Fine-tune domain-specific models for traffic control
- Integrate real-world sensor and V2X data
- Scale to city-level deployments

---

## Contributor

- Harshita Karnam

---

## License

MIT License
