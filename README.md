# RL-Based Traffic Signal Control

## Overview

This project focuses on optimizing traffic signal control using Reinforcement Learning (RL) techniques in both single and multi-intersection environments. The objective is to improve traffic flow efficiency and reduce congestion under varying traffic conditions.

---

## Features

- Supports **single and multi-intersection** traffic environments
- Implementation of multiple RL algorithms:
  - Deep Q-Network (**DQN**)
  - Double DQN (**DDQN**)
  - Proximal Policy Optimization (**PPO**)
- Performance evaluation using key metrics:
  - Reward
  - Queue Length
  - Average Travel Time
- Implemented **CoLight (Graph Attention-based RL)** for:
  - Single intersection
  - 4×4 multi-intersection grid

---

## Methodology

- Simulated traffic environments to model real-world scenarios
- Trained RL agents to control traffic signals dynamically
- Compared different algorithms based on efficiency and stability
- Implemented CoLight, a graph attention-based multi-agent RL approach, to enable coordination between intersections
- Modeled intersections as nodes and traffic flow as edges for efficient communication

---

## Results

- Improved traffic flow and reduced congestion compared to baseline methods
- Observed variations in performance across different RL algorithms
- Demonstrated effectiveness of RL in adaptive traffic signal control

---

## Tech Stack

- Python
- Reinforcement Learning frameworks (e.g., Stable-Baselines / PyTorch / TensorFlow)
- Traffic simulation tools (e.g., SUMO / custom environment)
- Graph Neural Networks (for CoLight implementation)

---

## Advanced Work

- Implemented **CoLight**, a state-of-the-art multi-agent RL approach using graph attention mechanisms
- Enabled communication between intersections for coordinated traffic signal control
- Evaluated performance on both single intersection and 4×4 grid environments

## Future Work

- Integration of **Large Language Models (LLMs)** for enhanced decision-making
- Scaling to more complex and real-world traffic networks
- Incorporating multi-modal traffic data

---

## Contributors

- Harshita Karnam

---

## License

This project is open-source and available under the MIT License.
