# RL-Based Traffic Signal Control

## Overview

This project focuses on optimizing traffic signal control using Reinforcement Learning (RL) for both single and multi-intersection environments. The goal is to improve traffic flow and reduce congestion under dynamic conditions.

The advanced system — **SafeGAT-iLLM** — combines a Graph Attention Network (GAT)-based RL agent with a selective LLM intervention layer that verifies and overrides decisions only when necessary.

Based on the paper:  
**SafeGAT: Uncertainty-Gated and Safety-Constrained LLM Supervision for Graph Attention Reinforcement Learning in Traffic Signal Control**

---

## Key Highlights

- Uses **Graph Attention Networks (GAT)** for multi-intersection coordination  
- Introduces **uncertainty-based LLM intervention** (not always active)  
- Uses **Top-K selective triggering** to reduce unnecessary LLM calls  
- Ensures **zero safety violations** using a safety constraint layer  
- Maintains **real-time performance** with minimal overhead  
- Separates **RL learning** from **decision execution**

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

- **SafeGAT-iLLM system**:
  - Confidence-based intervention (low-confidence decisions trigger LLM)  
  - Anomaly detection (queue spikes, emergency vehicles, sensor issues)  
  - Safety constraints (minimum green time, valid phase transitions)  
  - Decision logging for analysis  

---

## Methodology

- Simulate traffic using SUMO  
- Train RL agents to control signals dynamically  
- Use GAT to enable coordination between intersections  
- Add a **selective LLM layer** to refine uncertain decisions  

### SafeGAT-iLLM Pipeline

1. Detect anomalies (traffic spikes, errors, emergencies)  
2. Check RL confidence  
3. Trigger LLM only if needed  
4. Apply safety constraints  
5. Log decisions  

---

## Results

### Performance Comparison

| Method | Travel Time ↓ | Queue ↓ | Delay ↓ | Throughput ↑ |
|--------|-------------|--------|--------|-------------|
| Webster | 227 | 178 | 206 | 344 |
| Actuated | 162 | 119 | 142 | 419 |
| DQN | 178 | 133 | 156 | 402 |
| GAT-DQN | 151 | 109 | 132 | 432 |
| **SafeGAT-iLLM** | **145** | **101** | **123** | **441** |

---

### Ablation Study

| Variant | LLM Calls | Safety Violations |
|--------|----------|------------------|
| RL Only | 0 | 148 |
| Always LLM | 2570 | 193 |
| **SafeGAT-iLLM** | **640** | **0** |

---

## Key Insights

- ~4× fewer LLM calls compared to always using LLM  
- Eliminates unsafe traffic signal transitions  
- More stable and robust under traffic disturbances  
- Scales to large traffic networks (up to 7×28 grid)  

---

## Tech Stack

- Python  
- PyTorch / TensorFlow / Stable-Baselines  
- SUMO Traffic Simulator  
- Graph Neural Networks (GAT)  
- LLM APIs (Groq / OpenAI-compatible)  

---

## Future Work

- Fine-tune domain-specific LLM for traffic control  
- Integrate real-world sensor and V2X data  
- Scale to city-level deployments  

---

## Contributor

- Harshita Karnam  

---

## License

MIT License
