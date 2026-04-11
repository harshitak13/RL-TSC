# RL-Based Traffic Signal Control

## Overview

This project focuses on optimizing traffic signal control using Reinforcement Learning (RL) techniques in both single and multi-intersection environments. The objective is to improve traffic flow efficiency and reduce congestion under varying traffic conditions.

The advanced system — **SafeGAT-iLLM** — combines a Graph Attention Network-based DQN agent with a selective LLM intervention layer that verifies and, when necessary, overrides RL decisions at low-confidence junctions.

## Features
* Supports single and multi-intersection traffic environments
* Implementation of multiple RL algorithms:
   * Deep Q-Network (DQN)
   * Double DQN (DDQN)
   * Proximal Policy Optimization (PPO)
* Performance evaluation using key metrics:
   * Reward
   * Queue Length
   * Average Travel Time
* Implemented CoLight (Graph Attention-based RL) for:
   * Single intersection
   * 4×4 multi-intersection grid
* SafeGAT-iLLM — hybrid GAT-DQN with selective LLM safety/refinement layer
   * Confidence-based intervention gate (Q-margin threshold)
   * Anomaly detection (queue spikes, emergency vehicles, packet loss)
   * Hard safety shield (yellow-lock, min-green-hold, legal-action repair)
   * Full audit logging of every LLM decision

## Methodology
* Simulated traffic environments to model real-world scenarios
* Trained RL agents to control traffic signals dynamically
* Compared different algorithms based on efficiency and stability
* Implemented CoLight, a graph attention-based multi-agent RL approach, to enable coordination between intersections
* Modeled intersections as nodes and traffic flow as edges for efficient communication
* Built SafeGAT-iLLM — a 5-step pipeline wrapping GAT-DQN with selective LLM verification:

```
Step 1 │ ScenarioDetector    → detects anomaly tags (queue spike, emergency vehicle, packet loss …)
Step 2 │ InterventionGate    → opens when Q-margin Δ < τ_c (0.05) or anomaly is detected
Step 3 │ LLMRefine (if gate) → queries LLM for accept / override decision
Step 4 │ SafetyShield        → enforces yellow-lock, min-green-hold, legal-action constraints
Step 5 │ DecisionLogger      → logs every decision for audit and analysis
```

The LLM is queried **selectively** — only at junctions where the RL agent is uncertain — keeping the intervention rate at ~16.7% in the full system.

## Results

### Benchmark Comparison

| Method | ATT (s) ↓ | Queue Length (s) ↓ | Delay / Time-Loss (s) ↓ | Throughput (veh) ↑ |
|---|---|---|---|---|
| Webster (Fixed-Time) | 227.40 | 178.81 | 206.48 | 344 |
| Actuated / Webster-Adaptive | 162.43 | 119.21 | 142.40 | 419 |
| Plain DQN (no graph) | 178.67 | 133.51 | 156.64 | 402 |
| GAT-DQN (RL-only) | 151.06 | 109.67 | 132.43 | 432 |
| **SafeGAT-iLLM (ours)** | **145.02** | **101.02** | **123.83** | **441** |

### Ablation Study

| Variant | Total Reward | Mean Occupancy | LLM Calls | Safety Violations | Intervention Rate |
|---|---|---|---|---|---|
| V1: GAT-DQN Only | −8.90 | 0.0819 | 0 | 148 | 0% |
| V2: Uniform LLM | −9.41 | 0.0770 | 2,570 | 193 | 100% |
| **V3: Full SafeGAT-iLLM ★** | **−10.10** | **0.0719** | **640** | **0** | **16.7%** |

* Improved traffic flow and reduced congestion compared to baseline methods
* Observed variations in performance across different RL algorithms
* Demonstrated effectiveness of RL in adaptive traffic signal control
* V3 (Full SafeGAT-iLLM) achieves zero safety violations with only 640 LLM calls vs. 2,570 in the uniform LLM variant

## Tech Stack
* Python
* Reinforcement Learning frameworks (e.g., Stable-Baselines / PyTorch / TensorFlow)
* Traffic simulation tools (SUMO / TSHub environment wrapper)
* Graph Neural Networks (for CoLight and SafeGAT-iLLM implementation)
* LLM backend — Groq API (LLaMA 3.1 8B Instant) or any OpenAI-compatible endpoint

## Advanced Work
* Implemented **CoLight**, a state-of-the-art multi-agent RL approach using graph attention mechanisms
* Enabled communication between intersections for coordinated traffic signal control
* Evaluated performance on both single intersection and 4×4 grid environments
* Implemented **SafeGAT-iLLM** — a hybrid system with a selective LLM intervention gate, safety shield, and scenario detector layered on top of GAT-DQN
* Conducted ablation study across three variants (GAT-DQN only, Uniform LLM, Full SafeGAT-iLLM)
* Benchmarked against Webster fixed-time, actuated, and plain DQN baselines

## Setup & Usage

### Configuration

Edit `configs/config.yaml` to set your LLM API credentials:
```yaml
OPENAI_API_KEY: "your_api_key_here"
OPENAI_API_MODEL: "llama-3.1-8b-instant"
OPENAI_API_BASE: "https://api.groq.com/openai/v1"
```
> ⚠️ Do not commit real API keys to version control.

SafeGAT-iLLM parameters (confidence threshold, intervention budget, safety shield settings) are configured in `configs/safegat_llm.yaml`.

### Training
```bash
python train.py
```

### Running SafeGAT-iLLM Inference
```bash
python run_safegat.py
```

### CoLight
```bash
python Co-Light/1x1/run.py   # single intersection
python Co-Light/4x4/run.py   # 4×4 grid
```

### Ablation Study
```bash
python ablation/ablation_v1_gat_dqn_only.py
python ablation/ablation_v2_uniform_llm.py
python ablation/ablation_v3_full_safegat.py
```

## Future Work
* Integration of **Large Language Models (LLMs)** for enhanced decision-making *(partially realised in SafeGAT-iLLM)*
* Scaling to more complex and real-world traffic networks
* Incorporating multi-modal traffic data (V2X, sensors)
* Fine-tuning a traffic-domain-specific LLM to replace the general-purpose backend

---

## Contributors

- Harshita Karnam

---

## License

This project is open-source and available under the MIT License.
