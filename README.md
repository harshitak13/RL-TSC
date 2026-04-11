# SafeGAT-iLLM: Unified GAT-DQN + SafeGAT LLM Integration

A unified traffic signal control system combining:

- **iLLM-TSC2**: SUMO environment, CoLight-style GAT-DQN model, replay buffer, and training loop
- **SafeGAT-LLM**: Structured LLM integration pipeline with uncertainty gating, scenario detection,
  safety shielding, and decision logging

## Architecture

```
SUMO (GridEnv)
    │
    ▼
GAT-DQN (GATQNetwork)
    │  proposes actions + Q-values + attention weights
    ▼
SafeGATRefiner pipeline
    ├── ScenarioDetector  →  anomaly tags (queue spike, NaN, emergency)
    ├── InterventionGate  →  should LLM intervene? (low confidence / anomaly)
    ├── TrafficPromptBuilder → structured LLM prompt
    ├── LLMGateway        → calls Groq/OpenAI, parses JSON response
    └── SafetyShield      → hard enforcement (yellow-lock, min-green)
    │
    ▼
DecisionLogger  →  JSONL audit trail
    │
    ▼
env.step(safe_actions)
```

## Project Layout

```
SafeGAT_iLLM/
├── envs/
│   └── grid_env_wrapper.py     # SUMO multi-junction GridEnv (from iLLM-TSC2)
├── training/
│   ├── gat_network.py          # GATQNetwork: encoder + GATConv + Q-head
│   └── gat_dqn_trainer.py      # FastGATDQNTrainer: vectorised batched updates
├── llm/
│   ├── types.py                # RLDecisionInfo, LLMDecision, RefineResult
│   ├── scenario_detector.py    # Anomaly detection (NaN, queue spike, emergency)
│   ├── intervention_gate.py    # Uncertainty + anomaly gate (GateDecision)
│   ├── traffic_prompt_builder.py # LLM prompt construction
│   ├── llm_gateway.py          # LLM backend wrapper (Groq / OpenAI-compatible)
│   ├── safety_shield.py        # Post-LLM hard safety constraints
│   ├── action_refiner.py       # SafeGATRefiner: orchestrates full pipeline
│   └── decision_logger.py      # JSONL per-step audit logging
├── utils/
│   ├── make_tsc_env.py         # TSCEnvironment factory per junction
│   ├── readConfig.py           # config.yaml / env-var loader
│   └── margin.py               # Q-margin computation helper
├── configs/
│   └── config.yaml             # API key, model, base URL
├── train.py                    # Training entry point
└── run_safegat.py              # Inference entry point (SafeGAT + LLM)
```

## Setup

### 1. Install dependencies

```bash
pip install torch torch_geometric tshub langchain langchain-openai loguru pyyaml
```

TransSimHub (tshub):
```bash
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub && pip install -e .
```

### 2. Configure API key

Edit `configs/config.yaml`:
```yaml
OPENAI_API_KEY:   "gsk_YOUR_GROQ_KEY"
OPENAI_API_MODEL: "llama-3.1-8b-instant"
OPENAI_API_BASE:  "https://api.groq.com/openai/v1"
OPENAI_PROXY:     ""
```

### 3. Add your network files

Place `4x4.net.xml` and `4x4.sumocfg` in `network/`.
Update `network/net_config.py` with your junction IDs and topology.

## Running

### Training
```bash
python train.py
```

### Inference (SafeGAT + LLM)
```bash
python run_safegat.py
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `Q_MARGIN_TAU` | 0.05 | Q-margin threshold below which LLM is called |
| `LLM_BUDGET` | 1600 | Max LLM calls for an inference episode |
| `MAX_NODES_PER_STEP` | 2 | Max nodes sent to LLM per simulation step |
| `MIN_GREEN_STEPS` | 3 | Minimum green hold before phase switch |
| `confidence_threshold` | 0.15 | Gate confidence threshold |
| `intervention_budget` | 8 | Gate max interventions per gate scoring round |

## SafeGAT Decision Flow

1. GAT-DQN proposes actions and Q-values for all 12 junctions
2. Q-margins (Δ = Q(a\*) − Q(a2nd)) are computed per node
3. Nodes with Δ < τ or anomaly flags are flagged for LLM review
4. `SafeGATRefiner.refine()` runs the full pipeline per flagged node:
   - Detect scenario anomalies
   - Gate scores confidence + anomaly severity
   - If gate opens: prompt LLM, parse response, apply or accept
   - Safety shield validates legal phase and minimum green hold
5. Final actions are logged and executed in SUMO
