# ============================================================
# Imports (PURE NUMPY)
# ============================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SUMO_HOME
# ============================================================
os.environ["SUMO_HOME"] = r"C:\Program Files\Eclipse SUMO"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

# ============================================================
# SUMO config
# ============================================================
Sumo_config = [
    "sumo",
    "-c", r"C:\Users\harsh\Desktop\RL\Single intersection\RL.sumocfg",
    "--step-length", "0.10"
]

# ============================================================
# Hyperparameters
# ============================================================
TOTAL_STEPS = 10000
ROLLOUT_LENGTH = 200
EPOCHS = 5

GAMMA = 0.99
CLIP_EPS = 0.2
LR = 0.001

STATE_SIZE = 7
ACTION_SIZE = 2

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# ============================================================
# UCB Parameters
# ============================================================
UCB_C = 1.0
action_counts = np.zeros(ACTION_SIZE)
total_action_count = 0

# ============================================================
# PPO Parameters (Linear Policy + Value)
# ============================================================
W_policy = np.random.randn(STATE_SIZE, ACTION_SIZE) * 0.01
b_policy = np.zeros(ACTION_SIZE)

W_value = np.random.randn(STATE_SIZE) * 0.01
b_value = 0.0

# ============================================================
# Helper Functions
# ============================================================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    return np.array([
        get_queue_length("Node1_2_EB_0"),
        get_queue_length("Node1_2_EB_1"),
        get_queue_length("Node1_2_EB_2"),
        get_queue_length("Node2_7_SB_0"),
        get_queue_length("Node2_7_SB_1"),
        get_queue_length("Node2_7_SB_2"),
        get_current_phase("Node2")
    ], dtype=np.float32)

def get_reward(state):
    return -np.sum(state[:-1])

def apply_action(action, tls_id="Node2"):
    global last_switch_step, current_simulation_step
    if action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            next_phase = (get_current_phase(tls_id) + 1) % len(program.phases)
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

# ============================================================
# Rollout Storage
# ============================================================
states, actions, rewards, log_probs, values = [], [], [], [], []

# ============================================================
# PPO Update (NUMPY)
# ============================================================
def ppo_update():
    global W_policy, b_policy, W_value, b_value

    states_arr = np.array(states)
    actions_arr = np.array(actions)
    old_log_probs_arr = np.array(log_probs)
    rewards_arr = np.array(rewards)
    values_arr = np.array(values)

    # Compute returns
    returns = []
    G = 0
    for r in rewards_arr[::-1]:
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = np.array(returns)

    advantages = returns - values_arr
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(EPOCHS):
        for i in range(len(states_arr)):
            s = states_arr[i]
            a = actions_arr[i]
            adv = advantages[i]
            ret = returns[i]

            logits = s @ W_policy + b_policy
            probs = softmax(logits)

            log_prob = np.log(probs[a] + 1e-8)
            ratio = np.exp(log_prob - old_log_probs_arr[i])

            clipped_ratio = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_grad = -min(ratio * adv, clipped_ratio * adv)

            # Policy gradient
            dlogits = probs.copy()
            dlogits[a] -= 1
            dlogits *= policy_grad

            W_policy -= LR * np.outer(s, dlogits)
            b_policy -= LR * dlogits

            # Value function gradient
            value = s @ W_value + b_value
            value_error = value - ret

            W_value -= LR * value_error * s
            b_value -= LR * value_error

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)
print("\n=== Starting PPO + UCB Training (PURE NUMPY) ===")

step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    logits = state @ W_policy + b_policy
    probs = softmax(logits)

    # ===================== UCB ACTION SELECTION =====================
    ucb_bonus = UCB_C * np.sqrt(
        np.log(total_action_count + 1 + 1e-8) / (action_counts + 1e-8)
    )
    ucb_scores = probs + ucb_bonus
    action = np.argmax(ucb_scores)

    action_counts[action] += 1
    total_action_count += 1
    # ===============================================================

    log_prob = np.log(probs[action] + 1e-8)
    value = state @ W_value + b_value

    apply_action(action)
    traci.simulationStep()

    next_state = get_state()
    reward = get_reward(next_state)

    states.append(state)
    actions.append(action)
    rewards.append(reward)
    log_probs.append(log_prob)
    values.append(value)

    cumulative_reward += reward

    if (step + 1) % ROLLOUT_LENGTH == 0:
        ppo_update()
        states.clear()
        actions.clear()
        rewards.clear()
        log_probs.clear()
        values.clear()

    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(np.sum(next_state[:-1]))

    print(f"Step {step} | Action {action} | Reward {reward:.2f}")

# ============================================================
# Close SUMO
# ============================================================
traci.close()
print("\nPPO + UCB Training Completed")

# ============================================================
# Plots
# ============================================================
plt.figure()
plt.plot(step_history, reward_history)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("PPO + UCB – Cumulative Reward")
plt.grid()
plt.show()

plt.figure()
plt.plot(step_history, queue_history)
plt.xlabel("Step")
plt.ylabel("Total Queue Length")
plt.title("PPO + UCB – Queue Length")
plt.grid()
plt.show()
