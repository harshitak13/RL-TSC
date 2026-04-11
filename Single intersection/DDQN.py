# ============================================================
# Imports
# ============================================================
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

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
    'sumo',
    '-c', r"C:\Users\harsh\Desktop\RL\Single intersection\RL.sumocfg",
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# ============================================================
# RL Hyperparameters
# ============================================================
TOTAL_STEPS = 10000

GAMMA = 0.9
ALPHA = 0.001

STATE_SIZE = 7
ACTIONS = [0, 1]
ACTION_SIZE = len(ACTIONS)

MEMORY_SIZE = 5000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 500

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

device = torch.device("cpu")

# ============================================================
# UCB parameters
# ============================================================
C = 1.0  # Exploration coefficient
action_counts = {}  # Dictionary to store counts for each state-action pair

# ============================================================
# Replay Buffer
# ============================================================
memory = deque(maxlen=MEMORY_SIZE)

# ============================================================
# DDQN Network
# ============================================================
class DDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

online_net = DDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net = DDQN(STATE_SIZE, ACTION_SIZE).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=ALPHA)
loss_fn = nn.MSELoss()

# ============================================================
# Helper functions
# ============================================================
def to_tensor(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    return (
        get_queue_length("Node1_2_EB_0"),
        get_queue_length("Node1_2_EB_1"),
        get_queue_length("Node1_2_EB_2"),
        get_queue_length("Node2_7_SB_0"),
        get_queue_length("Node2_7_SB_1"),
        get_queue_length("Node2_7_SB_2"),
        get_current_phase("Node2")
    )

def get_reward(state):
    return -float(sum(state[:-1]))

def apply_action(action, tls_id="Node2"):
    global last_switch_step, current_simulation_step

    if action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            next_phase = (get_current_phase(tls_id) + 1) % len(program.phases)
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

# ============================================================
# UCB Action Selection
# ============================================================
def get_action(state):
    state_tuple = tuple(state)  # Use tuple as dict key

    if state_tuple not in action_counts:
        action_counts[state_tuple] = np.zeros(ACTION_SIZE)

    with torch.no_grad():
        q_vals = online_net(to_tensor(state)).numpy()[0]

    # Calculate UCB values
    total_counts = np.sum(action_counts[state_tuple]) + 1e-8
    ucb_values = q_vals + C * np.sqrt(np.log(total_counts + 1) / (action_counts[state_tuple] + 1e-8))

    # Choose action with max UCB
    action = int(np.argmax(ucb_values))

    # Update counts
    action_counts[state_tuple][action] += 1

    return action

# ============================================================
# DDQN Training Step
# ============================================================
def train_ddqn():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = torch.cat([to_tensor(s) for s, _, _, _ in batch])
    actions = torch.tensor([a for _, a, _, _ in batch]).unsqueeze(1)
    rewards = torch.tensor([r for _, _, r, _ in batch], dtype=torch.float32)
    next_states = torch.cat([to_tensor(ns) for _, _, _, ns in batch])

    # Current Q values
    q_values = online_net(states).gather(1, actions).squeeze()

    # Double DQN target
    with torch.no_grad():
        next_actions = torch.argmax(online_net(next_states), dim=1)
        next_q_values = target_net(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze()

        target_q = rewards + GAMMA * next_q_values

    loss = loss_fn(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)

print("\n=== Starting DDQN Training (PyTorch) with UCB ===")

step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    action = get_action(state)  # UCB-based action
    apply_action(action)

    traci.simulationStep()

    next_state = get_state()
    reward = get_reward(next_state)
    cumulative_reward += reward

    memory.append((state, action, reward, next_state))
    train_ddqn()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(sum(next_state[:-1]))

    print(
        f"Step {step} | Action {action} | "
        f"Reward {reward:.2f} | CumReward {cumulative_reward:.2f}"
    )

# ============================================================
# Close SUMO & Plot
# ============================================================
traci.close()
print("\nDDQN Training Completed with UCB")

plt.figure(figsize=(10, 5))
plt.plot(step_history, reward_history)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("DDQN (PyTorch with UCB): Cumulative Reward")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(step_history, queue_history)
plt.xlabel("Step")
plt.ylabel("Total Queue Length")
plt.title("DDQN (PyTorch with UCB): Queue Length")
plt.grid()
plt.show()

