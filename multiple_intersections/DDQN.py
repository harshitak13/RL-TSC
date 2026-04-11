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
    '-c', r"C:\Users\harsh\Desktop\RL\multiple_intersections\multiple_intersections.sumocfg",
    '--step-length', '0.10',
    '--delay', '1000'
]

# ============================================================
# RL Hyperparameters
# ============================================================
TOTAL_STEPS = 10000
GAMMA = 0.9
ALPHA = 0.001

STATE_SIZE = 7
ACTIONS = [0, 1]   # 0 = keep phase, 1 = switch
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
C = 1.0
action_counts = {}

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

# -----------------------------
# STATE (USING YOUR DETECTORS)
# -----------------------------
def get_state():
    return (
        get_queue_length("node10-11-EB-0"),
        get_queue_length("node10-11-EB-1"),
        get_queue_length("node10-11-EB-2"),
        get_queue_length("node01-11-NB-0"),
        get_queue_length("node01-11-NB-1"),
        get_queue_length("node01-11-NB-2"),
        get_current_phase("node11")
    )

def get_reward(state):
    return -float(sum(state[:-1]))

# ============================================================
# Apply Traffic Light Action
# ============================================================
def apply_action(action, tls_id="node11"):
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
    state_tuple = tuple(state)

    if state_tuple not in action_counts:
        action_counts[state_tuple] = np.zeros(ACTION_SIZE)

    with torch.no_grad():
        q_vals = online_net(to_tensor(state)).cpu().numpy()[0]

    total = np.sum(action_counts[state_tuple]) + 1e-8
    ucb = q_vals + C * np.sqrt(np.log(total + 1) / (action_counts[state_tuple] + 1e-8))

    action = int(np.argmax(ucb))
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

    q_vals = online_net(states).gather(1, actions).squeeze()

    with torch.no_grad():
        next_actions = torch.argmax(online_net(next_states), dim=1)
        next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target = rewards + GAMMA * next_q

    loss = loss_fn(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)
print("\n=== Starting DDQN + UCB Training ===")

cumulative_reward = 0
step_history, reward_history, queue_history = [], [], []
vehicle_depart_times = {}
travel_times = []
avg_travel_time_history = []

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    action = get_action(state)
    apply_action(action)

    traci.simulationStep()

    for veh_id in traci.simulation.getDepartedIDList():
        vehicle_depart_times[veh_id] = step

    for veh_id in traci.simulation.getArrivedIDList():
        if veh_id in vehicle_depart_times:
            travel_time = step - vehicle_depart_times[veh_id]
            travel_times.append(travel_time)
            del vehicle_depart_times[veh_id]


    if len(travel_times) > 0:
        avg_tt = np.mean(travel_times)
    else:
        avg_tt = 0

    avg_travel_time_history.append(avg_tt)

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

    print(f"Step {step} | Action {action} | Reward {reward:.2f} | Cum {cumulative_reward:.2f}")

# ============================================================
# Close SUMO & Plot
# ============================================================
traci.close()
print("\nTraining finished.")

plt.plot(step_history, reward_history)
plt.title("DDQN + UCB: Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.show()

plt.plot(step_history, queue_history)
plt.title("DDQN + UCB: Total Queue Length")
plt.xlabel("Step")
plt.ylabel("Queue")
plt.grid()
plt.show()

plt.plot(step_history, avg_travel_time_history)
plt.title("DDQN + UCB: Average Travel Time")
plt.xlabel("Step")
plt.ylabel("Average Travel Time")
plt.grid()
plt.show()
