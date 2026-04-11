# ============================================================
# Imports
# ============================================================
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

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
    'sumo-gui',
    '-c', r"C:\Users\harsh\Desktop\RL\multiple_intersections\multiple_intersections.sumocfg",
    '--step-length', '0.10',
    '--delay', '1000'
]

# ============================================================
# RL Hyperparameters
# ============================================================
TOTAL_STEPS = 10000

ALPHA = 0.001
GAMMA = 0.9

ACTIONS = [0, 1]   # 0 = keep phase, 1 = switch phase
STATE_SIZE = 7
ACTION_SIZE = len(ACTIONS)

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# ============================================================
# UCB variables
# ============================================================
action_counts = {}
c_ucb = 1.0

# ============================================================
# DQN Model
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

device = torch.device("cpu")
dqn_model = DQN(STATE_SIZE, ACTION_SIZE).to(device)

optimizer = optim.Adam(dqn_model.parameters(), lr=ALPHA)
loss_fn = nn.MSELoss()

# ============================================================
# Helper functions
# ============================================================
def to_tensor(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepHaltingNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    return (
        get_queue_length("node10-11-EB-0"),  # CHANGE if needed
        get_queue_length("node10-11-EB-1"),
        get_queue_length("node10-11-EB-2"),
        get_queue_length("node01-11-SB-0"),
        get_queue_length("node01-11-SB-1"),
        get_queue_length("node01-11-SB-2"),
        get_current_phase("node11")
    )

def get_reward(state):
    return -float(sum(state[:-1]))

# ============================================================
# UCB Action Selection
# ============================================================
def get_action_ucb(state, step):
    state_key = tuple(int(x) for x in state)

    if state_key not in action_counts:
        action_counts[state_key] = np.zeros(ACTION_SIZE)

    with torch.no_grad():
        q_values = dqn_model(to_tensor(state)).squeeze().numpy()

    ucb = q_values + c_ucb * np.sqrt(
        np.log(step + 1) / (action_counts[state_key] + 1e-6)
    )

    action = int(np.argmax(ucb))
    action_counts[state_key][action] += 1
    return action

# ============================================================
# Apply Action
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
# DQN Update (online TD)
# ============================================================
def update_dqn(old_state, action, reward, new_state):
    old_t = to_tensor(old_state)
    new_t = to_tensor(new_state)

    q_vals = dqn_model(old_t)

    with torch.no_grad():
        max_next_q = torch.max(dqn_model(new_t))

    target = q_vals.clone().detach()
    target[0, action] = reward + GAMMA * max_next_q

    loss = loss_fn(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)

print("\n=== DQN + UCB Online Training Started ===")

step_history = []
reward_history = []
queue_history = []
vehicle_depart_times = {}
travel_times = []

avg_travel_time_history = []
cumulative_reward = 0.0

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    action = get_action_ucb(state, step)

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

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_dqn(state, action, reward, new_state)

    with torch.no_grad():
        q_vals = dqn_model(to_tensor(state)).numpy()[0]

    print(
        f"Step {step} | Action {action} | "
        f"Reward {reward:.2f} | Cum {cumulative_reward:.2f} | Q {q_vals}"
    )

    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(sum(new_state[:-1]))

# ============================================================
# Close SUMO
# ============================================================
traci.close()

# ============================================================
# Plots
# ============================================================
plt.figure()
plt.plot(step_history, reward_history)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("DQN + UCB – Reward")
plt.grid()
plt.show()

plt.figure()
plt.plot(step_history, queue_history)
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.title("DQN + UCB – Queue")
plt.grid()
plt.show()

plt.figure()
plt.plot(step_history, avg_travel_time_history)
plt.xlabel("Step")
plt.ylabel("Average Travel Time")
plt.title("DQN + UCB – Average Travel Time")
plt.grid()
plt.show()
