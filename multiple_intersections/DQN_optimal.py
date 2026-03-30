# ============================================================
# Imports
# ============================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# SUMO setup
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
    '--step-length', '0.10'
]

# ============================================================
# RL Hyperparameters
# ============================================================
TOTAL_STEPS = 50000   # longer training
ALPHA = 0.001
GAMMA = 0.99

STATE_SIZE = 7
ACTION_SIZE = 2
ACTIONS = [0,1]

MIN_GREEN_STEPS = 40   # reduced from 100
last_switch_step = -MIN_GREEN_STEPS

# ============================================================
# UCB parameters (lower exploration)
# ============================================================
action_counts = {}
c_ucb = 0.3

# ============================================================
# Lane IDs
# ============================================================
LANES = [
    "node10-11-EB_0",
    "node10-11-EB_1",
    "node10-11-EB_2",
    "node01-11-SB_0",
    "node01-11-SB_1",
    "node01-11-SB_2"
]

# ============================================================
# Detector IDs
# ============================================================
DETECTORS = [
    "node10-11-EB-0",
    "node10-11-EB-1",
    "node10-11-EB-2",
    "node01-11-SB-0",
    "node01-11-SB-1",
    "node01-11-SB-2"
]

# ============================================================
# DQN Model
# ============================================================
class DQN(nn.Module):

    def __init__(self,state_size,action_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size,32)
        self.fc2 = nn.Linear(32,32)
        self.out = nn.Linear(32,action_size)

    def forward(self,x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

device=torch.device("cpu")
dqn_model=DQN(STATE_SIZE,ACTION_SIZE).to(device)

optimizer=optim.Adam(dqn_model.parameters(),lr=ALPHA)
loss_fn=nn.MSELoss()

# ============================================================
# Helper functions
# ============================================================
def to_tensor(state):
    return torch.tensor(state,dtype=torch.float32).unsqueeze(0)

def get_queue(det):
    return traci.lanearea.getLastStepVehicleNumber(det)

def get_wait(lane):
    return traci.lane.getWaitingTime(lane)

def get_phase():
    return traci.trafficlight.getPhase("node11")

# ============================================================
# State (normalized)
# ============================================================
def get_state():

    queues=[get_queue(d)/20 for d in DETECTORS]   # normalize queues

    phase=get_phase()/4                           # normalize phase

    return tuple(queues+[phase])

# ============================================================
# Improved Reward
# ============================================================
def get_reward():

    queue=sum(get_queue(d) for d in DETECTORS)

    waiting=sum(get_wait(l) for l in LANES)

    throughput=len(traci.simulation.getArrivedIDList())

    reward=-queue - 0.1*waiting + 2*throughput

    return reward

# ============================================================
# UCB Action Selection
# ============================================================
def get_action_ucb(state,step):

    state_key=tuple(int(x*10) for x in state)

    if state_key not in action_counts:
        action_counts[state_key]=np.zeros(ACTION_SIZE)

    with torch.no_grad():
        q_values=dqn_model(to_tensor(state)).squeeze().numpy()

    ucb=q_values + c_ucb*np.sqrt(
        np.log(step+1)/(action_counts[state_key]+1e-6)
    )

    action=int(np.argmax(ucb))
    action_counts[state_key][action]+=1

    return action

# ============================================================
# Apply Action
# ============================================================
def apply_action(action):

    global last_switch_step,current_simulation_step

    if action==1:

        if current_simulation_step-last_switch_step>=MIN_GREEN_STEPS:

            program=traci.trafficlight.getAllProgramLogics("node11")[0]

            next_phase=(get_phase()+1)%len(program.phases)

            traci.trafficlight.setPhase("node11",next_phase)

            last_switch_step=current_simulation_step

# ============================================================
# DQN Update
# ============================================================
def update_dqn(old_state,action,reward,new_state):

    old_t=to_tensor(old_state)
    new_t=to_tensor(new_state)

    q_vals=dqn_model(old_t)

    with torch.no_grad():
        max_next_q=torch.max(dqn_model(new_t))

    target=q_vals.clone().detach()
    target[0,action]=reward+GAMMA*max_next_q

    loss=loss_fn(q_vals,target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)

print("\n=== DQN + UCB Training Started ===")

step_history=[]
reward_history=[]
queue_history=[]
travel_time_history=[]

vehicle_depart_times={}
travel_times=[]

cumulative_reward=0

# ============================================================
# Training Loop
# ============================================================
for step in range(TOTAL_STEPS):

    current_simulation_step=step

    state=get_state()

    action=get_action_ucb(state,step)

    apply_action(action)

    traci.simulationStep()

    # travel time tracking
    for v in traci.simulation.getDepartedIDList():
        vehicle_depart_times[v]=step

    for v in traci.simulation.getArrivedIDList():

        if v in vehicle_depart_times:

            tt=step-vehicle_depart_times[v]

            travel_times.append(tt)

            del vehicle_depart_times[v]

    # average of recent vehicles
    avg_tt=np.mean(travel_times[-50:]) if len(travel_times)>50 else 0

    travel_time_history.append(avg_tt)

    new_state=get_state()

    reward=get_reward()

    cumulative_reward+=reward

    update_dqn(state,action,reward,new_state)

    step_history.append(step)
    reward_history.append(cumulative_reward)

    queue=sum(get_queue(d) for d in DETECTORS)
    queue_history.append(queue)

    print(f"Step {step} Reward {reward:.2f}")

# ============================================================
# Close SUMO
# ============================================================
traci.close()

# ============================================================
# Plots
# ============================================================
plt.plot(step_history,reward_history)
plt.title("Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.show()

plt.plot(step_history,queue_history)
plt.title("Queue Length")
plt.xlabel("Step")
plt.ylabel("Queue")
plt.grid()
plt.show()

plt.plot(step_history,travel_time_history)
plt.title("Average Travel Time (recent vehicles)")
plt.xlabel("Step")
plt.ylabel("Travel Time")
plt.grid()
plt.show()