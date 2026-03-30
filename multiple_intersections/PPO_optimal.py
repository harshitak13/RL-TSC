# ============================================================
# Imports
# ============================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SUMO setup
# ============================================================
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

# ============================================================
# SUMO Config
# ============================================================
Sumo_config = [
    "sumo-gui",
    "-c", r"C:\Users\harsh\Desktop\RL\multiple_intersections\multiple_intersections.sumocfg",
    "--step-length", "0.10"
]

# ============================================================
# Hyperparameters
# ============================================================
TOTAL_STEPS = 50000
ROLLOUT = 200

GAMMA = 0.99
LR = 0.001
CLIP = 0.2

STATE_SIZE = 7
ACTION_SIZE = 2

MIN_GREEN = 40
last_switch = -MIN_GREEN

# ============================================================
# Lane IDs (for waiting time)
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
# Detector IDs (for queue)
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
# PPO weights
# ============================================================
W_policy = np.random.randn(STATE_SIZE, ACTION_SIZE) * 0.01
b_policy = np.zeros(ACTION_SIZE)

W_value = np.random.randn(STATE_SIZE) * 0.01
b_value = 0

# ============================================================
# Helper functions
# ============================================================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def get_queue(det):
    return traci.lanearea.getLastStepVehicleNumber(det)

def get_wait(lane):
    return traci.lane.getWaitingTime(lane)

def get_phase():
    return traci.trafficlight.getPhase("node11")

# ============================================================
# State
# ============================================================
def get_state():

    queues = [get_queue(d)/20 for d in DETECTORS]

    phase = get_phase()/4

    return np.array(queues + [phase])

# ============================================================
# Reward
# ============================================================
def get_reward():

    queue = sum(get_queue(d) for d in DETECTORS)

    waiting = sum(get_wait(l) for l in LANES)

    throughput = len(traci.simulation.getArrivedIDList())

    reward = -queue - 0.1*waiting + 2*throughput

    return reward

# ============================================================
# Apply Action
# ============================================================
def apply_action(action):

    global last_switch, step

    if action == 1:

        if step - last_switch >= MIN_GREEN:

            phase = get_phase()
            next_phase = (phase + 1) % 4

            traci.trafficlight.setPhase("node11", next_phase)

            last_switch = step

# ============================================================
# PPO memory
# ============================================================
states=[]
actions=[]
rewards=[]
log_probs=[]
values=[]

# ============================================================
# PPO update
# ============================================================
def ppo_update():

    global W_policy,b_policy,W_value,b_value

    states_arr=np.array(states)
    actions_arr=np.array(actions)
    rewards_arr=np.array(rewards)
    old_log=np.array(log_probs)
    values_arr=np.array(values)

    returns=[]
    G=0

    for r in rewards_arr[::-1]:

        G=r+GAMMA*G
        returns.insert(0,G)

    returns=np.array(returns)

    adv=returns-values_arr
    adv=(adv-adv.mean())/(adv.std()+1e-8)

    for i in range(len(states_arr)):

        s=states_arr[i]
        a=actions_arr[i]

        logits=s@W_policy+b_policy
        probs=softmax(logits)

        logp=np.log(probs[a]+1e-8)

        ratio=np.exp(logp-old_log[i])

        clipped=np.clip(ratio,1-CLIP,1+CLIP)

        loss=-min(ratio*adv[i],clipped*adv[i])

        grad=probs.copy()
        grad[a]-=1
        grad*=loss

        W_policy-=LR*np.outer(s,grad)
        b_policy-=LR*grad

        value=s@W_value+b_value
        error=value-returns[i]

        W_value-=LR*error*s
        b_value-=LR*error

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)

print("\nStarting PPO Training")

# ============================================================
# Graph storage
# ============================================================
steps=[]
reward_hist=[]
queue_hist=[]
travel_hist=[]

veh_depart={}
travel_times=[]

cum_reward=0

# ============================================================
# Training Loop
# ============================================================
for step in range(TOTAL_STEPS):

    state=get_state()

    logits=state@W_policy+b_policy
    probs=softmax(logits)

    action=np.random.choice(ACTION_SIZE,p=probs)

    logp=np.log(probs[action]+1e-8)
    value=state@W_value+b_value

    apply_action(action)

    traci.simulationStep()

    # travel time tracking
    for v in traci.simulation.getDepartedIDList():
        veh_depart[v]=step

    for v in traci.simulation.getArrivedIDList():

        if v in veh_depart:

            tt=step-veh_depart[v]
            travel_times.append(tt)

            del veh_depart[v]

    avg_tt=np.mean(travel_times[-50:]) if len(travel_times)>50 else 0
    travel_hist.append(avg_tt)

    reward=get_reward()

    states.append(state)
    actions.append(action)
    rewards.append(reward)
    log_probs.append(logp)
    values.append(value)

    cum_reward+=reward

    if (step+1)%ROLLOUT==0:

        ppo_update()

        states.clear()
        actions.clear()
        rewards.clear()
        log_probs.clear()
        values.clear()

    steps.append(step)
    reward_hist.append(cum_reward)

    q=sum(get_queue(d) for d in DETECTORS)
    queue_hist.append(q)

    print(f"Step {step} Reward {reward:.2f}")

# ============================================================
# Close SUMO
# ============================================================
traci.close()

# ============================================================
# Plots
# ============================================================
plt.plot(steps,reward_hist)
plt.title("Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.show()

plt.plot(steps,queue_hist)
plt.title("Queue Length")
plt.xlabel("Step")
plt.ylabel("Queue")
plt.grid()
plt.show()

plt.plot(steps,travel_hist)
plt.title("Average Travel Time")
plt.xlabel("Step")
plt.ylabel("Travel Time")
plt.grid()
plt.show()




