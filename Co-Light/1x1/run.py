# =============================================================
# run.py  —  multi-episode CoLight training with SUMO/TraCI
# =============================================================
# IMPORTANT: copy BOTH run.py and colight_agent.py into the
# same folder, e.g. C:\Users\harsh\Desktop\RL\Co-Light\1x1\
# Then run:  python run.py
# =============================================================

import traci
import numpy as np
import matplotlib.pyplot as plt

from colight_agent import CoLightAgent   # must be in the same folder

# ── CONFIG ──────────────────────────────────────────────────────────────────
SUMO_CONFIG  = r"C:\Users\harsh\Desktop\RL\Co-Light\1x1\1x1.sumocfg"
NUM_EPISODES = 100
STEPS_PER_EP = 2000
SUMO_BINARY  = "sumo"       # change to "sumo-gui" to watch in the GUI
TL_ID        = "J1"
# ────────────────────────────────────────────────────────────────────────────


class Intersection:
    def __init__(self, tl_id):
        self.id     = tl_id
        self.phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases


class World:
    def __init__(self):
        self.intersections = [Intersection(TL_ID)]
        self.graph         = {"sparse_adj": np.array([[0, 0]])}


def run():
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    world = World()
    agent = CoLightAgent(world)

    ep_rewards, ep_queues, ep_travel_times = [], [], []

    for episode in range(NUM_EPISODES):

        if episode > 0:
            traci.load(["-c", SUMO_CONFIG])   # reset sim, keep process alive

        vehicle_enter_time   = {}
        vehicle_travel_times = []
        step_rewards         = []
        step_queues          = []
        lanes = traci.trafficlight.getControlledLanes(TL_ID)
        loss  = 0.0

        for step in range(STEPS_PER_EP):

            traci.simulationStep()            # exactly ONE step per iteration

            for vid in traci.vehicle.getIDList():
                if vid not in vehicle_enter_time:
                    vehicle_enter_time[vid] = step

            obs    = agent.get_obs(lanes)
            action = agent.get_action(obs)
            traci.trafficlight.setPhase(TL_ID, int(action[0]))

            queue    = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            reward   = -float(queue)
            next_obs = agent.get_obs(lanes)

            agent.store(obs, next_obs, reward, action)
            loss = agent.train()

            for vid in traci.simulation.getArrivedIDList():
                if vid in vehicle_enter_time:
                    vehicle_travel_times.append(step - vehicle_enter_time.pop(vid))

            step_rewards.append(reward)
            step_queues.append(queue)

        agent.update_target()   # sync target net once per episode

        avg_r  = float(np.mean(step_rewards))
        avg_q  = float(np.mean(step_queues))
        avg_tt = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else 0.0

        ep_rewards.append(avg_r)
        ep_queues.append(avg_q)
        ep_travel_times.append(avg_tt)

        print(f"Ep {episode+1:>3}/{NUM_EPISODES} | "
              f"reward {avg_r:>8.2f} | queue {avg_q:>5.2f} | "
              f"travel {avg_tt:>6.1f} steps | "
              f"eps {agent.epsilon:.4f} | loss {loss:.5f}")

    traci.close()

    # ── plots ──────────────────────────────────────────────────────────────
    eps = list(range(1, NUM_EPISODES + 1))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("CoLight training — per-episode averages")

    axes[0].plot(eps, ep_rewards,      color="steelblue");  axes[0].set_ylabel("Avg reward");       axes[0].grid(True, alpha=0.4)
    axes[1].plot(eps, ep_queues,       color="darkorange"); axes[1].set_ylabel("Avg queue");        axes[1].grid(True, alpha=0.4)
    axes[2].plot(eps, ep_travel_times, color="seagreen");   axes[2].set_ylabel("Avg travel time");  axes[2].grid(True, alpha=0.4)
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Saved training_results.png")


if __name__ == "__main__":
    run()
