# =============================================================
# run.py  —  4x4 CoLight training with SUMO/TraCI
# 12 traffic-light junctions: J1,J2,J3,J6,J7,J8,J11,J12,J13,J16,J17,J18
# =============================================================
# Place BOTH run.py and colight_agent.py in:
#   C:\Users\harsh\Desktop\RL\Co-Light\4x4\
# Then run:  python run.py
# =============================================================

import traci
import numpy as np
import matplotlib.pyplot as plt

from colight_agent import CoLightAgent, TL_IDS

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUMO_CONFIG  = r"C:\Users\harsh\Desktop\RL\Co-Light\4x4\4x4.sumocfg"
NUM_EPISODES = 100
STEPS_PER_EP = 2000
SUMO_BINARY  = "sumo"        # swap to "sumo-gui" to watch visually
# ─────────────────────────────────────────────────────────────────────────────


# ── World wrapper ─────────────────────────────────────────────────────────────
class Intersection:
    def __init__(self, tl_id: str):
        self.id     = tl_id
        self.phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases


class World:
    def __init__(self):
        # Build one Intersection object per traffic-light junction
        self.intersections = [Intersection(tid) for tid in TL_IDS]
        self.graph = {}   # edge_index is pre-built in colight_agent.py
# ─────────────────────────────────────────────────────────────────────────────


def run():
    # Start SUMO once; traci.load() resets it cheaply each episode
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    world = World()
    agent = CoLightAgent(world)

    # Cache controlled lanes per junction (static — doesn't change)
    tl_lanes = {
        tid: traci.trafficlight.getControlledLanes(tid)
        for tid in TL_IDS
    }

    ep_rewards, ep_queues, ep_travel_times = [], [], []

    for episode in range(NUM_EPISODES):

        if episode > 0:
            traci.load(["-c", SUMO_CONFIG])

        vehicle_enter_time   = {}
        vehicle_travel_times = []
        step_rewards         = []
        step_queues          = []
        loss = 0.0

        for step in range(STEPS_PER_EP):

            traci.simulationStep()     # single step per iteration

            # Track vehicle entry times
            for vid in traci.vehicle.getIDList():
                if vid not in vehicle_enter_time:
                    vehicle_enter_time[vid] = step

            # ── Observation: [12, ob_length] ──────────────────────
            obs = agent.get_obs()

            # ── Action: [12] int array ────────────────────────────
            actions = agent.get_action(obs)

            # Apply each action to its traffic light
            for i, tid in enumerate(TL_IDS):
                traci.trafficlight.setPhase(tid, int(actions[i]))

            # ── Per-agent rewards (negative queue per junction) ───
            rewards = np.array([
                -float(sum(traci.lane.getLastStepHaltingNumber(l)
                           for l in tl_lanes[tid]))
                for tid in TL_IDS
            ], dtype=np.float32)

            total_queue = -float(np.sum(rewards))

            # ── Next observation ──────────────────────────────────
            next_obs = agent.get_obs()

            # ── Store transition and train ────────────────────────
            agent.store(obs, next_obs, rewards, actions)
            loss = agent.train()

            # ── Vehicle exit tracking ─────────────────────────────
            for vid in traci.simulation.getArrivedIDList():
                if vid in vehicle_enter_time:
                    vehicle_travel_times.append(
                        step - vehicle_enter_time.pop(vid))

            step_rewards.append(float(np.sum(rewards)))
            step_queues.append(total_queue)

        # ── End of episode ────────────────────────────────────────
        agent.update_target()   # sync target network once per episode

        avg_r  = float(np.mean(step_rewards))
        avg_q  = float(np.mean(step_queues))
        avg_tt = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else 0.0

        ep_rewards.append(avg_r)
        ep_queues.append(avg_q)
        ep_travel_times.append(avg_tt)

        print(
            f"Ep {episode+1:>3}/{NUM_EPISODES} | "
            f"reward {avg_r:>9.2f} | "
            f"queue {avg_q:>6.2f} | "
            f"travel {avg_tt:>6.1f} steps | "
            f"eps {agent.epsilon:.4f} | "
            f"loss {loss:.5f}"
        )

    traci.close()

    # ── Plots ─────────────────────────────────────────────────────────────────
    eps = list(range(1, NUM_EPISODES + 1))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("CoLight 4x4 — per-episode training metrics")

    axes[0].plot(eps, ep_rewards,      color="steelblue")
    axes[0].set_ylabel("Sum reward (all junctions)")
    axes[0].set_title("Total reward (less negative = less waiting)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(eps, ep_queues,       color="darkorange")
    axes[1].set_ylabel("Total queue (vehicles)")
    axes[1].set_title("Network-wide queue length")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(eps, ep_travel_times, color="seagreen")
    axes[2].set_ylabel("Avg travel time (steps)")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Average vehicle travel time")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("training_results_4x4.png", dpi=150)
    plt.show()
    print("Saved training_results_4x4.png")


if __name__ == "__main__":
    run()
