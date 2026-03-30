# =========================
# run.py
# =========================
# Place this file in the SAME folder as colight_agent.py
# Run: python run.py
# =========================

import sys
import os

# Make sure Python finds colight_agent.py when run.py lives in the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traci
import numpy as np
import matplotlib.pyplot as plt

from colight_agent import CoLightAgent

# ========================
# CONFIG  — edit these
# ========================
SUMO_CONFIG  = r"C:\Users\harsh\Desktop\RL\Co-Light\1x1\1x1.sumocfg"
NUM_EPISODES = 100          # number of full simulation resets to train over
STEPS_PER_EP = 2000         # simulation steps per episode
SUMO_BINARY  = "sumo"       # swap to "sumo-gui" to watch one episode visually
TL_ID        = "J1"         # traffic-light junction ID in your .net.xml


# ========================
# WORLD WRAPPER
# ========================
class Intersection:
    """Lightweight wrapper around a single SUMO traffic light."""
    def __init__(self, tl_id: str):
        self.id     = tl_id
        self.phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases


class World:
    def __init__(self):
        self.intersections = [Intersection(TL_ID)]
        # For a multi-intersection grid, populate sparse_adj here.
        # Self-loop placeholder keeps the 1×1 case working.
        self.graph = {"sparse_adj": np.array([[0, 0]])}


# ========================
# MAIN
# ========================
def run():
    # Start SUMO once — subsequent episodes use traci.load() to reset cheaply
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    world = World()
    agent = CoLightAgent(world)

    # ── Per-episode metric storage ──────────────────────────────────────
    ep_rewards      = []
    ep_queues       = []
    ep_travel_times = []

    for episode in range(NUM_EPISODES):

        # Reset the simulation without restarting the SUMO process
        if episode > 0:
            traci.load(["-c", SUMO_CONFIG])

        vehicle_enter_time   = {}
        vehicle_travel_times = []
        step_rewards         = []
        step_queues          = []

        lanes = traci.trafficlight.getControlledLanes(TL_ID)

        # ── Episode step loop ───────────────────────────────────────────
        for step in range(STEPS_PER_EP):

            # ONE simulation step per iteration (was doubled in the old code)
            traci.simulationStep()

            # Track when each vehicle entered the network
            for veh_id in traci.vehicle.getIDList():
                if veh_id not in vehicle_enter_time:
                    vehicle_enter_time[veh_id] = step

            # ── Observation (normalised) ──────────────────────────────
            obs = agent.get_obs(lanes)          # shape: [1, ob_length]

            # ── Action ───────────────────────────────────────────────
            action = agent.get_action(obs)
            traci.trafficlight.setPhase(TL_ID, int(action[0]))

            # ── Reward: negative total halting vehicles ───────────────
            queue  = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
            reward = -float(queue)

            # ── Next observation (same step, no extra sim advance) ────
            next_obs = agent.get_obs(lanes)

            # ── Store transition and train ────────────────────────────
            agent.store(obs, next_obs, reward, action)
            loss = agent.train()

            # ── Vehicle exit tracking for travel-time metric ──────────
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in vehicle_enter_time:
                    vehicle_travel_times.append(step - vehicle_enter_time.pop(veh_id))

            step_rewards.append(reward)
            step_queues.append(queue)

        # ── End of episode ────────────────────────────────────────────────
        # Sync target network once per episode (correct cadence)
        agent.update_target()

        ep_avg_reward = float(np.mean(step_rewards))
        ep_avg_queue  = float(np.mean(step_queues))
        ep_avg_tt     = float(np.mean(vehicle_travel_times)) if vehicle_travel_times else 0.0

        ep_rewards.append(ep_avg_reward)
        ep_queues.append(ep_avg_queue)
        ep_travel_times.append(ep_avg_tt)

        print(
            f"Ep {episode+1:>3}/{NUM_EPISODES} | "
            f"Avg reward: {ep_avg_reward:>8.2f} | "
            f"Avg queue: {ep_avg_queue:>5.2f} | "
            f"Avg travel time: {ep_avg_tt:>6.1f} steps | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Loss: {loss:.5f}"
        )

    traci.close()

    # ========================
    # PLOTTING
    # ========================
    episodes = list(range(1, NUM_EPISODES + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("CoLight — per-episode training metrics", fontsize=14)

    axes[0].plot(episodes, ep_rewards, color="steelblue")
    axes[0].set_ylabel("Avg reward per step")
    axes[0].set_title("Reward (less negative = less waiting)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(episodes, ep_queues, color="darkorange")
    axes[1].set_ylabel("Avg queue (vehicles)")
    axes[1].set_title("Queue length")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(episodes, ep_travel_times, color="seagreen")
    axes[2].set_ylabel("Avg travel time (steps)")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Average vehicle travel time")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    run()