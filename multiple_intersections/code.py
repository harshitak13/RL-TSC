import os
import sys

# ============================================================
# SUMO_HOME
# ============================================================
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci

# ============================================================
# SUMO Config
# ============================================================
Sumo_config = [
    "sumo-gui",
    "-c", r"C:\Users\harsh\Desktop\RL\multiple_intersections\multiple_intersections.sumocfg"
]

# ============================================================
# Start SUMO
# ============================================================
traci.start(Sumo_config)

# run one step so simulation loads
traci.simulationStep()

# ============================================================
# Get Lane IDs
# ============================================================
lane_ids = traci.lane.getIDList()

print("\n===== LANE IDS =====")
for lane in lane_ids:
    print(lane)

# ============================================================
# Get Lane Area Detector IDs
# ============================================================
detector_ids = traci.lanearea.getIDList()

print("\n===== DETECTOR IDS =====")
for det in detector_ids:
    print(det)

# ============================================================
# Get Traffic Light IDs
# ============================================================
tls_ids = traci.trafficlight.getIDList()

print("\n===== TRAFFIC LIGHT IDS =====")
for tls in tls_ids:
    print(tls)

# ============================================================
# Close SUMO
# ============================================================
traci.close()