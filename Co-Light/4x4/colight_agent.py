# =============================================================
# colight_agent.py  —  CoLight DQN agent
# Network: 4x4 grid with 12 traffic-light junctions
# TL IDs (from net.xml tlLogic elements):
#   Row 0 (top):    J1,  J6,  J11, J16
#   Row 1:          J2,  J7,  J12, J17
#   Row 2 (bottom): J3,  J8,  J13, J18
# Adjacency (from edges in net.xml):
#   Horizontal: J1-J6, J6-J11, J11-J16
#               J2-J7, J7-J12, J12-J17
#               J3-J8, J8-J13, J13-J18
#   Vertical:   J1-J2, J2-J3
#               J6-J7, J7-J8
#               J11-J12, J12-J13
#               J16-J17, J17-J18
# =============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import deque
import random

try:
    import gymnasium as gym
    _GYM_MSG = "gymnasium"
except ImportError:
    import gym
    _GYM_MSG = "gym"

# Safe import of torch_geometric + torch_scatter
_HAS_GEO = False
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch_geometric.nn import MessagePassing
        from torch_geometric.utils import add_self_loops
        import torch_scatter
    _HAS_GEO = True
except Exception:
    _HAS_GEO = False

print(f"[CoLightAgent] torch_geometric: {'OK - real GAT' if _HAS_GEO else 'unavailable - MLP fallback'}")


# ---------------------------------------------------------------
# Junction layout (index order matches TL_IDS list below)
#
#  idx:  0    1    2    3
#  row0: J1   J6   J11  J16
#
#  idx:  4    5    6    7
#  row1: J2   J7   J12  J17
#
#  idx:  8    9    10   11
#  row2: J3   J8   J13  J18
#
# ---------------------------------------------------------------
TL_IDS = [
    "J1",  "J6",  "J11", "J16",   # row 0
    "J2",  "J7",  "J12", "J17",   # row 1
    "J3",  "J8",  "J13", "J18",   # row 2
]

_EDGES_UNDIRECTED = [
    # horizontal connections (from net.xml edges E17,E18,E19 etc.)
    (0, 1), (1, 2), (2, 3),    # J1-J6-J11-J16
    (4, 5), (5, 6), (6, 7),    # J2-J7-J12-J17
    (8, 9), (9, 10), (10, 11), # J3-J8-J13-J18
    # vertical connections
    (0, 4), (4, 8),             # J1-J2-J3
    (1, 5), (5, 9),             # J6-J7-J8
    (2, 6), (6, 10),            # J11-J12-J13
    (3, 7), (7, 11),            # J16-J17-J18
]


def _build_edge_index():
    src, dst = [], []
    for a, b in _EDGES_UNDIRECTED:
        src += [a, b]
        dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long)


EDGE_INDEX = _build_edge_index()   # shape [2, 2*E]


# ---------------------------------------------------------------
# Attention / MLP block
# ---------------------------------------------------------------
if _HAS_GEO:
    class _AttBlock(MessagePassing):
        def __init__(self, d, dv, d_out, nv):
            super().__init__(aggr='add')
            self.nv, self.dv = nv, dv
            self.W_t = nn.Linear(d, dv * nv)
            self.W_s = nn.Linear(d, dv * nv)
            self.W_h = nn.Linear(d, dv * nv)
            self.out = nn.Linear(dv, d_out)

        def forward(self, x, edge_index):
            ei, _ = add_self_loops(edge_index)
            return F.relu(self.out(self.propagate(x=x, edge_index=ei)))

        def message(self, x_i, x_j, edge_index):
            nv, dv = self.nv, self.dv
            ht  = F.relu(self.W_t(x_i)).view(-1, nv, dv).permute(1, 0, 2)
            hs  = F.relu(self.W_s(x_j)).view(-1, nv, dv).permute(1, 0, 2)
            idx = edge_index[1]
            e   = (ht * hs).sum(-1)
            mx  = torch_scatter.scatter_max(e, idx)[0]
            ec  = torch.exp(e - mx.index_select(1, idx))
            nm  = torch_scatter.scatter_sum(ec, idx)
            al  = ec / (nm.index_select(1, idx) + 1e-12)
            hh  = F.relu(self.W_h(x_j)).view(-1, nv, dv).permute(1, 0, 2)
            return (hh * al.unsqueeze(-1)).mean(0)
else:
    class _AttBlock(nn.Module):
        """MLP fallback when torch_geometric is unavailable."""
        def __init__(self, d, dv, d_out, nv):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d_out), nn.ReLU(),
                nn.Linear(d_out, d_out), nn.ReLU(),
            )

        def forward(self, x, edge_index=None):
            return self.net(x)


# ---------------------------------------------------------------
# Q-network: embedding -> ATT -> ATT -> Q-head
# ---------------------------------------------------------------
class CoLightNet(nn.Module):
    def __init__(self, ob_len, n_actions, phase_lengths,
                 emb=128, dv=32, heads=4, dout=128):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(ob_len, emb), nn.ReLU(),
            nn.Linear(emb, emb),   nn.ReLU(),
        )
        self.a1   = _AttBlock(emb,  dv, dout, heads)
        self.a2   = _AttBlock(dout, dv, dout, heads)
        self.head = nn.Linear(dout, n_actions)

        masks = [torch.ones(l) for l in phase_lengths]
        self.register_buffer(
            'mask', torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        )

    def forward(self, x, edge_index):
        h = self.emb(x)
        h = self.a1(h, edge_index)
        h = self.a2(h, edge_index)
        return self.head(h)          # [n_agents, n_actions]


# ---------------------------------------------------------------
# Agent
# ---------------------------------------------------------------
class CoLightAgent:
    def __init__(self, world, vehicle_max=20, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 batch_size=32, buffer_size=10000, grad_clip=1.0):

        self.vehicle_max   = vehicle_max
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.batch_size    = batch_size
        self.grad_clip     = grad_clip

        self.tl_ids     = [i.id for i in world.intersections]
        self.sub_agents = len(self.tl_ids)       # 12

        self.edge_idx = EDGE_INDEX               # pre-built adjacency

        self.phase_lengths = np.array([len(i.phases) for i in world.intersections])
        self.action_space  = gym.spaces.Discrete(int(max(self.phase_lengths)))
        self.ob_length     = 16    # lanes per intersection (padded/truncated)

        self.replay_buffer = deque(maxlen=buffer_size)

        self.model        = CoLightNet(self.ob_length, self.action_space.n, self.phase_lengths)
        self.target_model = CoLightNet(self.ob_length, self.action_space.n, self.phase_lengths)
        self.update_target()

        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=lr, alpha=0.9, eps=1e-7)
        self.criterion = nn.MSELoss()

        print(f"[CoLightAgent] gym={_GYM_MSG}  "
              f"GAT={'real' if _HAS_GEO else 'MLP fallback'}  "
              f"agents={self.sub_agents}  "
              f"phases={self.phase_lengths}  ob_len={self.ob_length}")

    # --- build observation for one junction ----------------------
    def _obs_one(self, tl_id: str) -> np.ndarray:
        import traci
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        raw = np.array([traci.lane.getLastStepHaltingNumber(l)
                        for l in lanes], dtype=np.float32)
        raw = raw / self.vehicle_max
        if len(raw) < self.ob_length:
            raw = np.pad(raw, (0, self.ob_length - len(raw)))
        else:
            raw = raw[:self.ob_length]
        return raw

    def get_obs(self) -> np.ndarray:
        """Returns [sub_agents, ob_length] array."""
        return np.stack([self._obs_one(tid) for tid in self.tl_ids])

    # --- action --------------------------------------------------
    def get_action(self, obs: np.ndarray, test: bool = False) -> np.ndarray:
        """obs: [sub_agents, ob_length]  ->  [sub_agents] int array"""
        if not test and np.random.rand() < self.epsilon:
            return np.array([
                np.random.randint(0, int(pl)) for pl in self.phase_lengths
            ])
        x = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            q = self.model(x, self.edge_idx)    # [12, n_actions]
        return np.array([
            int(torch.argmax(q[i, :pl]).item())
            for i, pl in enumerate(self.phase_lengths)
        ])

    # --- store ---------------------------------------------------
    def store(self, obs, next_obs, rewards, actions):
        self.replay_buffer.append((
            obs,
            next_obs,
            np.array(rewards, dtype=np.float32),
            np.array(actions, dtype=np.int64),
        ))

    # --- train ---------------------------------------------------
    def train(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch        = random.sample(self.replay_buffer, self.batch_size)
        s, ns, r, a = zip(*batch)

        # flatten [B, sub_agents, ob_length] -> [B*sub_agents, ob_length]
        s  = torch.tensor(np.array(s),  dtype=torch.float32).view(-1, self.ob_length)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).view(-1, self.ob_length)
        r  = torch.tensor(np.array(r),  dtype=torch.float32).view(-1)
        a  = torch.tensor(np.array(a),  dtype=torch.long).view(-1)

        q  = self.model(s, self.edge_idx)
        with torch.no_grad():
            tq = r + self.gamma * torch.max(
                self.target_model(ns, self.edge_idx), dim=1)[0]

        qt = q.clone().detach()
        for i, ai in enumerate(a):
            qt[i, ai] = tq[i]

        loss = self.criterion(q, qt)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    # --- sync target (call once per episode) ---------------------
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
