"""Core grid environment: 10x10 grid with obstacles, POIs, and two-phase episodes.

Two phases per episode, both controlled by RL:
  1. Negotiation -- agents take turns suggesting/accepting POIs.
  2. Navigation  -- agents move to reach the agreed POI.

Negotiation is turn-based: one agent has priority each step. The active
agent can *suggest* a POI or *accept* the peer's last suggestion (if any).
The other agent is passive (forced no-op) until it becomes their turn.
Agreement happens when the active agent accepts the peer's proposal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

GRID_SIZE = 10
NUM_OBSTACLES = 10
NUM_POIS = 3
NUM_AGENTS = 2
MAX_EPISODE_STEPS = 100

NUM_MOVE_ACTIONS = 5

# Negotiation actions: suggest POI 0/1/2, or accept the peer's last suggestion
ACT_SUGGEST_BASE = 5  # 5 = suggest POI 0, 6 = suggest POI 1, 7 = suggest POI 2
NUM_SUGGEST_ACTIONS = NUM_POIS
ACT_ACCEPT = ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS  # 8

ACTION_DIM = NUM_MOVE_ACTIONS + NUM_SUGGEST_ACTIONS + 1  # 9 (5 move + 3 suggest + 1 accept)

# Grid channel indices (C, H, W) with C=3
CH_OBSTACLE = 0
CH_POI = 1
CH_SELF = 2
NUM_CHANNELS = 3

# Negotiation obs components
# peer_action one-hot: [suggest_0, suggest_1, suggest_2, accept, no_action]
PEER_ACTION_DIM = NUM_SUGGEST_ACTIONS + 2  # 5
MY_TURN_DIM = 1  # 1 if it's this agent's turn, 0 otherwise

# Unified observation layout (always the same in both phases):
#   [phase_flag(1), my_turn(1), poi_scores(3), peer_action(5), agreed_poi_onehot(3), grid(3*10*10)]
PHASE_DIM = 1
NEG_INFO_DIM = MY_TURN_DIM + NUM_POIS + PEER_ACTION_DIM  # 9
AGREED_POI_DIM = NUM_POIS                                 # 3
GRID_FLAT_DIM = NUM_CHANNELS * GRID_SIZE * GRID_SIZE       # 300
OBS_FLAT_SIZE = PHASE_DIM + NEG_INFO_DIM + AGREED_POI_DIM + GRID_FLAT_DIM  # 313


class GridNegotiationEnv:
    """Two agents negotiate over POIs then navigate to the agreed one.

    Negotiation is turn-based. A random agent starts. The active agent
    can *suggest* a POI or *accept* the peer's last suggestion. The
    passive agent is forced to no-op. Turns alternate each step.
    Agreement happens when the active agent plays *accept*.
    """

    metadata = {"name": "grid_negotiation_v0", "render_modes": ["human"]}

    def __init__(
        self,
        agent_configs: dict | None = None,
        render_mode: Optional[str] = None,
        seed: int = 0,
    ):
        self.render_mode = render_mode
        self._seed = seed
        self.possible_agents = [f"agent_{i}" for i in range(NUM_AGENTS)]
        self.agents: List[str] = list(self.possible_agents)
        self.timestep = 0
        self.agent_configs = agent_configs or {}

        self._rng = np.random.RandomState(seed)
        self._build_static_map()

        self.poi_scores: Dict[str, np.ndarray] = {}
        self.last_suggestion: Dict[str, int] = {}
        self.phase = "negotiation"
        self.agreed_poi: Optional[int] = None
        self.neg_turn: Optional[str] = None  # which agent acts this step

    def reset(self, seed=None, options=None) -> tuple[Dict, Dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._build_static_map()

        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.agent_positions: Dict[str, list[int]] = {}
        for agent in self.possible_agents:
            self.agent_positions[agent] = self._random_free_cell()
        self.spawn_positions = {a: list(p) for a, p in self.agent_positions.items()}

        self.phase = "negotiation"
        self.agreed_poi = None
        self.last_suggestion = {}
        self.poi_scores = {}

        first = int(self._rng.randint(0, NUM_AGENTS))
        self.neg_turn = self.possible_agents[first]

        obs = {a: self._get_obs(a) for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, info

    def _peer(self, agent: str) -> str:
        return [a for a in self.possible_agents if a != agent][0]

    def step(self, actions: Dict[str, int]):
        self.timestep += 1

        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: self.timestep >= MAX_EPISODE_STEPS for a in self.agents}
        infos: Dict[str, dict] = {a: {"phase": self.phase} for a in self.agents}

        if self.phase == "negotiation":
            active = self.neg_turn
            assert active is not None
            act = actions.get(active, 0)
            peer = self._peer(active)

            if act == ACT_ACCEPT and peer in self.last_suggestion:
                self.agreed_poi = self.last_suggestion[peer]
                self.phase = "navigation"
                self.neg_turn = None
                for a in self.agents:
                    infos[a]["agreed_poi"] = self.agreed_poi
                    infos[a]["phase"] = "navigation"
            elif ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                poi_idx = act - ACT_SUGGEST_BASE
                self.last_suggestion[active] = poi_idx
                self.neg_turn = peer

        elif self.phase == "navigation":
            for agent, act in actions.items():
                if 0 <= act < NUM_MOVE_ACTIONS:
                    self._apply_move(agent, act)

            if self.agreed_poi is not None:
                target = self.poi_positions[self.agreed_poi]
                for a in self.agents:
                    if tuple(self.agent_positions[a]) == target:
                        terminated = {ag: True for ag in self.agents}
                        infos = {
                            ag: {"reached_poi": self.agreed_poi, "phase": self.phase}
                            for ag in self.agents
                        }
                        break

        obs = {a: self._get_obs(a) for a in self.agents}

        if any(truncated.values()):
            self.agents = []
        elif any(terminated.values()):
            self.agents = []

        return obs, rewards, terminated, truncated, infos

    def get_avail_actions(self, agent: str) -> List[int]:
        """Return a binary mask of length ACTION_DIM for valid actions."""
        mask = [0] * ACTION_DIM
        if self.phase == "negotiation":
            if agent == self.neg_turn:
                for i in range(NUM_SUGGEST_ACTIONS):
                    mask[ACT_SUGGEST_BASE + i] = 1
                peer = self._peer(agent)
                if peer in self.last_suggestion:
                    mask[ACT_ACCEPT] = 1
            else:
                mask[0] = 1  # passive agent: no-op (stay)
        else:
            for i in range(NUM_MOVE_ACTIONS):
                mask[i] = 1
        return mask

    # --- Helpers ----------------------------------------------------------

    def _build_static_map(self):
        self.obstacle_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        placed = 0
        while placed < NUM_OBSTACLES:
            r, c = self._rng.randint(0, GRID_SIZE, size=2)
            if not self.obstacle_map[r, c]:
                self.obstacle_map[r, c] = True
                placed += 1

        self.poi_positions: list[tuple[int, int]] = []
        while len(self.poi_positions) < NUM_POIS:
            r, c = int(self._rng.randint(0, GRID_SIZE)), int(self._rng.randint(0, GRID_SIZE))
            if not self.obstacle_map[r, c] and (r, c) not in self.poi_positions:
                self.poi_positions.append((r, c))

    def _random_free_cell(self) -> list[int]:
        while True:
            r, c = int(self._rng.randint(0, GRID_SIZE)), int(self._rng.randint(0, GRID_SIZE))
            if not self.obstacle_map[r, c] and (r, c) not in self.poi_positions:
                return [r, c]

    def _apply_move(self, agent: str, action: int):
        dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
        r, c = self.agent_positions[agent]
        nr, nc = max(0, min(GRID_SIZE - 1, r + dr)), max(0, min(GRID_SIZE - 1, c + dc))
        if not self.obstacle_map[nr, nc]:
            self.agent_positions[agent] = [nr, nc]

    def _get_obs(self, agent: str) -> Dict[str, np.ndarray]:
        """Unified observation: phase flag + turn + negotiation info + agreed POI + grid."""
        phase_flag = np.array(
            [1.0 if self.phase == "navigation" else 0.0], dtype=np.float32,
        )

        my_turn = np.array(
            [1.0 if self.neg_turn == agent else 0.0], dtype=np.float32,
        )

        own_scores = self.poi_scores.get(
            agent, np.zeros(NUM_POIS, dtype=np.float32),
        )
        peer = self._peer(agent)
        peer_action_onehot = np.zeros(PEER_ACTION_DIM, dtype=np.float32)
        if peer in self.last_suggestion:
            peer_action_onehot[self.last_suggestion[peer]] = 1.0
        else:
            peer_action_onehot[-1] = 1.0  # no_action slot

        agreed_onehot = np.zeros(NUM_POIS, dtype=np.float32)
        if self.agreed_poi is not None:
            agreed_onehot[self.agreed_poi] = 1.0

        grid = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[CH_OBSTACLE] = self.obstacle_map.astype(np.float32)
        for pr, pc in self.poi_positions:
            grid[CH_POI, pr, pc] = 1.0
        r, c = self.agent_positions[agent]
        grid[CH_SELF, r, c] = 1.0

        return {
            "phase": phase_flag,
            "my_turn": my_turn,
            "scores": own_scores.copy(),
            "peer_action": peer_action_onehot,
            "agreed_poi": agreed_onehot,
            "grid": grid,
        }

