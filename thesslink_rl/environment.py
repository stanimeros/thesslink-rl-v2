"""Core grid environment: 10x10 grid with obstacles, POIs, and comms.

Two phases per episode, both controlled by RL:
  1. Negotiation -- agents suggest POIs and agree on a target.
  2. Navigation  -- agents move to reach the agreed POI.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

GRID_SIZE = 10
NUM_OBSTACLES = 10
NUM_POIS = 3
NUM_AGENTS = 2
COMM_DIM = NUM_POIS
MAX_EPISODE_STEPS = 60

NUM_MOVE_ACTIONS = 5

# Negotiation actions
ACT_SUGGEST_BASE = 5  # 5 = suggest POI 0, 6 = suggest POI 1, 7 = suggest POI 2
NUM_SUGGEST_ACTIONS = NUM_POIS

ACTION_DIM = NUM_MOVE_ACTIONS + NUM_SUGGEST_ACTIONS  # 8

# Grid channel indices (C, H, W) with C=3
CH_OBSTACLE = 0
CH_POI = 1
CH_SELF = 2
NUM_CHANNELS = 3

# Negotiation obs: own POI scores (NUM_POIS) + peer last action one-hot (NUM_SUGGEST_ACTIONS + 1)
# The +1 is for "no action yet" at episode start
PEER_ACTION_DIM = NUM_SUGGEST_ACTIONS + 1  # 4: [suggest_0, suggest_1, suggest_2, no_action]
NEG_OBS_RAW_SIZE = NUM_POIS + PEER_ACTION_DIM  # 7

# Flat observation size: grid (C*H*W) + comm, negotiation obs is zero-padded to this
OBS_FLAT_SIZE = NUM_CHANNELS * GRID_SIZE * GRID_SIZE + COMM_DIM  # 303


class GridNegotiationEnv:
    """Two agents negotiate over POIs then navigate to the agreed one.

    Both phases are RL-controlled. During negotiation, agents choose
    suggest actions (5-7) to propose a POI. When both suggest the same
    POI on the same step, they agree and switch to navigation.
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

        self.comm_buffer: Dict[str, np.ndarray] = {
            a: np.zeros(COMM_DIM, dtype=np.float32) for a in self.agents
        }
        self.phase = "negotiation"
        self.agreed_poi = None
        self.last_suggestion = {}
        self.poi_scores = {}

        obs = {a: self._get_obs(a) for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, info

    def step(self, actions: Dict[str, int]):
        self.timestep += 1

        obs: Dict[str, Dict[str, np.ndarray]]
        rewards = {a: 0.0 for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: self.timestep >= MAX_EPISODE_STEPS for a in self.agents}
        infos: Dict[str, dict] = {a: {"phase": self.phase} for a in self.agents}

        if self.phase == "negotiation":
            suggestions: Dict[str, int] = {}
            for agent, act in actions.items():
                if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                    poi_idx = act - ACT_SUGGEST_BASE
                    suggestions[agent] = poi_idx
                    self.last_suggestion[agent] = poi_idx

            if len(suggestions) == NUM_AGENTS:
                values = list(suggestions.values())
                if values[0] == values[1]:
                    self.agreed_poi = values[0]
                    self.phase = "navigation"
                    for a in self.agents:
                        infos[a]["agreed_poi"] = self.agreed_poi
                        infos[a]["phase"] = "navigation"

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
            for i in range(NUM_SUGGEST_ACTIONS):
                mask[ACT_SUGGEST_BASE + i] = 1
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
        if self.phase == "negotiation":
            return self._get_negotiation_obs(agent)
        return self._get_navigation_obs(agent)

    def _get_negotiation_obs(self, agent: str) -> Dict[str, np.ndarray]:
        own_scores = self.poi_scores.get(
            agent, np.zeros(NUM_POIS, dtype=np.float32)
        )

        peer = [a for a in self.possible_agents if a != agent][0]
        peer_action_onehot = np.zeros(PEER_ACTION_DIM, dtype=np.float32)
        if peer in self.last_suggestion:
            peer_action_onehot[self.last_suggestion[peer]] = 1.0
        else:
            peer_action_onehot[-1] = 1.0  # "no action yet"

        return {"scores": own_scores.copy(), "peer_action": peer_action_onehot}

    def _get_navigation_obs(self, agent: str) -> Dict[str, np.ndarray]:
        grid = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[CH_OBSTACLE] = self.obstacle_map.astype(np.float32)
        for pr, pc in self.poi_positions:
            grid[CH_POI, pr, pc] = 1.0

        r, c = self.agent_positions[agent]
        grid[CH_SELF, r, c] = 1.0

        other = [a for a in self.possible_agents if a != agent][0]
        peer_comm = self.comm_buffer[other].copy()
        return {"grid": grid, "comm": peer_comm}

