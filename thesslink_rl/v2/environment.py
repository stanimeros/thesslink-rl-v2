"""Core grid environment v2: symbolic observations with GPS signal.

Observation vector (size 19):
  phase_flag    (1)  0.0 = Negotiation, 1.0 = Navigation
  self_scores   (3)  Agent's preference scores for the 3 POIs
  peer_action   (4)  One-hot: [No action, Suggest 0, Suggest 1, Suggest 2]
  agreed_poi    (3)  One-hot: which POI was agreed (all 0 during negotiation)
  self_pos      (2)  (row, col) normalised to [0, 1]
  relative_pos  (2)  (target_row - self_row, target_col - self_col) / (GRID_SIZE-1)
                     zero during negotiation; GPS towards agreed POI during navigation
  lidar         (4)  Distance to nearest obstacle in [N, S, E, W], normalised

Cooperative meeting: episode ends only when ALL agents reach the agreed POI.
Agents that arrive first are frozen (only Stay action available).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

ENV_TAG = "v2"
GRID_SIZE = 10
NUM_OBSTACLES = 10
NUM_POIS = 3
NUM_AGENTS = 2

NUM_MOVE_ACTIONS = 5

ACT_SUGGEST_BASE = 5
NUM_SUGGEST_ACTIONS = NUM_POIS
ACT_ACCEPT = ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS  # 8

ACTION_DIM = NUM_MOVE_ACTIONS + NUM_SUGGEST_ACTIONS + 1  # 9

PEER_ACTION_DIM = NUM_SUGGEST_ACTIONS + 1  # 4: [no_action, suggest_0, suggest_1, suggest_2]
AGREED_POI_DIM = NUM_POIS  # 3
PHASE_DIM = 1
SELF_SCORES_DIM = NUM_POIS  # 3
SELF_POS_DIM = 2
RELATIVE_POS_DIM = 2  # (target - self) normalised
LIDAR_DIM = 4  # N, S, E, W

OBS_FLAT_SIZE = (
    PHASE_DIM           # 1
    + SELF_SCORES_DIM   # 3
    + PEER_ACTION_DIM   # 4
    + AGREED_POI_DIM    # 3
    + SELF_POS_DIM      # 2
    + RELATIVE_POS_DIM  # 2
    + LIDAR_DIM         # 4
)  # = 19


class GridNegotiationEnv:
    """Two agents negotiate over POIs then both navigate to the agreed one.

    Cooperative meeting: the episode terminates only when ALL agents have
    reached the agreed POI.  An agent that arrives first is frozen in place
    (action mask allows only Stay).
    """

    metadata = {"name": ENV_TAG, "render_modes": ["human"]}

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
        self.neg_turn: Optional[str] = None
        self.agents_reached: Dict[str, bool] = {}

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
        self.agents_reached = {a: False for a in self.possible_agents}

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
        truncated = {a: False for a in self.agents}
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
                if not self.agents_reached.get(agent, False):
                    if 0 <= act < NUM_MOVE_ACTIONS:
                        self._apply_move(agent, act)

            if self.agreed_poi is not None:
                target = self.poi_positions[self.agreed_poi]
                for a in self.agents:
                    if tuple(self.agent_positions[a]) == target:
                        self.agents_reached[a] = True

                if all(self.agents_reached[a] for a in self.agents):
                    terminated = {ag: True for ag in self.agents}
                    infos = {
                        ag: {"reached_poi": self.agreed_poi, "phase": self.phase}
                        for ag in self.agents
                    }

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
                mask[0] = 1
        else:
            if self.agents_reached.get(agent, False):
                mask[0] = 1  # frozen — only Stay
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

    def _lidar(self, r: int, c: int) -> np.ndarray:
        """Cast rays in N, S, E, W; return normalised distance to nearest obstacle."""
        distances = np.ones(4, dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W
        max_dist = GRID_SIZE - 1
        for i, (dr, dc) in enumerate(directions):
            for step in range(1, GRID_SIZE):
                nr, nc = r + dr * step, c + dc * step
                if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
                    distances[i] = step / max_dist
                    break
                if self.obstacle_map[nr, nc]:
                    distances[i] = step / max_dist
                    break
            else:
                distances[i] = 1.0
        return distances

    def _get_obs(self, agent: str) -> np.ndarray:
        """Symbolic observation vector of size OBS_FLAT_SIZE (19)."""
        phase_flag = np.array(
            [1.0 if self.phase == "navigation" else 0.0], dtype=np.float32,
        )

        own_scores = self.poi_scores.get(
            agent, np.zeros(NUM_POIS, dtype=np.float32),
        )

        peer = self._peer(agent)
        peer_action_onehot = np.zeros(PEER_ACTION_DIM, dtype=np.float32)
        if peer in self.last_suggestion:
            peer_action_onehot[self.last_suggestion[peer] + 1] = 1.0
        else:
            peer_action_onehot[0] = 1.0  # no_action slot

        agreed_onehot = np.zeros(NUM_POIS, dtype=np.float32)
        if self.agreed_poi is not None:
            agreed_onehot[self.agreed_poi] = 1.0

        r, c = self.agent_positions[agent]
        norm = GRID_SIZE - 1
        self_pos = np.array([r / norm, c / norm], dtype=np.float32)

        relative_pos = np.zeros(RELATIVE_POS_DIM, dtype=np.float32)
        if self.agreed_poi is not None:
            tr, tc = self.poi_positions[self.agreed_poi]
            relative_pos[0] = (tr - r) / norm
            relative_pos[1] = (tc - c) / norm

        lidar = self._lidar(r, c)

        return np.concatenate([
            phase_flag,
            own_scores,
            peer_action_onehot,
            agreed_onehot,
            self_pos,
            relative_pos,
            lidar,
        ])
