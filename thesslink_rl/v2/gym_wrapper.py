"""Gymnasium multi-agent wrapper for EPyMARL -- v2 (potential-based reward shaping).

Symbolic 19-feature observation vector with GPS signal.

Negotiation shaping (individual per-agent rewards):
  - Optimal suggestion   (+0.10): agent suggests the POI with its highest score
  - Persistence bonus    (+0.05): agent re-suggests its own good POI when peer's
                                  offer scores poorly for this agent
  - Flexibility / accept (+0.10): agent accepts a peer offer that is fair
                                  (own score >= 0.6)

Negotiation common reward:
  - Agreement bonus  (+10.0 * quality): when both agents lock in, where
    quality = score_a * score_b (golden mean)

Navigation shaping:
  - Potential-based shaping: gamma * Phi(s') - Phi(s) each step, where
    Phi(s) = -BFS_distance(agent_pos, target_poi) / max_dist
  - Individual arrival reward: +10.0 * quality when an agent reaches the POI
  - Step penalty: -0.01 per navigation step
  - Terminal reward: +50.0 * quality when ALL agents reach the agreed POI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import (
    ACT_ACCEPT,
    ACT_SUGGEST_BASE,
    ACTION_DIM,
    GRID_SIZE,
    NUM_AGENTS,
    NUM_SUGGEST_ACTIONS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)
from ..evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    negotiation_quality,
    optimal_poi,
)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent

_SHAPING_GAMMA = 0.99
_MAX_BFS_DIST = float(GRID_SIZE * GRID_SIZE)


def _potential(agent_pos: tuple[int, int], bfs_grid: np.ndarray) -> float:
    """Phi(s) = -BFS distance from agent to target POI / max_dist."""
    d = bfs_grid[agent_pos[0], agent_pos[1]]
    if np.isinf(d):
        return -1.0
    return -d / _MAX_BFS_DIST


class GridNegotiationGymEnv(gym.Env):
    """Gymnasium wrapper around GridNegotiationEnv (v2) for EPyMARL."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        agent0_config: str | None = None,
        agent1_config: str | None = None,
        render_mode: str | None = None,
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__()

        default_models = _PACKAGE_DIR / "models"
        cfg_0 = AgentConfig.from_yaml(agent0_config or str(default_models / "human.yaml"))
        cfg_1 = AgentConfig.from_yaml(agent1_config or str(default_models / "taxi.yaml"))
        self._agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

        self._env = GridNegotiationEnv(
            agent_configs=self._agent_configs,
            render_mode=render_mode,
            seed=seed,
        )

        self.n_agents = NUM_AGENTS

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(OBS_FLAT_SIZE,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )

        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._optimal_poi: int = 0

        self._prev_potentials: Dict[str, float] = {}
        self._target_bfs: np.ndarray | None = None
        self._individual_arrived: Dict[str, bool] = {}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[tuple[np.ndarray, ...], dict]:
        super().reset(seed=seed)
        self._env.reset(seed=seed, options=options)

        agents = self._env.possible_agents
        for agent in agents:
            spawn = tuple(self._env.spawn_positions[agent])
            cfg = self._agent_configs.get(agent)
            scores = compute_poi_scores(
                spawn, spawn, self._env.poi_positions,
                self._env.obstacle_map, cfg,
            )
            self._poi_scores[agent] = scores
            self._env.poi_scores[agent] = scores

        self._agreed_poi = None
        self._optimal_poi = optimal_poi(self._poi_scores, agents)

        self._prev_potentials = {}
        self._target_bfs = None
        self._individual_arrived = {a: False for a in agents}

        obs_tuple = tuple(
            self._env._get_obs(a) for a in agents
        )

        info = {
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
            "optimal_poi": self._optimal_poi,
        }
        return obs_tuple, info

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}

        prev_phase = self._env.phase
        prev_neg_turn = self._env.neg_turn
        prev_suggestions = dict(self._env.last_suggestion)

        obs_d, rewards_d, terminated_d, truncated_d, infos_d = self._env.step(actions_dict)

        obs_tuple = tuple(obs_d[a] for a in agents)
        rewards = [0.0] * self.n_agents

        # ── Negotiation phase rewards ────────────────────────────────────
        if prev_phase == "negotiation" and prev_neg_turn is not None:
            active = prev_neg_turn
            act = actions_dict[active]
            agent_idx = agents.index(active)
            peer = agents[1 - agent_idx]
            my_scores = self._poi_scores[active]

            if ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                suggested_poi = act - ACT_SUGGEST_BASE

                if suggested_poi == int(np.argmax(my_scores)):
                    rewards[agent_idx] += 0.1

                if peer in prev_suggestions:
                    peer_suggested = prev_suggestions[peer]
                    if (my_scores[peer_suggested] < 0.4
                            and my_scores[suggested_poi] > my_scores[peer_suggested]):
                        rewards[agent_idx] += 0.05

            elif act == ACT_ACCEPT and peer in prev_suggestions:
                peer_suggested = prev_suggestions[peer]
                if my_scores[peer_suggested] >= 0.6:
                    rewards[agent_idx] += 0.1

            just_agreed = (
                self._agreed_poi is None and self._env.agreed_poi is not None
            )
            if just_agreed:
                self._agreed_poi = self._env.agreed_poi
                quality = negotiation_quality(
                    self._agreed_poi, self._poi_scores, agents,
                )
                for i in range(self.n_agents):
                    rewards[i] += quality * 10.0

                target = self._env.poi_positions[self._agreed_poi]
                self._target_bfs = bfs_distances(target, self._env.obstacle_map)
                for a in agents:
                    pos = tuple(self._env.agent_positions[a])
                    self._prev_potentials[a] = _potential(pos, self._target_bfs)

        # ── Navigation phase rewards ─────────────────────────────────────
        if prev_phase == "navigation" and self._agreed_poi is not None and self._target_bfs is not None:
            quality = negotiation_quality(
                self._agreed_poi, self._poi_scores, agents,
            )

            for i, a in enumerate(agents):
                if self._individual_arrived.get(a, False):
                    continue

                cur_pos = tuple(self._env.agent_positions[a])
                cur_phi = _potential(cur_pos, self._target_bfs)
                prev_phi = self._prev_potentials.get(a, cur_phi)

                rewards[i] += _SHAPING_GAMMA * cur_phi - prev_phi
                self._prev_potentials[a] = cur_phi

                rewards[i] -= 0.01

                if self._env.agents_reached.get(a, False) and not self._individual_arrived[a]:
                    self._individual_arrived[a] = True
                    rewards[i] += quality * 10.0

            all_reached = all(self._env.agents_reached[a] for a in agents)
            if all_reached:
                for i in range(self.n_agents):
                    rewards[i] += quality * 50.0

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)

        all_reached = all(
            "reached_poi" in infos_d.get(a, {}) for a in agents
        )
        negotiation_agreed = self._env.agreed_poi is not None
        agreed_optimal = (
            negotiation_agreed and self._agreed_poi == self._optimal_poi
        )

        info: dict[str, Any] = {
            "battle_won": all_reached,
            "reached_poi": int(all_reached),
            "negotiation_agreed": float(negotiation_agreed),
            "negotiation_optimal": float(agreed_optimal),
        }

        return obs_tuple, rewards, done, truncated, info

    def get_avail_actions(self) -> List[List[int]]:
        """Return per-agent action masks for EPyMARL."""
        return [
            self._env.get_avail_actions(a) for a in self._env.possible_agents
        ]

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._env._seed = seed
            self._env._rng = np.random.RandomState(seed)
