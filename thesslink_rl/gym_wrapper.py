"""Gymnasium multi-agent wrapper for EPyMARL compatibility.

EPyMARL's GymmaWrapper expects:
  - env.unwrapped.n_agents (int)
  - env.action_space = Tuple(Discrete, Discrete, ...)
  - env.observation_space = Tuple(Box, Box, ...) with flat 1D observations
  - reset() -> (tuple_of_obs, info)
  - step(list_of_actions) -> (tuple_of_obs, list_of_rewards, done, truncated, info)

Both negotiation and navigation are RL-controlled. Negotiation happens as
normal episode steps with suggest actions; navigation follows once agents agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import (
    ACTION_DIM,
    NUM_AGENTS,
    OBS_FLAT_SIZE,
    GridNegotiationEnv,
)
from .evaluation import (
    AgentConfig,
    bfs_distances,
    compute_poi_scores,
    golden_mean_reward,
)

_PACKAGE_DIR = Path(__file__).resolve().parent


class GridNegotiationGymEnv(gym.Env):
    """Gymnasium wrapper around GridNegotiationEnv for EPyMARL."""

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
                    low=-np.inf,
                    high=np.inf,
                    shape=(OBS_FLAT_SIZE,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            )
        )

        self._poi_scores: Dict[str, np.ndarray] = {}
        self._agreed_poi: int | None = None
        self._prev_dist: Dict[str, float] = {}

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the unified obs dict into a flat vector of size OBS_FLAT_SIZE."""
        return np.concatenate([
            obs_dict["phase"],
            obs_dict["my_turn"],
            obs_dict["scores"],
            obs_dict["peer_action"],
            obs_dict["agreed_poi"],
            obs_dict["grid"].flatten(),
        ])

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
        self._prev_dist = {}

        obs_tuple = tuple(
            self._flatten_obs(self._env._get_obs(a)) for a in agents
        )

        info = {
            "poi_scores": {k: v.tolist() for k, v in self._poi_scores.items()},
        }
        return obs_tuple, info

    def _bfs_dist_to_target(self, agent: str) -> float:
        """BFS distance from agent's current position to the agreed POI."""
        target = self._env.poi_positions[self._agreed_poi]
        pos = tuple(self._env.agent_positions[agent])
        dist_map = bfs_distances(pos, self._env.obstacle_map)
        d = dist_map[target[0], target[1]]
        return float(d) if np.isfinite(d) else float(self._env.obstacle_map.size)

    def step(
        self, actions: list[int] | tuple[int, ...] | np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], list[float], bool, bool, dict]:
        agents = self._env.possible_agents
        actions_dict = {agents[i]: int(actions[i]) for i in range(self.n_agents)}

        obs_d, rewards_d, terminated_d, truncated_d, infos_d = self._env.step(actions_dict)

        obs_tuple = tuple(self._flatten_obs(obs_d[a]) for a in agents)
        rewards = [rewards_d[a] for a in agents]

        just_agreed = (
            self._agreed_poi is None and self._env.agreed_poi is not None
        )
        if just_agreed:
            self._agreed_poi = self._env.agreed_poi
            sa = self._poi_scores[agents[0]][self._agreed_poi]
            sb = self._poi_scores[agents[1]][self._agreed_poi]
            gm = golden_mean_reward(float(sa), float(sb))
            rewards = [gm * 5.0] * self.n_agents
            for a in agents:
                self._prev_dist[a] = self._bfs_dist_to_target(a)

        if self._agreed_poi is not None and not just_agreed:
            target = self._env.poi_positions[self._agreed_poi]
            reached = False
            for i, a in enumerate(agents):
                pos = tuple(self._env.agent_positions.get(a, [-1, -1]))
                if pos == target:
                    reached = True
                new_dist = self._bfs_dist_to_target(a)
                old_dist = self._prev_dist.get(a, new_dist)
                rewards[i] += (old_dist - new_dist) * 0.05
                self._prev_dist[a] = new_dist

            if reached:
                sa = self._poi_scores[agents[0]][self._agreed_poi]
                sb = self._poi_scores[agents[1]][self._agreed_poi]
                gm = golden_mean_reward(float(sa), float(sb))
                rewards = [gm * 10.0] * self.n_agents

        done = all(terminated_d[a] for a in agents)
        truncated = all(truncated_d[a] for a in agents)

        reached = any("reached_poi" in infos_d.get(a, {}) for a in agents)
        # Logged by EPyMARL as test_negotiation_agreed_mean (fraction of eval
        # episodes where agents agreed on a POI before the episode ended).
        negotiation_agreed = self._env.agreed_poi is not None
        info: dict[str, Any] = {
            "battle_won": reached,
            "reached_poi": int(reached),
            "negotiation_agreed": float(negotiation_agreed),
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
