"""Preference scoring: each agent rates every POI based on energy and privacy.

Energy score is dynamic -- computed from the agent's current position to each
POI using BFS (respecting obstacles).

Privacy score is static -- computed from the agent's spawn location to each
POI using BFS.  A POI far from spawn is high privacy because an observer
cannot easily infer the agent's origin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .environment import GRID_SIZE, NUM_POIS


@dataclass
class AgentConfig:
    """Parsed agent model from a YAML file."""
    name: str
    privacy_emphasis: float         # 0-1, higher = prefers POIs far from spawn
    energy_model: str               # "linear" or "exponential"
    energy_per_step: float
    energy_exponential_gamma: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            privacy_emphasis=d.get("privacy_emphasis", 0.5),
            energy_model=d.get("energy_model", "linear"),
            energy_per_step=d.get("energy_per_step", 1.0),
            energy_exponential_gamma=d.get("energy_exponential_gamma", 0.12),
        )


def bfs_distances(
    origin: tuple[int, int],
    obstacle_map: np.ndarray,
) -> np.ndarray:
    """BFS shortest-path distance from *origin* to every reachable cell.

    Returns a float grid where unreachable / obstacle cells are ``np.inf``.
    """
    dist = np.full((GRID_SIZE, GRID_SIZE), np.inf, dtype=np.float64)
    dist[origin[0], origin[1]] = 0.0
    queue: deque[tuple[int, int]] = deque([origin])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                if not obstacle_map[nr, nc] and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
    return dist


def _energy_cost(dist: int, cfg: AgentConfig) -> float:
    """Total energy to travel *dist* steps under the agent's energy model."""
    if cfg.energy_model == "exponential":
        gamma = cfg.energy_exponential_gamma
        return cfg.energy_per_step * (1 - np.exp(-gamma * dist)) / gamma
    return cfg.energy_per_step * dist


def _energy_score(
    current_pos: tuple[int, int],
    poi: tuple[int, int],
    obstacle_map: np.ndarray,
    cfg: AgentConfig,
) -> float:
    """How cheap it is to reach the POI from the current position (BFS).

    Normalised against the maximum possible BFS cost on the grid so
    the result is in [0, 1]: 1 = very cheap, 0 = maximally expensive.
    """
    bfs = bfs_distances(current_pos, obstacle_map)
    dist = bfs[poi[0], poi[1]]
    if np.isinf(dist):
        return 0.0
    max_dist = float(np.max(bfs[np.isfinite(bfs)]))
    if max_dist == 0:
        return 1.0
    cost = _energy_cost(int(dist), cfg)
    max_cost = _energy_cost(int(max_dist), cfg)
    if max_cost == 0:
        return 1.0
    return float(np.clip(1.0 - cost / max_cost, 0.0, 1.0))


def _privacy_score(
    spawn: tuple[int, int],
    poi: tuple[int, int],
    obstacle_map: np.ndarray,
) -> float:
    """How well visiting this POI conceals the agent's spawn location (BFS).

    A POI close to spawn is low privacy -- an observer can narrow down
    the agent's origin.  A distant POI is high privacy.

    Returns 0.0 (POI at spawn) to 1.0 (maximally far).
    """
    bfs = bfs_distances(spawn, obstacle_map)
    dist = bfs[poi[0], poi[1]]
    if np.isinf(dist):
        return 0.0
    max_dist = float(np.max(bfs[np.isfinite(bfs)]))
    if max_dist == 0:
        return 0.0
    return float(dist / max_dist)


def compute_poi_scores(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig | None = None,
) -> np.ndarray:
    """Return an array of shape (NUM_POIS,) with scores in [0, 1].

    Energy is dynamic (BFS from *current_pos*), privacy is static (BFS
    from *spawn*).

    Formula:
        score = (1 - p) * energy + p * privacy
    where p = cfg.privacy_emphasis.
    """
    assert len(poi_positions) == NUM_POIS

    if cfg is None:
        cfg = AgentConfig(
            name="default", privacy_emphasis=0.0,
            energy_model="linear", energy_per_step=1.0,
            energy_exponential_gamma=0.12,
        )

    p = cfg.privacy_emphasis
    scores = np.zeros(NUM_POIS, dtype=np.float32)
    for i, poi in enumerate(poi_positions):
        e = _energy_score(current_pos, poi, obstacle_map, cfg)
        priv = _privacy_score(spawn, poi, obstacle_map)
        scores[i] = (1.0 - p) * e + p * priv
    return np.clip(scores, 0.0, 1.0)


def compute_eval_heatmap(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig,
) -> np.ndarray:
    """Compute a 2-D evaluation heatmap for visualization.

    Each cell's value reflects how desirable it is, based on proximity
    to the POIs weighted by their scores.  Energy is dynamic (BFS from
    *current_pos*), privacy is static (BFS from *spawn*).

    The best POI cell = 1.0; values fall off with BFS distance to each
    POI.  Obstacles = 0.
    """
    poi_scores = compute_poi_scores(
        current_pos, spawn, poi_positions, obstacle_map, cfg,
    )

    bfs_cur = bfs_distances(current_pos, obstacle_map)
    max_bfs = float(np.max(bfs_cur[np.isfinite(bfs_cur)]))
    if max_bfs == 0:
        max_bfs = 1.0

    poi_bfs: list[np.ndarray] = []
    for poi in poi_positions:
        poi_bfs.append(bfs_distances(poi, obstacle_map))

    best_poi_score = float(poi_scores.max())
    if best_poi_score == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obstacle_map[r, c]:
                continue
            best_val = 0.0
            for i, poi in enumerate(poi_positions):
                d_to_poi = poi_bfs[i][r, c]
                if np.isinf(d_to_poi):
                    continue
                falloff = max(1.0 - d_to_poi / max_bfs, 0.0)
                val = (poi_scores[i] / best_poi_score) * falloff
                best_val = max(best_val, val)
            heatmap[r, c] = best_val

    return np.clip(heatmap, 0.0, 1.0)


def golden_mean_reward(score_a: float, score_b: float) -> float:
    """Golden Mean reward: product of both agents' scores for the reached POI."""
    return score_a * score_b
