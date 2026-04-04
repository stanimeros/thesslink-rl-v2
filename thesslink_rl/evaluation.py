"""Preference scoring: each agent rates every POI based on energy and privacy.

Energy cost is dynamic — computed from the agent's current position to each
POI using BFS (respecting obstacles).  The ``energy_model`` shapes the cost
curve: ``linear`` means each step costs equally; ``exponential`` means cost
per step grows, so distant POIs are disproportionately expensive.

Privacy score is static — computed from the agent's spawn location to each
POI using BFS.  A POI far from spawn is high privacy because an observer
cannot easily infer the agent's origin.

Both components are **min-max normalised across the candidate POIs** so they
occupy the same [0, 1] scale.  This ensures ``privacy_emphasis`` blends them
fairly regardless of energy model or grid layout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .constants import GRID_SIZE, NUM_POIS


@dataclass
class AgentConfig:
    """Parsed agent model from a YAML file."""
    name: str
    privacy_emphasis: float         # 0-1, higher = prefers POIs far from spawn
    energy_model: str               # "linear" or "exponential"
    energy_exponential_gamma: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            privacy_emphasis=d.get("privacy_emphasis", 0.5),
            energy_model=d.get("energy_model", "linear"),
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


def _energy_cost(dist: float, cfg: AgentConfig) -> float:
    """Total energy to travel *dist* steps under the agent's energy model.

    ``linear``:      cost = dist
    ``exponential``:  cost = (1 - e^{-γ·dist}) / γ   (saturates for large dist)
    """
    if cfg.energy_model == "exponential":
        gamma = cfg.energy_exponential_gamma
        return float((1.0 - np.exp(-gamma * dist)) / gamma)
    return float(dist)


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise *arr* to [0, 1].  Returns zeros if range is 0."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def compute_poi_scores(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig | None = None,
) -> np.ndarray:
    """Return an array of shape (NUM_POIS,) with scores in [0, 1].

    1. Raw energy cost — BFS from *current_pos* to each POI, shaped by
       ``cfg.energy_model``.  Lower cost → higher energy score.
    2. Raw privacy distance — BFS from *spawn* to each POI.
       Greater distance → higher privacy score.
    3. Each component is **min-max normalised across the POIs** so the
       cheapest POI gets energy = 1 and the most expensive gets 0;
       the farthest-from-spawn POI gets privacy = 1 and the closest
       gets 0.
    4. Final score = (1 - p) · energy_norm + p · privacy_norm.
    """
    assert len(poi_positions) == NUM_POIS

    if cfg is None:
        cfg = AgentConfig(
            name="default", privacy_emphasis=0.0,
            energy_model="linear",
            energy_exponential_gamma=0.12,
        )

    bfs_cur = bfs_distances(current_pos, obstacle_map)
    bfs_spn = bfs_distances(spawn, obstacle_map)

    raw_energy = np.zeros(NUM_POIS, dtype=np.float64)
    raw_privacy = np.zeros(NUM_POIS, dtype=np.float64)

    for i, (pr, pc) in enumerate(poi_positions):
        d_cur = bfs_cur[pr, pc]
        d_spn = bfs_spn[pr, pc]
        raw_energy[i] = _energy_cost(d_cur, cfg) if np.isfinite(d_cur) else 1e9
        raw_privacy[i] = float(d_spn) if np.isfinite(d_spn) else 0.0

    energy_norm = 1.0 - _minmax(raw_energy)   # low cost → 1
    privacy_norm = _minmax(raw_privacy)        # far from spawn → 1

    p = cfg.privacy_emphasis
    scores = (1.0 - p) * energy_norm + p * privacy_norm
    return scores.astype(np.float32)


def compute_eval_heatmap(
    current_pos: tuple[int, int],
    spawn: tuple[int, int],
    poi_positions: list[tuple[int, int]],
    obstacle_map: np.ndarray,
    cfg: AgentConfig,
) -> np.ndarray:
    """Compute a 2-D evaluation heatmap for visualization.

    Each cell's value reflects how desirable it is as a destination,
    based on proximity to the POIs weighted by their scores.

    The best-scored POI cell = 1.0; values fall off with BFS distance
    to each POI, normalised per-POI so every POI's influence fades at
    the same rate.  Obstacles = 0.
    """
    poi_scores = compute_poi_scores(
        current_pos, spawn, poi_positions, obstacle_map, cfg,
    )

    poi_bfs: list[np.ndarray] = []
    poi_max_bfs: list[float] = []
    for poi in poi_positions:
        b = bfs_distances(poi, obstacle_map)
        poi_bfs.append(b)
        finite = b[np.isfinite(b)]
        poi_max_bfs.append(float(finite.max()) if len(finite) > 0 else 1.0)

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
                max_d = poi_max_bfs[i] if poi_max_bfs[i] > 0 else 1.0
                falloff = max(1.0 - d_to_poi / max_d, 0.0)
                val = (poi_scores[i] / best_poi_score) * falloff
                best_val = max(best_val, val)
            heatmap[r, c] = best_val

    return np.clip(heatmap, 0.0, 1.0)


def golden_mean_vector(
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> np.ndarray:
    """Golden mean for every POI — shape (NUM_POIS,)."""
    gm = np.ones(NUM_POIS, dtype=np.float64)
    for a in agents:
        gm *= scores[a].astype(np.float64)
    return gm


def optimal_poi(
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> int:
    """Return the POI index that maximises the golden-mean (product) of
    both agents' scores — i.e. the best common choice."""
    return int(np.argmax(golden_mean_vector(scores, agents)))


def negotiation_quality(
    poi_idx: int,
    scores: dict[str, np.ndarray],
    agents: list[str],
) -> float:
    """How good is the chosen POI relative to the best possible?

    Returns a value in [0, 1]:  1.0 = optimal choice, 0.0 = worst.
    The ratio ``gm_chosen / gm_best`` directly measures decision quality
    so the reward scales smoothly — a near-optimal pick is barely penalised,
    a terrible pick gets almost nothing.
    """
    gm = golden_mean_vector(scores, agents)
    best = float(gm.max())
    if best < 1e-12:
        return 0.0
    return float(np.clip(gm[poi_idx] / best, 0.0, 1.0))
