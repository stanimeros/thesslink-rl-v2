"""Preference scoring: each agent rates every POI based on energy and privacy.

Energy cost is dynamic — computed from the agent's current position to each
POI using BFS (respecting obstacles).  The ``energy_model`` shapes the cost
curve: ``linear`` means each step costs equally; ``exponential`` means cost
per step grows **geometrically**: step ``i`` costs ``gamma**(i-1)`` (first step
cost ``1``), so total for ``d`` steps is ``(gamma**d - 1) / (gamma - 1)`` when
``gamma != 1``.  YAML: ``energy_exponential_gamma`` is ``gamma``.

``energy_step`` multiplies the whole travel cost (e.g. gas vs electric units).

Privacy value is static — BFS distance from spawn to each POI, divided by the
maximum BFS distance reachable from spawn on the map (not min-max across only
the three POIs).

Energy scores are **min-max normalised across the candidate POIs**; privacy is
already in [0, 1].  Combined linearly: ``(1-p) · energy + p · privacy``.
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
    energy_exponential_gamma: float  # γ: geometric ratio between step costs (ignored if linear)
    energy_step: float              # scales total energy cost (e.g. gas vs electric)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            name=d["name"],
            privacy_emphasis=d.get("privacy_emphasis", 0.5),
            energy_model=d.get("energy_model", "linear"),
            energy_exponential_gamma=d.get("energy_exponential_gamma", 2.0),
            energy_step=float(d.get("energy_step", 1.0)),
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

    ``linear``:      cost = ``energy_step * dist``
    ``exponential``: step ``i`` costs ``γ^{i-1}`` (first step = 1); total for
    ``d`` steps is ``energy_step * (γ^d - 1) / (γ - 1)`` for ``γ != 1``, else
    ``energy_step * d``.
    """
    scale = float(cfg.energy_step)
    if scale <= 0:
        scale = 1.0
    if not np.isfinite(dist):
        return 1e9
    if dist <= 0:
        return 0.0
    d = int(round(dist))
    if d <= 0:
        return 0.0
    if cfg.energy_model == "exponential":
        gamma = float(cfg.energy_exponential_gamma)
        if gamma <= 0:
            return scale * float(d)
        if abs(gamma - 1.0) < 1e-12:
            return scale * float(d)
        return scale * float((gamma**d - 1.0) / (gamma - 1.0))
    return scale * float(dist)


def _minmax(arr: np.ndarray, *, equal_fill: float = 0.5) -> np.ndarray:
    """Min-max normalise *arr* to [0, 1].

    When all values are equal (zero range), returns ``equal_fill`` for each
    entry.  For **cost** (lower is better), use ``equal_fill=0`` before flipping.
    For **distance** (higher is better for privacy), use ``equal_fill=1`` so
    ties mean “all equally good,” not 0.5 which collapses the blend to 0.5.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, equal_fill, dtype=np.float64)
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
    2. Privacy value — BFS distance from *spawn* to each POI, scaled by the
       maximum BFS distance from *spawn* to any reachable cell (absolute scale,
       not min-max across only the three POIs).  Nearby meeting spots keep
       moderate privacy; only very far POIs approach 1.
    3. Energy — raw travel cost is **min-max normalised across the POIs**
       (cheapest → 1, most expensive → 0).  Exponential step costs make long
       paths expensive before that step.
    4. Final score = ``(1-p) · energy_norm + p · privacy_value`` (linear blend).
    """
    assert len(poi_positions) == NUM_POIS

    if cfg is None:
        cfg = AgentConfig(
            name="default", privacy_emphasis=0.0,
            energy_model="linear",
            energy_exponential_gamma=2.0,
            energy_step=1.0,
        )

    bfs_cur = bfs_distances(current_pos, obstacle_map)
    bfs_spn = bfs_distances(spawn, obstacle_map)

    finite_spawn = bfs_spn[np.isfinite(bfs_spn)]
    max_spawn_dist = float(finite_spawn.max()) if len(finite_spawn) > 0 else 1.0
    if max_spawn_dist < 1e-12:
        max_spawn_dist = 1.0

    raw_energy = np.zeros(NUM_POIS, dtype=np.float64)
    privacy_val = np.zeros(NUM_POIS, dtype=np.float64)

    for i, (pr, pc) in enumerate(poi_positions):
        d_cur = bfs_cur[pr, pc]
        d_spn = bfs_spn[pr, pc]
        raw_energy[i] = _energy_cost(d_cur, cfg) if np.isfinite(d_cur) else 1e9
        if np.isfinite(d_spn) and d_spn >= 0:
            privacy_val[i] = min(1.0, float(d_spn) / max_spawn_dist)
        else:
            privacy_val[i] = 0.0

    # equal_fill: ties → minmax raw cost = 0 → energy_norm = 1 (all equally cheap)
    energy_norm = 1.0 - _minmax(raw_energy, equal_fill=0.0)

    p = float(cfg.privacy_emphasis)
    scores = (1.0 - p) * energy_norm + p * privacy_val
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

    # Pin each POI cell to *that* POI's relative score.  Otherwise max_i
    # (score_i * falloff_i) lets a nearby high-scoring POI "paint" another POI's
    # cell bright green even when that POI has low preference (matches labels).
    for k, (pr, pc) in enumerate(poi_positions):
        if not obstacle_map[pr, pc]:
            heatmap[pr, pc] = float(poi_scores[k] / best_poi_score)

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
