"""Visualization: grid rendering, evaluation heatmaps, training curves, and episode replay.

All scoring and heatmap computation is delegated to ``evaluation.py``.
This module only handles rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .environment import GRID_SIZE, GridNegotiationEnv
from .evaluation import (
    AgentConfig,
    compute_eval_heatmap,
    compute_poi_scores,
)

COLORS = {
    "empty": "#f0f0f0",
    "obstacle": "#2d2d2d",
    "poi_0": "#2ecc71",   # green  (best)
    "poi_1": "#3498db",   # blue   (mid)
    "poi_2": "#e74c3c",   # red    (worst)
    "agent_0": "#f39c12",
    "agent_1": "#9b59b6",
}

AGENT_LABELS = {"agent_0": "A", "agent_1": "B"}

OUT_DIR = Path("plots")

ENV_TAG = "grid_negotiation"


def _ensure_out_dir():
    OUT_DIR.mkdir(exist_ok=True)


def _make_filename(
    plot_name: str, ext: str, algo: str | None = None, env_name: str | None = None,
) -> str:
    """Build ``<plot>-<algo>-<env>.<ext>`` following EPyMARL naming style."""
    parts = [plot_name]
    if algo:
        parts.append(algo)
    parts.append(env_name or ENV_TAG)
    return "-".join(parts) + f".{ext}"


# ── 1. Static grid snapshot ─────────────────────────────────────────────

def render_grid(
    env: GridNegotiationEnv,
    title: str = "",
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    algo: str | None = None,
    env_name: str | None = None,
) -> plt.Axes:
    """Draw the current grid state with obstacles, POIs, and agents."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    grid_rgb = np.full((GRID_SIZE, GRID_SIZE, 3), mcolors.to_rgb(COLORS["empty"]))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if env.obstacle_map[r, c]:
                grid_rgb[r, c] = mcolors.to_rgb(COLORS["obstacle"])

    poi_keys = ["poi_0", "poi_1", "poi_2"]
    for i, (pr, pc) in enumerate(env.poi_positions):
        grid_rgb[pr, pc] = mcolors.to_rgb(COLORS[poi_keys[i]])

    ax.imshow(grid_rgb, origin="upper", extent=(0, GRID_SIZE, GRID_SIZE, 0))

    for i, (pr, pc) in enumerate(env.poi_positions):
        marker = "^" if env.agreed_poi == i else "D"
        size = 200 if env.agreed_poi == i else 120
        ax.scatter(pc + 0.5, pr + 0.5, marker=marker, s=size,
                   c=COLORS[poi_keys[i]], edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(pc + 0.5, pr + 0.15, f"P{i}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    for agent in env.possible_agents:
        if agent not in env.agent_positions:
            continue
        r, c = env.agent_positions[agent]
        color = COLORS[agent]
        ax.scatter(c + 0.5, r + 0.5, marker="o", s=260,
                   c=color, edgecolors="white", linewidths=2, zorder=5)
        ax.text(c + 0.5, r + 0.5, AGENT_LABELS.get(agent, agent[-1]),
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)

    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)

    phase_tag = f"  [{env.phase}]" if hasattr(env, "phase") else ""
    neg_info = ""
    if env.phase == "negotiation":
        parts = []
        for a, poi in env.last_suggestion.items():
            parts.append(f"{AGENT_LABELS.get(a, a[-1])}->P{poi}")
        turn_str = ""
        if hasattr(env, "neg_turn") and env.neg_turn is not None:
            turn_str = f"turn:{AGENT_LABELS.get(env.neg_turn, env.neg_turn[-1])}"
        if parts or turn_str:
            neg_info = f"  ({', '.join(parts + ([turn_str] if turn_str else []))})"

    ax.set_title(title + phase_tag + neg_info, fontsize=11)

    if save_path is True:
        save_path = _make_filename("grid", "png", algo, env_name)
    if save_path:
        _ensure_out_dir()
        ax.figure.savefig(OUT_DIR / save_path, dpi=150, bbox_inches="tight")
    if show and standalone:
        plt.show()
    return ax


# ── 2. Heatmap panel drawing ────────────────────────────────────────────

def _draw_heatmap_panel(
    ax: plt.Axes,
    heatmap: np.ndarray,
    env: GridNegotiationEnv,
    agent: str,
    cfg: AgentConfig,
    spawn: tuple[int, int],
    poi_scores: np.ndarray,
    current_pos: tuple[int, int] | None = None,
    subtitle: str | None = None,
):
    """Draw a single heatmap panel with POI scores, obstacles, and agent marker."""
    poi_keys = ["poi_0", "poi_1", "poi_2"]
    heatmap_masked = np.ma.array(heatmap, mask=env.obstacle_map)

    ax.imshow(
        heatmap_masked, cmap="RdYlGn", vmin=0, vmax=1,
        origin="upper", extent=(0, GRID_SIZE, GRID_SIZE, 0),
        interpolation="nearest",
    )

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if env.obstacle_map[r, c]:
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1,
                    facecolor=COLORS["obstacle"], edgecolor="none", zorder=2,
                ))

    for i, (pr, pc) in enumerate(env.poi_positions):
        marker = "^" if env.agreed_poi == i else "D"
        size = 200 if env.agreed_poi == i else 120
        ax.scatter(pc + 0.5, pr + 0.5, marker=marker, s=size,
                   c=COLORS[poi_keys[i]], edgecolors="white",
                   linewidths=1.5, zorder=4)
        ax.text(pc + 0.5, pr + 0.15, f"P{i}: {poi_scores[i]:.2f}",
                ha="center", va="center", fontsize=6,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5),
                zorder=5)

    show_pos = current_pos if current_pos is not None else spawn
    pr, pc = show_pos
    sr, sc = spawn

    if current_pos is not None and (sr, sc) != (pr, pc):
        ax.scatter(sc + 0.5, sr + 0.5, marker="*", s=180,
                   c=COLORS[agent], edgecolors="white", linewidths=1,
                   zorder=6)

    ax.scatter(pc + 0.5, pr + 0.5, marker="o", s=260,
               c=COLORS[agent], edgecolors="white", linewidths=2, zorder=7)
    ax.text(pc + 0.5, pr + 0.5, AGENT_LABELS.get(agent, agent[-1]),
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=8)

    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(GRID_SIZE, 0)
    ax.set_title(subtitle or f"{cfg.name} eval", fontsize=10)


# ── 3. Static 3-panel heatmap image ────────────────────────────────────

def render_eval_heatmaps(
    env: GridNegotiationEnv,
    agent_configs: Dict[str, AgentConfig],
    title: str = "",
    show: bool = True,
    save_path: str | None = None,
    algo: str | None = None,
    env_name: str | None = None,
) -> plt.Figure:
    """Draw a 3-panel image: agent_0 heatmap | grid | agent_1 heatmap.

    Heatmaps are computed from spawn positions (current_pos = spawn).
    """
    agents = env.possible_agents
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(19, 6))
    agent_axes = {agents[0]: ax_left, agents[1]: ax_right}

    render_grid(env, title="Initial State", ax=ax_mid, show=False)

    for agent in agents:
        ax = agent_axes[agent]
        cfg = agent_configs[agent]
        spawn = tuple(env.spawn_positions[agent])
        heatmap = compute_eval_heatmap(
            spawn, spawn, env.poi_positions, env.obstacle_map, cfg,
        )
        scores = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map, cfg,
        )
        label = AGENT_LABELS.get(agent, agent[-1])
        _draw_heatmap_panel(
            ax, heatmap, env, agent, cfg, spawn,
            poi_scores=scores,
            subtitle=f"{cfg.name} (Agent {label})\np={cfg.privacy_emphasis}  energy={cfg.energy_model}",
        )

    plt.tight_layout()
    fig.suptitle(title or "Agent Evaluation Heatmaps", fontsize=13, y=1.02)

    if save_path is True:
        save_path = _make_filename("eval_heatmaps", "png", algo, env_name)
    if save_path:
        _ensure_out_dir()
        fig.savefig(OUT_DIR / save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── 4. Training curves ──────────────────────────────────────────────────


def rolling_mean_expanding(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over at most *window* points; same length as *values*.

    The first index uses a 1-point mean, then 2, … up to *window* points, so the
    smoothed curve spans the full x-range (unlike ``np.convolve(..., mode="valid")``).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    w = max(1, int(window))
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - w + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i - lo + 1)
    return out


def plot_training_curves(
    stats: Dict[str, list],
    window: int = 20,
    save_path: str | bool = True,
    show: bool = True,
    algo: str | None = None,
    env_name: str | None = None,
    timesteps: list | None = None,
):
    """Plot test metrics vs timesteps: common reward, negotiate/reach rates, ep length.

    (Policy-gradient / TD *losses* are training objectives logged separately by EPyMARL,
    not environment returns — omit here to focus on eval behaviour.)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    panels = [
        ("common_reward", "Mean common reward", "#2ecc71"),
        ("negotiate", "Negotiate rate (%)", "#9b59b6"),
        ("reach", "Reach rate (%)", "#3498db"),
        ("ep_len", "Episode length", "#e67e22"),
    ]

    x = np.asarray(timesteps, dtype=float) if timesteps is not None else None

    for ax, (key, label, color) in zip(axes, panels):
        data = np.array(stats.get(key, []))
        if len(data) == 0:
            ax.set_title(label)
            continue
        if x is not None and len(x) == len(data):
            xv = x
            ax.set_xlabel("Timesteps")
        else:
            xv = np.arange(1, len(data) + 1)
            ax.set_xlabel("Test index")
        ax.plot(xv, data, alpha=0.25, color=color, linewidth=0.8)
        smoothed = rolling_mean_expanding(data, window)
        ax.plot(xv, smoothed, color=color, linewidth=2,
                label=f"{window}-pt avg")
        ax.legend(fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if key in ("negotiate", "reach"):
            ax.set_ylim(0, 105)
        if key == "ep_len":
            ax.set_ylim(0, 105)

    fig.suptitle("Training Progress", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path is True:
        save_path = _make_filename("training_curves", "png", algo, env_name)
    if save_path:
        _ensure_out_dir()
        fig.savefig(OUT_DIR / save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── 5. Episode replay animation ─────────────────────────────────────────

def replay_episode(
    frames: list[dict],
    env: GridNegotiationEnv,
    agent_configs: Optional[Dict[str, AgentConfig]] = None,
    save_path: str | bool = True,
    interval_ms: int = 400,
    show: bool = True,
    algo: str | None = None,
    env_name: str | None = None,
):
    """Animate an episode from a list of frame snapshots.

    If *agent_configs* is provided, each frame renders the grid in the
    centre with **dynamic** evaluation heatmaps for each agent on either
    side (3-panel layout).  Heatmaps are recomputed every frame from the
    agent's current position (energy changes) while privacy stays fixed
    from spawn.
    """
    has_heatmaps = agent_configs is not None
    agents = env.possible_agents

    if has_heatmaps:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        ax_left, ax_mid, ax_right = axes
        agent_axes = {agents[0]: ax_left, agents[1]: ax_right}
        spawn_positions = frames[0].get(
            "spawn_positions", frames[0]["agent_positions"]
        )
    else:
        fig, ax_mid = plt.subplots(1, 1, figsize=(6, 6))

    def _draw(idx):
        frame = frames[idx]
        env.agent_positions = frame["agent_positions"]
        env.phase = frame["phase"]
        env.agreed_poi = frame.get("agreed_poi")
        env.last_suggestion = frame.get("last_suggestion", {})
        env.neg_turn = frame.get("neg_turn")

        ax_mid.clear()
        render_grid(env, title=f"Step {frame['timestep']}", ax=ax_mid, show=False)

        if has_heatmaps:
            for agent in agents:
                ax = agent_axes[agent]
                ax.clear()
                cfg = agent_configs[agent]
                spawn = tuple(spawn_positions[agent])
                cur_pos = tuple(frame["agent_positions"][agent])
                heatmap = compute_eval_heatmap(
                    cur_pos, spawn, env.poi_positions,
                    env.obstacle_map, cfg,
                )
                scores = compute_poi_scores(
                    cur_pos, spawn, env.poi_positions,
                    env.obstacle_map, cfg,
                )
                _draw_heatmap_panel(
                    ax, heatmap, env, agent, cfg, spawn,
                    poi_scores=scores,
                    current_pos=cur_pos,
                )

    anim = FuncAnimation(fig, _draw, frames=len(frames),
                         interval=interval_ms, repeat=False)
    if save_path is True:
        save_path = _make_filename("episode_replay", "gif", algo, env_name)
    if save_path:
        _ensure_out_dir()
        plt.tight_layout()
        anim.save(str(OUT_DIR / save_path), writer="pillow", dpi=100)
    if show:
        plt.show()
    plt.close(fig)


def capture_frame(env: GridNegotiationEnv) -> dict:
    """Snapshot the env state for replay_episode."""
    return {
        "agent_positions": {a: list(pos) for a, pos in env.agent_positions.items()},
        "spawn_positions": {a: list(pos) for a, pos in env.spawn_positions.items()},
        "phase": env.phase,
        "timestep": env.timestep,
        "agreed_poi": getattr(env, "agreed_poi", None),
        "last_suggestion": dict(getattr(env, "last_suggestion", {})),
        "neg_turn": getattr(env, "neg_turn", None),
    }
