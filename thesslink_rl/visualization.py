"""Visualization: grid rendering, evaluation heatmaps, training curves, and episode replay.

All scoring and heatmap computation is delegated to ``evaluation.py``.
This module only handles rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    # Same structural API across v0/v1/v2; v2 is representative for static typing only.
    from .v2.environment import GridNegotiationEnv

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .constants import GRID_SIZE, NUM_POIS

ACT_SUGGEST_BASE = 5
NUM_SUGGEST_ACTIONS = NUM_POIS
ACT_ACCEPT = ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS  # 8
NUM_MOVE_ACTIONS = 5
from .evaluation import (
    AgentConfig,
    compute_eval_heatmap,
    compute_poi_scores,
)

COLORS = {
    "empty": "#f0f0f0",
    "obstacle": "#2d2d2d",
    "agent_0": "#f39c12",
    "agent_1": "#9b59b6",
}

POI_RANK_COLORS = ["#2ecc71", "#3498db", "#e74c3c"]  # green (best), blue (mid), red (worst)

AGENT_LABELS = {"agent_0": "A", "agent_1": "B"}


def _heatmap_panel_subtitle(cfg: AgentConfig, agent_label: str) -> str:
    """Eval heatmap title: agent name, privacy emphasis p, energy model (γ if exponential)."""
    p = cfg.privacy_emphasis
    p_str = f"p={p:g}"
    if cfg.energy_model == "exponential":
        eg = f"energy={cfg.energy_model} γ={cfg.energy_exponential_gamma:g}"
    else:
        eg = f"energy={cfg.energy_model}"
    step = float(cfg.energy_step)
    step_s = f"  step={step:g}" if abs(step - 1.0) > 1e-9 else ""
    return f"{cfg.name} (Agent {agent_label})\n{p_str}  {eg}{step_s}"


def _poi_colors(scores: np.ndarray | None) -> list[str]:
    """Return a color per POI index, ranked by score: green=best, blue=mid, red=worst.

    If *scores* is None or all equal, falls back to index order.
    """
    n = len(POI_RANK_COLORS)
    if scores is None or len(scores) == 0:
        return list(POI_RANK_COLORS[:n])
    rank_order = np.argsort(-np.asarray(scores))  # highest score first
    colors = [""] * len(rank_order)
    for rank, poi_idx in enumerate(rank_order):
        colors[poi_idx] = POI_RANK_COLORS[min(rank, n - 1)]
    return colors

OUT_DIR = Path("plots")


def _env_out_dir(env_name: str | None) -> Path:
    """Return ``plots/<env_name>/``, creating it if needed."""
    if not env_name:
        raise ValueError(
            "env_name is required when saving plots (pass ENV_TAG from config, e.g. env_name=ENV_TAG)",
        )
    d = OUT_DIR / env_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_filename(
    plot_name: str, ext: str, algo: str | None = None, env_name: str | None = None,
) -> str:
    """Build ``<plot>-<algo>.<ext>`` (env name is the parent directory now)."""
    parts = [plot_name]
    if algo:
        parts.append(algo)
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
    poi_scores: np.ndarray | None = None,
) -> plt.Axes:
    """Draw the current grid state with obstacles, POIs, and agents.

    *poi_scores* determines POI colour ranking (green=best, red=worst).
    For the centre panel of 3-panel layouts, pass the merged (averaged)
    scores from both agents so the ranking reflects the common view.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    colors = _poi_colors(poi_scores)

    grid_rgb = np.full((GRID_SIZE, GRID_SIZE, 3), mcolors.to_rgb(COLORS["empty"]))

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if env.obstacle_map[r, c]:
                grid_rgb[r, c] = mcolors.to_rgb(COLORS["obstacle"])

    for i, (pr, pc) in enumerate(env.poi_positions):
        grid_rgb[pr, pc] = mcolors.to_rgb(colors[i])

    ax.imshow(grid_rgb, origin="upper", extent=(0, GRID_SIZE, GRID_SIZE, 0))

    for i, (pr, pc) in enumerate(env.poi_positions):
        marker = "^" if env.agreed_poi == i else "D"
        size = 200 if env.agreed_poi == i else 120
        ax.scatter(pc + 0.5, pr + 0.5, marker=marker, s=size,
                   c=colors[i], edgecolors="white", linewidths=1.5, zorder=3)
        score_label = f"P{i}: {poi_scores[i]:.3f}" if poi_scores is not None else f"P{i}"
        ax.text(pc + 0.5, pr + 0.15, score_label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5) if poi_scores is not None else None,
                zorder=4)

    for agent in env.possible_agents:
        if agent not in env.agent_positions:
            continue
        r, c = env.agent_positions[agent]
        color = COLORS[agent]

        if hasattr(env, "spawn_positions") and agent in env.spawn_positions:
            sr, sc = env.spawn_positions[agent]
            if (sr, sc) != (r, c):
                ax.scatter(sc + 0.5, sr + 0.5, marker="*", s=180,
                           c=color, edgecolors="white", linewidths=1, zorder=4)

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
        out = _env_out_dir(env_name)
        ax.figure.savefig(out / save_path, dpi=150, bbox_inches="tight")
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
    colors = _poi_colors(poi_scores)
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
                   c=colors[i], edgecolors="white",
                   linewidths=1.5, zorder=4)
        ax.text(pc + 0.5, pr + 0.15, f"P{i}: {poi_scores[i]:.3f}",
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

    all_scores = {}
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
        all_scores[agent] = scores
        label = AGENT_LABELS.get(agent, agent[-1])
        _draw_heatmap_panel(
            ax, heatmap, env, agent, cfg, spawn,
            poi_scores=scores,
            subtitle=_heatmap_panel_subtitle(cfg, label),
        )

    merged_scores = np.mean(list(all_scores.values()), axis=0)
    render_grid(env, title="Initial State", ax=ax_mid, show=False,
                poi_scores=merged_scores)

    plt.tight_layout()
    fig.suptitle(title or "Agent Evaluation Heatmaps", fontsize=13, y=1.02)

    if save_path is True:
        save_path = _make_filename("eval_heatmaps", "png", algo, env_name)
    if save_path:
        out = _env_out_dir(env_name)
        fig.savefig(out / save_path, dpi=150, bbox_inches="tight")
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
    """Plot test metrics vs timesteps: return, agreement / golden-mean / reach rates, ep length.

    (Policy-gradient / TD *losses* are training objectives logged separately by EPyMARL,
    not environment returns — omit here to focus on eval behaviour.)
    """
    panels = [
        ("common_reward", "Mean test return", "#2ecc71"),
        ("negotiate", "Agreement rate (%)", "#9b59b6"),
        ("negotiate_optimal", "Golden-mean agreement (%)", "#8e44ad"),
        ("reach", "Reach rate (%)", "#3498db"),
        ("ep_len", "Episode length", "#e67e22"),
    ]

    active_panels = [p for p in panels if len(stats.get(p[0], [])) > 0]
    if not active_panels:
        active_panels = panels

    n_panels = len(active_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    x = np.asarray(timesteps, dtype=float) if timesteps is not None else None

    for ax, (key, label, color) in zip(axes, active_panels):
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

    fig.suptitle("Training Progress", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path is True:
        save_path = _make_filename("training_curves", "png", algo, env_name)
    if save_path:
        out = _env_out_dir(env_name)
        fig.savefig(out / save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── 5. Episode replay animation ─────────────────────────────────────────

def replay_episode(
    frames: list[dict],
    env: GridNegotiationEnv,
    agent_configs: Optional[Dict[str, AgentConfig]] = None,
    save_path: str | bool = True,
    neg_interval_ms: int = 1200,
    nav_interval_ms: int = 250,
    show: bool = True,
    algo: str | None = None,
    env_name: str | None = None,
):
    """Animate an episode from a list of frame snapshots.

    Negotiation frames use *neg_interval_ms* (slow) and navigation frames
    use *nav_interval_ms* (fast).  Each frame's action description is
    rendered as a subtitle on the centre panel.

    If *agent_configs* is provided, each frame renders the grid in the
    centre with **dynamic** evaluation heatmaps for each agent on either
    side (3-panel layout).
    """
    from io import BytesIO
    from PIL import Image

    has_heatmaps = agent_configs is not None
    agents = env.possible_agents
    dpi = 100

    if has_heatmaps:
        fig, axes = plt.subplots(1, 3, figsize=(19, 7))
        ax_left, ax_mid, ax_right = axes
        agent_axes = {agents[0]: ax_left, agents[1]: ax_right}
        spawn_positions = frames[0].get(
            "spawn_positions", frames[0]["agent_positions"]
        )
    else:
        fig, ax_mid = plt.subplots(1, 1, figsize=(7, 7))

    pil_frames: list[Image.Image] = []
    durations: list[int] = []
    target_size: tuple[int, int] | None = None

    for idx, frame in enumerate(frames):
        env.agent_positions = frame["agent_positions"]
        env.phase = frame["phase"]
        env.agreed_poi = frame.get("agreed_poi")
        env.last_suggestion = frame.get("last_suggestion", {})
        env.neg_turn = frame.get("neg_turn")
        action_desc = frame.get("action_desc", "")

        if has_heatmaps:
            all_scores = {}
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
                all_scores[agent] = scores
                _draw_heatmap_panel(
                    ax, heatmap, env, agent, cfg, spawn,
                    poi_scores=scores,
                    current_pos=cur_pos,
                )
            merged_scores = np.mean(list(all_scores.values()), axis=0)
        else:
            merged_scores = None

        ax_mid.clear()
        title = f"Step {frame['timestep']}"
        if action_desc:
            title += f"\n{action_desc}"
        render_grid(env, title=title, ax=ax_mid,
                    show=False, poi_scores=merged_scores)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02,
                            wspace=0.15)
        buf = BytesIO()
        fig.savefig(buf, format="raw", dpi=dpi)
        buf.seek(0)
        w_px = int(fig.get_figwidth() * dpi)
        h_px = int(fig.get_figheight() * dpi)
        img = Image.frombuffer("RGBA", (w_px, h_px), buf.read(), "raw", "RGBA", 0, 1)
        img = img.convert("RGB")
        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        pil_frames.append(img)
        buf.close()

        is_neg = frame["phase"] == "negotiation"
        durations.append(neg_interval_ms if is_neg else nav_interval_ms)

    if save_path is True:
        save_path = _make_filename("episode_replay", "gif", algo, env_name)
    if save_path and pil_frames:
        out = _env_out_dir(env_name)
        pil_frames[0].save(
            str(out / save_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,
        )
    if show:
        plt.show()
    plt.close(fig)


MOVE_NAMES = ["stay", "up", "down", "left", "right"]


def describe_actions(
    env: GridNegotiationEnv,
    actions: Dict[str, int],
) -> str:
    """Build a human-readable one-liner for the actions about to be taken.

    Call this **before** ``env.step(actions)`` so that phase / neg_turn
    still reflect the pre-step state.
    """
    phase = env.phase
    neg_turn = getattr(env, "neg_turn", None)
    parts: list[str] = []
    for agent, act in actions.items():
        label = AGENT_LABELS.get(agent, agent[-1])
        if phase == "negotiation":
            if agent != neg_turn:
                continue
            if act == ACT_ACCEPT:
                peer = [a for a in env.possible_agents if a != agent][0]
                poi = env.last_suggestion.get(peer)
                parts.append(f"{label} accepted P{poi}" if poi is not None else f"{label} accepted")
            elif ACT_SUGGEST_BASE <= act < ACT_SUGGEST_BASE + NUM_SUGGEST_ACTIONS:
                poi_idx = act - ACT_SUGGEST_BASE
                parts.append(f"{label} suggested P{poi_idx}")
        else:
            if 0 <= act < NUM_MOVE_ACTIONS:
                parts.append(f"{label}:{MOVE_NAMES[act]}")
    return "  |  ".join(parts)


def capture_frame(
    env: GridNegotiationEnv,
    action_desc: str = "",
) -> dict:
    """Snapshot the env state for replay_episode.

    *action_desc* is a pre-built string from ``describe_actions`` describing
    what happened this step.  Pass an empty string for the initial frame.
    """
    return {
        "agent_positions": {a: list(pos) for a, pos in env.agent_positions.items()},
        "spawn_positions": {a: list(pos) for a, pos in env.spawn_positions.items()},
        "phase": env.phase,
        "timestep": env.timestep,
        "agreed_poi": getattr(env, "agreed_poi", None),
        "last_suggestion": dict(getattr(env, "last_suggestion", {})),
        "neg_turn": getattr(env, "neg_turn", None),
        "action_desc": action_desc,
    }
