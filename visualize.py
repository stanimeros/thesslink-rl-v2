#!/usr/bin/env python3
"""Visualize training results from Sacred metrics.

With **no** Sacred results (or no matching ``--algo``): writes **3** placeholder
files only — ``training_curves-example.png``, ``eval_heatmaps-example.png``,
``episode_replay-example.gif``.

With results: ``training_curves-all.png`` (comparison), **one** shared
``eval_heatmaps.png``, plus per-algorithm training curves and episode GIFs from
the **best** saved checkpoint under ``results/models`` (random rollout only if
missing checkpoint or load fails).

Usage:
    python visualize.py --env 3                    # required unless stdin is a TTY (then prompted)
    python visualize.py --env 2 --algo qmix mappo
    python visualize.py --env 3 --results epymarl/results
    python visualize.py --env 3 --models /path/to/models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from thesslink_rl.checkpoints import (
    describe_models_dir_status,
    find_best_checkpoint_timestep_dir,
    load_epymarl_config_for_algo,
    rollout_episode_frames_for_gif,
    test_reward_series,
)
from thesslink_rl.evaluation import AgentConfig, compute_poi_scores
from thesslink_rl.visualization import (
    _env_out_dir,
    _make_filename,
    capture_frame,
    describe_actions,
    plot_training_curves,
    render_eval_heatmaps,
    replay_episode,
    rolling_mean_expanding,
)

PROJECT = Path(__file__).resolve().parent
SEED = 105
EXAMPLE_TAG = "example"

ALGO_COLORS = {
    "iql": "#e74c3c",
    "qmix": "#3498db",
    "vdn": "#2ecc71",
    "mappo": "#f39c12",
    "coma": "#9b59b6",
}


def discover_runs(results_dir: Path) -> dict[str, dict]:
    """Find Sacred runs matching the active ENV_VERSION and parse their metrics.

    Sacred results live under:
      sacred/<algo>/thesslink_rl:thesslink/GridNegotiation-v<N>/<run_id>/metrics.json

    Only runs whose path contains ``GridNegotiation-v{ENV_VERSION}`` are returned.
    """
    from config import ENV_VERSION

    sacred_dir = results_dir / "sacred"
    if not sacred_dir.exists():
        print(f"No Sacred results at {sacred_dir}")
        return {}

    version_marker = f"GridNegotiation-v{ENV_VERSION}"

    runs = {}
    for alg_dir in sorted(sacred_dir.iterdir()):
        if not alg_dir.is_dir():
            continue
        metrics_files = [
            f for f in alg_dir.rglob("metrics.json")
            if version_marker in str(f)
        ]
        if not metrics_files:
            continue
        metrics_files.sort()
        with open(metrics_files[-1]) as f:
            metrics = json.load(f)
        runs[alg_dir.name] = metrics
    return runs


def _sync_poi_scores(env: Any, agent_configs: dict[str, AgentConfig]) -> None:
    """After ``reset``, fill ``env.poi_scores`` from spawn positions."""
    for agent in env.possible_agents:
        spawn = tuple(env.spawn_positions[agent])
        env.poi_scores[agent] = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )


def plot_comparison_curves(
    runs: dict[str, dict],
    window: int = 10,
):
    """Plot all algorithms on the same figure for comparison."""
    from config import ENV_TAG

    fig, axes = plt.subplots(1, 5, figsize=(27, 5))

    # Reward: common mean, or total Σ-agents mean when common_reward is off
    ax0 = axes[0]
    has_reward = False
    for algo, metrics in runs.items():
        steps, values = test_reward_series(metrics)
        if values.size == 0:
            continue
        has_reward = True
        color = ALGO_COLORS.get(algo, None)
        ax0.plot(steps, values, alpha=0.3, color=color, linewidth=0.8)
        smoothed = rolling_mean_expanding(values, window)
        ax0.plot(steps, smoothed, color=color, linewidth=2, label=algo.upper())
    ax0.set_xlabel("Timesteps")
    ax0.set_title("Mean test return (common or Σ agents)", fontsize=12)
    if has_reward:
        ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    panels = [
        ("test_negotiation_agreed_mean", "Agreement rate (%)", True),
        ("test_negotiation_optimal_mean", "Golden-mean agreement (%)", True),
        ("test_battle_won_mean", "Reach rate (%)", True),
        ("test_ep_length_mean", "Episode length", False),
    ]

    for ax, (metric_key, label, as_percent) in zip(axes[1:], panels):
        has_data = False
        for algo, metrics in runs.items():
            if metric_key not in metrics:
                continue
            has_data = True
            steps = np.array(metrics[metric_key]["steps"])
            values = np.array(metrics[metric_key]["values"])
            if as_percent:
                values = values * 100.0
            color = ALGO_COLORS.get(algo, None)
            ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.8)
            smoothed = rolling_mean_expanding(values, window)
            ax.plot(steps, smoothed, color=color, linewidth=2,
                    label=algo.upper())

        ax.set_xlabel("Timesteps")
        ax.set_title(label, fontsize=12)
        if has_data:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Algorithm Comparison — ThessLink Grid Negotiation (env {ENV_TAG})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    env_dir = _env_out_dir(ENV_TAG)
    out = "training_curves-all.png"
    fig.savefig(env_dir / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plots/{ENV_TAG}/{out}")


def plot_per_algo_curves(runs: dict[str, dict], window: int = 10):
    """Plot individual training curves per algorithm."""
    from config import ENV_TAG

    for algo, metrics in runs.items():
        steps_arr, vals_arr = test_reward_series(metrics)
        steps = steps_arr.tolist() if steps_arr.size else []
        gm = vals_arr.tolist() if vals_arr.size else []
        neg = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
        neg_opt = metrics.get("test_negotiation_optimal_mean", {}).get("values", [])
        reached = metrics.get("test_battle_won_mean", {}).get("values", [])
        epl = metrics.get("test_ep_length_mean", {}).get("values", [])

        stats = {
            "common_reward": gm,
            "negotiate": [v * 100.0 for v in neg],
            "negotiate_optimal": [v * 100.0 for v in neg_opt],
            "reach": [v * 100.0 for v in reached],
            "ep_len": epl,
        }

        w = min(window, max(1, len(gm)))
        plot_training_curves(
            stats, window=w,
            save_path=True, show=False, algo=algo,
            env_name=ENV_TAG,
            timesteps=steps if steps else None,
        )
        fname = _make_filename("training_curves", "png", algo)
        print(f"  -> plots/{ENV_TAG}/{fname}")


def generate_example_plots() -> None:
    """Exactly three files when no training metrics: one demo per plot type."""
    from config import ENV_TAG, GridNegotiationEnv

    models_dir = PROJECT / "thesslink_rl" / "models"
    cfg_0 = AgentConfig.from_yaml(str(models_dir / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(models_dir / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}
    env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED)
    env.reset(seed=SEED)
    _sync_poi_scores(env, agent_configs)
    agents = env.possible_agents

    n = 24
    steps = [i * 50_000 for i in range(1, n + 1)]
    stats = {
        "common_reward": [8.0 + i * 0.35 + (i % 3) * 0.2 for i in range(n)],
        "negotiate": [min(98.0, 15.0 + i * 2.8) for i in range(n)],
        "negotiate_optimal": [min(95.0, 12.0 + i * 2.2) for i in range(n)],
        "reach": [min(99.0, 8.0 + i * 3.2) for i in range(n)],
        "ep_len": [max(12.0, 130.0 - i * 3.5) for i in range(n)],
    }

    print(f"[1/3] Example training curves ({_make_filename('training_curves', 'png', EXAMPLE_TAG)})")
    plot_training_curves(
        stats,
        window=min(10, max(1, n)),
        save_path=True,
        show=False,
        algo=EXAMPLE_TAG,
        env_name=ENV_TAG,
        timesteps=steps,
    )
    print(f"  -> plots/{ENV_TAG}/{_make_filename('training_curves', 'png', EXAMPLE_TAG)}")

    print(f"[2/3] Example eval heatmaps ({_make_filename('eval_heatmaps', 'png', EXAMPLE_TAG)})")
    render_eval_heatmaps(
        env, agent_configs,
        save_path=True, show=False, algo=EXAMPLE_TAG,
        env_name=ENV_TAG,
        title="Example — add epymarl/results to plot real runs",
    )
    print(f"  -> plots/{ENV_TAG}/{_make_filename('eval_heatmaps', 'png', EXAMPLE_TAG)}")

    env.reset(seed=SEED)
    _sync_poi_scores(env, agent_configs)
    frames = _random_episode_frames(env)
    print(f"[3/3] Example episode replay ({_make_filename('episode_replay', 'gif', EXAMPLE_TAG)})")
    replay_episode(
        frames, env, agent_configs=agent_configs,
        save_path=True, show=False, algo=EXAMPLE_TAG,
        env_name=ENV_TAG,
    )
    print(f"  -> plots/{ENV_TAG}/{_make_filename('episode_replay', 'gif', EXAMPLE_TAG)}")


def _random_episode_frames(env: Any, max_steps: int = 40) -> list[dict]:
    """Fallback when no checkpoint or policy rollout fails (fixed RNG for reproducibility)."""
    rng = np.random.RandomState(99)
    frames = [capture_frame(env)]
    for _ in range(max_steps):
        if not env.agents:
            break
        actions = {}
        for agent in env.agents:
            avail = env.get_avail_actions(agent)
            valid = [i for i, a in enumerate(avail) if a == 1]
            actions[agent] = rng.choice(valid)
        desc = describe_actions(env, actions)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        frames.append(capture_frame(env, action_desc=desc))
        if all(terminated.values()) or all(truncated.values()):
            break
    return frames


def generate_heatmaps_and_replays(
    algos: list[str],
    *,
    results_dir: Path | None = None,
    runs: dict[str, dict] | None = None,
    models_root: Path | None = None,
):
    """Generate one eval heatmap PNG and per-algorithm episode replay GIFs on the same seed.

    Episode GIFs are written **only** when a checkpoint can be loaded from
    *models_root* (default: *results_dir*/models). No random-policy GIFs.
    """
    from config import ENV_CONFIG, ENV_TAG, ENV_VERSION, GridNegotiationEnv

    models_dir = PROJECT / "thesslink_rl" / "models"
    cfg_0 = AgentConfig.from_yaml(str(models_dir / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(models_dir / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

    env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED)

    env.reset(seed=SEED)
    _sync_poi_scores(env, agent_configs)

    render_eval_heatmaps(
        env, agent_configs,
        save_path=True, show=False, algo=None,
        env_name=ENV_TAG,
    )
    print(f"  -> plots/{ENV_TAG}/{_make_filename('eval_heatmaps', 'png', None)}")

    for raw_algo in algos:
        algo = raw_algo.lower()
        env.reset(seed=SEED)
        _sync_poi_scores(env, agent_configs)

        frames: list[dict] | None = None
        metrics = runs.get(algo) if runs else None
        if results_dir is not None and metrics is not None:
            ckpt = find_best_checkpoint_timestep_dir(
                algo, results_dir, metrics, ENV_VERSION,
                models_root=models_root,
            )
            if ckpt is not None:
                try:
                    cfg = load_epymarl_config_for_algo(algo, ENV_CONFIG, SEED)
                    frames = rollout_episode_frames_for_gif(
                        ckpt,
                        cfg,
                        SEED,
                        capture_frame,
                        describe_actions,
                    )
                    print(
                        f"  episode replay ({algo}): policy from {ckpt.name} "
                        f"({ckpt.parent.name})",
                    )
                except FileNotFoundError as e:
                    extra = getattr(e, "filename", None)
                    if extra:
                        extra = f" ({extra})"
                    else:
                        extra = ""
                    print(
                        f"  episode replay ({algo}): skipped — missing file{extra}: {e!r}",
                    )
                except Exception as e:
                    print(
                        f"  episode replay ({algo}): skipped — load/rollout failed: {e!r}",
                    )
            else:
                hint = describe_models_dir_status(models_root, results_dir)
                print(f"  episode replay ({algo}): skipped — {hint}")
        elif results_dir is not None:
            print(
                f"  episode replay ({algo}): skipped — no Sacred metrics for this algo",
            )

        if frames is not None:
            fname = _make_filename("episode_replay", "gif", algo)
            replay_episode(frames, env, agent_configs=agent_configs,
                           save_path=True, show=False, algo=algo,
                           env_name=ENV_TAG)
            print(f"  -> plots/{ENV_TAG}/{fname}")


def print_summary(runs: dict[str, dict]):
    """Print a results table."""
    print()
    header = (
        f"  {'ALG':<7} {'T_ENV':>8} {'RETURN':>8} {'AGR%':>7} "
        f"{'GM%':>8} {'REACH%':>8} {'EP_LEN':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for algo, metrics in sorted(runs.items()):
        steps_arr, vals_arr = test_reward_series(metrics)
        ret = vals_arr.tolist() if vals_arr.size else []
        steps = steps_arr.tolist() if steps_arr.size else []
        neg = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
        neg_opt = metrics.get("test_negotiation_optimal_mean", {}).get("values", [])
        bw = metrics.get("test_battle_won_mean", {}).get("values", [])
        epl = metrics.get("test_ep_length_mean", {}).get("values", [])

        def _pct(vals):
            return f"{(vals[-1] * 100):>7.1f}%" if vals else "     —  "

        print(
            f"  {algo.upper():<7} {steps[-1] if steps else 0:>8} "
            f"{ret[-1] if ret else 0:>8.4f} "
            f"{_pct(neg):>7} "
            f"{_pct(neg_opt):>8} "
            f"{_pct(bw):>8} "
            f"{epl[-1] if epl else 0:>8.1f}"
        )
    print()


def _resolve_env_version(cli_env: int | None) -> int:
    if cli_env is not None:
        return cli_env
    if sys.stdin.isatty():
        while True:
            raw = input("ThessLink env version [0-3]: ").strip()
            try:
                v = int(raw)
                if v in (0, 1, 2, 3):
                    return v
            except ValueError:
                pass
            print("  Enter 0, 1, 2, or 3.", file=sys.stderr)
    print(
        "Error: pass --env 0..3 (non-interactive).",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize ThessLink RL training results")
    parser.add_argument(
        "--env",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        metavar="N",
        help="ThessLink env version (0–3). If omitted, prompted when stdin is a TTY.",
    )
    parser.add_argument("--algo", nargs="+", default=None,
                        help="Algorithms to visualize (default: all found)")
    parser.add_argument("--results", type=str, default="epymarl/results",
                        help="Path to results directory")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Checkpoint directory (agent.th trees). Default: <results>/models",
    )
    parser.add_argument("--window", type=int, default=10,
                        help="Smoothing window for curves")
    args = parser.parse_args()

    v = _resolve_env_version(args.env)
    os.environ["THESSLINK_ENV_VERSION"] = str(v)
    from config import ENV_TAG, ENV_VERSION

    print(f"Using environment: {ENV_TAG} (ENV_VERSION={ENV_VERSION})")

    results_dir = Path(args.results)
    runs = discover_runs(results_dir)
    if args.algo:
        wanted = {a.lower() for a in args.algo}
        runs = {k: v for k, v in runs.items() if k in wanted}

    if not runs:
        print(
            "No Sacred metrics for this selection — writing exactly 3 example files "
            f"(*-{EXAMPLE_TAG}.*).",
        )
        generate_example_plots()
        print(f"Done! Plots saved to plots/{ENV_TAG}/")
        return

    algos = sorted(runs.keys())
    print(f"Found {len(algos)} algorithm(s): {', '.join(a.upper() for a in algos)}")
    models_root = Path(args.models) if args.models else None
    mr_display = models_root or (results_dir / "models")
    print(
        "Output: training_curves-all.png, eval_heatmaps.png, "
        "training_curves-<alg>.png per algorithm, "
        f"episode_replay-<alg>.gif per algorithm when checkpoints exist under {mr_display}.\n",
    )

    print_summary(runs)

    print("[1/3] Comparison — training_curves-all.png")
    plot_comparison_curves(runs, window=args.window)

    print("[2/3] Per-algorithm training curves (one PNG per algo)...")
    plot_per_algo_curves(runs, window=args.window)

    print("[3/3] Eval heatmap (one) + per-algorithm episode GIFs (best checkpoint)...")
    generate_heatmaps_and_replays(
        algos,
        results_dir=results_dir,
        runs=runs,
        models_root=models_root,
    )

    print(f"Done! Plots saved to plots/{ENV_TAG}/")


if __name__ == "__main__":
    main()
