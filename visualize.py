#!/usr/bin/env python3
"""Visualize training results from Sacred metrics.

Reads epymarl/results/sacred/ and generates:
  1. Multi-algorithm comparison training curves
  2. Per-algorithm eval heatmaps (same seed/scenario)
  3. Per-algorithm episode replay GIFs (same seed/scenario, random actions)

Usage:
    python visualize.py                            # all algos
    python visualize.py --algo qmix mappo          # specific algos
    python visualize.py --results epymarl/results  # custom path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesslink_rl.visualization import rolling_mean_expanding

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT / "plots"
SEED = 43

ALGO_COLORS = {
    "iql": "#e74c3c",
    "qmix": "#3498db",
    "vdn": "#2ecc71",
    "mappo": "#f39c12",
    "coma": "#9b59b6",
}


def discover_runs(results_dir: Path) -> dict[str, dict]:
    """Find all Sacred runs and parse their metrics."""
    sacred_dir = results_dir / "sacred"
    if not sacred_dir.exists():
        print(f"No Sacred results at {sacred_dir}")
        return {}

    runs = {}
    for alg_dir in sorted(sacred_dir.iterdir()):
        if not alg_dir.is_dir():
            continue
        metrics_files = list(alg_dir.rglob("metrics.json"))
        if not metrics_files:
            continue
        with open(metrics_files[-1]) as f:
            metrics = json.load(f)
        runs[alg_dir.name] = metrics
    return runs


def plot_comparison_curves(
    runs: dict[str, dict],
    window: int = 10,
    save_path: str | None = None,
):
    """Plot all algorithms on the same figure for comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Order: mean common reward, negotiate rate, reach rate, episode length.
    panels = [
        ("test_return_mean", "Mean common reward", False),
        ("test_negotiation_agreed_mean", "Negotiate rate (%)", True),
        ("test_battle_won_mean", "Reach rate (%)", True),
        ("test_ep_length_mean", "Episode length", False),
    ]

    for ax, (metric_key, label, as_percent) in zip(axes, panels):
        for algo, metrics in runs.items():
            if metric_key not in metrics:
                continue
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
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Algorithm Comparison -- ThessLink Grid Negotiation", fontsize=14, y=1.02)
    plt.tight_layout()

    PLOTS_DIR.mkdir(exist_ok=True)
    out = save_path or "training_curves-all-grid_negotiation.png"
    fig.savefig(PLOTS_DIR / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plots/{out}")


def plot_per_algo_curves(runs: dict[str, dict], window: int = 10):
    """Plot individual training curves per algorithm."""
    from thesslink_rl.visualization import _make_filename

    for algo, metrics in runs.items():
        steps = metrics.get("test_return_mean", {}).get("steps", [])
        gm = metrics.get("test_return_mean", {}).get("values", [])
        neg = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
        reached = metrics.get("test_battle_won_mean", {}).get("values", [])
        epl = metrics.get("test_ep_length_mean", {}).get("values", [])

        stats = {
            "common_reward": gm,
            "negotiate": [v * 100.0 for v in neg],
            "reach": [v * 100.0 for v in reached],
            "ep_len": epl,
        }

        from thesslink_rl.visualization import plot_training_curves
        w = min(window, max(1, len(gm)))
        plot_training_curves(
            stats, window=w,
            save_path=True, show=False, algo=algo,
            timesteps=steps if steps else None,
        )
        fname = _make_filename("training_curves", "png", algo)
        print(f"  -> plots/{fname}")


def generate_heatmaps_and_replays(algos: list[str]):
    """Generate eval heatmaps and episode replay GIFs on the same seed."""
    from thesslink_rl.environment import GridNegotiationEnv
    from thesslink_rl.evaluation import AgentConfig, compute_poi_scores
    from thesslink_rl.visualization import (
        _make_filename,
        capture_frame,
        render_eval_heatmaps,
        replay_episode,
    )

    models_dir = PROJECT / "thesslink_rl" / "models"
    cfg_0 = AgentConfig.from_yaml(str(models_dir / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(models_dir / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

    env = GridNegotiationEnv(agent_configs=agent_configs, seed=SEED)

    for algo in algos:
        env.reset(seed=SEED)
        agents = env.possible_agents
        for agent in agents:
            spawn = tuple(env.spawn_positions[agent])
            scores = compute_poi_scores(
                spawn, spawn, env.poi_positions, env.obstacle_map,
                agent_configs[agent],
            )
            env.poi_scores[agent] = scores

        fname = _make_filename("eval_heatmaps", "png", algo)
        render_eval_heatmaps(env, agent_configs,
                             save_path=True, show=False, algo=algo)
        print(f"  -> plots/{fname}")

        env.reset(seed=SEED)
        for agent in agents:
            spawn = tuple(env.spawn_positions[agent])
            scores = compute_poi_scores(
                spawn, spawn, env.poi_positions, env.obstacle_map,
                agent_configs[agent],
            )
            env.poi_scores[agent] = scores

        rng = np.random.RandomState(99)
        frames = [capture_frame(env)]

        for _ in range(40):
            if not env.agents:
                break
            actions = {}
            for agent in env.agents:
                avail = env.get_avail_actions(agent)
                valid = [i for i, a in enumerate(avail) if a == 1]
                actions[agent] = rng.choice(valid)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            frames.append(capture_frame(env))
            if all(terminated.values()) or all(truncated.values()):
                break

        fname = _make_filename("episode_replay", "gif", algo)
        replay_episode(frames, env, agent_configs=agent_configs,
                       save_path=True, show=False, algo=algo)
        print(f"  -> plots/{fname}")


def print_summary(runs: dict[str, dict]):
    """Print a results table."""
    print()
    header = (
        f"  {'ALG':<7} {'T_ENV':>8} {'RETURN':>8} {'NEG%':>7} "
        f"{'REACH%':>8} {'EP_LEN':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for algo, metrics in sorted(runs.items()):
        ret = metrics.get("test_return_mean", {}).get("values", [])
        neg = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
        bw = metrics.get("test_battle_won_mean", {}).get("values", [])
        epl = metrics.get("test_ep_length_mean", {}).get("values", [])
        steps = metrics.get("test_return_mean", {}).get("steps", [])
        neg_s = f"{(neg[-1] * 100):>6.1f}%" if neg else "   —  "
        print(
            f"  {algo.upper():<7} {steps[-1] if steps else 0:>8} "
            f"{ret[-1] if ret else 0:>8.4f} "
            f"{neg_s:>7} "
            f"{(bw[-1] * 100) if bw else 0:>7.1f}% "
            f"{epl[-1] if epl else 0:>8.1f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize ThessLink RL training results")
    parser.add_argument("--algo", nargs="+", default=None,
                        help="Algorithms to visualize (default: all found)")
    parser.add_argument("--results", type=str, default="epymarl/results",
                        help="Path to results directory")
    parser.add_argument("--window", type=int, default=10,
                        help="Smoothing window for curves")
    args = parser.parse_args()

    results_dir = Path(args.results)
    runs = discover_runs(results_dir)

    if not runs:
        print("No training results found. Generating random episode replay & heatmaps...")
        algos = args.algo or ["random"]
        generate_heatmaps_and_replays(algos)
        print("Done! Plots saved to plots/")
        return

    if args.algo:
        runs = {k: v for k, v in runs.items() if k in args.algo}
        if not runs:
            print(f"No results for: {args.algo}. Generating random episode replay & heatmaps...")
            generate_heatmaps_and_replays(args.algo)
            print("Done! Plots saved to plots/")
            return

    algos = list(runs.keys())
    print(f"Found {len(algos)} algorithm(s): {', '.join(a.upper() for a in algos)}")

    print_summary(runs)

    print("[1/3] Comparison training curves...")
    plot_comparison_curves(runs, window=args.window)

    print("[2/3] Per-algorithm training curves...")
    plot_per_algo_curves(runs, window=args.window)

    print("[3/3] Eval heatmaps & episode replays...")
    generate_heatmaps_and_replays(algos)

    print("Done! All plots saved to plots/")


if __name__ == "__main__":
    main()
