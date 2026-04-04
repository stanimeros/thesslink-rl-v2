#!/usr/bin/env python3
"""Smoke test: short QMIX training + generate all three plots.

Usage:
    source .venv/bin/activate
    python smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from config import ENV_CONFIG, ENV_TAG, ENV_VERSION, GridNegotiationEnv
from thesslink_rl.visualization import _make_filename

PROJECT = Path(__file__).resolve().parent
EPYMARL_SRC = PROJECT / "epymarl" / "src"
RESULTS_DIR = PROJECT / "epymarl" / "results"
PLOTS_DIR = PROJECT / "plots"

T_MAX = 4_000
TEST_INTERVAL = 1_000
SAVE_MODEL_INTERVAL = 1_000

SACRED_VERSION_MARKER = f"GridNegotiation-v{ENV_VERSION}"


def _latest_sacred_run_for_smoke() -> Path:
    """Sacred run for this smoke test: QMIX + current env version, newest mtime."""
    sacred_base = RESULTS_DIR / "sacred"
    candidates = [
        p for p in sacred_base.rglob("metrics.json")
        if SACRED_VERSION_MARKER in str(p) and "qmix" in p.parts
    ]
    if not candidates:
        candidates = [
            p for p in sacred_base.rglob("metrics.json")
            if SACRED_VERSION_MARKER in str(p)
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No Sacred metrics under {sacred_base} matching {SACRED_VERSION_MARKER!r}",
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def run_training() -> Path:
    """Launch a quick QMIX training and return the Sacred run directory."""
    cmd = [
        sys.executable, str(EPYMARL_SRC / "main.py"),
        "--config=qmix", f"--env-config={ENV_CONFIG}",
        "with",
        f"t_max={T_MAX}",
        f"test_interval={TEST_INTERVAL}",
        f"save_model=True",
        f"save_model_interval={SAVE_MODEL_INTERVAL}",
        f"test_nepisode=8",
    ]
    print(f"\n{'='*60}")
    print("STEP 1: Running short QMIX training")
    print(f"  t_max={T_MAX}  test_interval={TEST_INTERVAL}")
    print(f"  cmd: {' '.join(cmd[-7:])}")
    print(f"{'='*60}\n")

    proc = subprocess.run(
        cmd, cwd=str(EPYMARL_SRC),
        capture_output=False,
    )
    if proc.returncode != 0:
        print(f"Training failed with exit code {proc.returncode}")
        sys.exit(1)

    try:
        run_dir = _latest_sacred_run_for_smoke()
    except FileNotFoundError as e:
        print(f"No Sacred results found: {e}")
        sys.exit(1)
    print(f"\nSacred results at: {run_dir}")
    return run_dir


def load_sacred_metrics(run_dir: Path) -> dict:
    """Parse Sacred's metrics.json into {metric_name: {steps, values}}."""
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file) as f:
        raw = json.load(f)
    return raw


def print_results_table(metrics: dict):
    """Print a summary table of training results."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Results Summary")
    print(f"{'='*60}")

    keys_of_interest = [
        "test_return_mean", "test_return_std",
        "test_negotiation_agreed_mean",
        "test_battle_won_mean", "test_ep_length_mean",
        "test_reached_poi_mean",
    ]

    header = f"{'Metric':<30} {'Last Value':>12} {'Steps':>8}"
    print(header)
    print("-" * len(header))

    for key in keys_of_interest:
        if key in metrics:
            values = metrics[key]["values"]
            steps = metrics[key]["steps"]
            if values:
                print(f"{key:<30} {values[-1]:>12.4f} {steps[-1]:>8}")
        else:
            print(f"{key:<30} {'(not found)':>12}")

    all_keys = sorted(metrics.keys())
    other_keys = [k for k in all_keys if k not in keys_of_interest and not k.endswith("_T")]
    if other_keys:
        print(f"\nOther logged metrics: {', '.join(other_keys)}")


def generate_plots(metrics: dict, algo: str = "qmix"):
    """Generate the same 3 plots the project already has."""
    import numpy as np
    from thesslink_rl.evaluation import AgentConfig, compute_poi_scores
    from thesslink_rl.visualization import (
        capture_frame,
        describe_actions,
        plot_training_curves,
        render_eval_heatmaps,
        replay_episode,
    )

    print(f"\n{'='*60}")
    print("STEP 3: Generating Plots")
    print(f"{'='*60}")

    # --- 3a. Training curves from Sacred metrics ---
    steps = metrics.get("test_return_mean", {}).get("steps", [])
    gm_vals = metrics.get("test_return_mean", {}).get("values", [])
    neg_vals = metrics.get("test_negotiation_agreed_mean", {}).get("values", [])
    reached_vals = metrics.get("test_battle_won_mean", {}).get("values", [])
    epl_vals = metrics.get("test_ep_length_mean", {}).get("values", [])

    stats = {
        "common_reward": gm_vals,
        "negotiate": [v * 100.0 for v in neg_vals],
        "reach": [v * 100.0 for v in reached_vals],
        "ep_len": epl_vals,
    }
    fname = _make_filename("training_curves", "png", algo)
    print(f"  [1/3] Training curves...")
    plot_training_curves(
        stats,
        window=min(5, max(1, len(gm_vals))),
        save_path=True,
        show=False,
        algo=algo,
        env_name=ENV_TAG,
        timesteps=steps if steps else None,
    )
    print(f"         -> plots/{ENV_TAG}/{fname}")

    # --- 3b. Evaluation heatmaps ---
    models_dir = PROJECT / "thesslink_rl" / "models"
    cfg_0 = AgentConfig.from_yaml(str(models_dir / "human.yaml"))
    cfg_1 = AgentConfig.from_yaml(str(models_dir / "taxi.yaml"))
    agent_configs = {"agent_0": cfg_0, "agent_1": cfg_1}

    env = GridNegotiationEnv(agent_configs=agent_configs, seed=42)
    env.reset(seed=42)

    agents = env.possible_agents
    for agent in agents:
        spawn = tuple(env.spawn_positions[agent])
        scores = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )
        env.poi_scores[agent] = scores

    fname = _make_filename("eval_heatmaps", "png", algo)
    print(f"  [2/3] Evaluation heatmaps...")
    render_eval_heatmaps(env, agent_configs,
                         save_path=True, show=False, algo=algo,
                         env_name=ENV_TAG)
    print(f"         -> plots/{ENV_TAG}/{fname}")

    # --- 3c. Episode replay GIF (random-action demo, same env/map) ---
    env.reset(seed=42)
    for agent in agents:
        spawn = tuple(env.spawn_positions[agent])
        scores = compute_poi_scores(
            spawn, spawn, env.poi_positions, env.obstacle_map,
            agent_configs[agent],
        )
        env.poi_scores[agent] = scores

    rng = np.random.RandomState(99)
    frames = [capture_frame(env)]

    for step in range(40):
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

    fname = _make_filename("episode_replay", "gif", algo)
    print(f"  [3/3] Episode replay GIF...")
    replay_episode(frames, env, agent_configs=agent_configs,
                   save_path=True, show=False, algo=algo,
                   env_name=ENV_TAG)
    print(f"         -> plots/{ENV_TAG}/{fname}")


def main():
    print("ThessLink RL — Smoke Test")
    print(f"Project: {PROJECT}")
    print(f"Environment version: v{ENV_VERSION} (env-config={ENV_CONFIG})")

    run_dir = run_training()
    metrics = load_sacred_metrics(run_dir)
    print_results_table(metrics)
    generate_plots(metrics)

    algo = "qmix"
    env_plots = PLOTS_DIR / ENV_TAG
    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Plots saved in:   {env_plots}")
    print(f"  - {_make_filename('training_curves', 'png', algo)}")
    print(f"  - {_make_filename('eval_heatmaps', 'png', algo)}")
    print(f"  - {_make_filename('episode_replay', 'gif', algo)}")

    models_root = RESULTS_DIR / "models"
    if models_root.exists():
        th_files = [
            p for p in models_root.rglob("*.th")
            if SACRED_VERSION_MARKER in str(p) and "/qmix_" in str(p).replace("\\", "/")
        ]
        if not th_files:
            th_files = [
                p for p in models_root.rglob("*.th")
                if SACRED_VERSION_MARKER in str(p)
            ]
        if th_files:
            latest_th = max(th_files, key=lambda p: p.stat().st_mtime)
            print(
                f"\nLatest checkpoint (smoke, {SACRED_VERSION_MARKER}): "
                f"{latest_th.parent}",
            )
    print()


if __name__ == "__main__":
    main()
