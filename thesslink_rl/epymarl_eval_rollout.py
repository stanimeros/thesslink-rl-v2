"""Load EPyMARL checkpoints and roll out one episode for visualization GIFs."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import yaml

from config import GridNegotiationEnv

_LOGGER = logging.getLogger("thesslink.epymarl_rollout")


def _recursive_dict_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _test_reward_series(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same logic as visualize._test_reward_series — test return steps/values."""
    tr = metrics.get("test_total_return_mean", {})
    if tr.get("values"):
        return np.asarray(tr["steps"], dtype=float), np.asarray(tr["values"], dtype=float)
    rm = metrics.get("test_return_mean", {})
    if rm.get("values"):
        return np.asarray(rm["steps"], dtype=float), np.asarray(rm["values"], dtype=float)
    keys = sorted(
        k for k in metrics
        if k.startswith("test_agent_") and k.endswith("_return_mean")
    )
    if not keys:
        return np.array([]), np.array([])
    steps = np.asarray(metrics[keys[0]]["steps"], dtype=float)
    total = np.zeros(len(metrics[keys[0]]["values"]), dtype=float)
    for k in keys:
        m = metrics[k]
        if not m.get("values"):
            continue
        s = np.asarray(m["steps"], dtype=float)
        v = np.asarray(m["values"], dtype=float)
        if s.shape != steps.shape or (s != steps).any() or v.shape[0] != total.shape[0]:
            continue
        total += v
    if total.size == 0:
        return np.array([]), np.array([])
    return steps, total


def best_test_timestep_from_metrics(metrics: dict) -> int | None:
    """Timestep at which test return (or sum of agent returns) was highest."""
    steps, vals = _test_reward_series(metrics)
    if vals.size == 0:
        return None
    idx = int(np.argmax(vals))
    return int(steps[idx])


def _pick_checkpoint_timestep(available: list[int], target_t: int | None) -> int | None:
    if not available:
        return None
    if target_t is None:
        return max(available)
    le = [t for t in available if t <= target_t]
    if le:
        return max(le)
    return min(available, key=lambda t: abs(t - target_t))


def find_best_checkpoint_timestep_dir(
    algo: str,
    results_dir: Path,
    metrics: dict | None,
    env_version: int,
) -> Path | None:
    """Return path ``.../models/<token>/<timestep>`` with best test checkpoint, or None."""
    models_root = results_dir / "models"
    if not models_root.is_dir():
        return None

    version_marker = f"GridNegotiation-v{env_version}"
    candidates: list[tuple[Path, float]] = []
    for p in models_root.iterdir():
        if not p.is_dir():
            continue
        if version_marker not in p.name:
            continue
        if not p.name.startswith(f"{algo}_"):
            continue
        subs = [
            int(x.name)
            for x in p.iterdir()
            if x.is_dir() and x.name.isdigit()
        ]
        if not subs:
            continue
        mtime = p.stat().st_mtime
        candidates.append((p, mtime))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[1])
    run_dir = candidates[0][0]

    timesteps = sorted(
        int(x.name)
        for x in run_dir.iterdir()
        if x.is_dir() and x.name.isdigit()
    )
    if not timesteps:
        return None

    target = best_test_timestep_from_metrics(metrics) if metrics else None
    ts = _pick_checkpoint_timestep(timesteps, target)
    if ts is None:
        return None
    out = run_dir / str(ts)
    return out if out.is_dir() else None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_epymarl_src_on_path() -> Path:
    root = _project_root()
    ep = root / "epymarl" / "src"
    if not ep.is_dir():
        raise FileNotFoundError(f"EPyMARL src not found at {ep}")
    s = str(ep)
    if s not in sys.path:
        sys.path.insert(0, s)
    return ep


def load_epymarl_config_for_algo(
    algo: str,
    env_config_name: str,
    seed: int,
) -> dict:
    """Merge default + env + alg YAML (same order as epymarl main.py)."""
    root = _project_root()
    ep_cfg = root / "epymarl" / "src" / "config"
    env_fallback = root / "epymarl_config" / "envs"

    def read_yaml(path: Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    cfg = read_yaml(ep_cfg / "default.yaml")

    env_path = ep_cfg / "envs" / f"{env_config_name}.yaml"
    if not env_path.exists():
        env_path = env_fallback / f"{env_config_name}.yaml"
    if not env_path.exists():
        raise FileNotFoundError(
            f"No env config {env_config_name!r} under {ep_cfg / 'envs'} or {env_fallback}",
        )
    _recursive_dict_update(cfg, read_yaml(env_path))

    alg_path = ep_cfg / "algs" / f"{algo}.yaml"
    if not alg_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {alg_path}")
    _recursive_dict_update(cfg, read_yaml(alg_path))

    cfg["seed"] = seed
    if algo in ("iql", "mappo"):
        cfg["common_reward"] = False

    cfg["runner"] = "episode"
    cfg["batch_size_run"] = 1
    if cfg.get("test_nepisode", 0) < cfg["batch_size_run"]:
        cfg["test_nepisode"] = cfg["batch_size_run"]

    return cfg


def _unwrap_to_grid_negotiation_env(gymma_wrapper) -> GridNegotiationEnv:
    """Reach ``GridNegotiationEnv`` from EPyMARL's ``GymmaWrapper``."""
    cur = gymma_wrapper._env
    seen: set[int] = set()
    for _ in range(24):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if isinstance(cur, GridNegotiationEnv):
            return cur
        if hasattr(cur, "_env"):
            sub = cur._env
            if isinstance(sub, GridNegotiationEnv):
                return sub
        nxt = getattr(cur, "env", None)
        if nxt is not None:
            cur = nxt
            continue
        uw = getattr(cur, "unwrapped", None)
        if uw is not None and uw is not cur:
            cur = uw
            continue
        break

    raise RuntimeError(
        "Could not unwrap GymmaWrapper to GridNegotiationEnv "
        "(check gymnasium wrapper stack / ENV_VERSION).",
    )


def rollout_episode_frames_for_gif(
    model_timestep_dir: str | Path,
    config: dict,
    seed: int,
    capture_frame,
    describe_actions,
) -> list[dict]:
    """
    Run one greedy/test episode with a loaded checkpoint and build frame dicts
    for ``replay_episode`` (same as random rollout in visualize).
    """
    _ensure_epymarl_src_on_path()

    from types import SimpleNamespace as SN

    from components.episode_buffer import ReplayBuffer
    from components.transforms import OneHot
    from controllers import REGISTRY as mac_REGISTRY
    from learners import REGISTRY as le_REGISTRY
    from runners import REGISTRY as r_REGISTRY
    from run import args_sanity_check
    from utils.logging import Logger

    model_timestep_dir = Path(model_timestep_dir)
    if not model_timestep_dir.is_dir():
        raise FileNotFoundError(model_timestep_dir)

    cfg = deepcopy(config)
    cfg = args_sanity_check(cfg, _LOGGER)
    args = SN(**cfg)
    args.device = "cuda" if args.use_cuda else "cpu"
    args.env_args = dict(args.env_args)
    args.env_args["seed"] = seed

    np.random.seed(seed)
    th.manual_seed(seed)
    if args.use_cuda and th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    logger = Logger(_LOGGER)

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    mp = os.fspath(model_timestep_dir)
    _LOGGER.info("Loading policy from %s", mp)
    learner.load_models(mp)

    grid = _unwrap_to_grid_negotiation_env(runner.env)

    try:
        runner.reset()
        runner.mac.init_hidden(batch_size=runner.batch_size)
        frames: list[dict] = [capture_frame(grid)]
        terminated = False

        while not terminated:
            pre_transition_data = {
                "state": [runner.env.get_state()],
                "avail_actions": [runner.env.get_avail_actions()],
                "obs": [runner.env.get_obs()],
            }
            runner.batch.update(pre_transition_data, ts=runner.t)

            actions = runner.mac.select_actions(
                runner.batch,
                t_ep=runner.t,
                t_env=runner.t_env,
                test_mode=True,
            )

            row = actions[0]
            if isinstance(row, th.Tensor):
                act_list = row.detach().cpu().numpy().tolist()
            else:
                act_list = [int(x) for x in row]
            agents = grid.possible_agents
            actions_dict = {agents[i]: int(act_list[i]) for i in range(len(agents))}
            desc = describe_actions(grid, actions_dict)

            _, reward, term, trunc, env_info = runner.env.step(actions[0])
            terminated = bool(term or trunc)

            frames.append(capture_frame(grid, action_desc=desc))
            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            runner.batch.update(post_transition_data, ts=runner.t)
            runner.t += 1

        return frames
    finally:
        runner.close_env()
