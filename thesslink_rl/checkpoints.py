"""Resolve EPyMARL checkpoints under ``results/models`` and run policy rollouts for GIFs.

Requires a cloned ``epymarl/`` tree next to this package (see project README).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import yaml

# --- Sacred metrics → best environment timestep (matches visualize._test_reward_series) ---


def _test_reward_series(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Timesteps and values for test return (same precedence as ``visualize.py``)."""
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
            return np.array([]), np.array([])
        total += np.asarray(m["values"], dtype=float)
    return steps, total


def best_test_env_timestep(metrics: dict) -> int | None:
    """Return the logged ``t_env`` at which test return is highest (argmax)."""
    steps, vals = _test_reward_series(metrics)
    if steps.size == 0 or vals.size == 0:
        return None
    return int(steps[int(np.argmax(vals))])


def _checkpoint_dirs_for_run(run_dir: Path) -> dict[int, Path]:
    """Map timestep → checkpoint dir if it contains saved ``*.th`` files."""
    out: dict[int, Path] = {}
    if not run_dir.is_dir():
        return out
    for sub in run_dir.iterdir():
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        if any(sub.glob("*.th")):
            out[int(sub.name)] = sub
    return out


def _algo_run_dirs(models_root: Path, algo: str, env_version: int) -> list[Path]:
    """Directories that directly contain ``<t_env>/*.th`` for this algo and env version.

    EPyMARL saves under::

      models/<name>_seed<k>_<env_key>/GridNegotiation-vN_<timestamp>/<t_env>/*.th

    So the env version appears in a **nested** folder, not in the top-level name.
    """
    if not models_root.is_dir():
        return []
    algo_p = algo.lower() + "_"
    version_prefix = f"GridNegotiation-v{env_version}_"
    runs: list[Path] = []
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.lower().startswith(algo_p):
            continue
        # Flat layout (unusual): version marker in the top-level folder name
        if f"GridNegotiation-v{env_version}" in child.name:
            runs.append(child)
            continue
        # Normal layout: models/<algo>_.../GridNegotiation-vN_<datetime>/
        try:
            for sub in child.iterdir():
                if not sub.is_dir():
                    continue
                if sub.name.startswith(version_prefix):
                    runs.append(sub)
        except OSError:
            continue
    return runs


def find_best_checkpoint_timestep_dir(
    algo: str,
    results_dir: Path,
    metrics: dict,
    env_version: int,
    *,
    models_root: Path | None = None,
) -> Path | None:
    """Pick the saved checkpoint whose timestep is closest to best test return.

    Layout (EPyMARL): ``<models>/<name>_seed*_...GridNegotiation-vN.../<t_env>/*.th``
    """
    root = models_root if models_root is not None else (results_dir / "models")
    target_t = best_test_env_timestep(metrics)
    if target_t is None:
        return None
    best_path: Path | None = None
    best_dist: float | None = None
    for run_dir in _algo_run_dirs(root, algo, env_version):
        for t, path in _checkpoint_dirs_for_run(run_dir).items():
            dist = float(abs(t - target_t))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_path = path
            elif best_dist is not None and dist == best_dist and best_path is not None:
                if t > int(best_path.name):
                    best_path = path
    return best_path


def describe_models_dir_status(models_root: Path | None, results_dir: Path) -> str:
    """Short message when no checkpoint could be resolved."""
    root = models_root if models_root is not None else (results_dir / "models")
    if not root.exists():
        return f"no models directory at {root}"
    try:
        if not any(root.iterdir()):
            return f"models directory is empty ({root})"
    except OSError:
        return f"cannot read models directory ({root})"
    return f"no matching checkpoints under {root} for this algorithm and env version"


# --- EPyMARL config (YAML merge, aligned with ``epymarl/src/main.py``) ---


def _recursive_dict_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _recursive_dict_update(d[k], v)
        else:
            d[k] = v
    return d


def _resolve_env_config_yaml(project_root: Path, epymarl_src: Path, env_config_name: str) -> Path:
    """Prefer EPyMARL's ``config/envs/``; fall back to this repo's ``epymarl_config/envs/``.

    Training copies ThessLink YAMLs into ``epymarl``; users who only sync ``results/models``
    still need rollout config — the bundled ``epymarl_config/envs/*.yaml`` matches them.
    """
    in_epymarl = epymarl_src / "config" / "envs" / f"{env_config_name}.yaml"
    bundled = project_root / "epymarl_config" / "envs" / f"{env_config_name}.yaml"
    if in_epymarl.is_file():
        return in_epymarl
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"Missing env config {env_config_name}.yaml — expected {in_epymarl} "
        f"or {bundled}",
    )


def load_epymarl_config_for_algo(algo: str, env_config_name: str, seed: int) -> Any:
    """Load merged default + env + algorithm YAML into a ``SimpleNamespace`` (like EPyMARL)."""
    from types import SimpleNamespace

    import torch as th

    root = Path(__file__).resolve().parent.parent
    epymarl_src = root / "epymarl" / "src"
    if not epymarl_src.is_dir():
        raise FileNotFoundError(
            f"EPyMARL not found at {epymarl_src}. Clone it next to the project root.",
        )
    cfg_dir = epymarl_src / "config"
    with open(cfg_dir / "default.yaml") as f:
        config: dict = yaml.safe_load(f)
    env_yaml = _resolve_env_config_yaml(root, epymarl_src, env_config_name)
    with open(env_yaml) as f:
        _recursive_dict_update(config, yaml.safe_load(f))
    with open(cfg_dir / "algs" / f"{algo}.yaml") as f:
        _recursive_dict_update(config, yaml.safe_load(f))

    if algo in ("iql", "mappo"):
        config["common_reward"] = False

    config["seed"] = seed
    config["env_args"]["seed"] = seed
    config["test_nepisode"] = 1
    config["use_cuda"] = bool(config.get("use_cuda", True) and th.cuda.is_available())
    config["device"] = "cuda" if config["use_cuda"] else "cpu"

    # Single-env rollout for GIF capture (matches training stack but avoids parallel workers).
    config["runner"] = "episode"
    config["batch_size_run"] = 1

    # Nested dicts (e.g. env_args) stay as dicts — required for ``env_fn(**env_args)``.
    return SimpleNamespace(**config)


def _unwrap_grid_negotiation(marl_env: Any) -> Any:
    """Traverse Gymnasium / EPyMARL wrappers to ``GridNegotiationEnv``."""
    cur: Any = marl_env
    for _ in range(24):
        if hasattr(cur, "obstacle_map") and hasattr(cur, "possible_agents"):
            return cur
        nxt = getattr(cur, "_env", None)
        if nxt is not None:
            cur = nxt
            continue
        u = getattr(cur, "unwrapped", None)
        if u is not None and u is not cur:
            cur = u
            continue
        break
    raise RuntimeError("Could not find GridNegotiationEnv inside EPyMARL env wrappers")


def _load_epymarl_rollout_modules(epymarl_src: Path):
    """Import EPyMARL packages from ``epymarl/src`` via ``importlib`` (not on ``sys.path`` at analysis time).

    Static analyzers cannot resolve ``components.*`` as top-level imports because that tree is
    optional and only prepended at runtime.
    """
    if str(epymarl_src) not in sys.path:
        sys.path.insert(0, str(epymarl_src))
    episode_buffer = importlib.import_module("components.episode_buffer")
    transforms = importlib.import_module("components.transforms")
    controllers = importlib.import_module("controllers")
    learners = importlib.import_module("learners")
    runners = importlib.import_module("runners")
    logging_mod = importlib.import_module("utils.logging")
    return (
        episode_buffer.ReplayBuffer,
        transforms.OneHot,
        controllers.REGISTRY,
        learners.REGISTRY,
        runners.REGISTRY,
        logging_mod.Logger,
        logging_mod.get_logger,
    )


def rollout_episode_frames_for_gif(
    ckpt_dir: Path,
    cfg: Any,
    seed: int,
    capture_frame: Callable[..., dict],
    describe_actions: Callable[..., str],
) -> list[dict]:
    """Load checkpoint and return frames compatible with ``visualization.replay_episode``."""
    import torch as th

    root = Path(__file__).resolve().parent.parent
    epymarl_src = root / "epymarl" / "src"
    if not epymarl_src.is_dir():
        raise FileNotFoundError(
            f"EPyMARL not found at {epymarl_src}. Clone it next to the project root.",
        )

    (
        ReplayBuffer,
        OneHot,
        mac_REGISTRY,
        le_REGISTRY,
        r_REGISTRY,
        Logger,
        get_logger,
    ) = _load_epymarl_rollout_modules(epymarl_src)

    args = cfg
    args.seed = seed
    if isinstance(args.env_args, dict):
        args.env_args["seed"] = seed
    else:
        args.env_args.seed = seed

    logger = Logger(get_logger())
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    scheme: dict = {
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

    learner.load_models(str(ckpt_dir))
    runner.t_env = int(ckpt_dir.name)

    runner.reset()
    grid = _unwrap_grid_negotiation(runner.env)
    frames: list[dict] = [capture_frame(grid)]

    terminated = False
    runner.mac.init_hidden(batch_size=args.batch_size_run)

    while not terminated:
        pre_transition_data = {
            "state": [runner.env.get_state()],
            "avail_actions": [runner.env.get_avail_actions()],
            "obs": [runner.env.get_obs()],
        }
        runner.batch.update(pre_transition_data, ts=runner.t)

        actions = runner.mac.select_actions(
            runner.batch, t_ep=runner.t, t_env=runner.t_env, test_mode=True,
        )

        act0 = actions[0]
        act_list = act0.cpu().tolist() if hasattr(act0, "cpu") else act0
        if act_list and isinstance(act_list[0], list):
            flat = [int(x[0]) for x in act_list]
        else:
            flat = [int(x) for x in act_list]

        agents = grid.possible_agents
        actions_dict: Dict[str, int] = {
            agents[i]: flat[i] for i in range(min(len(agents), len(flat)))
        }
        desc = describe_actions(grid, actions_dict)

        _, reward, term, trunc, step_info = runner.env.step(actions[0])
        terminated = bool(term or trunc)

        post_transition_data = {
            "actions": actions,
            "terminated": [(terminated != step_info.get("episode_limit", False),)],
        }
        if args.common_reward:
            post_transition_data["reward"] = [(reward,)]
        else:
            post_transition_data["reward"] = [tuple(reward)]

        runner.batch.update(post_transition_data, ts=runner.t)
        frames.append(capture_frame(grid, action_desc=desc))

        runner.t += 1

        last_data = {
            "state": [runner.env.get_state()],
            "avail_actions": [runner.env.get_avail_actions()],
            "obs": [runner.env.get_obs()],
        }
        runner.batch.update(last_data, ts=runner.t)

        actions = runner.mac.select_actions(
            runner.batch, t_ep=runner.t, t_env=runner.t_env, test_mode=True,
        )
        runner.batch.update({"actions": actions}, ts=runner.t)

    runner.close_env()
    return frames
