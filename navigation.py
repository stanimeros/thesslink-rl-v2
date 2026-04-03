"""Navigation phase: agents move toward the agreed POI, avoiding obstacles."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from environment import MAX_EPISODE_STEPS, GridNegotiationEnv
from models import HybridAgent

NAV_STEP_PENALTY = -0.01
REACH_BONUS = 1.0


@torch.no_grad()
def run_navigation(
    env: GridNegotiationEnv,
    agents_models: Dict[str, HybridAgent],
    hidden_states: Dict[str, torch.Tensor | None],
    device: torch.device,
    max_steps: int | None = None,
) -> tuple[Dict[str, list], bool]:
    """Execute the navigation phase until the POI is reached or time runs out.

    Returns:
        rollout: {agent: [step_dicts]} with obs/action/logprob/value per step
        reached: whether any agent reached the target POI
    """
    if max_steps is None:
        max_steps = MAX_EPISODE_STEPS - env.timestep

    rollout: Dict[str, list] = {a: [] for a in env.possible_agents}
    reached = False

    alive_agents = list(env.possible_agents)

    for _ in range(max_steps):
        if not alive_agents:
            break

        actions: Dict[str, int] = {}
        step_data: Dict[str, dict] = {}

        for agent in alive_agents:
            model = agents_models[agent]
            obs = env._get_obs(agent)
            grid_t = torch.from_numpy(obs["grid"]).unsqueeze(0).to(device)
            comm_t = torch.from_numpy(obs["comm"]).unsqueeze(0).unsqueeze(0).to(device)

            action, logprob, entropy, value, h_new = model.get_action_and_value(
                grid_t, comm_t, hidden_states.get(agent)
            )
            hidden_states[agent] = h_new
            actions[agent] = int(action.item())

            step_data[agent] = {
                "grid": obs["grid"],
                "comm": obs["comm"],
                "action": action.cpu(),
                "logprob": logprob.cpu(),
                "value": value.cpu(),
                "entropy": entropy.cpu(),
            }

        obs_next, rewards, terminated, truncated, infos = env.step(actions)

        for agent in alive_agents:
            step_data[agent]["reward"] = rewards.get(agent, 0.0) + NAV_STEP_PENALTY
            step_data[agent]["terminated"] = terminated.get(agent, False)
            step_data[agent]["truncated"] = truncated.get(agent, False)
            rollout[agent].append(step_data[agent])

        if any(terminated.values()):
            reached = True
            break
        if any(truncated.values()):
            break

        alive_agents = list(env.agents)

    return rollout, reached


def compute_nav_rewards(
    nav_rollout: Dict[str, list],
    reached: bool,
    golden_mean: float,
) -> Dict[str, list]:
    """Assign final rewards: Golden Mean bonus on success, step penalties otherwise."""
    for agent, steps in nav_rollout.items():
        for i, step in enumerate(steps):
            if reached and i == len(steps) - 1:
                step["reward"] = golden_mean + REACH_BONUS + NAV_STEP_PENALTY
            # step penalties already applied during collection
    return nav_rollout


def collect_navigation_rollout(
    env: GridNegotiationEnv,
    agents_models: Dict[str, HybridAgent],
    hidden_states: Dict[str, torch.Tensor | None],
    device: torch.device,
    golden_mean: float,
) -> tuple[Dict[str, list], bool]:
    """Full navigation collection with reward shaping applied."""
    nav_rollout, reached = run_navigation(env, agents_models, hidden_states, device)
    nav_rollout = compute_nav_rewards(nav_rollout, reached, golden_mean)
    return nav_rollout, reached
