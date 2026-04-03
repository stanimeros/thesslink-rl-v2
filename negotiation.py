"""Negotiation phase: agents exchange POI scores over fixed rounds via GRU."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from environment import COMM_DIM, NUM_POIS, GridNegotiationEnv
from evaluation import compute_poi_scores
from models import HybridAgent

NEGOTIATION_ROUNDS = 5
AGREEMENT_THRESHOLD = 0.8  # cosine similarity threshold to declare consensus


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / max(norm, 1e-8)


@torch.no_grad()
def run_negotiation(
    env: GridNegotiationEnv,
    agents_models: Dict[str, HybridAgent],
    device: torch.device,
) -> tuple[int, Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    """Execute the negotiation phase and return the agreed POI index.

    Returns:
        agreed_poi: index of the chosen POI
        poi_scores: {agent_name: scores array} for reward computation
        hidden_states: {agent_name: final GRU hidden} carried into navigation
    """
    poi_scores: Dict[str, np.ndarray] = {}
    for agent in env.agents:
        pos = tuple(env.agent_positions[agent])
        other = [a for a in env.agents if a != agent][0]
        peer_pos = tuple(env.agent_positions[other])
        scores = compute_poi_scores(pos, peer_pos, env.poi_positions, env.obstacle_map)
        poi_scores[agent] = scores
        env.set_comm(agent, scores)

    hidden: Dict[str, torch.Tensor | None] = {a: None for a in env.agents}

    rollout_data: Dict[str, list] = {a: [] for a in env.agents}

    for rnd in range(NEGOTIATION_ROUNDS):
        for agent in env.agents:
            model = agents_models[agent]
            obs = env._get_obs(agent)

            grid_t = torch.from_numpy(obs["grid"]).unsqueeze(0).to(device)
            comm_t = torch.from_numpy(obs["comm"]).unsqueeze(0).unsqueeze(0).to(device)

            logits, value, h_new = model(grid_t, comm_t, hidden[agent])
            hidden[agent] = h_new

            updated_scores = poi_scores[agent] + 0.05 * np.random.randn(NUM_POIS)
            updated_scores = np.clip(updated_scores, 0.0, 1.0).astype(np.float32)
            poi_scores[agent] = updated_scores
            env.set_comm(agent, updated_scores)

            rollout_data[agent].append({
                "grid": obs["grid"],
                "comm": obs["comm"],
                "value": value.cpu(),
            })

    agreed_poi = _select_poi(poi_scores)
    env.switch_to_navigation(agreed_poi)

    hidden_out = {a: hidden[a] for a in env.agents}
    return agreed_poi, poi_scores, hidden_out


def _select_poi(poi_scores: Dict[str, np.ndarray]) -> int:
    """Pick the POI that maximises the Golden Mean (product of scores)."""
    agents = list(poi_scores.keys())
    sa = poi_scores[agents[0]]
    sb = poi_scores[agents[1]]
    products = sa * sb
    return int(np.argmax(products))


def collect_negotiation_rollout(
    env: GridNegotiationEnv,
    agents_models: Dict[str, HybridAgent],
    device: torch.device,
) -> tuple[int, Dict[str, np.ndarray], Dict[str, torch.Tensor], Dict[str, list]]:
    """Like run_negotiation but also returns per-step rollout data for PPO."""
    poi_scores: Dict[str, np.ndarray] = {}
    for agent in env.agents:
        pos = tuple(env.agent_positions[agent])
        other = [a for a in env.agents if a != agent][0]
        peer_pos = tuple(env.agent_positions[other])
        scores = compute_poi_scores(pos, peer_pos, env.poi_positions, env.obstacle_map)
        poi_scores[agent] = scores
        env.set_comm(agent, scores)

    hidden: Dict[str, torch.Tensor | None] = {a: None for a in env.agents}
    rollout: Dict[str, list] = {a: [] for a in env.agents}

    for rnd in range(NEGOTIATION_ROUNDS):
        for agent in env.agents:
            model = agents_models[agent]
            obs = env._get_obs(agent)
            grid_t = torch.from_numpy(obs["grid"]).unsqueeze(0).to(device)
            comm_t = torch.from_numpy(obs["comm"]).unsqueeze(0).unsqueeze(0).to(device)

            action, logprob, entropy, value, h_new = model.get_action_and_value(
                grid_t, comm_t, hidden[agent]
            )
            hidden[agent] = h_new

            rollout[agent].append({
                "grid": obs["grid"], "comm": obs["comm"],
                "action": action.cpu(), "logprob": logprob.cpu(),
                "value": value.cpu(), "entropy": entropy.cpu(),
            })

            updated = poi_scores[agent] + 0.05 * np.random.randn(NUM_POIS)
            poi_scores[agent] = np.clip(updated, 0.0, 1.0).astype(np.float32)
            env.set_comm(agent, poi_scores[agent])

    agreed_poi = _select_poi(poi_scores)
    env.switch_to_navigation(agreed_poi)
    return agreed_poi, poi_scores, {a: hidden[a] for a in env.agents}, rollout
