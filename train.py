"""CleanRL-style PPO training loop coordinating Negotiation -> Navigation."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import COMM_DIM, GridNegotiationEnv
from evaluation import golden_mean_reward
from models import HybridAgent
from negotiation import collect_negotiation_rollout
from navigation import collect_navigation_rollout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-episodes", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _flatten_rollout(rollout: Dict[str, list], device: torch.device):
    """Merge per-agent step dicts into batched tensors for PPO update."""
    grids, comms, actions, logprobs, values, rewards, dones = [], [], [], [], [], [], []

    for agent, steps in rollout.items():
        for s in steps:
            grids.append(torch.from_numpy(s["grid"]))
            comms.append(torch.from_numpy(s["comm"]))
            actions.append(s["action"])
            logprobs.append(s["logprob"])
            values.append(s["value"])
            rewards.append(s.get("reward", 0.0))
            dones.append(float(s.get("terminated", False)))

    if not grids:
        return None

    return {
        "grids": torch.stack(grids).to(device),
        "comms": torch.stack(comms).unsqueeze(1).to(device),
        "actions": torch.stack(actions).to(device),
        "logprobs": torch.stack(logprobs).to(device),
        "values": torch.stack(values).to(device),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "dones": torch.tensor(dones, dtype=torch.float32, device=device),
    }


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, batch, args):
    grids = batch["grids"]
    comms = batch["comms"]
    actions = batch["actions"]
    old_logprobs = batch["logprobs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    for _ in range(args.update_epochs):
        _, new_logprob, entropy, new_value, _ = model.get_action_and_value(
            grids, comms, action=actions
        )
        ratio = (new_logprob - old_logprobs).exp()

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_value - returns) ** 2).mean()
        ent_loss = entropy.mean()
        loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

    return {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": ent_loss.item()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = GridNegotiationEnv(seed=args.seed)
    model = HybridAgent().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agents_models = {a: model for a in env.possible_agents}  # shared weights

    stats = defaultdict(list)

    for ep in range(1, args.total_episodes + 1):
        env.reset(seed=args.seed + ep)

        agreed_poi, poi_scores, hidden, neg_rollout = collect_negotiation_rollout(
            env, agents_models, device
        )

        sa = poi_scores[env.possible_agents[0]][agreed_poi]
        sb = poi_scores[env.possible_agents[1]][agreed_poi]
        gm = golden_mean_reward(float(sa), float(sb))

        nav_rollout, reached = collect_navigation_rollout(
            env, agents_models, hidden, device, golden_mean=gm
        )

        merged: Dict[str, list] = {a: neg_rollout[a] + nav_rollout[a] for a in env.possible_agents}
        for agent in env.possible_agents:
            for step in merged[agent]:
                step.setdefault("reward", gm / max(len(merged[agent]), 1))

        batch = _flatten_rollout(merged, device)
        if batch is None:
            continue

        adv, ret = compute_gae(
            batch["rewards"], batch["values"], batch["dones"], args.gamma, args.gae_lambda
        )
        batch["advantages"] = adv
        batch["returns"] = ret

        metrics = ppo_update(model, optimizer, batch, args)

        stats["gm"].append(gm)
        stats["reached"].append(int(reached))
        stats["pg_loss"].append(metrics["pg_loss"])

        if ep % 20 == 0:
            avg_gm = np.mean(stats["gm"][-20:])
            avg_reach = np.mean(stats["reached"][-20:])
            print(
                f"Ep {ep:4d} | GM {avg_gm:.3f} | Reach {avg_reach:.2f} "
                f"| PG {metrics['pg_loss']:.4f} | Ent {metrics['entropy']:.4f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
