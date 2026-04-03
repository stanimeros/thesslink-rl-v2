"""Hybrid CNN + GRU model with shared backbone for Policy and Value heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import COMM_DIM, GRID_SIZE, NUM_CHANNELS


class SpatialEncoder(nn.Module):
    """Small CNN that processes the (C, H, W) grid observation."""

    def __init__(self, in_channels: int = NUM_CHANNELS, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


class CommEncoder(nn.Module):
    """GRU that maintains state across negotiation rounds."""

    def __init__(self, input_dim: int = COMM_DIM, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(
        self, comm: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """comm: (B, seq_len, COMM_DIM), h: (1, B, hidden)"""
        out, h_new = self.gru(comm, h)
        return out[:, -1, :], h_new  # last step output, new hidden


class HybridAgent(nn.Module):
    """Combines CNN (spatial) + GRU (comm) into Policy + Value heads."""

    def __init__(
        self,
        action_dim: int = 5,
        spatial_dim: int = 64,
        comm_hidden: int = 32,
    ):
        super().__init__()
        self.spatial_enc = SpatialEncoder(out_dim=spatial_dim)
        self.comm_enc = CommEncoder(hidden_dim=comm_hidden)
        trunk_dim = spatial_dim + comm_hidden

        self.policy_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        grid: torch.Tensor,
        comm: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns logits, value, new_hidden."""
        spatial_feat = self.spatial_enc(grid)
        comm_feat, h_new = self.comm_enc(comm, h)
        trunk = torch.cat([spatial_feat, comm_feat], dim=-1)
        logits = self.policy_head(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return logits, value, h_new

    def get_action_and_value(
        self,
        grid: torch.Tensor,
        comm: torch.Tensor,
        h: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action; returns action, logprob, entropy, value, h."""
        logits, value, h_new = self.forward(grid, comm, h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_new

    def get_value(
        self,
        grid: torch.Tensor,
        comm: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, value, _ = self.forward(grid, comm, h)
        return value


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer
