from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class GraphTemporalBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.channel_mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)
        h = self.channel_mix(h)
        return self.act(h) + x


class UGNetDiffuser(BaseDiffuser):
    diffusion_profile = {
        "source_model": "DiffSTG/UGNet-style graph-temporal diffusion denoiser",
        "backbone_family": "graph_temporal_conv",
        "preserved_operations": [
            "diffusion timestep embedding",
            "future/history concatenation conditioning",
            "temporal convolution",
            "channel mixing as graph-temporal proxy",
            "noise prediction head",
        ],
        "limitations": [
            "Uses learned channel mixing as a lightweight graph proxy.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__(seq_length, input_channels)

        self.in_proj = nn.Conv1d(input_channels * 2, hidden_dim, kernel_size=1)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([GraphTemporalBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Conv1d(hidden_dim, input_channels, kernel_size=1)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1)
        h = self.in_proj(x)
        h = h + self.time_mlp(self.time_emb(t)).unsqueeze(-1)

        for block in self.blocks:
            h = block(h)

        return self.out_proj(h)
