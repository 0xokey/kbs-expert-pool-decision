from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseDiffuser, SinusoidalTimeEmbedding


class RNNDiffuser(BaseDiffuser):
    diffusion_profile = {
        "source_model": "TimeGrad-style recurrent diffusion denoiser",
        "backbone_family": "rnn_lstm",
        "preserved_operations": [
            "diffusion timestep embedding",
            "future/history concatenation conditioning",
            "recurrent temporal modeling",
            "noise prediction head",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__(seq_length, input_channels)

        self.in_proj = nn.Linear(input_channels * 2, hidden_dim)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, input_channels)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1).transpose(1, 2)  # [B,L,2C]
        h = self.in_proj(x)
        h = h + self.time_emb(t).unsqueeze(1)

        h, _ = self.rnn(h)
        y = self.out_proj(h).transpose(1, 2)
        return y
