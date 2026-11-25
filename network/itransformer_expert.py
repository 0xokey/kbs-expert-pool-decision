"""iTransformer-lite: variate-token transformer expert (full context).

Each target channel's lookback window is embedded as one token; a Transformer
encoder mixes information *across channels* (the opposite inductive bias of the
channel-independent PatchTST expert), and a shared head decodes each token to the
forecast horizon.  Deterministic; used as a routed expert.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser


class ITransformerLiteDiffuser(BaseDiffuser):
    uses_full_context = True

    diffusion_profile = {
        "source_model": "iTransformer-lite (variates as tokens)",
        "backbone_family": "variate_transformer",
        "preserved_operations": [
            "full lookback embedding per variate",
            "cross-variate self-attention",
            "shared multi-horizon head",
        ],
        "limitations": [
            "Deterministic; ignores the diffusion timestep.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        context_length: int = 336,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_instance_norm: bool = True,
    ):
        super().__init__(seq_length, input_channels)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.use_instance_norm = bool(use_instance_norm)

        self.embed = nn.Linear(self.context_length, self.d_model)
        self.channel_emb = nn.Parameter(torch.zeros(1, input_channels, self.d_model))
        nn.init.trunc_normal_(self.channel_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=4 * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, int(seq_length))

    def forward(self, noisy_future, t, past):
        B, C, L = past.shape
        if L != self.context_length:
            past = F.interpolate(past, size=self.context_length, mode="linear", align_corners=False)
        x = past
        if self.use_instance_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
            x = (x - mean) / std
        tokens = self.embed(x) + self.channel_emb[:, :C]
        z = self.norm(self.encoder(tokens))
        out = self.head(z)
        if self.use_instance_norm:
            out = out * std + mean
        return out
