"""Channel-independent PatchTST-style expert with full-context access.

Unlike the diffusion denoisers, this expert conditions on the *raw* lookback
window (e.g. 336 steps) instead of the compressed 96-step context, and predicts
the forecast window directly (channel-independent patching + Transformer).
It is deterministic and is used as a routed expert under the same gate; with
`training_mode=regression` its objective is plain MSE against the residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseDiffuser


class PatchTSTDiffuser(BaseDiffuser):
    uses_full_context = True

    diffusion_profile = {
        "source_model": "PatchTST-style channel-independent transformer",
        "backbone_family": "patch_transformer_full_context",
        "preserved_operations": [
            "full lookback patching",
            "channel-independent tokenisation",
            "transformer encoder over patches",
            "direct multi-horizon head",
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
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        head_type: str = "flatten",
        use_instance_norm: bool = True,
    ):
        super().__init__(seq_length, input_channels)
        self.context_length = int(context_length)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        num_patches = (self.context_length - self.patch_len) // self.stride + 1
        self.num_patches = int(max(1, num_patches))

        # Channel-independent: each target channel is patched separately.
        self.patch_embed = nn.Conv1d(
            1, self.d_model, kernel_size=self.patch_len, stride=self.stride
        )
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
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
        self.head_type = str(head_type or "flatten").strip().lower()
        self.use_instance_norm = bool(use_instance_norm)
        if self.head_type == "flatten":
            self.head = nn.Linear(self.d_model * self.num_patches, int(seq_length))
        else:
            self.head = nn.Linear(self.d_model, int(seq_length))

    def forward(self, noisy_future, t, past):
        # past: [B, C, context_length] (full lookback); channel-independent.
        B, C, L = past.shape
        x = past.reshape(B * C, 1, L)
        if self.use_instance_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
            x = (x - mean) / std
        tokens = self.patch_embed(x).transpose(1, 2)          # [B*C, N, D]
        tokens = tokens + self.pos[:, : tokens.size(1)]
        z = self.norm(self.encoder(tokens))                   # [B*C, N, D]
        if self.head_type == "flatten":
            out = self.head(z.reshape(z.size(0), -1))         # [B*C, seq]
        else:
            out = self.head(z.mean(dim=1))
        if self.use_instance_norm:
            out = out * std.squeeze(-1) + mean.squeeze(-1)
        return out.view(B, C, int(self.seq_length))
