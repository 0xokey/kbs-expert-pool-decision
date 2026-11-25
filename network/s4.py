from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDiffuser, SinusoidalTimeEmbedding


@torch.jit.script
def _selective_scan(a_bar: torch.Tensor, b_bar: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Diagonal selective scan: h_t = a_t * h_{t-1} + b_t, y_t = <c_t, h_t>.

    Compiled with TorchScript: the 168-step Python loop was the dominant cost of
    this expert (~8x slower than the scripted kernel on a 4080S).
    """
    B, L, N = a_bar.shape
    h = torch.zeros(B, N, dtype=a_bar.dtype, device=a_bar.device)
    ys = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(L):
        h = a_bar[:, t] * h + b_bar[:, t]
        ys.append((c[:, t] * h).sum(dim=-1))
    return torch.stack(ys, dim=-1)


class SelectiveSSM(nn.Module):
    """Selective diagonal state space model (Mamba-inspired).

    - A (diagonal, fixed): HiPPO-like negative values for long-range stability.
    - B, C (input-dependent): selective gating — what to remember / what to output.
    - Δ (per-channel, input-dependent): discretisation step size.

    Runs per-channel recurrence over L=168 steps — a for-loop here is
    cheaper than the FFT overhead for this sequence length.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_rank: int | None = None):
        super().__init__()
        if dt_rank is None:
            dt_rank = max(d_model // 4, 8)
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)

        # Diagonal A — negative so exp(Δ·A) is always < 1 (stable decay)
        A = -(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.register_buffer("A", A.view(1, 1, d_state))  # [1, 1, N]

        # Single projection → B, C, Δ_input
        self.x_proj = nn.Linear(d_model, d_state * 2 + dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.D = nn.Parameter(torch.ones(d_model))

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.constant_(self.dt_proj.bias, 1.0)  # softplus(1) ≈ 1.31

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        residual = x
        x = self.norm(x)
        B_b, L, D = x.shape
        N = self.d_state

        proj = self.x_proj(x)  # [B, L, 2N + dt_rank]
        B_ssm = proj[..., :N]                                # [B, L, N]
        C_ssm = proj[..., N : 2 * N]                         # [B, L, N]
        dt = F.softplus(self.dt_proj(proj[..., 2 * N :]))     # [B, L, D]

        # Per-channel SSM — reshape to [B*D, L, ...]
        dt_flat = dt.permute(0, 2, 1).reshape(B_b * D, L, 1)          # [B*D, L, 1]
        B_flat = B_ssm.unsqueeze(1).expand(-1, D, -1, -1).reshape(B_b * D, L, N)
        C_flat = C_ssm.unsqueeze(1).expand(-1, D, -1, -1).reshape(B_b * D, L, N)

        # ZOH discretisation (diagonal A → element-wise)
        A_bar = torch.exp(dt_flat * self.A)                              # [B*D, L, N]
        B_bar = (A_bar - 1.0) / self.A.clamp(-1e2) * B_flat * dt_flat   # [B*D, L, N]

        # Linear recurrence via the compiled selective scan.
        y = _selective_scan(A_bar, B_bar, C_flat)                        # [B*D, L]
        y = y.view(B_b, D, L).transpose(1, 2)                            # [B, L, D]
        y = y * self.D + residual
        return y


class S4Diffuser(BaseDiffuser):
    """Historical SSM expert name kept for checkpoint compatibility.

    The implementation is a selective diagonal SSM, closer to a lightweight
    Mamba-style block than to the original S4 kernel. Prefer
    `SelectiveSSMDiffuser` in new configs.
    """
    diffusion_profile = {
        "source_model": "Selective SSM/Mamba-style diffusion denoiser",
        "backbone_family": "selective_ssm",
        "preserved_operations": [
            "diffusion timestep embedding",
            "future/history concatenation conditioning",
            "input-dependent B/C/dt projections",
            "diagonal state-space recurrence",
            "noise prediction head",
        ],
        "limitations": [
            "Not a strict original S4 kernel reproduction.",
        ],
    }

    def __init__(
        self,
        seq_length: int,
        input_channels: int,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 2,
    ):
        super().__init__(seq_length, input_channels)

        self.in_proj = nn.Linear(input_channels * 2, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.ssm_blocks = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, input_channels)

    def forward(self, noisy_future, t, past):
        noisy_future, t = self._normalize_inputs(noisy_future, t, past)
        past = self.channel_mix(past)

        x = torch.cat([noisy_future, past], dim=1).transpose(1, 2)  # [B, L, 2C]
        h = self.in_proj(x)                                          # [B, L, D]
        h = h + self.time_emb(t).unsqueeze(1)

        for block in self.ssm_blocks:
            h = block(h)

        y = self.out_norm(h)
        y = self.out_proj(y).transpose(1, 2)                         # [B, C, L]
        return y


class SelectiveSSMDiffuser(S4Diffuser):
    """Explicit name for the selective SSM denoiser expert."""

    pass
