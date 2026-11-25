import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_rows(table: torch.Tensor) -> torch.Tensor:
    return table / (table.sum(dim=-1, keepdim=True) + 1e-8)


def _broadcast_scale(scale: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if scale.dim() != 1:
        scale = scale.view(-1)
    view_shape = [1] * ref.dim()
    view_shape[-1] = scale.numel()
    return scale.view(*view_shape).to(device=ref.device, dtype=ref.dtype)


def _uniform_from_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.float32)
    denom = mask.sum(dim=-1, keepdim=True)
    safe = mask / denom.clamp(min=1.0)
    zero_mask = denom <= 0
    if zero_mask.any():
        fill = torch.full_like(mask, 1.0 / max(mask.size(-1), 1))
        safe = torch.where(zero_mask.expand_as(mask), fill, safe)
    return safe


def _topk_mask(weights: torch.Tensor, k: int | None) -> torch.Tensor | None:
    if k is None:
        return None
    k = int(k)
    if k <= 0 or k >= weights.size(-1):
        return None
    idx = torch.topk(weights, k, dim=-1).indices
    mask = torch.zeros_like(weights)
    mask.scatter_(-1, idx, 1.0)
    return mask


def _apply_sparse_routing(
    weights: torch.Tensor,
    scale: torch.Tensor | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    out = weights
    allow_mask = torch.ones_like(out)
    if scale is not None:
        gate_scale = _broadcast_scale(scale, out)
        out = out * gate_scale
        allow_mask = allow_mask * (gate_scale > 0).to(dtype=out.dtype)

    topk_mask = _topk_mask(out, top_k)
    if topk_mask is not None:
        out = out * topk_mask
        allow_mask = allow_mask * topk_mask

    denom = out.sum(dim=-1, keepdim=True)
    fallback = _uniform_from_mask(allow_mask)
    safe = out / denom.clamp(min=1e-8)
    return torch.where((denom > 1e-8).expand_as(out), safe, fallback)


def _apply_hard_routing(
    weights: torch.Tensor,
    scale: torch.Tensor | None = None,
    straight_through: bool = True,
) -> torch.Tensor:
    soft = _apply_sparse_routing(weights, scale=scale, top_k=None)
    hard_idx = torch.argmax(soft, dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, hard_idx, 1.0)

    if not straight_through:
        return hard
    return hard.detach() - soft.detach() + soft


def _masked_average(probs: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
    alive = _broadcast_scale(alive_mask.to(dtype=probs.dtype), probs)
    avg = (probs * alive).mean(dim=tuple(range(probs.dim() - 1)))
    denom = alive.mean(dim=tuple(range(probs.dim() - 1))).clamp(min=1e-8)
    return avg / denom


def make_default_rule_table(expert_names: list[str]) -> torch.Tensor:
    E = len(expert_names)
    table = torch.ones(4, E, dtype=torch.float32) * (1.0 / max(E, 1))

    def boost(row: int, names: str | tuple[str, ...], amount: float):
        if isinstance(names, str):
            allowed = {names.lower()}
        else:
            allowed = {str(name).lower() for name in names}
        for i, name in enumerate(expert_names):
            if name.lower() in allowed:
                table[row, i] += amount

    csdi_names = ("CSDIDiffuser", "CSDIAttentionDiffuser")
    ssm_names = ("S4Diffuser", "SelectiveSSMDiffuser")

    boost(0, "UNetDiffuser", 0.35)
    boost(0, csdi_names, 0.25)
    boost(0, ssm_names, 0.25)
    boost(0, "RNNDiffuser", 0.15)
    boost(0, "UGNetDiffuser", 0.10)

    boost(1, "DiffWaveDiffuser", 0.45)
    boost(1, "UNetDiffuser", 0.20)
    boost(1, "UGNetDiffuser", 0.25)
    boost(1, csdi_names, 0.10)

    boost(2, "PatchTransformerDiffuser", 0.45)
    boost(2, ssm_names, 0.20)
    boost(2, "DiMTSMambaDiffuser", 0.25)
    boost(2, "RNNDiffuser", 0.10)
    boost(2, "UGNetDiffuser", 0.20)
    boost(2, csdi_names, 0.10)

    boost(3, ssm_names, 0.30)
    boost(3, "DiMTSMambaDiffuser", 0.35)
    boost(3, "SpectralDiffuser", 0.25)
    boost(3, "PatchTransformerDiffuser", 0.20)
    boost(3, "DiffWaveDiffuser", 0.15)
    boost(3, "UNetDiffuser", 0.10)
    boost(3, "UGNetDiffuser", 0.20)
    boost(3, csdi_names, 0.20)

    return _normalize_rows(table)


class RuleCoarseRouter(nn.Module):
    def __init__(self, expert_names: list[str], rule_table=None):
        super().__init__()
        self.expert_names = expert_names

        if rule_table is None:
            table = make_default_rule_table(expert_names)
        else:
            if isinstance(rule_table, dict):
                rows = [rule_table[str(i)] for i in sorted([int(k) for k in rule_table.keys()])]
                table = torch.tensor(rows, dtype=torch.float32)
            else:
                table = torch.tensor(rule_table, dtype=torch.float32)

            if table.dim() != 2:
                raise RuntimeError(f"rule_table should be 2D, got {tuple(table.shape)}")
            if table.size(1) != len(expert_names):
                raise RuntimeError(
                    f"rule_table expert dim mismatch: got {table.size(1)}, expected {len(expert_names)}"
                )
            table = _normalize_rows(table)

        self.register_buffer("rule_table", table)

    def forward(self, coarse_label: torch.Tensor) -> torch.Tensor:
        if coarse_label is None:
            raise RuntimeError("coarse_label cannot be None when using RuleCoarseRouter")

        if coarse_label.dim() == 0:
            coarse_label = coarse_label.unsqueeze(0)

        max_label = self.rule_table.size(0) - 1
        coarse_label = coarse_label.long().clamp(min=0, max=max_label)
        return self.rule_table[coarse_label]


class CoarseGating(nn.Module):
    def __init__(self, seq_len: int, num_channels: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1)
        return self.fc(h)


class FineTaskGating(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        num_experts: int,
        num_tasks: int,
        target_indices: list[int] | None = None,
        hidden_dim: int = 128,
        use_channel_stats: bool = True,
        channel_stat_hidden_dim: int = 32,
        use_channel_embedding: bool = True,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.num_experts = int(num_experts)
        self.use_channel_stats = bool(use_channel_stats)
        self.use_channel_embedding = bool(use_channel_embedding)
        if target_indices is None or len(target_indices) != self.num_tasks:
            target_indices = list(range(self.num_tasks))
        self.register_buffer("target_indices", torch.tensor([int(i) for i in target_indices], dtype=torch.long))

        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim, num_tasks * num_experts)
        if self.use_channel_stats:
            stat_hidden = int(max(1, channel_stat_hidden_dim))
            self.channel_stat_mlp = nn.Sequential(
                nn.Linear(4, stat_hidden),
                nn.SiLU(),
                nn.Linear(stat_hidden, num_experts),
            )
        else:
            self.channel_stat_mlp = None
        if self.use_channel_embedding:
            self.channel_embedding = nn.Embedding(num_tasks, num_experts)
        else:
            self.channel_embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1)
        logits = self.fc(h).view(-1, self.num_tasks, self.num_experts)

        if self.channel_stat_mlp is not None:
            idx = self.target_indices.to(device=x.device)
            idx = idx.clamp(min=0, max=max(x.size(1) - 1, 0))
            channel_x = x.index_select(dim=1, index=idx)
            mean = channel_x.mean(dim=-1)
            std = channel_x.std(dim=-1, unbiased=False)
            last = channel_x[:, :, -1]
            first = channel_x[:, :, 0]
            slope = last - first
            stats = torch.stack([mean, std, last, slope], dim=-1)
            logits = logits + self.channel_stat_mlp(stats)

        if self.channel_embedding is not None:
            task_ids = torch.arange(self.num_tasks, device=x.device)
            logits = logits + self.channel_embedding(task_ids).unsqueeze(0).to(dtype=logits.dtype)
        return logits


class TaskTower(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        hidden_dim = int(max(1, hidden_dim))
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x.unsqueeze(1)).squeeze(1)


class KnownFutureAdapter(nn.Module):
    def __init__(self, target_channels: int, known_future_channels: int, hidden_dim: int = 128):
        super().__init__()
        hidden_dim = int(max(hidden_dim, target_channels))
        in_channels = int(target_channels + known_future_channels)
        self.in_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.mix = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out_proj = nn.Conv1d(hidden_dim, target_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, past: torch.Tensor, known_future: torch.Tensor | None = None) -> torch.Tensor:
        if known_future is None or known_future.numel() == 0:
            return past
        input_dtype = past.dtype
        x = torch.cat([past, known_future], dim=1).contiguous()
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=torch.float32)
            h = self.in_proj(x)
            h = self.mix(h)
            delta = self.out_proj(h)
        return past + delta.to(dtype=input_dtype)


class HierarchicalMMoEGating(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        num_experts: int,
        num_tasks: int,
        expert_names: list[str],
        target_indices: list[int] | None = None,
        hidden_dim: int = 128,
        use_rule_coarse: bool = True,
        coarse_mix_alpha: float = 1.0,
        rule_table=None,
        channel_expert_prior=None,
        channel_prior_alpha: float = 0.0,
        use_channel_stats: bool = True,
        channel_stat_hidden_dim: int = 32,
        use_channel_embedding: bool = True,
        coarse_top_k: int | None = None,
        final_top_k: int | None = None,
        hard_final_routing: bool = False,
        hard_routing_straight_through: bool = True,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_rule_coarse = bool(use_rule_coarse)
        self.coarse_mix_alpha = float(coarse_mix_alpha)
        self.channel_prior_alpha = float(max(0.0, min(1.0, channel_prior_alpha)))
        self.coarse_top_k = coarse_top_k
        self.final_top_k = final_top_k
        self.hard_final_routing = bool(hard_final_routing)
        self.hard_routing_straight_through = bool(hard_routing_straight_through)
        self.gate_temperature = float(max(gate_temperature, 1e-3))

        self.learned_coarse_gate = CoarseGating(
            seq_len=seq_len,
            num_channels=num_channels,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
        )
        self.fine_gate = FineTaskGating(
            seq_len=seq_len,
            num_channels=num_channels,
            num_experts=num_experts,
            num_tasks=num_tasks,
            target_indices=target_indices,
            hidden_dim=hidden_dim,
            use_channel_stats=use_channel_stats,
            channel_stat_hidden_dim=channel_stat_hidden_dim,
            use_channel_embedding=use_channel_embedding,
        )
        self.rule_router = RuleCoarseRouter(expert_names=expert_names, rule_table=rule_table)
        if channel_expert_prior is None:
            prior = torch.ones(num_tasks, num_experts, dtype=torch.float32) * (1.0 / max(num_experts, 1))
        else:
            prior = torch.as_tensor(channel_expert_prior, dtype=torch.float32)
            if prior.dim() != 2:
                raise RuntimeError(f"channel_expert_prior should be 2D, got {tuple(prior.shape)}")
            if prior.shape != (num_tasks, num_experts):
                raise RuntimeError(
                    f"channel_expert_prior shape mismatch: got {tuple(prior.shape)}, "
                    f"expected {(num_tasks, num_experts)}"
                )
            prior = _normalize_rows(prior)
        self.register_buffer("channel_expert_prior", prior)

    def set_channel_expert_prior(self, prior: torch.Tensor):
        prior = torch.as_tensor(prior, dtype=self.channel_expert_prior.dtype, device=self.channel_expert_prior.device)
        if prior.shape != self.channel_expert_prior.shape:
            raise RuntimeError(
                f"channel_expert_prior shape mismatch: got {tuple(prior.shape)}, "
                f"expected {tuple(self.channel_expert_prior.shape)}"
            )
        self.channel_expert_prior.copy_(_normalize_rows(prior))

    def forward(
        self,
        x: torch.Tensor,
        coarse_label: torch.Tensor | None = None,
        expert_gate_scale: torch.Tensor | None = None,
    ):
        learned_coarse_logits = self.learned_coarse_gate(x)
        fine_logits = self.fine_gate(x)

        learned_coarse = F.softmax(learned_coarse_logits / self.gate_temperature, dim=-1)
        fine = F.softmax(fine_logits / self.gate_temperature, dim=-1)

        if self.use_rule_coarse and coarse_label is not None:
            rule_coarse = self.rule_router(coarse_label).to(x.device)
            learned_for_mix = learned_coarse
            if rule_coarse.dim() == 3:
                learned_for_mix = learned_coarse.unsqueeze(1)
            coarse_dense = self.coarse_mix_alpha * rule_coarse + (1.0 - self.coarse_mix_alpha) * learned_for_mix
            coarse_dense = coarse_dense / (coarse_dense.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            rule_coarse = None
            coarse_dense = learned_coarse

        if coarse_dense.dim() == 2:
            coarse_for_tasks = coarse_dense.unsqueeze(1).expand(-1, fine.size(1), -1)
        else:
            coarse_for_tasks = coarse_dense

        final_dense = coarse_for_tasks * fine
        final_dense = final_dense / (final_dense.sum(dim=-1, keepdim=True) + 1e-8)
        if self.channel_prior_alpha > 0.0:
            channel_prior = self.channel_expert_prior.to(device=x.device, dtype=final_dense.dtype).unsqueeze(0)
            final_dense = (1.0 - self.channel_prior_alpha) * final_dense + self.channel_prior_alpha * channel_prior
            final_dense = final_dense / (final_dense.sum(dim=-1, keepdim=True) + 1e-8)

        coarse_sparse = _apply_sparse_routing(coarse_for_tasks, scale=expert_gate_scale, top_k=self.coarse_top_k)
        if self.hard_final_routing:
            final_sparse = _apply_hard_routing(
                final_dense,
                scale=expert_gate_scale,
                straight_through=self.hard_routing_straight_through,
            )
        else:
            final_sparse = _apply_sparse_routing(final_dense, scale=expert_gate_scale, top_k=self.final_top_k)

        return {
            "final": final_sparse,
            "coarse": coarse_sparse,
            "fine": fine,
            "final_dense": final_dense,
            "coarse_dense": coarse_for_tasks,
            "fine_dense": fine,
            "learned_coarse": learned_coarse,
            "learned_coarse_logits": learned_coarse_logits,
            "fine_logits": fine_logits,
            "rule_coarse": rule_coarse,
        }


class ContextCompressor(nn.Module):
    """Learnable long-context compression (replaces linear interpolation).

    The dataset provides a `context_len` history (e.g. 336 steps) while the
    denoisers operate on a `pred_len` window (e.g. 96). Linearly interpolating
    336 -> 96 destroys high-frequency structure; this module learns a residual
    correction on top of the last `out_len` steps, so it starts as the recent
    window and can only add information.
    """

    def __init__(self, channels: int, out_len: int, hidden_dim: int = 64):
        super().__init__()
        self.out_len = int(out_len)
        hidden_dim = int(max(8, hidden_dim))
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(self.out_len),
            nn.Conv1d(hidden_dim, channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < self.out_len:
            x = F.interpolate(x, size=self.out_len, mode="linear", align_corners=False)
        tail = x[:, :, -self.out_len:]
        return tail + self.net(x)


class MoEDiffuser(nn.Module):
    STATUS_ACTIVE = 0
    STATUS_SOFT_RETIRED = 1
    STATUS_HARD_PRUNED = 2

    def __init__(
        self,
        expert_instances: list[nn.Module],
        seq_length: int,
        num_channels: int,
        device: torch.device,
        target_indices: list[int] | None = None,
        known_future_channels: int = 0,
        gate_hidden_dim: int = 128,
        use_rule_coarse: bool = True,
        coarse_mix_alpha: float = 1.0,
        rule_table=None,
        channel_expert_prior=None,
        channel_prior_alpha: float = 0.0,
        use_channel_stats: bool = True,
        channel_stat_hidden_dim: int = 32,
        use_channel_embedding: bool = True,
        coarse_top_k: int | None = None,
        final_top_k: int | None = 2,
        hard_final_routing: bool = False,
        hard_routing_straight_through: bool = True,
        gate_temperature: float = 1.0,
        soft_retire_scale: float = 0.05,
        min_alive_experts: int = 3,
        use_task_heads: bool = False,
        task_head_hidden_dim: int = 16,
        uniform_ensemble: bool = False,
        use_context_compressor: bool = True,
    ):
        super().__init__()
        self.experts = nn.ModuleList(expert_instances)
        self.seq_length = int(seq_length)
        self.num_channels = int(num_channels)
        self.device = device
        self.known_future_channels = int(max(0, known_future_channels))
        self.soft_retire_scale = float(soft_retire_scale)
        self.min_alive_experts = int(max(1, min_alive_experts))

        if target_indices is None or len(target_indices) == 0:
            target_indices = list(range(num_channels))
        self.target_indices = [int(i) for i in target_indices]

        self.num_experts = len(self.experts)
        self.num_tasks = len(self.target_indices)
        self.expert_names = [type(expert).__name__ for expert in self.experts]
        self.use_task_heads = bool(use_task_heads)
        # Baseline mode: every alive expert contributes equally. Routing then
        # degenerates to a plain ensemble and the gate receives no gradient.
        self.uniform_ensemble = bool(uniform_ensemble)

        self.mmoe_gating = HierarchicalMMoEGating(
            seq_len=self.seq_length,
            num_channels=self.num_channels,
            num_experts=self.num_experts,
            num_tasks=self.num_tasks,
            expert_names=self.expert_names,
            target_indices=self.target_indices,
            hidden_dim=gate_hidden_dim,
            use_rule_coarse=use_rule_coarse,
            coarse_mix_alpha=coarse_mix_alpha,
            rule_table=rule_table,
            channel_expert_prior=channel_expert_prior,
            channel_prior_alpha=channel_prior_alpha,
            use_channel_stats=use_channel_stats,
            channel_stat_hidden_dim=channel_stat_hidden_dim,
            use_channel_embedding=use_channel_embedding,
            coarse_top_k=coarse_top_k,
            final_top_k=final_top_k,
            hard_final_routing=hard_final_routing,
            hard_routing_straight_through=hard_routing_straight_through,
            gate_temperature=gate_temperature,
        )
        if self.known_future_channels > 0:
            self.known_future_adapter = KnownFutureAdapter(
                target_channels=self.num_channels,
                known_future_channels=self.known_future_channels,
                hidden_dim=gate_hidden_dim,
            )
        else:
            self.known_future_adapter = None

        self.context_compressor = (
            ContextCompressor(self.num_channels, self.seq_length, hidden_dim=64)
            if bool(use_context_compressor)
            else None
        )

        # Optional fixed per-channel assignment produced by validation-based
        # expert selection; -1 means "use the learned gate".
        self.register_buffer(
            "fixed_channel_experts",
            torch.full((self.num_tasks,), -1, dtype=torch.long),
            persistent=False,
        )
        # Soft per-channel expert weights (rows sum to one); used by the
        # certified router, which commits a channel only when the validation
        # margin is significant and otherwise falls back to a mixture.
        self.register_buffer(
            "fixed_channel_weights",
            torch.full((self.num_tasks, self.num_experts), -1.0),
            persistent=False,
        )
        self.register_buffer("expert_status", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("expert_gate_scale", torch.ones(self.num_experts, dtype=torch.float32))
        self.register_buffer("expert_bad_epochs", torch.zeros(self.num_experts, dtype=torch.long))
        self.register_buffer("expert_soft_retired_epoch", torch.full((self.num_experts,), -1, dtype=torch.long))
        self.register_buffer("expert_hard_pruned_epoch", torch.full((self.num_experts,), -1, dtype=torch.long))

        if self.use_task_heads:
            self.task_heads = nn.ModuleList(
                [TaskTower(hidden_dim=task_head_hidden_dim) for _ in range(self.num_tasks)]
            )
        else:
            self.task_heads = nn.ModuleList()

    def set_fixed_channel_experts(self, indices):
        """Force one expert per target channel (validated outside the gate)."""
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.fixed_channel_experts.device)
        if idx.numel() != self.num_tasks:
            raise ValueError(f"expected {self.num_tasks} channel choices, got {idx.numel()}")
        if int(idx.max().item()) >= self.num_experts or int(idx.min().item()) < -1:
            raise ValueError(f"expert index out of range: {idx.tolist()}")
        self.fixed_channel_experts.copy_(idx)
        self.fixed_channel_weights.fill_(-1)

    def set_fixed_channel_weights(self, weights):
        """Per-channel expert weights, shape [num_tasks, num_experts]."""
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.fixed_channel_weights.device)
        if tuple(w.shape) != (self.num_tasks, self.num_experts):
            raise ValueError(f"expected [{self.num_tasks},{self.num_experts}], got {tuple(w.shape)}")
        row_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.fixed_channel_weights.copy_(w / row_sum)
        self.fixed_channel_experts.fill_(-1)

    def clear_fixed_channel_experts(self):
        self.fixed_channel_experts.fill_(-1)
        self.fixed_channel_weights.fill_(-1)

    def _alive_mask(self) -> torch.Tensor:
        return (self.expert_status != self.STATUS_HARD_PRUNED)

    def _routing_scale(self) -> torch.Tensor:
        return self.expert_gate_scale * self._alive_mask().to(dtype=self.expert_gate_scale.dtype)

    def _align_tensor(self, x: torch.Tensor, num_channels: int, resize: str = "interp") -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(f"Expected [B,C,L], got {tuple(x.shape)}")

        B, C, L = x.shape
        if C > num_channels:
            x = x[:, :num_channels, :]
        elif C < num_channels:
            x = torch.cat([x, x.new_zeros((B, num_channels - C, L))], dim=1)

        if resize == "interp" and x.size(-1) != self.seq_length:
            x = F.interpolate(x, size=self.seq_length, mode="linear", align_corners=False)
        return x.contiguous()

    def _align_to_CL(self, x: torch.Tensor) -> torch.Tensor:
        return self._align_tensor(x, self.num_channels)

    def _align_known_future(self, x: torch.Tensor) -> torch.Tensor:
        return self._align_tensor(x, self.known_future_channels)

    def _build_condition_context(
        self,
        past: torch.Tensor,
        known_future: torch.Tensor | None = None,
    ) -> torch.Tensor:
        past = self._align_tensor(past, self.num_channels, resize="none")
        if past.size(-1) != self.seq_length:
            if self.context_compressor is not None:
                past = self.context_compressor(past)
            else:
                past = F.interpolate(past, size=self.seq_length, mode="linear", align_corners=False)
        past = past.contiguous()
        if self.known_future_adapter is None:
            return past
        if known_future is None:
            return past
        known_future = self._align_known_future(known_future)
        return self.known_future_adapter(past, known_future)

    def num_alive_experts(self) -> int:
        return int(self._alive_mask().sum().item())

    def can_prune_more(self) -> bool:
        return self.num_alive_experts() > self.min_alive_experts

    def get_expert_status_strings(self) -> list[str]:
        names = {
            self.STATUS_ACTIVE: "active",
            self.STATUS_SOFT_RETIRED: "soft_retired",
            self.STATUS_HARD_PRUNED: "hard_pruned",
        }
        return [names.get(int(v), "unknown") for v in self.expert_status.detach().cpu().tolist()]

    def soft_retire_expert(self, expert_idx: int, epoch: int | None = None) -> bool:
        expert_idx = int(expert_idx)
        if expert_idx < 0 or expert_idx >= self.num_experts:
            return False
        if int(self.expert_status[expert_idx].item()) != self.STATUS_ACTIVE:
            return False
        self.expert_status[expert_idx] = self.STATUS_SOFT_RETIRED
        self.expert_gate_scale[expert_idx] = self.soft_retire_scale
        if epoch is not None:
            self.expert_soft_retired_epoch[expert_idx] = int(epoch)
        return True

    def hard_prune_expert(self, expert_idx: int, epoch: int | None = None, force: bool = False) -> bool:
        expert_idx = int(expert_idx)
        if expert_idx < 0 or expert_idx >= self.num_experts:
            return False
        if not force and not self.can_prune_more():
            return False
        if int(self.expert_status[expert_idx].item()) == self.STATUS_HARD_PRUNED:
            return False

        self.expert_status[expert_idx] = self.STATUS_HARD_PRUNED
        self.expert_gate_scale[expert_idx] = 0.0
        if epoch is not None:
            self.expert_hard_pruned_epoch[expert_idx] = int(epoch)

        expert = self.experts[expert_idx]
        expert.eval()
        for p in expert.parameters():
            p.requires_grad = False
        return True

    def reset_expert_bad_epoch(self, expert_idx: int):
        self.expert_bad_epochs[int(expert_idx)] = 0

    def increment_expert_bad_epoch(self, expert_idx: int):
        self.expert_bad_epochs[int(expert_idx)] += 1

    def expert_diffusion_profiles(self) -> list[dict]:
        profiles = []
        for idx, expert in enumerate(self.experts):
            if hasattr(expert, "get_diffusion_profile"):
                profile = expert.get_diffusion_profile()
            else:
                profile = {"class_name": type(expert).__name__}
            profile = dict(profile)
            profile["expert_idx"] = int(idx)
            profile["expert_name"] = self.expert_names[idx] if idx < len(self.expert_names) else type(expert).__name__
            profiles.append(profile)
        return profiles

    def forward(
        self,
        noisy_future,
        t,
        past,
        coarse_label=None,
        return_weights: bool = False,
        known_future: torch.Tensor | None = None,
    ):
        if past is None:
            raise RuntimeError("past cannot be None")

        raw_past = self._align_tensor(past, self.num_channels, resize="none")
        past = self._build_condition_context(raw_past, known_future=known_future)
        gate_scale = self._routing_scale()
        weight_dict = self.mmoe_gating(past, coarse_label=coarse_label, expert_gate_scale=gate_scale)
        if float(self.fixed_channel_weights.min().item()) >= 0:
            forced = self.fixed_channel_weights.view(
                1, self.num_tasks, self.num_experts
            ).expand_as(weight_dict["final_dense"])
            for key in ("final_dense", "coarse_dense", "fine_dense", "final", "coarse", "fine"):
                weight_dict[key] = forced
        elif int(self.fixed_channel_experts.min().item()) >= 0:
            forced = torch.zeros_like(weight_dict["final_dense"])
            for task_i, expert_i in enumerate(self.fixed_channel_experts.tolist()):
                forced[:, task_i, int(expert_i)] = 1.0
            for key in ("final_dense", "coarse_dense", "fine_dense", "final", "coarse", "fine"):
                weight_dict[key] = forced
        if self.uniform_ensemble:
            alive = self._alive_mask().to(dtype=weight_dict["final_dense"].dtype)
            uniform = alive / alive.sum().clamp(min=1.0)
            dense = uniform.view(1, 1, -1).expand_as(weight_dict["final_dense"])
            weight_dict["final_dense"] = dense
            weight_dict["coarse_dense"] = dense
            weight_dict["fine_dense"] = dense
            weight_dict["final"] = dense
            weight_dict["coarse"] = dense
            weight_dict["fine"] = dense
        weight_dict["expert_gate_scale"] = gate_scale
        weight_dict["expert_status"] = self.expert_status

        if return_weights and noisy_future is None:
            return None, weight_dict

        if noisy_future is None or t is None:
            raise RuntimeError("MoEDiffuser.forward requires noisy_future and t during training/sampling.")

        noisy_future = self._align_to_CL(noisy_future)

        final_weights = weight_dict["final"]
        coarse_weights = weight_dict["coarse"]
        active_mask = (final_weights > 0).any(dim=(0, 1))
        active_mask = active_mask | (coarse_weights > 0).any(dim=(0, 1))
        active_mask = active_mask & (gate_scale > 0)
        active_indices = torch.nonzero(active_mask, as_tuple=False).view(-1)

        if active_indices.numel() == 0:
            fallback_scores = weight_dict["final_dense"].mean(dim=(0, 1)) * (gate_scale > 0).to(weight_dict["final_dense"].dtype)
            if fallback_scores.sum() <= 0:
                fallback_scores = (self._alive_mask()).to(dtype=weight_dict["final_dense"].dtype)
            active_indices = torch.topk(fallback_scores, k=1, dim=-1).indices.view(-1)

        expert_outs = []
        kept_indices: list[int] = []
        for idx in active_indices.detach().cpu().tolist():
            if float(gate_scale[idx].item()) <= 0:
                continue
            expert = self.experts[idx]
            expert_past = raw_past if getattr(expert, "uses_full_context", False) else past
            out = expert(noisy_future, t, expert_past)
            if out.dim() != 3:
                raise RuntimeError(
                    f"Expert[{idx}]({type(self.experts[idx]).__name__}) returned {tuple(out.shape)}, expected [B,C,L]."
                )
            expert_outs.append(self._align_to_CL(out))
            kept_indices.append(int(idx))

        if len(expert_outs) == 0:
            raise RuntimeError("No active experts available for routing.")

        expert_outs = torch.stack(expert_outs, dim=1)
        active_idx_tensor = torch.tensor(kept_indices, device=noisy_future.device, dtype=torch.long)
        final_out = torch.zeros_like(noisy_future)

        for task_i, ch in enumerate(self.target_indices):
            w = final_weights[:, task_i, active_idx_tensor]
            y = torch.einsum("be,bel->bl", w, expert_outs[:, :, ch, :])
            if self.use_task_heads:
                y = self.task_heads[task_i](y)
            y = y.to(dtype=final_out.dtype)
            final_out[:, ch, :] = y

        mask = torch.ones(self.num_channels, dtype=torch.bool, device=final_out.device)
        mask[self.target_indices] = False
        if mask.any():
            coarse_channel_weights = coarse_weights.mean(dim=1)
            coarse_mix = torch.einsum("be,becl->bcl", coarse_channel_weights[:, active_idx_tensor], expert_outs)
            coarse_mix = coarse_mix.to(dtype=final_out.dtype)
            final_out[:, mask, :] = coarse_mix[:, mask, :]

        weight_dict["active_expert_indices"] = active_idx_tensor
        # Kept for the expert-wise auxiliary loss: hard routing at inference
        # needs every expert to be a decent standalone predictor, not just a
        # component of the mixture.
        weight_dict["expert_outputs"] = expert_outs
        weight_dict["expert_output_indices"] = active_idx_tensor
        if return_weights:
            return final_out, weight_dict
        return final_out

    def freeze_experts(self):
        print("[MoE] Freeze expert parameters.")
        for idx, expert in enumerate(self.experts):
            if int(self.expert_status[idx].item()) == self.STATUS_HARD_PRUNED:
                continue
            for p in expert.parameters():
                p.requires_grad = False

    def unfreeze_experts(self):
        print("[MoE] Unfreeze expert parameters.")
        for idx, expert in enumerate(self.experts):
            if int(self.expert_status[idx].item()) == self.STATUS_HARD_PRUNED:
                continue
            for p in expert.parameters():
                p.requires_grad = True

    def gate_balance_loss(self, weight_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        alive = self._alive_mask().to(dtype=weight_dict["coarse_dense"].dtype)

        coarse = weight_dict.get("coarse_dense", weight_dict["coarse"])
        avg_coarse = _masked_average(coarse, alive)
        target = alive / alive.sum().clamp(min=1.0)
        losses.append(((avg_coarse - target) ** 2).mean())

        fine = weight_dict.get("fine_dense", weight_dict["fine"])
        avg_fine = _masked_average(fine, alive)
        losses.append(((avg_fine - target) ** 2).mean())

        final = weight_dict.get("final_dense", weight_dict["final"])
        avg_final = _masked_average(final, alive)
        losses.append(((avg_final - target) ** 2).mean())
        return sum(losses) / len(losses)

    def gate_entropy_loss(self, weight_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        def entropy_penalty(probs: torch.Tensor) -> torch.Tensor:
            alive = self._alive_mask().to(dtype=probs.dtype, device=probs.device)
            p = probs * _broadcast_scale(alive, probs)
            p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            entropy = -(p.clamp(min=1e-8) * torch.log(p.clamp(min=1e-8))).sum(dim=-1)
            max_entropy = math.log(max(int(alive.sum().item()), 1))
            if max_entropy <= 0:
                return torch.zeros((), device=probs.device, dtype=probs.dtype)
            return 1.0 - (entropy / max_entropy).mean()

        losses = [
            entropy_penalty(weight_dict.get("coarse_dense", weight_dict["coarse"])),
            entropy_penalty(weight_dict.get("fine_dense", weight_dict["fine"])),
            entropy_penalty(weight_dict.get("final_dense", weight_dict["final"])),
        ]
        return sum(losses) / len(losses)

    def gate_min_usage_loss(self, weight_dict: dict[str, torch.Tensor], min_usage: float = 0.0) -> torch.Tensor:
        min_usage = float(max(0.0, min_usage))
        if min_usage <= 0.0:
            return torch.zeros((), device=self.expert_gate_scale.device)

        alive = self._alive_mask().to(dtype=weight_dict["final"].dtype, device=weight_dict["final"].device)
        final = weight_dict["final"]
        avg_final = _masked_average(final, alive)
        target = torch.full_like(avg_final, min_usage)
        return F.relu(target - avg_final).mean()

    def channel_prior_loss(self, weight_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        final_dense = weight_dict.get("final_dense", None)
        if final_dense is None:
            return torch.zeros((), device=self.expert_gate_scale.device)

        prior = self.mmoe_gating.channel_expert_prior.to(
            device=final_dense.device,
            dtype=final_dense.dtype,
        )
        alive = self._alive_mask().to(device=final_dense.device, dtype=final_dense.dtype)
        prior = prior * alive.view(1, -1)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        probs = final_dense * alive.view(1, 1, -1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = prior.unsqueeze(0).expand_as(probs)
        return -(target * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()
