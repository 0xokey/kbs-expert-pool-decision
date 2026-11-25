import inspect

import torch

from utils.common import instantiate_model, resolve_context_pred_lengths
from network import MODEL_REGISTRY, DEFAULT_EXPERTS
from network.moe import MoEDiffuser


def _expert_smoke_test(expert: torch.nn.Module, C: int, L: int, device: torch.device, context_len: int | None = None):
    expert.eval()
    past_len = int(context_len or L) if getattr(expert, "uses_full_context", False) else L
    with torch.no_grad():
        x = torch.randn(2, C, L, device=device)
        past = torch.randn(2, C, past_len, device=device)
        t = torch.randint(0, 1000, (2,), device=device).long()
        y = expert(x, t, past)

    if not isinstance(y, torch.Tensor):
        raise TypeError("expert output is not a Tensor")
    if y.shape != x.shape:
        raise ValueError(f"expert output shape {tuple(y.shape)} != expected {tuple(x.shape)}")
    expert.train()


def build_dataset(DatasetClass, seq_len, csv_path, mode, train_ds=None, **extra_kwargs):
    sig = inspect.signature(DatasetClass.__init__)
    params = sig.parameters

    kwargs = {}

    if "seq_len" in params:
        kwargs["seq_len"] = int(seq_len)
    if "context_len" in params:
        kwargs["context_len"] = int(extra_kwargs.get("context_len", seq_len))
    if "pred_len" in params:
        kwargs["pred_len"] = int(extra_kwargs.get("pred_len", seq_len))
    if "csv_file_path" in params:
        kwargs["csv_file_path"] = csv_path
    if "mode" in params:
        kwargs["mode"] = mode

    for k, v in extra_kwargs.items():
        if k in {"context_len", "pred_len"}:
            continue
        if k in params:
            kwargs[k] = v

    if train_ds is not None:
        if "scaler" in params and hasattr(train_ds, "scaler"):
            kwargs["scaler"] = train_ds.scaler
        if "feature_cols" in params and hasattr(train_ds, "feature_cols"):
            kwargs["feature_cols"] = train_ds.feature_cols
        if "target_cols" in params and hasattr(train_ds, "target_cols"):
            kwargs["target_cols"] = train_ds.target_cols
        if "time_col" in params and hasattr(train_ds, "time_col") and train_ds.time_col is not None:
            kwargs["time_col"] = train_ds.time_col

    return DatasetClass(**kwargs)


def unpack_batch(batch, device):
    if isinstance(batch, dict):
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        known_future = batch.get("known_future", None)
        if known_future is not None:
            known_future = known_future.to(device)
        coarse_label = batch.get("coarse_label", None)
        if coarse_label is not None:
            coarse_label = coarse_label.to(device).long()
            if coarse_label.dim() == 0:
                coarse_label = coarse_label.view(1)
        return past, future, coarse_label, known_future
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        past, future, coarse_label = batch
        coarse_label = coarse_label.to(device).long()
        if coarse_label.dim() == 0:
            coarse_label = coarse_label.view(1)
        return past.to(device), future.to(device), coarse_label, None
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        past, future = batch
        return past.to(device), future.to(device), None, None
    raise RuntimeError("Unexpected batch format")


def unpack_sample(item, device):
    if isinstance(item, dict):
        past = item["past"].to(device)
        future = item["future"].to(device)
        known_future = item.get("known_future", None)
        if known_future is not None:
            known_future = known_future.to(device)
        coarse_label = item.get("coarse_label", None)
        if coarse_label is not None:
            coarse_label = coarse_label.to(device).long()
            if coarse_label.dim() == 0:
                coarse_label = coarse_label.view(1)
        return past, future, coarse_label, known_future
    if isinstance(item, (list, tuple)) and len(item) == 3:
        past, future, coarse_label = item
        coarse_label = coarse_label.to(device).long()
        if coarse_label.dim() == 0:
            coarse_label = coarse_label.view(1)
        return past.to(device), future.to(device), coarse_label, None
    if isinstance(item, (list, tuple)) and len(item) == 2:
        past, future = item
        return past.to(device), future.to(device), None, None
    raise RuntimeError("Unexpected sample format")


def forward_diffusion_model(
    model,
    noisy_future,
    t,
    past,
    coarse_label=None,
    known_future=None,
    return_weights: bool = False,
):
    kwargs = {"return_weights": return_weights}
    if coarse_label is not None:
        kwargs["coarse_label"] = coarse_label
    if known_future is not None:
        kwargs["known_future"] = known_future

    forward = getattr(model, "forward", model)
    sig = inspect.signature(forward)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not accepts_kwargs:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    return model(noisy_future, t, past, **kwargs)


def get_model_instance(config, num_channels, device, target_indices=None, known_future_channels: int = 0):
    model_name = config["model"]
    _, pred_len = resolve_context_pred_lengths(config)

    if model_name == "MoEDiffuser":
        expert_names = config.get("moe_experts", DEFAULT_EXPERTS)

        print(f"\n⚡ [MoE Assembly] 当前启用专家: {expert_names}")
        expert_instances = []

        for name in expert_names:
            if name not in MODEL_REGISTRY:
                raise ValueError(f"❌ moe_experts 中的模型未注册: {name}")

            ModelClass = MODEL_REGISTRY[name]
            print(f"   -> 添加专家: {name}")

            expert_kwargs = (config.get("expert_kwargs") or {}).get(name) or {}
            instance = instantiate_model(
                ModelClass,
                seq_len=pred_len,
                num_channels=num_channels,
                device=device,
                extra_kwargs=expert_kwargs,
            )
            _expert_smoke_test(
                instance,
                C=num_channels,
                L=pred_len,
                device=device,
                context_len=resolve_context_pred_lengths(config)[0],
            )
            expert_instances.append(instance)

        if len(expert_instances) == 0:
            raise RuntimeError("❌ 没有可用专家，无法启动 MoE。")

        return MoEDiffuser(
            expert_instances=expert_instances,
            seq_length=pred_len,
            num_channels=num_channels,
            device=device,
            target_indices=target_indices,
            known_future_channels=int(max(0, known_future_channels)),
            use_rule_coarse=config.get("use_rule_coarse", True),
            coarse_mix_alpha=float(config.get("coarse_mix_alpha", 1.0)),
            rule_table=config.get("rule_table", None),
            channel_expert_prior=config.get("channel_expert_prior", None),
            channel_prior_alpha=float(config.get("channel_prior_alpha", 0.0)),
            use_channel_stats=bool(config.get("channel_gate_use_stats", True)),
            channel_stat_hidden_dim=int(config.get("channel_gate_stat_hidden_dim", 32)),
            use_channel_embedding=bool(config.get("channel_gate_use_embedding", True)),
            coarse_top_k=config.get("coarse_top_k", None),
            final_top_k=config.get("final_top_k", 2),
            hard_final_routing=bool(config.get("hard_final_routing", False)),
            hard_routing_straight_through=bool(config.get("hard_routing_straight_through", True)),
            gate_temperature=float(config.get("gate_temperature", 1.0)),
            soft_retire_scale=float(config.get("soft_retire_scale", 0.05)),
            min_alive_experts=int(config.get("min_alive_experts", 3)),
            use_task_heads=bool(config.get("use_task_heads", False)),
            task_head_hidden_dim=int(config.get("task_head_hidden_dim", 16)),
            uniform_ensemble=bool(config.get("uniform_ensemble", False)),
            use_context_compressor=bool(config.get("use_context_compressor", True)),
        ).to(device)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"❌ 未知模型: {model_name}")

    ModelClass = MODEL_REGISTRY[model_name]
    return instantiate_model(ModelClass, seq_len=pred_len, num_channels=num_channels, device=device)
