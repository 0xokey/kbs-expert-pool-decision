def _as_int(config: dict, key: str, default=None, *, minimum: int | None = None):
    raw = config.get(key, default)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[Config] {key} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"[Config] {key} must be >= {minimum}, got {value}")
    return value


def _as_float(config: dict, key: str, default=None, *, minimum: float | None = None):
    raw = config.get(key, default)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[Config] {key} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"[Config] {key} must be >= {minimum}, got {value}")
    return value


def _sampler(config: dict, key: str, fallback: str = "ddim") -> str:
    value = str(config.get(key, fallback)).strip().lower()
    if value not in {"ddpm", "ddim"}:
        raise ValueError(f"[Config] {key} must be 'ddpm' or 'ddim', got {value!r}")
    config[key] = value
    return value


SUPPORTED_EXPERT_DIFFUSION_PROFILE_POLICIES = {"require_global_alignment"}


def validate_config(config: dict, *, mode: str = "train", num_timesteps: int = 1000) -> dict:
    """Validate and normalize high-risk runtime options in-place."""
    if not isinstance(config, dict):
        raise TypeError("[Config] config must be a dict")

    context_len = _as_int(config, "context_len", config.get("seq_len", 168), minimum=1)
    pred_len = _as_int(config, "pred_len", config.get("seq_len", 168), minimum=1)
    seq_len = _as_int(config, "seq_len", pred_len, minimum=1)
    config["context_len"] = int(context_len)
    config["pred_len"] = int(pred_len)
    config["seq_len"] = int(seq_len)

    _as_int(config, "batch_size", config.get("batch_size", 1), minimum=1)
    _as_int(config, "eval_batch_size", config.get("eval_batch_size", 1), minimum=1)
    _as_int(config, "debug_max_train_windows", None, minimum=2)
    _as_int(config, "phase0_rule_t_values", config.get("phase0_rule_t_values", 5), minimum=1)
    _as_int(config, "phase0_prior_max_windows", config.get("phase0_prior_max_windows", 256), minimum=1)
    _as_int(config, "final_top_k", config.get("final_top_k", 1), minimum=1)
    _as_int(config, "coarse_top_k", config.get("coarse_top_k", None), minimum=1)
    _as_int(config, "crps_num_windows", config.get("crps_num_windows", 16), minimum=1)
    _as_int(config, "crps_samples", config.get("crps_samples", 0), minimum=0)
    _as_int(config, "eval_ensemble_samples", config.get("eval_ensemble_samples", 1), minimum=1)
    prediction_type = str(config.get("prediction_type", "epsilon") or "epsilon").strip().lower()
    if prediction_type not in {"epsilon", "x0"}:
        raise ValueError(f"[Config] prediction_type must be 'epsilon' or 'x0', got {prediction_type!r}")
    config["prediction_type"] = prediction_type

    training_mode = str(config.get("training_mode", "diffusion") or "diffusion").strip().lower()
    if training_mode not in {"diffusion", "regression"}:
        raise ValueError(f"[Config] training_mode must be 'diffusion' or 'regression', got {training_mode!r}")
    config["training_mode"] = training_mode
    inference_mode = str(config.get("inference_mode", "diffusion") or "diffusion").strip().lower()
    if inference_mode not in {"diffusion", "direct"}:
        raise ValueError(f"[Config] inference_mode must be 'diffusion' or 'direct', got {inference_mode!r}")
    config["inference_mode"] = inference_mode

    scaler_type = str(config.get("scaler_type", "minmax") or "minmax").strip().lower()
    if scaler_type not in {"minmax", "standard"}:
        raise ValueError(f"[Config] scaler_type must be 'minmax' or 'standard', got {scaler_type!r}")
    config["scaler_type"] = scaler_type

    split_ratios = config.get("split_ratios", None)
    if split_ratios is not None:
        ratios = [float(v) for v in split_ratios]
        if len(ratios) != 3 or any(v <= 0 for v in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"[Config] split_ratios must be 3 positive values summing to 1, got {split_ratios}")
        config["split_ratios"] = ratios

    split_borders = config.get("split_borders", None)
    if split_borders is not None:
        borders = [int(v) for v in split_borders]
        if len(borders) not in {2, 3} or any(v <= 0 for v in borders):
            raise ValueError(
                f"[Config] split_borders must be [train_end, val_end(, test_end)] positive ints, got {split_borders}"
            )
        if any(borders[i] >= borders[i + 1] for i in range(len(borders) - 1)):
            raise ValueError(f"[Config] split_borders must be strictly increasing, got {split_borders}")
        config["split_borders"] = borders
    seasonal_period = _as_int(config, "seasonal_period", config.get("seasonal_period", 24), minimum=1)
    _as_float(config, "eval_eta", config.get("eval_eta", 0.0), minimum=0.0)
    _as_float(config, "final_eval_eta", config.get("final_eval_eta", config.get("eval_eta", 0.0)), minimum=0.0)
    _as_float(config, "predict_eta", config.get("predict_eta", config.get("final_eval_eta", 0.0)), minimum=0.0)
    _as_float(config, "lambda_channel_prior", config.get("lambda_channel_prior", 0.0), minimum=0.0)
    _as_float(config, "phase0_prior_temperature", config.get("phase0_prior_temperature", 0.1), minimum=1e-8)
    _as_float(config, "phase0_prior_margin", config.get("phase0_prior_margin", 0.05), minimum=0.0)
    prior_winrate_threshold = _as_float(
        config,
        "phase0_prior_winrate_threshold",
        config.get("phase0_prior_winrate_threshold", 0.6),
        minimum=0.0,
    )
    if prior_winrate_threshold is not None and prior_winrate_threshold > 1.0:
        raise ValueError(
            "[Config] phase0_prior_winrate_threshold must be <= 1.0, "
            f"got {prior_winrate_threshold}"
        )
    prior_hard_mix = _as_float(
        config,
        "phase0_prior_hard_mix",
        config.get("phase0_prior_hard_mix", 0.7),
        minimum=0.0,
    )
    if prior_hard_mix is not None and prior_hard_mix > 1.0:
        raise ValueError(f"[Config] phase0_prior_hard_mix must be <= 1.0, got {prior_hard_mix}")
    channel_prior_alpha = _as_float(
        config,
        "channel_prior_alpha",
        config.get("channel_prior_alpha", 0.0),
        minimum=0.0,
    )
    if channel_prior_alpha is not None and channel_prior_alpha > 1.0:
        raise ValueError(f"[Config] channel_prior_alpha must be <= 1.0, got {channel_prior_alpha}")

    eval_sampler = _sampler(config, "eval_sampler", "ddim")
    final_sampler = _sampler(config, "final_eval_sampler", eval_sampler)
    predict_sampler = _sampler(config, "predict_sampler", final_sampler)
    active_samplers = {eval_sampler, final_sampler}
    if mode == "predict":
        active_samplers.add(predict_sampler)

    profile_policy = str(
        config.get("expert_diffusion_profile_policy", "require_global_alignment")
    ).strip().lower()
    if profile_policy not in SUPPORTED_EXPERT_DIFFUSION_PROFILE_POLICIES:
        raise ValueError(
            "[Config] expert_diffusion_profile_policy currently supports only "
            f"{sorted(SUPPORTED_EXPERT_DIFFUSION_PROFILE_POLICIES)}, got {profile_policy!r}"
        )
    config["expert_diffusion_profile_policy"] = profile_policy

    routing_mode = str(config.get("routing_mode", "") or "").strip().lower()
    if routing_mode:
        if routing_mode not in {"hard_channel", "soft_topk", "legacy"}:
            raise ValueError(
                "[Config] routing_mode must be 'hard_channel', 'soft_topk', or 'legacy', "
                f"got {routing_mode!r}"
            )
        config["routing_mode"] = routing_mode
        if routing_mode == "hard_channel":
            config["hard_final_routing"] = True
            config["final_top_k"] = 1
            legacy_experts = {"CSDIDiffuser", "S4Diffuser"}
            configured_experts = set(config.get("moe_experts") or [])
            used_legacy = sorted(legacy_experts.intersection(configured_experts))
            if used_legacy:
                raise ValueError(
                    "[Config] hard_channel runs must use current expert names; "
                    f"legacy aliases are only allowed with routing_mode=legacy: {used_legacy}"
                )
    _as_int(config, "channel_gate_stat_hidden_dim", config.get("channel_gate_stat_hidden_dim", 32), minimum=1)

    _as_int(config, "eval_inference_steps", config.get("eval_inference_steps", 100), minimum=1)
    _as_int(
        config,
        "final_eval_inference_steps",
        config.get("final_eval_inference_steps", config.get("eval_inference_steps", 100)),
        minimum=1,
    )
    _as_int(
        config,
        "predict_inference_steps",
        config.get("predict_inference_steps", config.get("final_eval_inference_steps", 100)),
        minimum=1,
    )

    if "ddim" in active_samplers:
        t_start = config.get("ddim_t_start", None)
        if t_start is None and bool(config.get("require_ddim_t_start", True)):
            raise ValueError(
                "[Config] ddim_t_start is required for DDIM sampling. "
                "Set it explicitly, e.g. 800, or set require_ddim_t_start=false for legacy reproduction."
            )
        if t_start is not None:
            t_start = _as_int(config, "ddim_t_start", minimum=0)
            if t_start >= int(num_timesteps):
                raise ValueError(
                    f"[Config] ddim_t_start must be < num_timesteps ({num_timesteps}), got {t_start}"
                )
            config["ddim_t_start"] = int(t_start)

    allow_legacy_feature_config = bool(config.get("allow_legacy_feature_config", False))
    if (
        bool(config.get("add_cyc_features", False))
        and not bool(config.get("add_time_features", False))
        and not allow_legacy_feature_config
    ):
        raise ValueError("[Config] add_cyc_features=true requires add_time_features=true.")

    if bool(config.get("hard_final_routing", False)):
        config["final_top_k"] = 1

    if bool(config.get("add_time_features", False)) and not bool(config.get("target_only_diffusion", False)):
        if bool(config.get("require_target_only_with_time_features", True)):
            raise ValueError(
                "[Config] add_time_features=true should be paired with target_only_diffusion=true "
                "so time features are used as known_future covariates instead of diffusion targets."
            )

    forecast_objective = str(config.get("forecast_objective", "direct") or "direct").strip().lower()
    aliases = {
        "none": "direct",
        "raw": "direct",
        "absolute": "direct",
        "seasonal_residual": "daily_seasonal_residual",
        "daily_residual": "daily_seasonal_residual",
    }
    forecast_objective = aliases.get(forecast_objective, forecast_objective)
    if forecast_objective not in {"direct", "daily_seasonal_residual"}:
        raise ValueError(
            "[Config] forecast_objective must be 'direct' or "
            f"'daily_seasonal_residual', got {config.get('forecast_objective')!r}"
        )
    if forecast_objective == "daily_seasonal_residual" and int(seasonal_period) > int(context_len):
        raise ValueError(
            "[Config] seasonal_period must be <= context_len when using "
            f"daily_seasonal_residual, got seasonal_period={seasonal_period}, context_len={context_len}"
        )
    config["forecast_objective"] = forecast_objective
    config["seasonal_period"] = int(seasonal_period)

    return config
