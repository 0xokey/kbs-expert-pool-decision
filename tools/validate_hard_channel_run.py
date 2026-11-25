import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_hard_channel_run(run_dir: Path):
    config = _read_json(run_dir / "config.json")
    return isinstance(config, dict) and str(config.get("routing_mode", "")).lower() == "hard_channel"


def _run_report_row(run_dir: Path, failures):
    config = _read_json(run_dir / "config.json") or {}
    summary = _read_json(run_dir / "evaluation" / "Final" / "metrics" / "summary.json") or {}
    profiles = _read_json(run_dir / "expert_diffusion_profiles.json") or {}
    global_diffusion = profiles.get("global_diffusion", {}) if isinstance(profiles, dict) else {}
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "run_tag": config.get("run_tag", ""),
        "config_model": config.get("model", ""),
        "routing_mode": config.get("routing_mode", ""),
        "expert_diffusion_profile_policy": config.get(
            "expert_diffusion_profile_policy",
            global_diffusion.get("profile_policy", ""),
        ),
        "target_only_diffusion": config.get("target_only_diffusion", ""),
        "hard_final_routing": config.get("hard_final_routing", ""),
        "final_top_k": config.get("final_top_k", ""),
        "num_experts": len(config.get("moe_experts") or []),
        "forecast_objective": summary.get("forecast_objective", config.get("forecast_objective", "")),
        "num_windows": summary.get("num_windows", ""),
        "RMSE": summary.get("RMSE", ""),
        "R2": summary.get("R2", ""),
        "status": "OK" if not failures else "FAILED",
        "num_failures": int(len(failures)),
        "failures": " | ".join(failures),
    }


def _check_channel_best_json_csv_consistency(run_dir: Path, best_rows, failures):
    payload = _read_json(run_dir / "channel_best_expert.json")
    if not isinstance(payload, dict):
        failures.append("channel_best_expert.json is not an object")
        return
    channels = payload.get("channels")
    if not isinstance(channels, list):
        failures.append("channel_best_expert.json missing channels list")
        return
    if best_rows and len(channels) != len(best_rows):
        failures.append(f"channel_best_expert.json channels={len(channels)} csv_rows={len(best_rows)}")
    csv_by_pos = {str(row.get("channel_pos", "")): row for row in best_rows}
    for channel in channels:
        if not isinstance(channel, dict):
            failures.append("channel_best_expert.json contains non-object channel")
            continue
        channel_pos = str(channel.get("channel_pos", ""))
        csv_row = csv_by_pos.get(channel_pos)
        if csv_row is None:
            failures.append(f"channel_best_expert.json channel_pos={channel_pos} missing in csv")
            continue
        for key in [
            "channel_name",
            "prior_expert_name",
            "top_selected_expert_name",
            "top_dense_expert_name",
            "top_contribution_expert_name",
        ]:
            if str(channel.get(key, "")) != str(csv_row.get(key, "")):
                failures.append(f"channel_best_expert.json {key} differs from csv at channel_pos={channel_pos}")
        for key in [
            "prior_weight",
            "prior_avg_mse_loss",
            "top_selected_freq",
            "top_dense_weight",
            "top_contribution_score",
            "phase1_recon_mse",
            "phase2_recon_mse",
        ]:
            lhs = _float_or_none(channel.get(key))
            rhs = _float_or_none(csv_row.get(key))
            if lhs is None or rhs is None:
                failures.append(f"channel_best_expert.json {key} is not numeric at channel_pos={channel_pos}")
            elif abs(lhs - rhs) > 1e-6:
                failures.append(f"channel_best_expert.json {key} differs from csv at channel_pos={channel_pos}")


def _check_expert_diffusion_profiles(run_dir: Path, config: dict, failures):
    payload = _read_json(run_dir / "expert_diffusion_profiles.json")
    if not isinstance(payload, dict):
        failures.append("missing or invalid expert_diffusion_profiles.json")
        return
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        failures.append("expert_diffusion_profiles.json missing profiles list")
        return
    architecture_state = payload.get("architecture_state", {})
    if not isinstance(architecture_state, dict):
        failures.append("expert_diffusion_profiles architecture_state is not an object")
    else:
        if str(architecture_state.get("expert_selection", "")) != "hard_one_expert_per_channel":
            failures.append("expert_diffusion_profiles architecture_state expert_selection is not hard_one_expert_per_channel")
        if str(architecture_state.get("routing_optimization_status", "")) != "deferred":
            failures.append("expert_diffusion_profiles architecture_state routing_optimization_status is not deferred")
        if str(architecture_state.get("target_expert_integration", "")) != "full_diffusion_expert_per_channel":
            failures.append("expert_diffusion_profiles architecture_state target_expert_integration is not full_diffusion_expert_per_channel")
    configured_experts = list(config.get("moe_experts") or [])
    if len(profiles) != len(configured_experts):
        failures.append(f"expert_diffusion_profiles count={len(profiles)} expected={len(configured_experts)}")
    by_name = {str(profile.get("expert_name", "")): profile for profile in profiles if isinstance(profile, dict)}
    global_diffusion = payload.get("global_diffusion", {})
    if not isinstance(global_diffusion, dict):
        failures.append("expert_diffusion_profiles global_diffusion is not an object")
        global_diffusion = {}
    expected_policy = str(config.get("expert_diffusion_profile_policy", "require_global_alignment"))
    if str(global_diffusion.get("profile_policy", "")) != expected_policy:
        failures.append("expert_diffusion_profiles global profile_policy differs from config")
    expected_noise_schedule = str(config.get("noise_schedule", "linear"))
    expected_num_timesteps = int(global_diffusion.get("num_timesteps", 1000) or 1000)
    expected_ddim_t_start = config.get("ddim_t_start", None)
    expected_final_steps = config.get("final_eval_inference_steps", config.get("eval_inference_steps", 100))
    expected_final_eta = config.get("final_eval_eta", config.get("eval_eta", 0.0))
    for name in configured_experts:
        profile = by_name.get(str(name))
        if profile is None:
            failures.append(f"expert_diffusion_profiles missing expert={name}")
            continue
        pred_type = str(config.get("prediction_type", "epsilon")).lower()
        expected_objective = "noise_prediction" if pred_type == "epsilon" else f"{pred_type}_prediction"
        if str(profile.get("training_objective", "")) != expected_objective:
            failures.append(
                f"expert_diffusion_profiles {name} training_objective != {expected_objective}"
            )
        if str(profile.get("scheduler_scope", "")) != "shared_global":
            failures.append(f"expert_diffusion_profiles {name} scheduler_scope is not shared_global")
        if profile.get("shared_scheduler_aligned", False) is not True:
            failures.append(f"expert_diffusion_profiles {name} is not marked shared_scheduler_aligned")
        if str(profile.get("effective_diffusion_protocol", "")) != "shared_global_scheduler_expert_denoiser":
            failures.append(f"expert_diffusion_profiles {name} has unexpected effective_diffusion_protocol")
        if str(profile.get("current_integration_level", "")) != "denoiser_extracted":
            failures.append(f"expert_diffusion_profiles {name} current_integration_level is not denoiser_extracted")
        if str(profile.get("target_integration_level", "")) != "full_diffusion_expert":
            failures.append(f"expert_diffusion_profiles {name} target_integration_level is not full_diffusion_expert")
        if not profile.get("source_reference_path"):
            failures.append(f"expert_diffusion_profiles {name} missing source_reference_path")
        if not isinstance(profile.get("missing_full_protocol_operations", None), list):
            failures.append(f"expert_diffusion_profiles {name} missing_full_protocol_operations is not a list")
        if str(profile.get("noise_schedule", "")) != expected_noise_schedule:
            failures.append(f"expert_diffusion_profiles {name} noise_schedule differs from config")
        if int(profile.get("num_timesteps", -1)) != expected_num_timesteps:
            failures.append(f"expert_diffusion_profiles {name} num_timesteps differs from global diffusion")
        if not profile.get("preserved_operations"):
            failures.append(f"expert_diffusion_profiles {name} has empty preserved_operations")
        sampling = profile.get("sampling", {})
        if not isinstance(sampling, dict):
            failures.append(f"expert_diffusion_profiles {name} sampling is not an object")
        else:
            if sampling.get("ddim_t_start", None) != expected_ddim_t_start:
                failures.append(f"expert_diffusion_profiles {name} sampling.ddim_t_start differs from config")
            if int(sampling.get("inference_steps", -1)) != int(expected_final_steps):
                failures.append(f"expert_diffusion_profiles {name} sampling.inference_steps differs from config")
            if abs(float(sampling.get("eta", 0.0)) - float(expected_final_eta)) > 1e-12:
                failures.append(f"expert_diffusion_profiles {name} sampling.eta differs from config")

    if str(global_diffusion.get("noise_schedule", "")) != expected_noise_schedule:
        failures.append("expert_diffusion_profiles global noise_schedule differs from config")
    if global_diffusion.get("ddim_t_start", None) != expected_ddim_t_start:
        failures.append("expert_diffusion_profiles global ddim_t_start differs from config")
    pred_type_global = str(config.get("prediction_type", "epsilon")).lower()
    expected_global_objective = "noise_prediction" if pred_type_global == "epsilon" else f"{pred_type_global}_prediction"
    if str(global_diffusion.get("training_objective", "")) != expected_global_objective:
        failures.append(
            f"expert_diffusion_profiles global training_objective != {expected_global_objective}"
        )


def _failures_for_run(run_dir: Path):
    failures = []
    config = _read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        return ["missing config.json"]

    if str(config.get("routing_mode", "")).lower() != "hard_channel":
        failures.append("config.routing_mode is not hard_channel")
    if not bool(config.get("hard_final_routing", False)):
        failures.append("config.hard_final_routing is not true")
    if int(config.get("final_top_k", 0)) != 1:
        failures.append("config.final_top_k is not 1")
    if not bool(config.get("target_only_diffusion", False)):
        failures.append("config.target_only_diffusion is not true")
    legacy_experts = {"CSDIDiffuser", "S4Diffuser"}
    configured_experts = list(config.get("moe_experts") or [])
    used_legacy = sorted(legacy_experts.intersection(set(configured_experts)))
    if used_legacy:
        failures.append(f"config.moe_experts contains legacy aliases: {used_legacy}")

    gate_path = run_dir / "gate_final.npy"
    gate = None
    if not gate_path.exists():
        failures.append("missing gate_final.npy")
    else:
        gate = np.load(gate_path)
        if gate.ndim != 3:
            failures.append(f"gate_final.npy should be 3D, got {gate.shape}")
        else:
            if not np.allclose(gate.sum(axis=-1), 1.0):
                failures.append("gate_final rows do not sum to 1")
            if not np.all((gate == 0.0) | (gate == 1.0)):
                failures.append("gate_final contains values other than 0/1")
            if int((gate > 0).sum(axis=-1).min()) != 1 or int((gate > 0).sum(axis=-1).max()) != 1:
                failures.append("gate_final is not exactly one-hot per sample/channel")

    required_root = [
        "channel_expert_prior.csv",
        "channel_train_loss.csv",
        "channel_best_expert.csv",
        "channel_best_expert.json",
        "expert_diffusion_profiles.json",
        "gate_task_summary.csv",
    ]
    for rel in required_root:
        if not (run_dir / rel).exists():
            failures.append(f"missing {rel}")

    gate_summary_rows = _read_csv(run_dir / "gate_task_summary.csv")
    if gate_summary_rows and gate is not None and gate.ndim == 3:
        task_ids = sorted(
            {
                value
                for value in (_int_or_none(row.get("task_idx")) for row in gate_summary_rows)
                if value is not None
            }
        )
        expert_ids = sorted(
            {
                value
                for value in (_int_or_none(row.get("expert_idx")) for row in gate_summary_rows)
                if value is not None
            }
        )
        if len(task_ids) != int(gate.shape[1]):
            failures.append(f"gate_task_summary task count={len(task_ids)} expected={int(gate.shape[1])}")
        if len(expert_ids) != int(gate.shape[2]):
            failures.append(f"gate_task_summary expert count={len(expert_ids)} expected={int(gate.shape[2])}")
        selection_by_task = {}
        for row in gate_summary_rows:
            task_idx = _int_or_none(row.get("task_idx"))
            if task_idx is None:
                continue
            selection = _float_or_none(row.get("selection_freq", row.get("selected_freq")))
            if selection is None:
                failures.append(f"gate_task_summary selection_freq is not numeric at task_idx={task_idx}")
                continue
            selection_by_task[task_idx] = selection_by_task.get(task_idx, 0.0) + selection
        for task_idx, total in selection_by_task.items():
            if abs(total - 1.0) > 1e-6:
                failures.append(f"gate_task_summary selection_freq sum={total:.6f} at task_idx={task_idx}")

    final_metrics_dir = run_dir / "evaluation" / "Final" / "metrics"
    manifest = _read_json(final_metrics_dir / "metrics_manifest.json")
    if not isinstance(manifest, dict):
        failures.append("missing Final metrics_manifest.json")
    else:
        for key in [
            "channel_expert_selection",
            "channel_expert_health",
            "channel_expert_summary",
            "expert_health",
        ]:
            value = manifest.get(key)
            if not value:
                failures.append(f"manifest missing {key}")
            elif not (final_metrics_dir / value).exists():
                failures.append(f"manifest file missing for {key}: {value}")

    best_rows = _read_csv(run_dir / "channel_best_expert.csv")
    if (run_dir / "channel_best_expert.json").exists():
        _check_channel_best_json_csv_consistency(run_dir, best_rows, failures)
    if (run_dir / "expert_diffusion_profiles.json").exists():
        _check_expert_diffusion_profiles(run_dir, config, failures)
    if best_rows and gate_path.exists():
        gate = np.load(gate_path)
        expected_channels = int(gate.shape[1]) if gate.ndim == 3 else None
        if expected_channels is not None and len(best_rows) != expected_channels:
            failures.append(
                f"channel_best_expert.csv rows={len(best_rows)} expected={expected_channels}"
            )
    final_summary_rows = _read_csv(final_metrics_dir / "channel_expert_summary.csv")
    if best_rows and final_summary_rows:
        final_by_pos = {str(row.get("channel_pos", "")): row for row in final_summary_rows}
        for row in best_rows:
            channel_pos = str(row.get("channel_pos", ""))
            final_row = final_by_pos.get(channel_pos)
            if final_row is None:
                failures.append(f"channel_best_expert channel_pos={channel_pos} missing in Final summary")
                continue
            for key in [
                "top_selected_expert_name",
                "top_dense_expert_name",
                "top_contribution_expert_name",
            ]:
                if str(row.get(key, "")) != str(final_row.get(key, "")):
                    failures.append(f"channel_best_expert {key} differs from Final summary at channel_pos={channel_pos}")
            for key in ["top_selected_freq", "top_dense_weight", "top_contribution_score"]:
                lhs = _float_or_none(row.get(key))
                rhs = _float_or_none(final_row.get(key))
                if lhs is None or rhs is None:
                    failures.append(f"channel_best_expert {key} is not numeric at channel_pos={channel_pos}")
                elif abs(lhs - rhs) > 1e-6:
                    failures.append(f"channel_best_expert {key} differs from Final summary at channel_pos={channel_pos}")

    selection_rows = _read_csv(final_metrics_dir / "channel_expert_selection.csv")
    if selection_rows:
        bad = [row for row in selection_rows if str(row.get("is_hard_one_hot", "")).lower() != "true"]
        if bad:
            failures.append("channel_expert_selection has non-hard rows")
        if gate is not None and gate.ndim == 3:
            expected_channels = int(gate.shape[1])
            expected_experts = int(gate.shape[2])
            sample_ids = {
                value
                for value in (_int_or_none(row.get("sample_idx")) for row in selection_rows)
                if value is not None
            }
            channel_positions = {
                value
                for value in (_int_or_none(row.get("channel_pos")) for row in selection_rows)
                if value is not None
            }
            if len(selection_rows) != len(sample_ids) * expected_channels:
                failures.append(
                    "channel_expert_selection row count="
                    f"{len(selection_rows)} expected={len(sample_ids) * expected_channels}"
                )
            if channel_positions != set(range(expected_channels)):
                failures.append(
                    "channel_expert_selection channel_pos coverage="
                    f"{sorted(channel_positions)} expected={list(range(expected_channels))}"
                )
            for row in selection_rows:
                channel_pos = _int_or_none(row.get("channel_pos"))
                expert_idx = _int_or_none(row.get("selected_expert_idx"))
                if channel_pos is None or channel_pos < 0 or channel_pos >= expected_channels:
                    failures.append(f"channel_expert_selection invalid channel_pos={row.get('channel_pos')}")
                if expert_idx is None or expert_idx < 0 or expert_idx >= expected_experts:
                    failures.append(f"channel_expert_selection invalid selected_expert_idx={row.get('selected_expert_idx')}")
                selected_weight = _float_or_none(row.get("selected_weight"))
                if selected_weight is None or abs(selected_weight - 1.0) > 1e-6:
                    failures.append(
                        "channel_expert_selection selected_weight "
                        f"expected 1.0 got {row.get('selected_weight')}"
                    )
                dense_weight = _float_or_none(row.get("selected_dense_weight"))
                if dense_weight is None or dense_weight < -1e-6 or dense_weight > 1.0 + 1e-6:
                    failures.append(
                        "channel_expert_selection selected_dense_weight out of range "
                        f"got {row.get('selected_dense_weight')}"
                    )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate hard-channel run artifacts.")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--results-dir", default=None, help="Validate all hard-channel runs under this directory.")
    parser.add_argument("--out", default=None, help="Optional CSV validation report path for batch mode.")
    parser.add_argument("--allow-failed", action="store_true", help="In batch mode, write the report but do not exit non-zero for failed runs.")
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        rows = []
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir() or not _is_hard_channel_run(run_dir):
                continue
            failures = _failures_for_run(run_dir)
            rows.append(_run_report_row(run_dir, failures))
        out = Path(args.out) if args.out else results_dir / "hard_channel_validation_report.csv"
        _write_csv(
            out,
            rows,
            [
                "run_name",
                "run_dir",
                "run_tag",
                "config_model",
                "routing_mode",
                "expert_diffusion_profile_policy",
                "target_only_diffusion",
                "hard_final_routing",
                "final_top_k",
                "num_experts",
                "forecast_objective",
                "num_windows",
                "RMSE",
                "R2",
                "status",
                "num_failures",
                "failures",
            ],
        )
        failed = [row for row in rows if row["status"] != "OK"]
        print(f"Validated {len(rows)} hard-channel runs. Report: {out}")
        if failed:
            print(f"FAILED runs: {len(failed)}")
            if not args.allow_failed:
                raise SystemExit(1)
        return

    if args.run_dir is None:
        parser.error("run_dir is required unless --results-dir is provided")

    run_dir = Path(args.run_dir)
    failures = _failures_for_run(run_dir)
    if failures:
        print(f"FAILED: {run_dir}")
        for item in failures:
            print(f" - {item}")
        raise SystemExit(1)
    print(f"OK: {run_dir}")


if __name__ == "__main__":
    main()
