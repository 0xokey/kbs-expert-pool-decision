import os
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # GBK consoles cannot print the dataset's emoji banners
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.generic_csv import GenericCSVDataset
from network import DEFAULT_EXPERTS
from network.moe import _apply_hard_routing, make_default_rule_table
from train import _load_config_file, forward_diffusion_model, get_model_instance
from training.baselines import make_residual_target_bcl, restore_residual_forecast_bcl
from training.evaluation import run_evaluation
from training.loops import _export_expert_diffusion_profiles
from utils.config import validate_config
from utils.metrics import MetricsCalculator
from utils.scheduler import DDPMScheduler
from tools.validate_hard_channel_run import (
    _check_expert_diffusion_profiles,
    _check_channel_best_json_csv_consistency,
    _failures_for_run,
    _run_report_row,
)
from tools.report_hard_channel_runs import _is_validated, _read_json as _report_read_json, _should_refresh_summary
from predict import _resolve_channel_best_file


class ZeroNoiseModel(torch.nn.Module):
    def forward(self, x, t, condition, **kwargs):
        return torch.zeros_like(x)


class BadShapeModel(torch.nn.Module):
    def forward(self, x, t, condition, **kwargs):
        return torch.zeros(x.size(0), max(1, x.size(1) - 1), x.size(2), device=x.device)


class InternalTypeErrorModel(torch.nn.Module):
    def forward(self, x, t, condition, known_future=None):
        if known_future is not None:
            raise TypeError("internal known_future bug")
        return torch.zeros_like(x)


class TinyEvalDataset:
    context_len = 4
    pred_len = 4
    seq_len = 4
    num_channels = 2
    target_indices = [0, 1]
    target_cols = ["a", "b"]

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        base = float(idx)
        return {
            "past": torch.zeros(2, 4) + base,
            "future": torch.ones(2, 4) * (base + 1.0),
        }


class CoreSmokeTests(unittest.TestCase):
    def _make_csv(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        dates = pd.date_range("2024-01-01", periods=32, freq="h")
        data = {
            "date": dates,
            "HUFL": np.linspace(1.0, 2.0, 32),
            "HULL": np.linspace(2.0, 3.0, 32),
            "MUFL": np.linspace(3.0, 4.0, 32),
            "MULL": np.linspace(4.0, 5.0, 32),
            "LUFL": np.linspace(5.0, 6.0, 32),
            "LULL": np.linspace(6.0, 7.0, 32),
            "OT": np.linspace(7.0, 8.0, 32),
        }
        pd.DataFrame(data).to_csv(tmp.name, index=False)
        return tmp.name

    def _write_csv_rows(self, path, rows, fields):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    def test_validate_config_guards_critical_sampling_options(self):
        config = {
            "seq_len": 16,
            "context_len": 16,
            "pred_len": 16,
            "eval_sampler": "ddim",
            "final_eval_sampler": "ddim",
            "predict_sampler": "ddim",
            "eval_inference_steps": 4,
            "final_eval_inference_steps": 4,
            "predict_inference_steps": 4,
        }
        with self.assertRaises(ValueError):
            validate_config(dict(config), mode="train")

        config["ddim_t_start"] = 8
        validate_config(dict(config), mode="train")

        hard_config = dict(config, final_top_k=3, hard_final_routing=True)
        validate_config(hard_config, mode="train")
        self.assertEqual(hard_config["final_top_k"], 1)

        routing_config = dict(config, final_top_k=3, routing_mode="hard_channel")
        validate_config(routing_config, mode="train")
        self.assertTrue(routing_config["hard_final_routing"])
        self.assertEqual(routing_config["final_top_k"], 1)

        hard_legacy_alias = dict(config, routing_mode="hard_channel", moe_experts=["CSDIDiffuser"])
        with self.assertRaises(ValueError):
            validate_config(hard_legacy_alias, mode="train")

        bad_profile_policy = dict(config, expert_diffusion_profile_policy="per_expert_scheduler")
        with self.assertRaises(ValueError):
            validate_config(bad_profile_policy, mode="train")

        legacy_alias = dict(config, routing_mode="legacy", moe_experts=["CSDIDiffuser", "S4Diffuser"])
        validate_config(legacy_alias, mode="train")

        bad_debug_limit = dict(config, debug_max_train_windows=1)
        with self.assertRaises(ValueError):
            validate_config(bad_debug_limit, mode="train")

        bad_features = dict(config, add_time_features=False, add_cyc_features=True)
        with self.assertRaises(ValueError):
            validate_config(bad_features, mode="train")

        legacy_feature_config = dict(
            config,
            routing_mode="legacy",
            add_time_features=False,
            add_cyc_features=True,
            allow_legacy_feature_config=True,
        )
        validate_config(legacy_feature_config, mode="train")

        residual_config = dict(config, forecast_objective="daily_seasonal_residual", seasonal_period=24)
        with self.assertRaises(ValueError):
            validate_config(residual_config, mode="train")

    def test_explicit_missing_config_fails_fast(self):
        with self.assertRaises(FileNotFoundError):
            _load_config_file(os.path.join(tempfile.gettempdir(), "missing_mmoe_config.json"))

    def test_predict_channel_best_epoch_resolution(self):
        class Args:
            channel_best_file = None
            channel_best_epoch = "Final From Checkpoint"

        self.assertEqual(
            _resolve_channel_best_file(Args),
            "channel_best_expert_Final_From_Checkpoint.json",
        )

        Args.channel_best_file = "custom.json"
        self.assertEqual(_resolve_channel_best_file(Args), "custom.json")

    def test_report_treats_legacy_summary_without_status_as_valid(self):
        self.assertTrue(_is_validated({"validation_failures": ""}))
        self.assertTrue(_is_validated({"validation_status": "OK", "validation_failures": "ignored"}))
        self.assertFalse(_is_validated({"validation_status": "FAILED", "validation_failures": ""}))
        self.assertFalse(_is_validated({"validation_failures": "missing gate_final.npy"}))

    def test_report_validated_only_forces_summary_refresh(self):
        class Args:
            refresh_summary = False
            validated_only = True

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "run_summary.csv").write_text("run_name\nold\n", encoding="utf-8")
            self.assertTrue(_should_refresh_summary(Args, out_dir))

            Args.validated_only = False
            self.assertFalse(_should_refresh_summary(Args, out_dir))

            Args.refresh_summary = True
            self.assertTrue(_should_refresh_summary(Args, out_dir))

    def test_report_read_json_missing_file_defaults_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(_report_read_json(Path(tmpdir) / "missing.json"), {})

    def test_hard_channel_validation_catches_non_final_channel_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "routing_mode": "hard_channel",
                        "hard_final_routing": True,
                        "final_top_k": 1,
                        "target_only_diffusion": True,
                        "moe_experts": ["ExpertA", "ExpertB"],
                    }
                ),
                encoding="utf-8",
            )
            np.save(run_dir / "gate_final.npy", np.array([[[1.0, 0.0]]], dtype=np.float32))
            for name in ["channel_expert_prior.csv", "channel_train_loss.csv"]:
                (run_dir / name).write_text("ok\n", encoding="utf-8")
            self._write_csv_rows(
                run_dir / "gate_task_summary.csv",
                [
                    {"task_idx": 0, "expert_idx": 0, "selection_freq": 1.0},
                    {"task_idx": 0, "expert_idx": 1, "selection_freq": 0.0},
                ],
                ["task_idx", "expert_idx", "selection_freq"],
            )
            metrics_dir = run_dir / "evaluation" / "Final" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics_manifest.json").write_text(
                json.dumps(
                    {
                        "channel_expert_selection": "channel_expert_selection.csv",
                        "channel_expert_health": "channel_expert_health.csv",
                        "channel_expert_summary": "channel_expert_summary.csv",
                        "expert_health": "expert_health.csv",
                    }
                ),
                encoding="utf-8",
            )
            for name in ["channel_expert_health.csv", "expert_health.csv"]:
                (metrics_dir / name).write_text("ok\n", encoding="utf-8")

            summary_fields = [
                "channel_idx",
                "channel_pos",
                "channel_name",
                "top_selected_expert_name",
                "top_selected_freq",
                "top_dense_expert_name",
                "top_dense_weight",
                "top_contribution_expert_name",
                "top_contribution_score",
            ]
            final_row = {
                "channel_idx": 0,
                "channel_pos": 0,
                "channel_name": "a",
                "top_selected_expert_name": "ExpertA",
                "top_selected_freq": 1.0,
                "top_dense_expert_name": "ExpertA",
                "top_dense_weight": 0.75,
                "top_contribution_expert_name": "ExpertA",
                "top_contribution_score": 0.5,
            }
            self._write_csv_rows(metrics_dir / "channel_expert_summary.csv", [final_row], summary_fields)
            self._write_csv_rows(
                metrics_dir / "channel_expert_selection.csv",
                [{"is_hard_one_hot": "True"}],
                ["is_hard_one_hot"],
            )

            polluted_row = dict(final_row, top_dense_weight=0.25)
            self._write_csv_rows(run_dir / "channel_best_expert.csv", [polluted_row], summary_fields)
            (run_dir / "channel_best_expert.json").write_text(
                json.dumps({"channels": [polluted_row]}),
                encoding="utf-8",
            )

            failures = _failures_for_run(run_dir)
            self.assertTrue(any("top_dense_weight differs from Final summary" in item for item in failures))

    def test_hard_channel_validation_catches_channel_best_json_csv_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            csv_row = {
                "channel_pos": "0",
                "channel_name": "a",
                "prior_expert_name": "ExpertA",
                "prior_weight": "0.5",
                "prior_avg_mse_loss": "1.0",
                "top_selected_expert_name": "ExpertA",
                "top_selected_freq": "1.0",
                "top_dense_expert_name": "ExpertA",
                "top_dense_weight": "0.75",
                "top_contribution_expert_name": "ExpertA",
                "top_contribution_score": "0.25",
                "phase1_recon_mse": "0.1",
                "phase2_recon_mse": "0.2",
            }
            (run_dir / "channel_best_expert.json").write_text(
                json.dumps({"channels": [dict(csv_row, top_dense_weight=0.25)]}),
                encoding="utf-8",
            )
            failures = []
            _check_channel_best_json_csv_consistency(run_dir, [csv_row], failures)
            self.assertTrue(any("top_dense_weight differs from csv" in item for item in failures))

    def test_hard_channel_validation_catches_missing_profile_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = {
                "moe_experts": ["ExpertA"],
                "noise_schedule": "linear",
                "ddim_t_start": 800,
                "final_eval_inference_steps": 100,
                "final_eval_eta": 0.0,
            }
            (run_dir / "expert_diffusion_profiles.json").write_text(
                json.dumps(
                    {
                        "global_diffusion": {
                            "noise_schedule": "linear",
                            "num_timesteps": 1000,
                            "training_objective": "noise_prediction",
                            "ddim_t_start": 800,
                            "profile_policy": "require_global_alignment",
                        },
                        "profiles": [
                            {
                                "expert_name": "ExpertA",
                                "training_objective": "noise_prediction",
                                "scheduler_scope": "shared_global",
                                "noise_schedule": "linear",
                                "num_timesteps": 1000,
                                "shared_scheduler_aligned": True,
                                "effective_diffusion_protocol": "shared_global_scheduler_expert_denoiser",
                                "sampling": {"ddim_t_start": 800, "inference_steps": 100, "eta": 0.0},
                                "preserved_operations": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            failures = []
            _check_expert_diffusion_profiles(run_dir, config, failures)
            self.assertTrue(any("empty preserved_operations" in item for item in failures))

    def test_exported_expert_profiles_align_to_run_ddim_config(self):
        class DummyMoE(torch.nn.Module):
            def expert_diffusion_profiles(self):
                return [
                    {
                        "expert_name": "MambaExpert",
                        "training_objective": "noise_prediction",
                        "scheduler_scope": "shared_global",
                        "noise_schedule": "linear",
                        "num_timesteps": 1000,
                        "sampling": {"sampler": "ddim", "ddim_t_start": 800, "eta": 0.0},
                        "preserved_operations": ["selective_state_space_temporal_blocks"],
                        "source_reference_path": r"D:\桌面\MMoediffusionmodels\SSSD-S4 Mamba SSM",
                    }
                ]

        config = {
            "moe_experts": ["MambaExpert"],
            "noise_schedule": "linear",
            "ddim_t_start": 700,
            "eval_sampler": "ddim",
            "eval_inference_steps": 50,
            "final_eval_sampler": "ddim",
            "final_eval_inference_steps": 75,
            "eval_eta": 0.0,
            "final_eval_eta": 0.2,
            "expert_diffusion_profile_policy": "require_global_alignment",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _export_expert_diffusion_profiles(DummyMoE(), config, tmpdir)
            self.assertEqual(payload["architecture_state"]["expert_selection"], "hard_one_expert_per_channel")
            self.assertEqual(payload["architecture_state"]["routing_optimization_status"], "deferred")
            self.assertEqual(
                payload["architecture_state"]["target_expert_integration"],
                "full_diffusion_expert_per_channel",
            )
            profile = payload["profiles"][0]
            self.assertEqual(profile["sampling"]["ddim_t_start"], 700)
            self.assertEqual(profile["sampling"]["inference_steps"], 75)
            self.assertAlmostEqual(profile["sampling"]["eta"], 0.2)
            self.assertTrue(profile["shared_scheduler_aligned"])
            self.assertEqual(profile["current_integration_level"], "denoiser_extracted")
            self.assertEqual(profile["target_integration_level"], "full_diffusion_expert")
            self.assertIsInstance(profile["missing_full_protocol_operations"], list)
            failures = []
            _check_expert_diffusion_profiles(Path(tmpdir), config, failures)
            self.assertEqual(failures, [])

    def test_hard_channel_validation_catches_gate_summary_selection_sum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "routing_mode": "hard_channel",
                        "hard_final_routing": True,
                        "final_top_k": 1,
                        "target_only_diffusion": True,
                        "moe_experts": ["ExpertA", "ExpertB"],
                    }
                ),
                encoding="utf-8",
            )
            np.save(run_dir / "gate_final.npy", np.array([[[1.0, 0.0]]], dtype=np.float32))
            self._write_csv_rows(
                run_dir / "gate_task_summary.csv",
                [
                    {"task_idx": 0, "expert_idx": 0, "selection_freq": 0.25},
                    {"task_idx": 0, "expert_idx": 1, "selection_freq": 0.25},
                ],
                ["task_idx", "expert_idx", "selection_freq"],
            )
            failures = _failures_for_run(run_dir)
            self.assertTrue(any("gate_task_summary selection_freq sum=0.500000" in item for item in failures))

    def test_hard_channel_validation_catches_bad_selection_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "routing_mode": "hard_channel",
                        "hard_final_routing": True,
                        "final_top_k": 1,
                        "target_only_diffusion": True,
                        "moe_experts": ["ExpertA", "ExpertB"],
                    }
                ),
                encoding="utf-8",
            )
            np.save(run_dir / "gate_final.npy", np.array([[[1.0, 0.0]]], dtype=np.float32))
            metrics_dir = run_dir / "evaluation" / "Final" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "metrics_manifest.json").write_text(
                json.dumps(
                    {
                        "channel_expert_selection": "channel_expert_selection.csv",
                        "channel_expert_health": "channel_expert_health.csv",
                        "channel_expert_summary": "channel_expert_summary.csv",
                        "expert_health": "expert_health.csv",
                    }
                ),
                encoding="utf-8",
            )
            for name in ["channel_expert_health.csv", "channel_expert_summary.csv", "expert_health.csv"]:
                (metrics_dir / name).write_text("ok\n", encoding="utf-8")
            self._write_csv_rows(
                metrics_dir / "channel_expert_selection.csv",
                [
                    {
                        "sample_idx": 0,
                        "channel_pos": 0,
                        "selected_expert_idx": 0,
                        "selected_weight": 0.5,
                        "selected_dense_weight": 0.5,
                        "is_hard_one_hot": "True",
                    }
                ],
                [
                    "sample_idx",
                    "channel_pos",
                    "selected_expert_idx",
                    "selected_weight",
                    "selected_dense_weight",
                    "is_hard_one_hot",
                ],
            )
            failures = _failures_for_run(run_dir)
            self.assertTrue(any("selected_weight expected 1.0" in item for item in failures))

    def test_hard_channel_validation_report_row_includes_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model": "MoEDiffuser",
                        "routing_mode": "hard_channel",
                        "target_only_diffusion": True,
                        "hard_final_routing": True,
                        "final_top_k": 1,
                        "moe_experts": ["ExpertA", "ExpertB"],
                        "forecast_objective": "direct",
                    }
                ),
                encoding="utf-8",
            )
            metrics_dir = run_dir / "evaluation" / "Final" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "summary.json").write_text(
                json.dumps({"num_windows": 3, "RMSE": 1.25, "R2": 0.5}),
                encoding="utf-8",
            )
            row = _run_report_row(run_dir, ["missing gate_final.npy"])
            self.assertEqual(row["config_model"], "MoEDiffuser")
            self.assertEqual(row["num_experts"], 2)
            self.assertEqual(row["num_windows"], 3)
            self.assertEqual(row["status"], "FAILED")

    def test_dataset_target_only_shapes(self):
        csv_path = self._make_csv()
        try:
            ds = GenericCSVDataset(
                seq_len=4,
                context_len=4,
                pred_len=4,
                csv_file_path=csv_path,
                mode="train",
                channels_first=True,
                add_time_features=True,
                add_cyc_features=True,
                target_only_diffusion=True,
            )
            item = ds[0]
            self.assertEqual(tuple(item["past"].shape), (7, 4))
            self.assertEqual(tuple(item["future"].shape), (7, 4))
            self.assertEqual(tuple(item["known_future"].shape), (12, 4))
            self.assertEqual(ds.num_channels, 7)
            self.assertEqual(ds.known_future_channels, 12)
        finally:
            os.remove(csv_path)

    def test_moe_forward_with_known_future(self):
        prior = np.zeros((7, len(DEFAULT_EXPERTS)), dtype=np.float32)
        for channel_pos in range(7):
            prior[channel_pos, channel_pos % len(DEFAULT_EXPERTS)] = 1.0
        config = {
            "model": "MoEDiffuser",
            "moe_experts": list(DEFAULT_EXPERTS),
            "pred_len": 16,
            "use_rule_coarse": True,
            "coarse_mix_alpha": 0.7,
            "channel_expert_prior": prior.tolist(),
            "channel_prior_alpha": 1.0,
            "channel_gate_use_stats": True,
            "channel_gate_stat_hidden_dim": 8,
            "channel_gate_use_embedding": True,
            "use_task_heads": True,
            "task_head_hidden_dim": 8,
            "coarse_top_k": 3,
            "final_top_k": 1,
            "hard_final_routing": True,
            "hard_routing_straight_through": True,
            "gate_temperature": 1.25,
        }
        model = get_model_instance(
            config,
            7,
            torch.device("cpu"),
            target_indices=list(range(7)),
            known_future_channels=12,
        )
        model.eval()
        with torch.no_grad():
            out, weights = model(
                torch.randn(2, 7, 16),
                torch.randint(0, 1000, (2,)),
                torch.randn(2, 7, 16),
                coarse_label=torch.zeros(2, 7, dtype=torch.long),
                known_future=torch.randn(2, 12, 16),
                return_weights=True,
            )
        self.assertEqual(tuple(out.shape), (2, 7, 16))
        self.assertEqual(tuple(weights["final"].shape), (2, 7, len(DEFAULT_EXPERTS)))
        self.assertTrue(torch.allclose(weights["final"].sum(dim=-1), torch.ones(2, 7), atol=1e-6))
        self.assertTrue(torch.all((weights["final"].detach() == 0.0) | (weights["final"].detach() == 1.0)))
        picked = torch.argmax(weights["final"], dim=-1)
        expected = torch.arange(7).view(1, 7).expand(2, 7)
        self.assertTrue(torch.equal(picked.cpu(), expected))
        prior_loss = model.channel_prior_loss(weights)
        self.assertTrue(torch.isfinite(prior_loss))
        self.assertGreaterEqual(float(prior_loss.item()), 0.0)

    def test_hard_routing_is_one_hot_with_straight_through_gradient(self):
        logits = torch.randn(2, 3, 4, requires_grad=True)
        weights = torch.softmax(logits, dim=-1)
        routed = _apply_hard_routing(weights, straight_through=True)

        self.assertTrue(torch.allclose(routed.detach().sum(dim=-1), torch.ones(2, 3), atol=1e-6))
        self.assertTrue(torch.all((routed.detach() == 0.0) | (routed.detach() == 1.0)))

        reward = torch.arange(4, dtype=routed.dtype).view(1, 1, 4)
        loss = (routed * reward).sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum().item()), 0.0)

    def test_scheduler_strict_shape_and_explicit_layout(self):
        scheduler = DDPMScheduler(num_timesteps=10, device="cpu")
        condition_blc = torch.randn(2, 5, 7)
        out = scheduler.sample(
            ZeroNoiseModel(),
            condition_blc,
            seq_len=5,
            sampler="ddim",
            num_inference_steps=2,
            t_start=4,
            condition_format="BLC",
        )
        self.assertEqual(tuple(out.shape), (2, 7, 5))

        with self.assertRaises(RuntimeError):
            scheduler.sample(
                BadShapeModel(),
                torch.randn(2, 7, 5),
                seq_len=5,
                sampler="ddim",
                num_inference_steps=2,
                t_start=4,
            )

    def test_crps_is_nonnegative_for_basic_ensemble(self):
        ensemble = np.array([[[1.0, 2.0]], [[2.0, 4.0]]])
        y_true = np.array([[1.5, 3.0]])
        self.assertGreaterEqual(MetricsCalculator.crps(ensemble, y_true), 0.0)

    def test_predict_restores_residual_forecast(self):
        from predict import _restore_generated_forecast

        past = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        residual = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
        restored = _restore_generated_forecast(residual, past, "daily_seasonal_residual", 2)
        self.assertTrue(torch.allclose(restored, torch.tensor([[[3.5, 4.5, 3.25, 4.25]]])))
        untouched = _restore_generated_forecast(residual, past, "direct", 2)
        self.assertTrue(torch.equal(untouched, residual))

    def test_daily_seasonal_residual_round_trip(self):
        past = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        future = torch.tensor([[[3.5, 4.5, 3.25, 4.25]]])
        residual = make_residual_target_bcl(past, future, period=2)
        self.assertTrue(torch.allclose(residual, torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])))
        restored = restore_residual_forecast_bcl(past, residual, period=2)
        self.assertTrue(torch.allclose(restored, future))

    def test_forward_kwargs_do_not_swallow_internal_typeerror(self):
        with self.assertRaisesRegex(TypeError, "internal known_future bug"):
            forward_diffusion_model(
                InternalTypeErrorModel(),
                torch.randn(1, 2, 4),
                torch.zeros(1, dtype=torch.long),
                torch.randn(1, 2, 4),
                known_future=torch.randn(1, 1, 4),
            )

        scheduler = DDPMScheduler(num_timesteps=10, device="cpu")
        with self.assertRaisesRegex(TypeError, "internal known_future bug"):
            scheduler.sample(
                InternalTypeErrorModel(),
                torch.randn(1, 2, 4),
                seq_len=4,
                known_future=torch.randn(1, 1, 4),
                sampler="ddim",
                num_inference_steps=2,
                t_start=4,
            )

    def test_context_compressor_starts_from_tail_window(self):
        from network.moe import ContextCompressor

        compressor = ContextCompressor(channels=3, out_len=8, hidden_dim=8)
        x = torch.randn(2, 3, 24)
        out = compressor(x)
        self.assertEqual(tuple(out.shape), (2, 3, 8))
        self.assertTrue(torch.allclose(out, x[:, :, -8:], atol=1e-6))

    def test_patchtst_expert_full_context(self):
        from network import MODEL_REGISTRY

        expert = MODEL_REGISTRY["PatchTSTDiffuser"](
            seq_length=12, input_channels=4, context_length=48, patch_len=8, stride=4
        )
        past = torch.randn(2, 4, 48)
        out = expert(torch.zeros(2, 4, 12), torch.zeros(2, dtype=torch.long), past)
        self.assertEqual(tuple(out.shape), (2, 4, 12))

    def test_dlinear_expert_forward_shape(self):
        from network import MODEL_REGISTRY

        expert = MODEL_REGISTRY["DLinearDiffuser"](seq_length=12, input_channels=4)
        noisy = torch.randn(2, 4, 12)
        out = expert(noisy, torch.zeros(2, dtype=torch.long), torch.randn(2, 4, 12))
        self.assertEqual(tuple(out.shape), (2, 4, 12))

    def test_default_rule_table_uses_exact_expert_names(self):
        names = ["S4Diffuser", "SelectiveSSMDiffuser", "CSDIDiffuser", "CSDIAttentionDiffuser"]
        table = make_default_rule_table(names)
        self.assertEqual(tuple(table.shape), (4, 4))
        self.assertTrue(torch.allclose(table.sum(dim=1), torch.ones(4), atol=1e-6))
        self.assertGreater(table[3, names.index("SelectiveSSMDiffuser")].item(), 0.0)

    def test_evaluation_writes_summary_csv_manifest_and_crps_without_health(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_evaluation(
                ZeroNoiseModel(),
                DDPMScheduler(num_timesteps=10, device="cpu"),
                TinyEvalDataset(),
                "Unit",
                torch.device("cpu"),
                tmpdir,
                export_all_windows=False,
                max_eval_windows=1,
                index_seed=1,
                sample_seed=2,
                save_preview_plot=False,
                save_window_plots=False,
                save_overview_plots=False,
                sample_sampler="ddim",
                sample_inference_steps=2,
                sample_t_start=4,
                crps_samples=2,
                save_expert_health=False,
                eval_batch_size=1,
            )
            self.assertIsNotNone(metrics["CRPS"])
            metrics_dir = os.path.join(tmpdir, "evaluation", "Unit", "metrics")
            self.assertTrue(os.path.exists(os.path.join(metrics_dir, "summary.json")))
            self.assertTrue(os.path.exists(os.path.join(metrics_dir, "summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(metrics_dir, "metrics_manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(metrics_dir, "baseline_metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(metrics_dir, "baseline_skill.json")))
            with open(os.path.join(metrics_dir, "metrics_manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("channel_expert_selection", manifest)


if __name__ == "__main__":
    unittest.main()
