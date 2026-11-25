"""One-page analysis of a finished run: metrics, routing, per-channel view.

    python tools/analyze_run.py results/<run>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _csv(path: Path):
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _f(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze one experiment run.")
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    cfg = _json(run / "config.json")

    print(f"run: {run.name}")
    print(
        f"config: model={cfg.get('model')} experts={len(cfg.get('moe_experts') or [])} "
        f"routing={cfg.get('routing_mode')} pred={cfg.get('prediction_type', 'epsilon')} "
        f"objective={cfg.get('forecast_objective')} eval_space={cfg.get('eval_space')} "
        f"ensemble={cfg.get('eval_ensemble_samples', 1)} seed={cfg.get('seed')}"
    )

    for split in ("Final", "FinalTest"):
        summary = _json(run / "evaluation" / split / "metrics" / "summary.json")
        if not summary:
            continue
        print(
            f"{split:9s} windows={summary.get('num_windows')} "
            f"RMSE={_f(summary.get('RMSE')):.4f} MAE={_f(summary.get('MAE')):.4f} "
            f"R2={_f(summary.get('R2')):.4f} CRPS={summary.get('CRPS')} "
            f"| baseline({summary.get('baseline_reference')})={_f(summary.get('reference_baseline_RMSE')):.4f} "
            f"skill={_f(summary.get('RMSE_skill_vs_reference_baseline')):+.3f}"
        )

    gate_rows = _csv(run / "gate_task_summary.csv")
    if gate_rows:
        best = {}
        for row in gate_rows:
            freq = _f(row.get("selection_freq"), 0.0)
            task = str(row.get("task_name"))
            if freq > best.get(task, (0.0, ""))[0]:
                best[task] = (freq, str(row.get("expert_name")))
        print("routing (top-1 expert per channel):")
        for task, (freq, name) in best.items():
            print(f"  {task:6s} -> {name:28s} {freq:.2f}")

        usage = {}
        for row in gate_rows:
            name = str(row.get("expert_name"))
            usage[name] = usage.get(name, 0.0) + _f(
                row.get("avg_final_dense_weight", row.get("avg_final_weight")), 0.0
            )
        total = sum(usage.values()) or 1.0
        print("expert mean dense weight:")
        for name, value in sorted(usage.items(), key=lambda kv: -kv[1]):
            print(f"  {name:28s} {value / total:.3f}")

    channel_rows = _csv(run / "evaluation" / "Final" / "metrics" / "channel_metrics.csv")
    if channel_rows:
        print("per-channel RMSE (val):")
        for row in channel_rows:
            print(f"  {str(row.get('channel_name')):6s} {_f(row.get('RMSE')):.4f}  R2={_f(row.get('R2')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
