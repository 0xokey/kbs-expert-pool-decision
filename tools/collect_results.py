"""Collect and aggregate experiment results across seeds.

    python tools/collect_results.py --results-dir results --tag main
    python tools/collect_results.py --results-dir results --out results/table_main.csv --markdown

The aggregator reads only persisted artifacts:
`config.json`, `evaluation/<split>/metrics/summary.json` and
`evaluation/<split>/metrics/baseline_metrics.csv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
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


METRIC_KEYS = ["RMSE", "MAE", "R2", "CRPS", "sMAPE"]
NUMERIC_FIELDS = [
    "val_RMSE", "val_MAE", "val_R2", "val_CRPS",
    "test_RMSE", "test_MAE", "test_R2", "test_CRPS",
    "val_baseline_RMSE", "test_baseline_RMSE",
    "val_skill_RMSE", "test_skill_RMSE",
]


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_rmse(split_dir: Path, reference: str):
    for row in _read_csv(split_dir / "metrics" / "baseline_metrics.csv"):
        if str(row.get("baseline", "")).strip().lower() == str(reference).strip().lower():
            return _to_float(row.get("RMSE"))
    return None


def _load_baseline_row(run_dir: Path):
    payload = _read_json(run_dir / "baseline_metrics.json")
    if not payload:
        return None
    val = payload.get("val", {}) or {}
    test = payload.get("test", {}) or {}
    return {
        "run_name": run_dir.name,
        "run_tag": str(payload.get("run_tag", "")),
        "seed": payload.get("seed", ""),
        "dataset": Path(str(payload.get("config", ""))).stem,
        "forecast_objective": "",
        "routing_mode": str(payload.get("model", "baseline")),
        "num_experts": 0,
        "val_RMSE": val.get("RMSE"),
        "val_MAE": val.get("MAE"),
        "val_R2": val.get("R2"),
        "val_CRPS": None,
        "test_RMSE": test.get("RMSE"),
        "test_MAE": test.get("MAE"),
        "test_R2": test.get("R2"),
        "test_CRPS": None,
        "val_baseline_RMSE": None,
        "test_baseline_RMSE": None,
        "val_skill_RMSE": None,
        "test_skill_RMSE": None,
    }


def load_runs(results_dir: Path, tag_filter=None):
    rows = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        baseline_row = _load_baseline_row(run_dir)
        if baseline_row is not None:
            tag = baseline_row["run_tag"]
            if tag_filter and tag_filter not in tag:
                continue
            rows.append(baseline_row)
            continue
        config = _read_json(run_dir / "config.json")
        if not config:
            continue
        tag = str(config.get("run_tag", ""))
        if tag_filter and tag_filter not in tag:
            continue
        reference = str(config.get("baseline_reference", "daily_seasonal"))
        val_dir = run_dir / "evaluation" / "Final"
        test_dir = run_dir / "evaluation" / "FinalTest"
        val = _read_json(val_dir / "metrics" / "summary.json")
        test = _read_json(test_dir / "metrics" / "summary.json")
        row = {
            "run_name": run_dir.name,
            "run_tag": tag,
            "seed": config.get("seed", ""),
            "dataset": Path(str(config.get("csv_path", ""))).stem,
            "forecast_objective": config.get("forecast_objective", ""),
            "routing_mode": config.get("routing_mode", ""),
            "num_experts": len(config.get("moe_experts") or []),
            "val_RMSE": val.get("RMSE"),
            "val_MAE": val.get("MAE"),
            "val_R2": val.get("R2"),
            "val_CRPS": val.get("CRPS"),
            "test_RMSE": test.get("RMSE"),
            "test_MAE": test.get("MAE"),
            "test_R2": test.get("R2"),
            "test_CRPS": test.get("CRPS"),
        }
        row["val_baseline_RMSE"] = _baseline_rmse(val_dir, reference)
        row["test_baseline_RMSE"] = _baseline_rmse(test_dir, reference)
        for split in ("val", "test"):
            base = row.get(f"{split}_baseline_RMSE")
            value = row.get(f"{split}_RMSE")
            row[f"{split}_skill_RMSE"] = (
                None if base in (None, 0) or value is None else 1.0 - float(value) / float(base)
            )
        rows.append(row)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["run_tag"] or "untagged", []).append(row)
    out = []
    for tag, items in sorted(groups.items()):
        agg = {"run_tag": tag, "num_runs": len(items), "seeds": ",".join(str(item["seed"]) for item in items)}
        for field in NUMERIC_FIELDS:
            values = [float(item[field]) for item in items if item.get(field) is not None]
            if not values:
                agg[f"{field}_mean"] = ""
                agg[f"{field}_std"] = ""
                continue
            agg[f"{field}_mean"] = statistics.fmean(values)
            agg[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(agg)
    return out


def _fmt(value, digits=4):
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(agg_rows):
    lines = [
        "| tag | runs | val RMSE | val R2 | test RMSE | test R2 | test skill vs baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in agg_rows:
        lines.append(
            "| {tag} | {runs} | {v} | {vr} | {t} | {tr} | {s} |".format(
                tag=row["run_tag"],
                runs=row["num_runs"],
                v=f"{_fmt(row['val_RMSE_mean'])} ± {_fmt(row['val_RMSE_std'])}",
                vr=f"{_fmt(row['val_R2_mean'], 3)}",
                t=f"{_fmt(row['test_RMSE_mean'])} ± {_fmt(row['test_RMSE_std'])}",
                tr=f"{_fmt(row['test_R2_mean'], 3)}",
                s=f"{_fmt(row['test_skill_RMSE_mean'], 3)}",
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate results across runs/seeds.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tag", default=None, help="Only include run_tag values containing this substring.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--per-run-out", default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    rows = load_runs(results_dir, tag_filter=args.tag)
    if not rows:
        print("[Collect] no runs found")
        return 1
    agg = aggregate(rows)

    per_run = results_dir / (args.per_run_out or "per_run_metrics.csv")
    write_csv(
        per_run,
        rows,
        [
            "run_name", "run_tag", "seed", "dataset", "forecast_objective", "routing_mode", "num_experts",
            "val_RMSE", "val_MAE", "val_R2", "val_CRPS",
            "test_RMSE", "test_MAE", "test_R2", "test_CRPS",
            "val_baseline_RMSE", "test_baseline_RMSE", "val_skill_RMSE", "test_skill_RMSE",
        ],
    )

    agg_fields = ["run_tag", "num_runs", "seeds"] + [
        f"{field}_{stat}" for field in NUMERIC_FIELDS for stat in ("mean", "std")
    ]
    out_path = results_dir / (args.out or "aggregate_metrics.csv")
    write_csv(out_path, agg, agg_fields)

    print(f"[Collect] {len(rows)} runs -> {per_run}")
    print(f"[Collect] {len(agg)} groups -> {out_path}")
    table = markdown_table(agg)
    if args.markdown:
        md_path = results_dir / "aggregate_metrics.md"
        md_path.write_text(table + "\n", encoding="utf-8")
        print(f"[Collect] markdown -> {md_path}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
