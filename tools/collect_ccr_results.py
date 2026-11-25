"""Aggregate Certified/Active-CCR per-run CSVs into per-dataset mean/std."""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def dataset_key(name: str) -> str:
    n = name.lower()
    for key, label in (("etth1", "ETTh1"), ("etth2", "ETTh2"), ("ettm1", "ETTm1"),
                       ("ettm2", "ETTm2"), ("weather", "Weather"),
                       ("synthetic_balanced", "synth-balanced"),
                       ("synthetic_hetero", "synth-hetero"),
                       ("synthetic_homo", "synth-homo")):
        if key in n:
            return label
    return "other"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate CCR results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/ccr_by_dataset.csv")
    args = parser.parse_args(argv)

    groups = defaultdict(list)
    for path in sorted(Path(args.results_dir).glob("*/active_ccr_*.csv")):
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["dataset"] = dataset_key(path.parent.name)
                groups[row["dataset"]].append(row)
    for path in sorted(Path(args.results_dir).glob("*/certified_routing_*.csv")):
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("method") in ("gate", "best_single", "oracle"):
                    row["dataset"] = dataset_key(path.parent.name)
                    groups[row["dataset"] + ":refs"].append(row)

    ref_means = defaultdict(dict)
    for key, items in groups.items():
        if not key.endswith(":refs"):
            continue
        base = key[: -len(":refs")]
        by_method = defaultdict(list)
        for item in items:
            by_method[item["method"]].append(float(item["rmse"]))
        for method, vals in by_method.items():
            ref_means[base][method] = (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    for key, items in sorted(groups.items()):
        if key.endswith(":refs"):
            continue
        vals = [float(r["rmse"]) for r in items]
        gains = [float(r["gain"]) for r in items if r.get("gain") not in (None, "")]
        cert = [float(r["certified"]) for r in items]
        ns = [float(r["mean_n"]) for r in items]
        rows.append({
            "dataset": key, "runs": len(items),
            "ccr_rmse_mean": st.fmean(vals),
            "ccr_rmse_std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "gain_mean": st.fmean(gains) if gains else "",
            "gain_std": st.stdev(gains) if len(gains) > 1 else 0.0,
            "certified_mean": st.fmean(cert), "mean_n": st.fmean(ns),
        })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    def fm(key, method):
        v = ref_means.get(key, {}).get(method)
        return f"{v[0]:.4f}" if v else "--"

    print(f"{'dataset':16s} {'n':>2s} {'gate':>7s} {'best':>7s} {'oracle':>7s} {'CCR':>16s} {'gain':>7s}")
    for r in rows:
        print(f"{r['dataset']:16s} {r['runs']:>2d} {fm(r['dataset'],'gate'):>7s} "
              f"{fm(r['dataset'],'best_single'):>7s} {fm(r['dataset'],'oracle'):>7s} "
              f"{r['ccr_rmse_mean']:.4f}±{r['ccr_rmse_std']:.4f} {r['gain_mean']:+.3f}")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
