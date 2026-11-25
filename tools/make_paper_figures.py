"""Render paper-ready figures from a single run directory.

    python tools/make_paper_figures.py --run-dir results/<run> --out figures/<name>

Figures (PNG + PDF):
  1. routing_heatmap      channel x expert selection frequency
  2. expert_usage         average gate weight / top-1 frequency per expert
  3. channel_rmse         per-channel RMSE, model vs baselines
  4. horizon_rmse         RMSE by forecast horizon, model vs baselines
  5. gate_entropy         per-channel routing entropy distribution
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


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


def _float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Figures] {out_dir / name}.png / .pdf")


def plot_routing_heatmap(run_dir: Path, out_dir: Path):
    rows = _read_csv(run_dir / "gate_task_summary.csv")
    if not rows:
        print("[Figures] skip routing heatmap (gate_task_summary.csv missing)")
        return
    tasks = sorted({str(row["task_name"]) for row in rows})
    experts = sorted({str(row["expert_name"]) for row in rows})
    matrix = np.zeros((len(tasks), len(experts)))
    for row in rows:
        if "selection_freq" not in row:
            continue
        i = tasks.index(str(row["task_name"]))
        j = experts.index(str(row["expert_name"]))
        matrix[i, j] += _float(row.get("selection_freq"), 0.0)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(experts) + 2.2), 0.5 * len(tasks) + 1.6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(experts, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_xlabel("expert")
    ax.set_ylabel("target channel")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0.005:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="selection frequency")
    ax.set_title("Hard-channel routing: selected expert per channel")
    _save(fig, out_dir, "routing_heatmap")


def plot_expert_usage(run_dir: Path, out_dir: Path):
    rows = _read_csv(run_dir / "evaluation" / "Final" / "metrics" / "expert_health.csv")
    if not rows:
        print("[Figures] skip expert usage (expert_health.csv missing)")
        return
    names = [str(row["expert_name"]) for row in rows]
    weight = np.array([_float(row.get("avg_final_dense_weight"), 0.0) for row in rows])
    top1 = np.array([_float(row.get("top1_freq"), 0.0) for row in rows])
    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(names) + 2.5), 3.4))
    ax.bar(x - width / 2, weight, width, label="mean dense gate weight")
    ax.bar(x + width / 2, top1, width, label="top-1 frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("share")
    ax.set_title("Expert utilisation under hard-channel routing")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "expert_usage")


def plot_channel_rmse(run_dir: Path, out_dir: Path, split="Final"):
    metrics_dir = run_dir / "evaluation" / split / "metrics"
    rows = _read_csv(metrics_dir / "channel_metrics.csv")
    if not rows:
        print(f"[Figures] skip channel RMSE ({split} channel_metrics.csv missing)")
        return
    channels = [str(row["channel_name"]) for row in rows]
    model = np.array([_float(row["RMSE"]) for row in rows])
    baselines = _read_csv(metrics_dir / "baseline_channel_metrics.csv")
    series = {"MoE (ours)": model}
    for baseline in sorted({str(row["baseline"]) for row in baselines}):
        by_name = {str(row["channel_name"]): _float(row["RMSE"]) for row in baselines if str(row["baseline"]) == baseline}
        series[baseline] = np.array([by_name.get(ch, np.nan) for ch in channels])

    x = np.arange(len(channels))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(channels) + 2.4), 3.6))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title(f"Per-channel RMSE ({'validation' if split == 'Final' else 'test'})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, f"channel_rmse_{'val' if split == 'Final' else 'test'}")


def plot_horizon_rmse(run_dir: Path, out_dir: Path, split="Final"):
    metrics_dir = run_dir / "evaluation" / split / "metrics"
    rows = _read_csv(metrics_dir / "horizon_metrics.csv")
    if not rows:
        print(f"[Figures] skip horizon RMSE ({split} horizon_metrics.csv missing)")
        return
    horizons = [str(row["horizon"]) for row in rows]
    model = np.array([_float(row["RMSE"]) for row in rows])
    baselines = _read_csv(metrics_dir / "baseline_horizon_metrics.csv")

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(horizons, model, marker="o", linewidth=2.0, label="MoE (ours)")
    for baseline in sorted({str(row["baseline"]) for row in baselines}):
        values = np.array([
            _float(row["RMSE"]) for row in baselines if str(row["baseline"]) == baseline
        ])
        labels = [str(row["horizon"]) for row in baselines if str(row["baseline"]) == baseline]
        if len(values) == len(horizons) and labels == horizons:
            ax.plot(horizons, values, marker="s", linewidth=1.2, linestyle="--", label=baseline)
    ax.set_xlabel("forecast horizon (steps)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE by horizon ({'validation' if split == 'Final' else 'test'})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, f"horizon_rmse_{'val' if split == 'Final' else 'test'}")


def plot_gate_entropy(run_dir: Path, out_dir: Path):
    gate_path = run_dir / "gate_final.npy"
    if not gate_path.exists():
        print("[Figures] skip gate entropy (gate_final.npy missing)")
        return
    gate = np.load(gate_path)
    if gate.ndim != 3:
        print("[Figures] skip gate entropy (unexpected gate shape)")
        return
    probs = np.clip(gate, 0.0, 1.0)
    entropy = -(np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0)).sum(axis=-1)
    num_alive = float((gate.sum(axis=-1) > 0).sum(axis=-1).mean())
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(entropy.reshape(-1), bins=30, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("routing entropy (nats)")
    ax.set_ylabel("route count")
    ax.set_title(f"Hard routing entropy (mean alive slots={num_alive:.2f})")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "gate_entropy")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render paper figures for one run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "paper_figures"
    for split in ("Final", "FinalTest"):
        plot_channel_rmse(run_dir, out_dir, split=split)
        plot_horizon_rmse(run_dir, out_dir, split=split)
    plot_routing_heatmap(run_dir, out_dir)
    plot_expert_usage(run_dir, out_dir)
    plot_gate_entropy(run_dir, out_dir)
    print(f"[Figures] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
