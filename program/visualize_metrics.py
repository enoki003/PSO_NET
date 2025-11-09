"""Visualize training/evaluation metrics from saved JSON artefacts.

Supported inputs:
  - Single CNN baseline: results dir containing metrics.json with
    {"history": {...}, "val_metrics": {...}, "test_metrics": {...}}
  - Aggregated baselines: results/baselines.json produced by
    program.aggregate_results (bar charts).

Outputs PNG files to a target directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/eval metrics")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a metrics.json directory or to baselines.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/viz"),
        help="Directory to write figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively as well as saving PNGs",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_history(history: dict, out_dir: Path, show: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plot loss and accuracy curves if present
    keys = list(history.keys())
    # normalize common aliases
    acc_keys = [k for k in keys if "acc" in k]
    loss_keys = [k for k in keys if "loss" in k]

    if loss_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in sorted(loss_keys):
            ax.plot(history[k], label=k)
        ax.set_title("Loss curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "loss_curves.png"
        fig.savefig(path, dpi=160)
        print(f"Saved {path}")
        if not show:
            plt.close(fig)

    if acc_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in sorted(acc_keys):
            ax.plot(history[k], label=k)
        ax.set_title("Accuracy curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "accuracy_curves.png"
        fig.savefig(path, dpi=160)
        print(f"Saved {path}")
        if not show:
            plt.close(fig)


def visualize_single_run(run_dir: Path, out_dir: Path, show: bool) -> None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found in {run_dir}")
    payload = _load_json(metrics_path)
    history = payload.get("history")
    if isinstance(history, dict):
        plot_history(history, out_dir, show)

    # Also export a small text summary of val/test metrics if present
    lines: List[str] = []
    if "val_metrics" in payload:
        lines.append("Validation metrics:")
        for k, v in payload["val_metrics"].items():
            lines.append(f"  {k}: {v:.4f}")
    if "test_metrics" in payload:
        lines.append("Test metrics:")
        for k, v in payload["test_metrics"].items():
            lines.append(f"  {k}: {v:.4f}")
    if lines:
        txt = "\n".join(lines)
        with (out_dir / "summary.txt").open("w", encoding="utf-8") as fp:
            fp.write(txt)
        print(f"Saved {out_dir / 'summary.txt'}")


def visualize_aggregate(agg_file: Path, out_dir: Path, show: bool) -> None:
    data = _load_json(agg_file)
    entries = data.get("entries", [])
    if not entries:
        raise SystemExit("No entries in aggregate file")

    # Build bar chart for a few common metrics if present
    labels: List[str] = []
    top1: List[float] = []
    top5: List[float] = []
    for rec in entries:
        labels.append(rec.get("label", rec.get("path", "run")))
        m = rec.get("metrics", {})
        top1.append(float(m.get("top1", np.nan)))
        top5.append(float(m.get("top5", np.nan)))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(x - width / 2, top1, width, label="Top-1")
    ax.bar(x + width / 2, top5, width, label="Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline comparison")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "baseline_bars.png"
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")
    if not show:
        plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input.is_dir():
        visualize_single_run(args.input, out_dir, args.show)
    else:
        visualize_aggregate(args.input, out_dir, args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
