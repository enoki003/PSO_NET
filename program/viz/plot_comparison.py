from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot comparison CSV from compare_experiments")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./results/viz/comparison.png"))
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rows = []
    with open(args.csv, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append(r)
    exps = [r["experiment"] for r in rows]
    accs = [float(r["accuracy"]) if r["accuracy"] not in ["", "None", None] else np.nan for r in rows]
    diag = [float(r["diag_mean"]) if r["diag_mean"] not in ["", "None", None] else np.nan for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(exps))
    ax1.bar(x, accs, color="#1f78b4", alpha=0.8)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(exps, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(x, diag, label="diag_mean", color="#33a02c", marker="o")
    ax2.set_ylabel("Diag mean")
    ax2.set_ylim(0, 1.0)
    ax2.legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
