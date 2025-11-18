from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-iteration diffs (L2, Linf) for C_history.npy")
    p.add_argument("--c-history", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./results/viz"))
    p.add_argument("--threshold", type=float, default=1e-4, help="Threshold for near-zero change")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    arr = np.load(args.c_history)
    diffs = [float(np.linalg.norm(arr[i + 1] - arr[i])) for i in range(arr.shape[0] - 1)]
    maxdiff = [float(np.max(np.abs(arr[i + 1] - arr[i]))) for i in range(arr.shape[0] - 1)]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"l2": diffs, "linf": maxdiff}, open(out_dir / "C_diffs.json", "w"), indent=2)

    # plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(diffs, label="L2 change")
    ax.plot(maxdiff, label="Linf change")
    ax.axhline(args.threshold, color="k", linestyle="--", label=f"thr={args.threshold}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Change norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C_diffs.png", dpi=160)
    print(f"Saved C_diffs.json & C_diffs.png to {out_dir}")


if __name__ == "__main__":
    main()
