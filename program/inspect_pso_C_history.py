from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


try:
    from scipy.linalg import eigvals
except Exception:
    from numpy.linalg import eigvals


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PSO average gating matrix C_history.npy")
    parser.add_argument("--c-history", type=Path, required=True, help="Path to C_history.npy (iterations, N, N)")
    parser.add_argument("--out", type=Path, default=Path("./results/viz/pso"), help="Output dir for stats & plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args(argv)


def entropy_rows(mat: np.ndarray) -> np.ndarray:
    # mat shape (N,N) row stochastic
    p = np.clip(mat, 1e-12, None)
    return -np.sum(p * np.log(p), axis=1)


def sparsity_frac(mat: np.ndarray, thresh: float = 1e-3) -> float:
    return float(np.mean((mat < thresh).astype(float)))


def spectral_radius(mat: np.ndarray) -> float:
    try:
        es = eigvals(mat)
        return float(np.max(np.abs(es)))
    except Exception:
        return float(np.nan)


def inspect_c_history(c_path: Path, out_dir: Path, show: bool = False) -> None:
    arr = np.load(c_path)
    assert arr.ndim == 3
    iters = arr.shape[0]
    ent_means = []
    sparsities = []
    spectral = []
    diag_means = []

    for i in range(iters):
        mat = arr[i]
        ent = entropy_rows(mat)
        ent_means.append(float(np.mean(ent)))
        sparsities.append(sparsity_frac(mat))
        diag_means.append(float(np.mean(np.diag(mat))))
        spectral.append(spectral_radius(mat))

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "iterations": int(iters),
        "ent_means": ent_means,
        "sparsities": sparsities,
        "diag_means": diag_means,
        "spectral_radius": spectral,
    }
    with (out_dir / "C_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    # Plot stats
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(ent_means, label="Row entropy (mean)")
    axes[0].set_ylabel("Entropy")
    axes[0].legend()
    axes[1].plot(sparsities, label="Sparsity fraction")
    axes[1].set_ylabel("Sparsity")
    axes[1].legend()
    axes[2].plot(diag_means, label="Diag mean")
    axes[2].plot(spectral, label="Spectral radius")
    axes[2].set_ylabel("Diag / spectral")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C_stats.png", dpi=160)
    print("Saved C_stats.png and C_stats.json in", out_dir)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    inspect_c_history(args.c_history, args.out, show=args.show)


if __name__ == "__main__":
    main()
