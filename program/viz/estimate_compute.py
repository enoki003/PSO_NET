from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate compute cost from gating matrices (avg active experts) and approximate FLOPs ratio.")
    p.add_argument("--gating", type=Path, required=True, help="Path to PSO output dir with gating_weights.npy and optionally pso_history.json")
    p.add_argument("--threshold", type=float, default=1e-3, help="Threshold to consider edge active")
    p.add_argument("--top-k", type=int, default=0, help="If >0, compute average of top-k active experts per row")
    p.add_argument("--out", type=Path, default=Path("./results/viz"))
    return p.parse_args(argv)


def estimate_from_history(history_path: Path, threshold: float, top_k: int):
    history = json.loads(history_path.read_text(encoding="utf-8"))
    gating_mats_all = []
    for entry in history:
        mat = entry.get("avg_gating")
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=np.float32)
        gating_mats_all.append(arr)
    if not gating_mats_all:
        return None
    arr = np.stack(gating_mats_all, axis=0)  # (iters, N, N)
    # Compute per-row active expert counts
    if top_k > 0:
        # top-k active per row
        topk_counts = np.sum(np.argsort(-arr, axis=-1) < top_k, axis=-1)  # boolean
        # Actually above is incorrect; easier: for each row, pick top-k indices and count
        n_iters, n, _ = arr.shape
        counts = np.zeros((n_iters, n), dtype=np.float32)
        for i in range(n_iters):
            row = arr[i]
            for r in range(n):
                topk = np.sort(row[r])[-top_k:]
                counts[i, r] = np.sum(row[r] >= topk[0])
    else:
        counts = np.sum(arr > threshold, axis=-1)
    # average per sample is average number of active experts per row
    counts_mean = float(np.mean(counts))
    # expected number of expert forward passes per sample = counts_mean (one row per expert?)
    return {
        "avg_active_per_row": counts_mean,
        "threshold": threshold,
        "top_k": top_k,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = args.gating / "pso_history.json"
    if not history_path.exists():
        print(f"pso_history.json not found in {args.gating}. Trying C_history.npy...")
        # fallback: use C_history
        cpath = args.gating / "C_history.npy"
        if not cpath.exists():
            raise SystemExit("No history found in gating dir")
        arr = np.load(cpath)
        counts = np.sum(arr > args.threshold, axis=-1)  # shape (iters,N)
        avg_active = float(np.mean(counts))
        res = {"avg_active_per_row": avg_active, "threshold": args.threshold}
    else:
        res = estimate_from_history(history_path, args.threshold, args.top_k)
        if res is None:
            raise SystemExit("No avg_gating found in history")

    with (out_dir / f"compute_estimate_{Path(args.gating).name}.json").open("w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2)
    print(f"Saved compute estimate to {out_dir / f'compute_estimate_{Path(args.gating).name}.json'}")


if __name__ == "__main__":
    main()
