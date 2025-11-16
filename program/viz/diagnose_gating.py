from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from ..gating import build_gating_model
from ..pso_train import WeightAdapter, normalize_images
from ..config import (
    CIFAR10_CHANNEL_MEAN,
    CIFAR10_CHANNEL_STD,
    CIFAR_CHANNEL_MEAN,
    CIFAR_CHANNEL_STD,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose PSO gating: diag-dominance & entropy stats")
    p.add_argument("--gating", type=Path, required=True, help="Path to PSO output dir containing gating_weights.npy")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--sample-count", type=int, default=0, help="0 means use full test set")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path("./results/viz/pso_diag"))
    return p.parse_args(argv)


def load_test_dataset(dataset: str):
    if dataset == "cifar100":
        from keras.datasets import cifar100

        (x_train, _), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean = CIFAR_CHANNEL_MEAN
        std = CIFAR_CHANNEL_STD
    else:
        from keras.datasets import cifar10

        (x_train, _), (x_test, y_test) = cifar10.load_data()
        mean = CIFAR10_CHANNEL_MEAN
        std = CIFAR10_CHANNEL_STD
    y_test = y_test.squeeze().astype(np.int32)
    return x_test, y_test, mean, std


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    # load dataset
    x_test, y_test, mean, std = load_test_dataset(args.dataset)
    if args.sample_count and args.sample_count < x_test.shape[0]:
        idxs = np.arange(x_test.shape[0])
        rng = np.random.default_rng(0)
        rng.shuffle(idxs)
        idxs = idxs[: args.sample_count]
        x_test = x_test[idxs]
        y_test = y_test[idxs]

    samples_norm = normalize_images(x_test, mean, std)

    # gating model
    gating_model = build_gating_model(num_experts=args.num_experts, hidden_units=384)
    weights_path = args.gating / "gating_weights.npy"
    if not weights_path.exists():
        raise SystemExit(f"gating_weights.npy not found at {weights_path}")
    vec = np.load(weights_path)
    adapter = WeightAdapter(gating_model)
    adapter.assign_from_vector(vec)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    gating_logits = gating_model.predict(samples_norm, batch_size=args.batch_size, verbose=0)
    gating_logits = gating_logits.reshape(-1, args.num_experts, args.num_experts)
    gating_mats = softmax(gating_logits, axis=-1)

    sample_stats = []
    diag_counts = []
    ent_means = []
    sparsity_vals = []

    for i in range(gating_mats.shape[0]):
        mat = gating_mats[i]
        ent = -np.sum((mat + 1e-12) * np.log(mat + 1e-12), axis=1)
        diag = np.diag(mat)
        dominated = [int(np.argmax(row) == j) for j, row in enumerate(mat)]
        diag_frac = float(np.mean(dominated))
        diag_counts.append(diag_frac)
        ent_means.append(float(np.mean(ent)))
        sparsity_vals.append(float(np.mean((mat < 1e-3).astype(float))))
        sample_stats.append({
            "index": int(i),
            "diag_frac": diag_frac,
            "diag_mean": float(np.mean(diag)),
            "row_entropy_mean": float(np.mean(ent)),
            "sparsity": float(np.mean((mat < 1e-3).astype(float))),
        })

    stats = {
        "samples": int(gating_mats.shape[0]),
        "mean_diag_frac": float(np.mean(diag_counts)),
        "median_diag_frac": float(np.median(diag_counts)),
        "mean_row_entropy": float(np.mean(ent_means)),
        "mean_sparsity": float(np.mean(sparsity_vals)),
        "sample_stats": sample_stats,
    }

    with (out / "gating_diag_stats.json").open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    # histogram of diag dominance fraction
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diag_counts, bins=20)
    ax.set_title("Histogram of per-sample diagonal-dominance fraction")
    ax.set_xlabel("Fraction of rows where argmax==row index")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out / "diag_frac_hist.png", dpi=160)
    plt.close(fig)

    # row entropy histogram
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(ent_means, bins=20)
    ax2.set_title("Histogram of per-sample mean row entropy")
    ax2.set_xlabel("Mean entropy")
    ax2.set_ylabel("Count")
    fig2.tight_layout()
    fig2.savefig(out / "row_entropy_hist.png", dpi=160)
    plt.close(fig2)

    print(f"Saved gating diag stats to {out / 'gating_diag_stats.json'} and histograms")


if __name__ == "__main__":
    main()
