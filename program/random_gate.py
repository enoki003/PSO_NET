from __future__ import annotations

"""Evaluate random gating mixtures over fixed expert logits."""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from keras.datasets import cifar10, cifar100

from . import config
from .pso_train import load_expert_models, normalize_images, precompute_expert_logits


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random gating baseline evaluation")
    parser.add_argument("--experts", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=0, help="Optional random subset size")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("./results/random_gate"))
    return parser.parse_args(argv)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], int]:
    if name == "cifar100":
        (_, _), (images, labels) = cifar100.load_data(label_mode="fine")
        mean, std = config.CIFAR_CHANNEL_MEAN, config.CIFAR_CHANNEL_STD
        num_classes = config.CIFAR_NUM_CLASSES
    else:
        (_, _), (images, labels) = cifar10.load_data()
        mean, std = config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
        num_classes = config.CIFAR10_NUM_CLASSES
    images = normalize_images(images, mean, std)
    labels = labels.squeeze().astype(np.int32)
    return images, labels, mean, num_classes


def run_trials(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    images, labels, _, num_classes = load_dataset(args.dataset)
    if args.sample_count and args.sample_count < images.shape[0]:
        idx = rng.choice(images.shape[0], size=args.sample_count, replace=False)
        images = images[idx]
        labels = labels[idx]

    expert_models = load_expert_models(
        args.experts,
        args.num_experts,
        learning_rate=1e-3,
        img_shape=config.CIFAR_IMG_SHAPE if args.dataset == "cifar100" else config.CIFAR10_IMG_SHAPE,
        num_classes=num_classes,
    )
    logits = precompute_expert_logits(expert_models, images, batch_size=args.batch_size)

    trial_results: list[dict[str, float]] = []
    for trial in range(args.trials):
        weights = rng.dirichlet(np.ones(args.num_experts), size=logits.shape[0])
        combined = np.einsum("sa,sac->sc", weights, logits)
        predictions = np.argmax(combined, axis=1)
        top1 = float(np.mean(predictions == labels))

        trial_results.append({
            "trial": trial,
            "top1": top1,
        })

    top1_mean = float(np.mean([r["top1"] for r in trial_results]))
    top1_std = float(np.std([r["top1"] for r in trial_results]))

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset": args.dataset,
                "num_experts": args.num_experts,
                "trials": args.trials,
                "top1_mean": top1_mean,
                "top1_std": top1_std,
                "per_trial": trial_results,
            },
            fp,
            indent=2,
        )

    print("Random gating evaluation complete")
    print("Top-1 mean:", top1_mean)
    print("Top-1 std:", top1_std)


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    run_trials(argv)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
