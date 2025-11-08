from __future__ import annotations

"""Evaluate simple averaging ensemble over trained sub experts."""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from keras.datasets import cifar10, cifar100

from . import config
from .pso_train import load_expert_models, normalize_images, precompute_expert_logits


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate averaged ensemble of experts")
    parser.add_argument("--experts", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=0, help="Optional random subset size")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("./results/ensemble"))
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


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    if k >= logits.shape[1]:
        # if k exceeds classes, consider all predictions correct if true label in logits
        return 1.0
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    match = (topk == labels[:, None]).any(axis=1)
    return float(np.mean(match))


def evaluate(argv: Iterable[str] | None = None) -> None:
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
    ensemble_logits = np.mean(logits, axis=1)

    top1 = float(np.mean(np.argmax(ensemble_logits, axis=1) == labels))
    top5 = top_k_accuracy(ensemble_logits, labels, k=5)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset": args.dataset,
                "num_experts": args.num_experts,
                "top1": top1,
                "top5": top5,
                "samples": int(labels.shape[0]),
            },
            fp,
            indent=2,
        )

    print("Ensemble evaluation complete")
    print("Top-1:", top1)
    print("Top-5:", top5)


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    evaluate(argv)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
