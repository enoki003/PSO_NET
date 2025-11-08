from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from keras.datasets import cifar100
from keras.utils import to_categorical

from . import config
from .sub_expert import build_sub_expert_model


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class TrainConfig:
    """Configuration values for CIFAR-100 specialist training."""

    train_with_softmax: bool = True
    num_experts: int = 8
    subset_pool_fraction: float = 0.2
    epochs: int = 3
    batch_size: int = 128
    label_smoothing: float = 0.1
    learning_rate: float = 1e-3
    subset_seed: int = 42
    output_root: Path = Path("./models/cifar_sub_experts")
    noise_std: float = 0.05


def parse_args(argv: Iterable[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CIFAR-100 sub experts.")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--subset-pool-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--no-softmax", dest="train_with_softmax", action="store_false")
    parser.set_defaults(train_with_softmax=True)
    args = parser.parse_args(argv)
    return TrainConfig(
        train_with_softmax=args.train_with_softmax,
        num_experts=args.num_experts,
        subset_pool_fraction=args.subset_pool_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        subset_seed=args.subset_seed,
        output_root=args.output_root,
        noise_std=args.noise_std,
    )


def normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    mean = np.asarray(config.CIFAR_CHANNEL_MEAN, dtype=np.float32)
    std = np.asarray(config.CIFAR_CHANNEL_STD, dtype=np.float32)
    return (images - mean) / std


def stratified_subsets(labels: np.ndarray, experts: int, pool_fraction: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = labels.squeeze().astype(np.int32)
    expert_indices = [[] for _ in range(experts)]

    for cls in range(config.CIFAR_NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        if cls_indices.size == 0:
            continue
        rng.shuffle(cls_indices)
        pool_target = int(cls_indices.size * pool_fraction)
        pool_target = max(experts, pool_target)
        pool_target = min(cls_indices.size, pool_target)
        pooled = cls_indices[:pool_target]
        for offset, idx in enumerate(pooled):
            expert_id = offset % experts
            expert_indices[expert_id].append(int(idx))

    result = []
    for idxs in expert_indices:
        rng.shuffle(idxs)
        result.append(np.array(idxs, dtype=np.int32))
    return result


def augment_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="CONSTANT")
    image = tf.image.random_crop(image, size=config.CIFAR_IMG_SHAPE)
    image = tf.image.random_flip_left_right(image)
    return image


def build_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=images.shape[0], seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda img, lbl: (augment_image(img), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def train(cfg: TrainConfig) -> None:
    tf.random.set_seed(cfg.subset_seed)

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    index_sets = stratified_subsets(y_train, cfg.num_experts, cfg.subset_pool_fraction, cfg.subset_seed)

    if cfg.train_with_softmax:
        # Keras 3: to_categorical no longer accepts dtype; cast explicitly
        y_train_proc = to_categorical(y_train, num_classes=config.CIFAR_NUM_CLASSES).astype("float32")
        y_test_proc = to_categorical(y_test, num_classes=config.CIFAR_NUM_CLASSES).astype("float32")
    else:
        y_train_proc = y_train.squeeze().astype("int32")
        y_test_proc = y_test.squeeze().astype("int32")

    val_dataset = build_dataset(
        x_test,
        y_test_proc,
        batch_size=cfg.batch_size,
        augment=False,
        shuffle=False,
        seed=cfg.subset_seed,
    )

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "train_with_softmax": cfg.train_with_softmax,
        "num_experts": cfg.num_experts,
        "subset_pool_fraction": cfg.subset_pool_fraction,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "label_smoothing": cfg.label_smoothing if cfg.train_with_softmax else 0.0,
        "learning_rate": cfg.learning_rate,
        "subset_seed": cfg.subset_seed,
        "noise_std": cfg.noise_std,
    }
    with open(cfg.output_root / "config.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    for expert_id, indices in enumerate(index_sets):
        expert_dir = cfg.output_root / f"expert_{expert_id:02d}"
        expert_dir.mkdir(parents=True, exist_ok=True)

        expert_images = x_train[indices]
        expert_labels = y_train_proc[indices]

        train_dataset = build_dataset(
            expert_images,
            expert_labels,
            batch_size=cfg.batch_size,
            augment=True,
            shuffle=True,
            seed=cfg.subset_seed + expert_id,
        )

        model = build_sub_expert_model(
            use_softmax=cfg.train_with_softmax,
            smoothing=cfg.label_smoothing,
            learning_rate=cfg.learning_rate,
            noise_std=cfg.noise_std,
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.3,
                patience=2,
                min_lr=1e-5,
                monitor="val_accuracy",
                mode="max",  # 明示: 精度は最大化
                verbose=1,
            )
        ]

        history = model.fit(
            train_dataset,
            epochs=cfg.epochs,
            validation_data=val_dataset,
            verbose=1,
            callbacks=callbacks,
        )

        history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
        with open(expert_dir / "history.json", "w", encoding="utf-8") as fp:
            json.dump(history_dict, fp, indent=2)

        primary_weights = expert_dir / ("softmax.weights.h5" if cfg.train_with_softmax else "logits.weights.h5")
        model.save_weights(primary_weights)

        with open(expert_dir / "model.json", "w", encoding="utf-8") as fp:
            fp.write(model.to_json())

        np.save(expert_dir / "train_indices.npy", indices, allow_pickle=False)

        if cfg.train_with_softmax:
            logits_model = build_sub_expert_model(
                use_softmax=False,
                smoothing=0.0,
                learning_rate=cfg.learning_rate,
                noise_std=0.0,
            )
            logits_model.set_weights(model.get_weights())
            logits_model.save_weights(expert_dir / "logits.weights.h5")

        print(f"Expert {expert_id:02d} finished. Samples={indices.size}")

    print("All experts trained successfully.")


def main(argv: Iterable[str] | None = None) -> None:
    cfg = parse_args(argv)
    train(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
