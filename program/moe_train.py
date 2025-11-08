from __future__ import annotations

"""Gradient-based Mixture-of-Experts baseline using fixed sub experts."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
from keras import callbacks, layers, optimizers
from keras.datasets import cifar10, cifar100
from tensorflow import keras

from . import config
from .pso_train import load_expert_models, normalize_images, precompute_expert_logits

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class Arguments:
    experts: Path
    num_experts: int
    dataset: str
    hidden_units: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    val_fraction: float
    sample_count: int
    output: Path


def parse_args(argv: Iterable[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description="Train gradient-based MoE gating network")
    parser.add_argument("--experts", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    parser.add_argument("--hidden-units", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--sample-count", type=int, default=0, help="Optional random subset size")
    parser.add_argument("--output", type=Path, default=Path("./results/moe"))
    args = parser.parse_args(argv)
    return Arguments(
        experts=args.experts,
        num_experts=args.num_experts,
        dataset=args.dataset,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_fraction=args.val_fraction,
        sample_count=args.sample_count,
        output=args.output,
    )


def load_dataset(name: str, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], tuple[int, int, int], int]:
    if name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean, std = config.CIFAR_CHANNEL_MEAN, config.CIFAR_CHANNEL_STD
        img_shape = config.CIFAR_IMG_SHAPE
        num_classes = config.CIFAR_NUM_CLASSES
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean, std = config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
        img_shape = config.CIFAR10_IMG_SHAPE
        num_classes = config.CIFAR10_NUM_CLASSES

    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in range(num_classes):
        cls_indices = np.where(y_train == cls)[0]
        rng.shuffle(cls_indices)
        val_count = max(1, int(cls_indices.size * val_fraction))
        val_idx.extend(cls_indices[:val_count])
        train_idx.extend(cls_indices[val_count:])
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)

    x_train_split = normalize_images(x_train[train_idx], mean, std)
    y_train_split = y_train[train_idx]
    x_val_split = normalize_images(x_train[val_idx], mean, std)
    y_val_split = y_train[val_idx]
    x_test = normalize_images(x_test, mean, std)

    return (
        x_train_split,
        y_train_split,
        x_val_split,
        y_val_split,
        x_test,
        y_test,
        mean,
        img_shape,
        num_classes,
    )


def build_gating_model(num_experts: int, hidden_units: int, img_shape: tuple[int, int, int]) -> keras.Model:
    inputs = keras.Input(shape=img_shape)
    x = layers.Conv2D(48, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(hidden_units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_experts, name="gate_logits")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="moe_gating")


class MoEModel(keras.Model):
    def __init__(self, gating: keras.Model):
        super().__init__()
        self.gating = gating

    def call(self, inputs, training: bool = False):  # type: ignore[override]
        images, expert_logits = inputs
        gate_logits = self.gating(images, training=training)
        weights = tf.nn.softmax(gate_logits, axis=-1)
        return tf.einsum("be,bec->bc", weights, expert_logits)


def build_dataset(
    images: np.ndarray,
    expert_logits: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, expert_logits.astype(np.float32), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=images.shape[0], seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda img, log, lbl: ((img, log), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def train(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        _,
        img_shape,
        num_classes,
    ) = load_dataset(args.dataset, args.val_fraction, args.seed)

    if args.sample_count and args.sample_count < x_train.shape[0]:
        idx = rng.choice(x_train.shape[0], size=args.sample_count, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]

    expert_models = load_expert_models(
        args.experts,
        args.num_experts,
        learning_rate=args.lr,
        img_shape=img_shape,
        num_classes=num_classes,
    )

    train_logits = precompute_expert_logits(expert_models, x_train, batch_size=args.batch_size)
    val_logits = precompute_expert_logits(expert_models, x_val, batch_size=args.batch_size)
    test_logits = precompute_expert_logits(expert_models, x_test, batch_size=args.batch_size)

    gating = build_gating_model(args.num_experts, args.hidden_units, img_shape)
    model = MoEModel(gating)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name="top5"),
    ]
    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_ds = build_dataset(x_train, train_logits, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_ds = build_dataset(x_val, val_logits, y_val, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_ds = build_dataset(x_test, test_logits, y_test, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    early_stop = callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True)

    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=[early_stop])

    test_metrics = model.evaluate(test_ds, return_dict=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "history.json", "w", encoding="utf-8") as fp:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, fp, indent=2)
    with open(args.output / "test_metrics.json", "w", encoding="utf-8") as fp:
        json.dump({k: float(v) for k, v in test_metrics.items()}, fp, indent=2)
    with open(args.output / "config.json", "w", encoding="utf-8") as fp:
        config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(args).items()}
        json.dump(config_dict, fp, indent=2)

    model.gating.save_weights(args.output / "gating.weights.h5")

    print("MoE training complete")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    train(argv)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
