from __future__ import annotations

"""Train stacking meta-learner on precomputed expert logits."""

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
    output: Path


def parse_args(argv: Iterable[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description="Train stacking meta-learner over expert logits")
    parser.add_argument("--experts", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--dataset", choices=["cifar100", "cifar10"], default="cifar100")
    parser.add_argument("--hidden-units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("./results/stacking"))
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
        output=args.output,
    )


def load_dataset(name: str, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean, std = config.CIFAR_CHANNEL_MEAN, config.CIFAR_CHANNEL_STD
        num_classes = config.CIFAR_NUM_CLASSES
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean, std = config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
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

    return x_train_split, y_train_split, x_val_split, y_val_split, x_test, y_test, num_classes


def build_dataset(features: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features.astype(np.float32), labels.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=features.shape[0], seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_meta_model(input_dim: int, num_classes: int, hidden_units: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_units, activation="relu")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hidden_units // 2, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, outputs, name="stacking_meta")


def train(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    tf.random.set_seed(args.seed)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        num_classes,
    ) = load_dataset(args.dataset, args.val_fraction, args.seed)

    expert_models = load_expert_models(
        args.experts,
        args.num_experts,
        learning_rate=args.lr,
        img_shape=config.CIFAR_IMG_SHAPE if args.dataset == "cifar100" else config.CIFAR10_IMG_SHAPE,
        num_classes=num_classes,
    )

    train_logits = precompute_expert_logits(expert_models, x_train, batch_size=args.batch_size)
    val_logits = precompute_expert_logits(expert_models, x_val, batch_size=args.batch_size)
    test_logits = precompute_expert_logits(expert_models, x_test, batch_size=args.batch_size)

    input_dim = args.num_experts * num_classes
    train_features = train_logits.reshape(train_logits.shape[0], input_dim)
    val_features = val_logits.reshape(val_logits.shape[0], input_dim)
    test_features = test_logits.reshape(test_logits.shape[0], input_dim)

    model = build_meta_model(input_dim, num_classes, args.hidden_units)
    optimizer = optimizers.Adam(learning_rate=args.lr)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name="top5"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_ds = build_dataset(train_features, y_train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_ds = build_dataset(val_features, y_val, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_ds = build_dataset(test_features, y_test, batch_size=args.batch_size, shuffle=False, seed=args.seed)

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

    model.save_weights(args.output / "meta.weights.h5")

    print("Stacking training complete")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI
    train(argv)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
