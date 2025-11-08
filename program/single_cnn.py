"""Single CNN baseline training for CIFAR datasets.

Supports (a) standard supervised training on CIFAR-10 / CIFAR-100 and
(b) Split-CIFAR-10 Class-IL sequential fine-tuning with optional replay
buffer for continual learning baselines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10, cifar100

from . import config
from .data_utils import TASK_SPLIT, build_split_tasks, load_cifar10_split
from .sub_expert import build_sub_expert_model
from .train_sub import build_dataset, normalize_images

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class TrainArgs:
    dataset: str
    split: bool
    memory_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    noise_std: float
    val_fraction: float
    early_stop_patience: int
    seed: int
    output: Path


def parse_args(argv: Iterable[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train single CNN baseline on CIFAR datasets"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument(
        "--split", action="store_true", help="Use Split-CIFAR-10 sequential training"
    )
    parser.add_argument(
        "--memory-size", type=int, default=0, help="Replay buffer size for split mode"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Epochs per run (per task when --split)"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation split fraction for standard mode",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("./results/single"))
    args = parser.parse_args(argv)
    return TrainArgs(
        dataset=args.dataset,
        split=args.split,
        memory_size=args.memory_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        val_fraction=args.val_fraction,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        output=args.output,
    )


def history_to_dict(history: tf.keras.callbacks.History) -> dict[str, List[float]]:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def evaluate_dataset(
    model: tf.keras.Model, images: np.ndarray, labels: np.ndarray, batch_size: int
) -> dict[str, float]:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    result = model.evaluate(ds, verbose=0, return_dict=True)
    return {key: float(val) for key, val in result.items()}


def stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    classes: int,
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in range(classes):
        cls_indices = indices[labels[indices] == cls]
        if cls_indices.size == 0:
            continue
        rng.shuffle(cls_indices)
        val_count = int(cls_indices.size * val_fraction)
        val_idx.extend(cls_indices[:val_count])
        train_idx.extend(cls_indices[val_count:])
    return np.array(train_idx, dtype=np.int32), np.array(val_idx, dtype=np.int32)


def update_memory(
    mem_images: np.ndarray,
    mem_labels: np.ndarray,
    new_images: np.ndarray,
    new_labels: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if capacity <= 0:
        return np.empty((0,) + new_images.shape[1:], dtype=new_images.dtype), np.empty(
            (0,), dtype=new_labels.dtype
        )
    if mem_images.size == 0:
        mem_images = new_images.copy()
        mem_labels = new_labels.copy()
    else:
        mem_images = np.concatenate([mem_images, new_images], axis=0)
        mem_labels = np.concatenate([mem_labels, new_labels], axis=0)
    if mem_images.shape[0] <= capacity:
        return mem_images, mem_labels
    idx = rng.choice(mem_images.shape[0], size=capacity, replace=False)
    return mem_images[idx], mem_labels[idx]


def prepare_model(
    args: TrainArgs, img_shape: Tuple[int, int, int], num_classes: int
) -> tf.keras.Model:
    return build_sub_expert_model(
        use_softmax=False,
        smoothing=0.0,
        noise_std=args.noise_std,
        learning_rate=args.learning_rate,
        img_shape=img_shape,
        num_classes=num_classes,
    )


def make_callbacks(args: TrainArgs) -> List[tf.keras.callbacks.Callback]:
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.3,
            patience=3,
            min_lr=1e-5,
            monitor="val_acc",
            mode="max",
            verbose=1,
        )
    ]
    if args.early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=args.early_stop_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


def train_standard(args: TrainArgs) -> Tuple[dict[str, object], tf.keras.Model]:
    if args.dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean, std = config.CIFAR_CHANNEL_MEAN, config.CIFAR_CHANNEL_STD
        num_classes = config.CIFAR_NUM_CLASSES
        img_shape = config.CIFAR_IMG_SHAPE
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean, std = config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
        num_classes = config.CIFAR10_NUM_CLASSES
        img_shape = config.CIFAR10_IMG_SHAPE

    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(x_train.shape[0], dtype=np.int32)
    train_idx, val_idx = stratified_split(
        indices, y_train, num_classes, args.val_fraction, rng
    )

    x_train_norm = normalize_images(x_train[train_idx], mean, std)
    y_train_split = y_train[train_idx]
    x_val_norm = normalize_images(x_train[val_idx], mean, std)
    y_val_split = y_train[val_idx]
    x_test_norm = normalize_images(x_test, mean, std)

    train_ds = build_dataset(
        x_train_norm,
        y_train_split,
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = build_dataset(
        x_val_norm,
        y_val_split,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )

    model = prepare_model(args, img_shape, num_classes)
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=make_callbacks(args),
        verbose=1,
    )

    test_metrics = evaluate_dataset(model, x_test_norm, y_test, args.batch_size)
    val_metrics = evaluate_dataset(model, x_val_norm, y_val_split, args.batch_size)

    return (
        {
            "mode": "standard",
            "dataset": args.dataset,
            "train_samples": int(x_train_norm.shape[0]),
            "val_samples": int(x_val_norm.shape[0]),
            "test_samples": int(x_test_norm.shape[0]),
            "history": history_to_dict(history),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        model,
    )


def train_split_cifar10(args: TrainArgs) -> Tuple[dict[str, object], tf.keras.Model]:
    if args.dataset != "cifar10":
        raise ValueError("Split mode only supports CIFAR-10")

    data = load_cifar10_split(val_fraction=args.val_fraction, seed=args.seed)
    x_train = normalize_images(
        data.x_train, config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
    )
    x_val = normalize_images(
        data.x_val, config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
    )
    x_test = normalize_images(
        data.x_test, config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
    )
    y_train = data.y_train.astype(np.int32)
    y_val = data.y_val.astype(np.int32)
    y_test = data.y_test.astype(np.int32)

    tasks = build_split_tasks(y_train)
    model = prepare_model(args, config.CIFAR10_IMG_SHAPE, config.CIFAR10_NUM_CLASSES)

    val_ds = build_dataset(
        x_val,
        y_val,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )
    test_ds = build_dataset(
        x_test,
        y_test,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )

    memory_images = np.empty((0,) + config.CIFAR10_IMG_SHAPE, dtype=np.float32)
    memory_labels = np.empty((0,), dtype=np.int32)
    rng = np.random.default_rng(args.seed)

    task_records: List[dict[str, object]] = []

    for task_id, indices in enumerate(tasks):
        task_images = x_train[indices]
        task_labels = y_train[indices]

        if args.memory_size > 0 and memory_images.size > 0:
            train_images = np.concatenate([task_images, memory_images], axis=0)
            train_labels = np.concatenate([task_labels, memory_labels], axis=0)
        else:
            train_images = task_images
            train_labels = task_labels

        train_ds = build_dataset(
            train_images,
            train_labels,
            batch_size=args.batch_size,
            augment=True,
            shuffle=True,
            seed=args.seed + task_id,
        )

        history = model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=make_callbacks(args),
            verbose=1,
        )

        val_metrics = {
            k: float(v)
            for k, v in model.evaluate(val_ds, verbose=0, return_dict=True).items()
        }
        test_metrics = {
            k: float(v)
            for k, v in model.evaluate(test_ds, verbose=0, return_dict=True).items()
        }

        task_records.append(
            {
                "task_id": task_id,
                "classes": [int(cls) for cls in TASK_SPLIT[task_id]],
                "train_samples": int(train_images.shape[0]),
                "history": history_to_dict(history),
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            }
        )

        if args.memory_size > 0:
            memory_images, memory_labels = update_memory(
                memory_images,
                memory_labels,
                task_images,
                task_labels,
                args.memory_size,
                rng,
            )

    final_test = task_records[-1]["test_metrics"] if task_records else {}
    return (
        {
            "mode": "split",
            "dataset": args.dataset,
            "tasks": task_records,
            "memory_size": args.memory_size,
            "final_test": final_test,
        },
        model,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.split:
        results, model = train_split_cifar10(args)
    else:
        results, model = train_standard(args)

    with open(args.output / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    with open(args.output / "run_config.json", "w", encoding="utf-8") as fp:
        record = asdict(args)
        record["output"] = str(args.output)
        json.dump(record, fp, indent=2)

    model.save_weights(args.output / "single_cnn.weights.h5")

    print("Training finished. Metrics saved to", args.output)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
