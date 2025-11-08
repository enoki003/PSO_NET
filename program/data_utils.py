"""Data loading and task splitting utilities for CIFAR-10 experiments.

Provides:
- load_cifar10_split: returns train/val/test arrays with stratified splits.
- build_split_tasks: yields task-specific index sets for Split-CIFAR-10 Class-IL setting.
- build_tf_dataset: creates tf.data.Dataset with augmentation & normalization order:
    augment -> normalize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10

from . import config

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class CIFAR10Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_cifar10_split(val_fraction: float = 0.1, seed: int = 42) -> CIFAR10Data:
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    y_train_full = y_train_full.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)

    rng = np.random.default_rng(seed)
    # stratified split train -> train/val
    train_indices: List[int] = []
    val_indices: List[int] = []
    for cls in range(config.CIFAR10_NUM_CLASSES):
        cls_indices = np.where(y_train_full == cls)[0]
        rng.shuffle(cls_indices)
        val_count = int(cls_indices.size * val_fraction)
        val_indices.extend(cls_indices[:val_count])
        train_indices.extend(cls_indices[val_count:])
    train_indices = np.array(train_indices, dtype=np.int32)
    val_indices = np.array(val_indices, dtype=np.int32)

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    return CIFAR10Data(x_train, y_train, x_val, y_val, x_test, y_test)


TASK_SPLIT = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


def build_split_tasks(labels: np.ndarray) -> List[np.ndarray]:
    """Return list of index arrays for each Split-CIFAR-10 task (Class-IL).

    Labels always remain 0..9; tasks are only used for incremental ordering.
    """
    tasks: List[np.ndarray] = []
    for pair in TASK_SPLIT:
        idx = np.where(np.isin(labels, pair))[0]
        tasks.append(idx.astype(np.int32))
    return tasks


def _augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    # pad to 40x40 then random crop back to 32x32
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=config.CIFAR10_IMG_SHAPE)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def _normalize(image: tf.Tensor) -> tf.Tensor:
    mean = tf.constant(config.CIFAR10_CHANNEL_MEAN, dtype=tf.float32)
    std = tf.constant(config.CIFAR10_CHANNEL_STD, dtype=tf.float32)
    return (image / 255.0 - mean) / std


def build_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    images_tf = tf.convert_to_tensor(images, dtype=tf.uint8)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((images_tf, labels_tf))
    if shuffle:
        ds = ds.shuffle(buffer_size=images.shape[0], seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda x, y: (_augment(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (_normalize(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
