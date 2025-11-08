"""Shared utilities for PSO gating optimisation and evaluation.

This module provides small helpers used by the PSO runner and later the
evaluation/visualisation steps: an in-memory evaluation set, a Keras
weight <-> vector adapter, and expert model loaders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tensorflow import keras

from . import config
from .sub_expert import build_sub_expert_model


@dataclass
class EvalBatch:
    images: np.ndarray
    labels: np.ndarray
    expert_logits: np.ndarray


class EvaluationSet:
    """Simple iterable wrapper over an in-memory evaluation split.

    Keeps data as numpy arrays for fast repeated access during PSO.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, expert_logits: np.ndarray, batch_size: int) -> None:
        self.images = images
        self.labels = labels
        self.expert_logits = expert_logits
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.images.shape[0] / self.batch_size)

    def iterate(self) -> Iterable[EvalBatch]:
        total = self.images.shape[0]
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            yield EvalBatch(
                images=self.images[start:end],
                labels=self.labels[start:end],
                expert_logits=self.expert_logits[start:end],
            )


class WeightAdapter:
    """Flatten / restore Keras model weights to a 1-D numpy vector.

    Usage:
      adapter = WeightAdapter(model)
      vec = adapter.sample_initial(rng)
      adapter.assign_from_vector(vec)
    """

    def __init__(self, model: keras.Model) -> None:
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.counts = [int(np.prod(shape)) for shape in self.shapes]
        self.dimension = int(sum(self.counts))

    def assign_from_vector(self, vector: np.ndarray) -> None:
        if vector.size != self.dimension:
            raise ValueError(f"Vector length {vector.size} != expected {self.dimension}")
        weights: List[np.ndarray] = []
        offset = 0
        for shape, count in zip(self.shapes, self.counts):
            segment = vector[offset : offset + count]
            weights.append(segment.reshape(shape))
            offset += count
        self.model.set_weights(weights)

    def sample_initial(self, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
        weights = self.model.get_weights()
        flat = np.concatenate([w.reshape(-1) for w in weights])
        noise = rng.normal(scale=scale, size=flat.shape)
        return flat + noise


def normalize_images(images: np.ndarray) -> np.ndarray:
    mean = np.asarray(config.CIFAR_CHANNEL_MEAN, dtype=np.float32)
    std = np.asarray(config.CIFAR_CHANNEL_STD, dtype=np.float32)
    return (images.astype("float32") / 255.0 - mean) / std


def load_expert_models(expert_root: Path, num_experts: int, learning_rate: float) -> List[keras.Model]:
    models: List[keras.Model] = []
    for expert_id in range(num_experts):
        expert_dir = expert_root / f"expert_{expert_id:02d}"
        weights_path = expert_dir / "logits.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found for expert {expert_id}: {weights_path}")
        model = build_sub_expert_model(use_softmax=False, smoothing=0.0, learning_rate=learning_rate, noise_std=0.0)
        model.load_weights(weights_path)
        models.append(model)
    return models


def precompute_expert_logits(models: List[keras.Model], images: np.ndarray, batch_size: int) -> np.ndarray:
    logits_per_expert = []
    for model in models:
        logits = model.predict(images, batch_size=batch_size, verbose=0)
        logits_per_expert.append(logits)
    stacked = np.stack(logits_per_expert, axis=1)
    return stacked