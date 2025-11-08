"""Utilities for constructing gating networks used in PSO and evaluation.

`img_shape` をパラメータ化し CIFAR-10/100 両対応。既存呼び出しはデフォルト値を使用。
"""

from __future__ import annotations

from tensorflow import keras

from . import config


def build_gating_model(num_experts: int, hidden_units: int, *, img_shape: tuple[int, int, int] = config.CIFAR_IMG_SHAPE) -> keras.Model:
    inputs = keras.Input(shape=img_shape)
    x = keras.layers.Conv2D(48, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(hidden_units, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_experts * num_experts, name="gating_logits")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="pso_gating_network")
