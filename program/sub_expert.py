"""Model architecture helpers for CIFAR-100 sub experts."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def _conv_block(filters: int, x: tf.Tensor) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation('relu')(x)
    return x


def build_sub_expert_model(
    *,
    use_softmax: bool,
    smoothing: float,
    noise_std: float = 0.05,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Construct the CIFAR-100 specialist model used by training and PSO."""
    inputs = layers.Input(shape=config.CIFAR_IMG_SHAPE)
    x = layers.GaussianNoise(noise_std)(inputs)

    for filters in (64, 96, 128):
        x = _conv_block(filters, x)
        x = _conv_block(filters, x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = _conv_block(160, x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(config.CIFAR_NUM_CLASSES, name='logits')(x)

    if use_softmax:
        outputs = layers.Activation('softmax', name='softmax')(logits)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
        metrics = [
            keras.metrics.CategoricalAccuracy(name='acc'),
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top5'),
        ]
    else:
        outputs = logits
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='acc'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
        ]

    model = keras.Model(inputs=inputs, outputs=outputs, name='cifar_sub_expert')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
