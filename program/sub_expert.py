"""Model architecture helpers for CIFAR sub experts.

デフォルトは CIFAR-100 前提だが、CIFAR-10 実験向けに `img_shape` と `num_classes` を
パラメータ化している。既存呼び出し互換のためデフォルト引数は従来値。
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def _conv_block(filters: int, x: tf.Tensor) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    return x


def build_sub_expert_model(
    *,
    use_softmax: bool,
    smoothing: float,
    noise_std: float = 0.05,
    learning_rate: float = 1e-3,
    img_shape: tuple[int, int, int] = config.CIFAR_IMG_SHAPE,
    num_classes: int = config.CIFAR_NUM_CLASSES,
) -> keras.Model:
    """Construct a CIFAR specialist model.

    Parameters
    ----------
    use_softmax : bool
        出力層に softmax を付与するか（事前学習用） / False の場合 logits を返す。
    smoothing : float
        ラベルスムージング係数 (softmax 使用時のみ有効)。
    noise_std : float
        入力 GaussianNoise の標準偏差。PSO 評価時は 0 にする。
    learning_rate : float
        Adam の学習率。
    img_shape : tuple[int,int,int]
        入力画像形状。CIFAR-10/100 で (32,32,3)。
    num_classes : int
        クラス数。CIFAR-10=10, CIFAR-100=100。
    """
    inputs = layers.Input(shape=img_shape)
    x = layers.GaussianNoise(noise_std)(inputs)

    for filters in (64, 96, 128):
        x = _conv_block(filters, x)
        x = _conv_block(filters, x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = _conv_block(160, x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_classes, name="logits")(x)

    if use_softmax:
        outputs = layers.Activation("softmax", name="softmax")(logits)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
        metrics = [
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ]
    else:
        outputs = logits
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ]

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_sub_expert")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
