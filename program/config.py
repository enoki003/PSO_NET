"""Centralised configuration values shared across training scripts."""

from __future__ import annotations

CIFAR_CHANNEL_MEAN: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR_CHANNEL_STD: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
CIFAR_IMG_SHAPE: tuple[int, int, int] = (32, 32, 3)
CIFAR_NUM_CLASSES: int = 100

PSO_RECURRENT_STEPS: int = 3
PSO_DEFAULT_PARTICLES: int = 24
PSO_DEFAULT_ITERATIONS: int = 120
PSO_INERTIA_MAX: float = 0.9
PSO_INERTIA_MIN: float = 0.4
PSO_COGNITIVE_COEFF: float = 1.7
PSO_SOCIAL_COEFF: float = 1.7

FITNESS_ALPHA: float = 1.0
FITNESS_BETA: float = 0.1
FITNESS_GAMMA: float = 0.05
FITNESS_DELTA: float = 0.05
