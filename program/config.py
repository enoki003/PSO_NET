"""Centralised configuration values shared across training scripts."""

from __future__ import annotations

# CIFAR-100 (legacy path; used by existing scripts)
CIFAR_CHANNEL_MEAN: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR_CHANNEL_STD: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)
CIFAR_IMG_SHAPE: tuple[int, int, int] = (32, 32, 3)
CIFAR_NUM_CLASSES: int = 100

# CIFAR-10 constants for new experiments
CIFAR10_CHANNEL_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_CHANNEL_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
CIFAR10_IMG_SHAPE: tuple[int, int, int] = (32, 32, 3)
CIFAR10_NUM_CLASSES: int = 10

PSO_RECURRENT_STEPS: int = 3
PSO_DEFAULT_PARTICLES: int = 24
PSO_DEFAULT_ITERATIONS: int = 120
PSO_INERTIA_MAX: float = 0.9
PSO_INERTIA_MIN: float = 0.4
PSO_COGNITIVE_COEFF: float = 1.7
PSO_SOCIAL_COEFF: float = 1.7
PSO_VELOCITY_MAX: float = 0.05  # velocity clipping (L-infinity)
PSO_EARLY_STOP_PATIENCE: int = 20

FITNESS_ALPHA: float = 1.0
FITNESS_BETA: float = 0.1
FITNESS_GAMMA: float = 0.05
FITNESS_DELTA: float = 0.05
