"""Program package: PSO-driven dynamic expert routing sandbox.

Submodules:
- train_sub: CIFAR-100 sub expert training
- pso_train: PSO gating optimisation
- train_pipeline: Orchestrated experts + PSO pipeline
- visualize_gating: Animation of gating evolution
- train_sub_nn: MNIST baseline (if present)
"""

__all__ = [
    "config",
    "sub_expert",
    "train_sub",
    "pso_train",
    "train_pipeline",
    "visualize_gating",
]
