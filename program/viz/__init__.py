"""program.viz

Small convenience wrapper for visualization subcommands. This module re-exports
existing plotting/animation functions that live in `program.*` so users can run
"python -m program.viz.cli" for a unified visualization CLI.
"""
from __future__ import annotations

# Re-export key functions from existing modules to avoid changing core file layout
from ..visualize_gating import animate_matrices, plot_static, load_history, extract_matrices
from ..visualize_metrics import visualize_single_run, visualize_aggregate
from ..auto_visualize import auto_visualize
from ..visualize_input_dynamics import visualize_for_samples

__all__ = [
    "animate_matrices",
    "plot_static",
    "load_history",
    "extract_matrices",
    "visualize_single_run",
    "visualize_aggregate",
    "auto_visualize",
    "visualize_for_samples",
]
