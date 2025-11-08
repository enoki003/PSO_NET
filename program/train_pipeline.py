"""End-to-end training pipeline orchestrating experts and PSO gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from . import config
from . import pso_train
from .train_sub import TrainConfig, train


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experts and optimise gating via PSO.")
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert training stage.")
    parser.add_argument("--skip-pso", action="store_true", help="Skip PSO optimisation stage.")

    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--subset-pool-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--logits-only", action="store_true", help="Train experts without softmax head.")

    parser.add_argument("--experts-output", type=Path, default=Path("./models/cifar_sub_experts"))
    parser.add_argument("--gating-output", type=Path, default=Path("./models/pso_gating"))

    parser.add_argument("--pso-sample-count", type=int, default=4096)
    parser.add_argument("--pso-batch-size", type=int, default=128)
    parser.add_argument("--pso-hidden-units", type=int, default=384)
    parser.add_argument("--pso-iterations", type=int, default=config.PSO_DEFAULT_ITERATIONS)
    parser.add_argument("--pso-particles", type=int, default=config.PSO_DEFAULT_PARTICLES)
    parser.add_argument("--pso-seed", type=int, default=123)

    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    train_with_softmax = not args.logits_only

    if not args.skip_experts:
        cfg = TrainConfig(
            train_with_softmax=train_with_softmax,
            num_experts=args.num_experts,
            subset_pool_fraction=args.subset_pool_fraction,
            epochs=args.epochs,
            batch_size=args.batch_size,
            label_smoothing=args.label_smoothing,
            learning_rate=args.learning_rate,
            subset_seed=args.subset_seed,
            output_root=args.experts_output,
            noise_std=args.noise_std,
        )
        train(cfg)

    if not args.skip_pso:
        pso_args = [
            "--experts",
            str(args.experts_output),
            "--num-experts",
            str(args.num_experts),
            "--sample-count",
            str(args.pso_sample_count),
            "--batch-size",
            str(args.pso_batch_size),
            "--hidden-units",
            str(args.pso_hidden_units),
            "--lr",
            str(args.learning_rate),
            "--seed",
            str(args.pso_seed),
            "--output",
            str(args.gating_output),
            "--iterations",
            str(args.pso_iterations),
            "--particles",
            str(args.pso_particles),
        ]
        pso_train.main(pso_args)

    summary = {
        "experts_trained": not args.skip_experts,
        "pso_ran": not args.skip_pso,
        "experts_output": str(args.experts_output),
        "gating_output": str(args.gating_output),
        "num_experts": args.num_experts,
        "pso_iterations": args.pso_iterations,
        "pso_particles": args.pso_particles,
    }
    args.gating_output.mkdir(parents=True, exist_ok=True)
    with open(args.gating_output / "pipeline_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
