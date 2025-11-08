"""Configuration-driven experiment orchestrator.

Reads a TOML configuration file to execute the full workflow:
1. Train sub-expert networks.
2. Optimise the PSO gating network.
3. Run baseline evaluations (single CNN, ensemble, random gate, MoE, stacking).

Each stage can be toggled on/off in the configuration. Artefacts from every
stage are collected into a single summary JSON file under the experiment's
result directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import tomllib

from . import ensemble_eval, moe_train, pso_train, random_gate, single_cnn, stacking_fit
from .train_sub import TrainConfig, train as train_experts


@dataclass
class Defaults:
    dataset: str
    num_experts: int
    seed: int
    learning_rate: float


@dataclass
class Paths:
    config_dir: Path
    results_dir: Path
    experts_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment pipeline from a TOML configuration")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML configuration file")
    return parser.parse_args(argv)


def resolve_path(value: Any, base_dir: Path) -> Path:
    if value is None:
        raise ValueError("Path value is required in configuration")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return None


def ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_experts_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
) -> Tuple[Dict[str, Any], Path]:
    output_dir = resolve_path(cfg.get("output_dir", paths.experts_dir), paths.config_dir)
    if not cfg.get("enabled", True):
        return {"status": "skipped", "output": str(output_dir)}, output_dir

    train_cfg = TrainConfig(
        train_with_softmax=cfg.get("train_with_softmax", True),
        num_experts=int(cfg.get("num_experts", defaults.num_experts)),
        subset_pool_fraction=float(cfg.get("subset_pool_fraction", 0.2)),
        epochs=int(cfg.get("epochs", 3)),
        batch_size=int(cfg.get("batch_size", 128)),
        label_smoothing=float(cfg.get("label_smoothing", 0.1)),
        learning_rate=float(cfg.get("learning_rate", defaults.learning_rate)),
        subset_seed=int(cfg.get("subset_seed", defaults.seed)),
        output_root=output_dir,
        noise_std=float(cfg.get("noise_std", 0.05)),
        dataset=str(cfg.get("dataset", defaults.dataset)),
    )
    print(f"[run_experiment] Training experts -> {output_dir}")
    ensure_parent(output_dir)
    train_experts(train_cfg)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "num_experts": train_cfg.num_experts,
        "dataset": train_cfg.dataset,
    }
    config_snapshot = load_json_if_exists(output_dir / "config.json")
    if config_snapshot is not None:
        summary["config"] = config_snapshot
    return summary, output_dir


def run_pso_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
    experts_dir: Path,
) -> Tuple[Dict[str, Any], Path | None]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "pso"), paths.config_dir)
    if not cfg.get("enabled", True):
        return {"status": "skipped", "output": str(output_dir)}, output_dir

    if not experts_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {experts_dir}")

    dataset = str(cfg.get("dataset", defaults.dataset))
    args = [
        "--experts",
        str(experts_dir),
        "--num-experts",
        str(cfg.get("num_experts", defaults.num_experts)),
        "--sample-count",
        str(cfg.get("sample_count", 4096)),
        "--batch-size",
        str(cfg.get("batch_size", 128)),
        "--hidden-units",
        str(cfg.get("hidden_units", 384)),
        "--lr",
        str(cfg.get("learning_rate", defaults.learning_rate)),
        "--seed",
        str(cfg.get("seed", defaults.seed)),
        "--output",
        str(output_dir),
        "--iterations",
        str(cfg.get("iterations", cfg.get("max_iters", 120))),
        "--particles",
        str(cfg.get("particles", 24)),
        "--dataset",
        dataset,
    ]

    print(f"[run_experiment] Running PSO optimisation -> {output_dir}")
    ensure_parent(output_dir)
    pso_train.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "dataset": dataset,
        "cli_args": args,
    }
    fitness = load_json_if_exists(output_dir / "fitness.json")
    if fitness is not None:
        summary["metrics"] = fitness
    return summary, output_dir


def run_single_cnn_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
) -> Dict[str, Any]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "single_cnn"), paths.config_dir)
    if not cfg.get("enabled", False):
        return {"status": "skipped", "output": str(output_dir)}

    args = [
        "--dataset",
        str(cfg.get("dataset", defaults.dataset)),
        "--epochs",
        str(cfg.get("epochs", 60)),
        "--batch-size",
        str(cfg.get("batch_size", 128)),
        "--learning-rate",
        str(cfg.get("learning_rate", defaults.learning_rate)),
        "--noise-std",
        str(cfg.get("noise_std", 0.05)),
        "--val-fraction",
        str(cfg.get("val_fraction", 0.1)),
        "--early-stop-patience",
        str(cfg.get("early_stop_patience", 8)),
        "--seed",
        str(cfg.get("seed", defaults.seed)),
        "--output",
        str(output_dir),
    ]
    if cfg.get("split", False):
        args.append("--split")
        args.extend(["--memory-size", str(cfg.get("memory_size", 0))])

    print(f"[run_experiment] Training single CNN baseline -> {output_dir}")
    ensure_parent(output_dir)
    single_cnn.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "cli_args": args,
    }
    metrics = load_json_if_exists(output_dir / "metrics.json")
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def run_ensemble_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
    experts_dir: Path,
) -> Dict[str, Any]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "ensemble"), paths.config_dir)
    if not cfg.get("enabled", False):
        return {"status": "skipped", "output": str(output_dir)}

    if not experts_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {experts_dir}")

    args = [
        "--experts",
        str(experts_dir),
        "--num-experts",
        str(cfg.get("num_experts", defaults.num_experts)),
        "--dataset",
        str(cfg.get("dataset", defaults.dataset)),
        "--batch-size",
        str(cfg.get("batch_size", 256)),
        "--output",
        str(output_dir),
    ]
    sample_count = int(cfg.get("sample_count", 0))
    if sample_count > 0:
        args.extend(["--sample-count", str(sample_count)])
    seed = cfg.get("seed")
    if seed is not None:
        args.extend(["--seed", str(seed)])

    print(f"[run_experiment] Evaluating ensemble baseline -> {output_dir}")
    ensure_parent(output_dir)
    ensemble_eval.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "cli_args": args,
    }
    metrics = load_json_if_exists(output_dir / "metrics.json")
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def run_random_gate_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
    experts_dir: Path,
) -> Dict[str, Any]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "random_gate"), paths.config_dir)
    if not cfg.get("enabled", False):
        return {"status": "skipped", "output": str(output_dir)}

    if not experts_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {experts_dir}")

    args = [
        "--experts",
        str(experts_dir),
        "--num-experts",
        str(cfg.get("num_experts", defaults.num_experts)),
        "--dataset",
        str(cfg.get("dataset", defaults.dataset)),
        "--trials",
        str(cfg.get("trials", 10)),
        "--batch-size",
        str(cfg.get("batch_size", 256)),
        "--output",
        str(output_dir),
    ]
    sample_count = int(cfg.get("sample_count", 0))
    if sample_count > 0:
        args.extend(["--sample-count", str(sample_count)])
    args.extend(["--seed", str(cfg.get("seed", defaults.seed))])

    print(f"[run_experiment] Evaluating random gate baseline -> {output_dir}")
    ensure_parent(output_dir)
    random_gate.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "cli_args": args,
    }
    metrics = load_json_if_exists(output_dir / "metrics.json")
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def run_moe_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
    experts_dir: Path,
) -> Dict[str, Any]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "moe"), paths.config_dir)
    if not cfg.get("enabled", False):
        return {"status": "skipped", "output": str(output_dir)}

    if not experts_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {experts_dir}")

    args = [
        "--experts",
        str(experts_dir),
        "--num-experts",
        str(cfg.get("num_experts", defaults.num_experts)),
        "--dataset",
        str(cfg.get("dataset", defaults.dataset)),
        "--hidden-units",
        str(cfg.get("hidden_units", 384)),
        "--epochs",
        str(cfg.get("epochs", 30)),
        "--batch-size",
        str(cfg.get("batch_size", 128)),
        "--lr",
        str(cfg.get("lr", defaults.learning_rate)),
        "--seed",
        str(cfg.get("seed", defaults.seed)),
        "--val-fraction",
        str(cfg.get("val_fraction", 0.1)),
        "--output",
        str(output_dir),
    ]
    sample_count = int(cfg.get("sample_count", 0))
    if sample_count > 0:
        args.extend(["--sample-count", str(sample_count)])

    print(f"[run_experiment] Training MoE baseline -> {output_dir}")
    ensure_parent(output_dir)
    moe_train.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "cli_args": args,
    }
    metrics = load_json_if_exists(output_dir / "test_metrics.json")
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def run_stacking_stage(
    cfg: Dict[str, Any],
    defaults: Defaults,
    paths: Paths,
    experts_dir: Path,
) -> Dict[str, Any]:
    output_dir = resolve_path(cfg.get("output_dir", paths.results_dir / "stacking"), paths.config_dir)
    if not cfg.get("enabled", False):
        return {"status": "skipped", "output": str(output_dir)}

    if not experts_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {experts_dir}")

    args = [
        "--experts",
        str(experts_dir),
        "--num-experts",
        str(cfg.get("num_experts", defaults.num_experts)),
        "--dataset",
        str(cfg.get("dataset", defaults.dataset)),
        "--hidden-units",
        str(cfg.get("hidden_units", 512)),
        "--epochs",
        str(cfg.get("epochs", 40)),
        "--batch-size",
        str(cfg.get("batch_size", 256)),
        "--lr",
        str(cfg.get("lr", defaults.learning_rate)),
        "--seed",
        str(cfg.get("seed", defaults.seed)),
        "--val-fraction",
        str(cfg.get("val_fraction", 0.1)),
        "--output",
        str(output_dir),
    ]

    print(f"[run_experiment] Training stacking baseline -> {output_dir}")
    ensure_parent(output_dir)
    stacking_fit.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "cli_args": args,
    }
    metrics = load_json_if_exists(output_dir / "test_metrics.json")
    if metrics is not None:
        summary["metrics"] = metrics
    return summary


def run_experiment(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "rb") as fp:
        data = tomllib.load(fp)

    experiment = data.get("experiment", {})
    dataset = str(experiment.get("dataset", "cifar100"))
    num_experts = int(experiment.get("num_experts", 8))
    seed = int(experiment.get("seed", 123))
    learning_rate = float(experiment.get("learning_rate", 1e-3))

    config_dir = config_path.resolve().parent
    results_dir = resolve_path(
        experiment.get("results_dir", Path("./results") / experiment.get("name", "experiment")),
        config_dir,
    )
    experts_dir = resolve_path(experiment.get("experts_dir", results_dir / "experts"), config_dir)

    ensure_parent(results_dir)

    defaults = Defaults(dataset=dataset, num_experts=num_experts, seed=seed, learning_rate=learning_rate)
    paths = Paths(config_dir=config_dir, results_dir=results_dir, experts_dir=experts_dir)

    summary: Dict[str, Any] = {
        "experiment": {
            "name": experiment.get("name", "experiment"),
            "config": str(config_path.resolve()),
            "dataset": dataset,
            "num_experts": num_experts,
            "seed": seed,
            "results_dir": str(results_dir),
            "experts_dir": str(experts_dir),
        },
        "stages": {},
    }

    experts_summary, experts_output = run_experts_stage(data.get("experts", {}), defaults, paths)
    summary["stages"]["experts"] = experts_summary
    paths.experts_dir = experts_output
    summary["experiment"]["experts_dir"] = str(experts_output)

    pso_summary, _ = run_pso_stage(data.get("pso", {}), defaults, paths, experts_output)
    summary["stages"]["pso"] = pso_summary

    baselines_cfg = data.get("baselines", {})
    baselines_summary: Dict[str, Any] = {}
    if baselines_cfg:
        baselines_summary["single_cnn"] = run_single_cnn_stage(baselines_cfg.get("single_cnn", {}), defaults, paths)
        baselines_summary["ensemble"] = run_ensemble_stage(baselines_cfg.get("ensemble", {}), defaults, paths, experts_output)
        baselines_summary["random_gate"] = run_random_gate_stage(baselines_cfg.get("random_gate", {}), defaults, paths, experts_output)
        baselines_summary["moe"] = run_moe_stage(baselines_cfg.get("moe", {}), defaults, paths, experts_output)
        baselines_summary["stacking"] = run_stacking_stage(baselines_cfg.get("stacking", {}), defaults, paths, experts_output)
    summary["stages"]["baselines"] = baselines_summary

    shutil.copy2(config_path, results_dir / "experiment_config.toml")
    with open(results_dir / "experiment_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[run_experiment] Summary written to {results_dir / 'experiment_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
