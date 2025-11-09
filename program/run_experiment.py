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

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib  # type: ignore[assignment]

from . import config, ensemble_eval, moe_train, pso_train, random_gate, single_cnn, stacking_fit
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
    parser.add_argument("--skip-experts", action="store_true", help="Skip expert training stage regardless of config")
    parser.add_argument("--skip-pso", action="store_true", help="Skip PSO optimisation stage regardless of config")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip all baseline evaluations regardless of config")
    parser.add_argument("--skip-single-cnn", action="store_true", help="Skip the single CNN baseline stage")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip the ensemble baseline stage")
    parser.add_argument("--skip-random-gate", action="store_true", help="Skip the random gate baseline stage")
    parser.add_argument("--skip-moe", action="store_true", help="Skip the MoE baseline stage")
    parser.add_argument("--skip-stacking", action="store_true", help="Skip the stacking baseline stage")
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
        val_fraction=float(cfg.get("val_fraction", 0.1)),
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
    recurrent_steps = int(cfg.get("recurrent_steps", config.PSO_RECURRENT_STEPS))
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
        "--recurrent-steps",
        str(recurrent_steps),
    ]
    # New: allow choosing optimisation split; default to 'train' for unbiased testing later
    optimize_split = str(cfg.get("optimize_split", "train"))
    args.extend(["--optimize-split", optimize_split])

    print(f"[run_experiment] Running PSO optimisation -> {output_dir}")
    ensure_parent(output_dir)
    pso_train.main(args)

    summary: Dict[str, Any] = {
        "status": "completed",
        "output": str(output_dir),
        "dataset": dataset,
        "recurrent_steps": recurrent_steps,
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


def run_experiment(config_path: Path, overrides: Dict[str, bool] | None = None) -> Dict[str, Any]:
    overrides = overrides or {}
    with open(config_path, "rb") as fp:
        data = tomllib.load(fp)

    experiment = dict(data.get("experiment", {}))
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

    experts_cfg = dict(data.get("experts", {}))
    if overrides.get("skip_experts"):
        experts_cfg["enabled"] = False
    experts_summary, experts_output = run_experts_stage(experts_cfg, defaults, paths)
    summary["stages"]["experts"] = experts_summary
    paths.experts_dir = experts_output
    summary["experiment"]["experts_dir"] = str(experts_output)

    pso_cfg = dict(data.get("pso", {}))
    if overrides.get("skip_pso"):
        pso_cfg["enabled"] = False
    pso_summary, _ = run_pso_stage(pso_cfg, defaults, paths, experts_output)
    summary["stages"]["pso"] = pso_summary

    baselines_cfg = {
        key: dict(value) for key, value in data.get("baselines", {}).items()
    }
    baselines_summary: Dict[str, Any] = {}
    if overrides.get("skip_baselines"):
        baselines_cfg = {}
    else:
        if overrides.get("skip_single_cnn") and "single_cnn" in baselines_cfg:
            baselines_cfg["single_cnn"]["enabled"] = False
        if overrides.get("skip_ensemble") and "ensemble" in baselines_cfg:
            baselines_cfg["ensemble"]["enabled"] = False
        if overrides.get("skip_random_gate") and "random_gate" in baselines_cfg:
            baselines_cfg["random_gate"]["enabled"] = False
        if overrides.get("skip_moe") and "moe" in baselines_cfg:
            baselines_cfg["moe"]["enabled"] = False
        if overrides.get("skip_stacking") and "stacking" in baselines_cfg:
            baselines_cfg["stacking"]["enabled"] = False

    if baselines_cfg:
        # Only execute stages explicitly enabled to avoid overwriting prior results with 'skipped'
        if baselines_cfg.get("single_cnn", {}).get("enabled", False):
            baselines_summary["single_cnn"] = run_single_cnn_stage(baselines_cfg.get("single_cnn", {}), defaults, paths)
        if baselines_cfg.get("ensemble", {}).get("enabled", False):
            baselines_summary["ensemble"] = run_ensemble_stage(baselines_cfg.get("ensemble", {}), defaults, paths, experts_output)
        if baselines_cfg.get("random_gate", {}).get("enabled", False):
            baselines_summary["random_gate"] = run_random_gate_stage(baselines_cfg.get("random_gate", {}), defaults, paths, experts_output)
        if baselines_cfg.get("moe", {}).get("enabled", False):
            baselines_summary["moe"] = run_moe_stage(baselines_cfg.get("moe", {}), defaults, paths, experts_output)
        if baselines_cfg.get("stacking", {}).get("enabled", False):
            baselines_summary["stacking"] = run_stacking_stage(baselines_cfg.get("stacking", {}), defaults, paths, experts_output)
    summary["stages"]["baselines"] = baselines_summary

    shutil.copy2(config_path, results_dir / "experiment_config.toml")
    # Append/merge into existing summary if present (do not clobber completed stages with 'skipped')
    final_summary: Dict[str, Any] = summary
    existing_path = results_dir / "experiment_summary.json"
    if existing_path.exists():
        with existing_path.open("r", encoding="utf-8") as fp:
            existing = json.load(fp)
        final = existing
        # Update experiment meta minimally
        final.setdefault("experiment", {}).update({
            "config": summary["experiment"].get("config"),
            "results_dir": summary["experiment"].get("results_dir"),
            "experts_dir": summary["experiment"].get("experts_dir"),
        })
        final.setdefault("stages", {})
        # Experts stage
        new_experts = summary["stages"].get("experts")
        if new_experts and new_experts.get("status") == "completed":
            final["stages"]["experts"] = new_experts
        # PSO stage
        new_pso = summary["stages"].get("pso")
        if new_pso and new_pso.get("status") == "completed":
            final["stages"]["pso"] = new_pso
        # Baselines: update only completed ones to preserve prior results
        new_base = summary["stages"].get("baselines", {})
        base_final = final["stages"].setdefault("baselines", {})
        for key, val in new_base.items():
            if isinstance(val, dict) and val.get("status") == "completed":
                base_final[key] = val
        final_summary = final
    with open(existing_path, "w", encoding="utf-8") as fp:
        json.dump(final_summary, fp, indent=2)
    print(f"[run_experiment] Summary written to {results_dir / 'experiment_summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {
        "skip_experts": args.skip_experts,
        "skip_pso": args.skip_pso,
        "skip_baselines": args.skip_baselines,
        "skip_single_cnn": args.skip_single_cnn,
        "skip_ensemble": args.skip_ensemble,
        "skip_random_gate": args.skip_random_gate,
        "skip_moe": args.skip_moe,
        "skip_stacking": args.skip_stacking,
    }
    run_experiment(args.config, overrides)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
