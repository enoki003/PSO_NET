from __future__ import annotations

"""Auto-visualize experiment outputs described in an experiment_summary.json.

Uses existing tools:
- program.visualize_metrics.visualize_single_run for baseline / single-cnn histories
- program.visualize_gating.plot_static for PSO results (and optionally animation)

This script writes to `results/.../viz` by default and will not delete previous outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from . import visualize_metrics
from .visualize_gating import plot_static, load_history, extract_matrices


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-generate figures from experiment_summary.json")
    parser.add_argument("--summary", type=Path, required=True, help="Path to experiment_summary.json")
    parser.add_argument("--out", type=Path, default=None, help="Optional output dir to override results/viz")
    parser.add_argument("--show", action="store_true", help="Display plots interactively (local) - not recommended on headless CI")
    parser.add_argument("--psograph", action="store_true", help="Export gating graph snapshot as well")
    parser.add_argument("--threshold", type=float, default=0.01, help="Edge threshold for gating graph")
    parser.add_argument("--input-dynamics", action="store_true", help="Also run input dynamics GIF generation for a sample per class")
    return parser.parse_args(argv)


def auto_visualize(summary_path: Path, out_override: Path | None = None, show: bool = False, psograph: bool = False, threshold: float = 0.01) -> None:
    if not summary_path.exists():
        raise SystemExit(f"Summary file not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    results_dir = Path(summary["experiment"]["results_dir"]) if "experiment" in summary and "results_dir" in summary["experiment"] else summary_path.parent
    viz_dir = out_override or (Path(results_dir) / "viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Baselines: single CNN -> plot history
    baselines = summary.get("stages", {}).get("baselines", {})
    single = baselines.get("single_cnn")
    if single and "output" in single:
        run_dir = Path(single["output"]) if isinstance(single["output"], str) else Path(single["output"])
        out = viz_dir / "single_cnn"
        print(f"Visualizing single CNN results from {run_dir} -> {out}")
        visualize_metrics.visualize_single_run(run_dir, out, show)

    # Ensemble / random gate: just copy metrics file -> optionally could draw bar plots; for now we reuse aggregate
    # We will collect paths for aggregate
    aggregate_entries = []
    for name in ("ensemble", "random_gate", "moe", "stacking"):
        rec = baselines.get(name)
        if rec and "output" in rec:
            path = Path(rec["output"]) if isinstance(rec["output"], str) else rec["output"]
            metrics = path / ("metrics.json" if (path / "metrics.json").exists() else "test_metrics.json")
            if metrics.exists():
                aggregate_entries.append({"label": name, "path": str(path), "metrics": json.load(metrics.open("r", encoding="utf-8"))})

    if aggregate_entries:
        # Write a baselines.json file for viz
        agg_path = viz_dir / "baselines.json"
        with agg_path.open("w", encoding="utf-8") as fp:
            json.dump({"entries": aggregate_entries}, fp, indent=2)
        print(f"Wrote aggregated baseline summary -> {agg_path}")
        # Use existing viz tool to make bar charts
        visualize_metrics.visualize_aggregate(agg_path, viz_dir, show)

    # PSO gating: static summary
    pso = summary.get("stages", {}).get("pso")
    if pso and "output" in pso:
        pso_dir = Path(pso["output"]) if isinstance(pso["output"], str) else pso["output"]
        history_json = pso_dir / "pso_history.json"
        if history_json.exists():
            history = load_history(history_json)
            matrices = extract_matrices(history)
            out_pso_dir = viz_dir / "pso"
            print(f"Visualizing PSO gating matrices -> {out_pso_dir}")
            plot_static(history, matrices, out_pso_dir, export_graph=psograph, threshold=threshold, show=show)
        else:
            print(f"No PSO pso_history.json found at {history_json}")
        if args.input_dynamics:
            # call visualize_input_dynamics for one sample per class
            from .visualize_input_dynamics import main as viz_input_main

            print("Running input dynamics visualization for a sample per class...")
            viz_input_main([
                "--experts",
                str(Path(summary["experiment"]["experts_dir"])),
                "--gating",
                str(Path(pso["output"])),
                "--dataset",
                summary["experiment"].get("dataset", "cifar10"),
                "--per-class",
                "1",
                "--out",
                str(out / "input_dynamics"),
            ])

    print("Auto-visualize finished; outputs in:", viz_dir)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    auto_visualize(args.summary, out_override=args.out, show=args.show, psograph=args.psograph, threshold=args.threshold)


if __name__ == "__main__":
    main()
