"""Aggregate metrics from multiple experiment runs into a single report.

Each input directory is expected to contain at least a ``metrics.json`` file
(and optionally a ``run_config.json``).  The script collects these artefacts
and emits a consolidated JSON document so downstream analysis notebooks only
need to parse one file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline/result metrics")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Directories containing metrics.json (and optionally run_config.json)",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for each input (defaults to directory name)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/baselines.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any metrics.json is missing (default: skip silently)",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate(paths: List[Path], labels: List[str], strict: bool) -> List[dict]:
    records: List[dict] = []
    for idx, run_dir in enumerate(paths):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            message = f"Skipping {run_dir}: metrics.json not found"
            if strict:
                raise FileNotFoundError(message)
            print(message)
            continue

        metrics = _load_json(metrics_path)
        record = {
            "label": labels[idx],
            "path": str(run_dir),
            "metrics": metrics,
        }

        config_path = run_dir / "run_config.json"
        if config_path.exists():
            record["config"] = _load_json(config_path)

        records.append(record)
    return records


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = [path.resolve() for path in args.inputs]

    if args.labels and len(args.labels) != len(inputs):
        raise ValueError("--labels count must match number of input directories")

    labels = args.labels if args.labels else [path.name for path in inputs]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": aggregate(inputs, labels, args.strict),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote aggregate results to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
