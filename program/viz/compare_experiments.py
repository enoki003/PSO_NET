from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare experiments under results dir and print ACC/diag_mean table")
    p.add_argument("--results-dir", type=Path, default=Path("configs/results/cifar10_full"))
    p.add_argument("--experiments", nargs="*", type=str, default=None, help="Subdirs to compare (default: pso,pso_temp3_beta0,ensemble,moe,random_gate,single_cnn,stacking)")
    p.add_argument("--out", type=Path, default=Path("./results/viz"))
    return p.parse_args(argv)


def load_metrics(exp_dir: Path) -> dict:
    m = {}
    f = exp_dir / "fitness.json"
    if f.exists():
        j = json.load(open(f, "r", encoding="utf-8"))
        # top-level should contain accuracy + train/test blocks
        m["score"] = j.get("score")
        m["accuracy"] = j.get("accuracy")
        m["diag_mean"] = j.get("diag_mean", None)
    else:
        # try metrics.json e.g ensemble or single cnn
        f2 = exp_dir / "metrics.json"
        if f2.exists():
            j = json.load(open(f2, "r", encoding="utf-8"))
            # different structures exist
            m["accuracy"] = j.get("test_accuracy") or j.get("accuracy") or j.get("acc")
        else:
            f3 = exp_dir / "test_metrics.json"
            if f3.exists():
                j = json.load(open(f3, "r", encoding="utf-8"))
                m["accuracy"] = j.get("accuracy") or j.get("acc")
    return m


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    default_exps = ["pso", "pso_temp3_beta0", "ensemble", "moe", "random_gate", "single_cnn", "stacking"]
    exps = args.experiments if args.experiments else default_exps
    results_dir = args.results_dir

    rows = []
    for e in exps:
        d = results_dir / e
        if not d.exists():
            print(f"Skipping missing experiment: {d}")
            continue
        metrics = load_metrics(d)
        rows.append((e, d, metrics))

    # Print summary table
    print("Experiment,Accuracy,Diag_mean,Source")
    for e, d, m in rows:
        acc = m.get("accuracy")
        diag = m.get("diag_mean")
        src = d
        print(f"{e},{acc},{diag},{src}")

    # Save CSV
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "comparison.csv"
    with open(csv, "w", encoding="utf-8") as fp:
        fp.write("experiment,accuracy,diag_mean,source\n")
        for e, d, m in rows:
            acc = m.get("accuracy")
            diag = m.get("diag_mean")
            fp.write(f"{e},{acc},{diag},{d}\n")
    print(f"Saved comparison CSV to {csv}")


if __name__ == "__main__":
    main()
