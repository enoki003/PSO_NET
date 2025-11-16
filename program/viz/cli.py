"""Visualization CLI

Provides a single entrypoint for the most common visualization tasks so the
project is easier to navigate. Keeps existing modules intact and merely calls
into their existing `main` functions for backwards compatibility.

Usage examples:

python -m program.viz.cli gating --history path/to/pso_history.json --out out.gif
python -m program.viz.cli metrics --input results/.../metrics.json
python -m program.viz.cli auto --summary results/.../experiment_summary.json
python -m program.viz.cli input --experts models/cifar_sub_experts --gating results/.../pso

"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

from .. import auto_visualize as _auto_module
from . import extract_matrices, load_history  # re-exported from visualize_gating
from .. import visualize_gating as _vg
from .. import visualize_metrics as _vm
from .. import visualize_input_dynamics as _vid


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("program.viz CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gating", help="Visualize PSO gating history/avg matrices")
    g.add_argument("--history", type=str, required=True)
    g.add_argument("--out", type=str, default="./results/viz/gating_anim.mp4")
    g.add_argument("--fps", type=int, default=6)
    g.add_argument("--static", action="store_true")
    g.add_argument("--graph", action="store_true")
    g.add_argument("--show", action="store_true")

    m = sub.add_parser("metrics", help="Plot training and evaluation metrics")
    m.add_argument("--input", type=str, required=True)
    m.add_argument("--output", type=str, default="./results/viz")
    m.add_argument("--show", action="store_true")

    a = sub.add_parser("auto", help="Auto visualize from experiment_summary.json")
    a.add_argument("--summary", type=str, required=True)
    a.add_argument("--out", type=str, default=None)
    a.add_argument("--show", action="store_true")
    a.add_argument("--psograph", action="store_true")
    a.add_argument("--threshold", type=float, default=0.01)

    i = sub.add_parser("input", help="Visualize input-dependent dynamics and GIFs")
    i.add_argument("--experts", type=str, required=True)
    i.add_argument("--gating", type=str, required=True)
    i.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    i.add_argument("--per-class", type=int, default=0)
    i.add_argument("--sample-ids", nargs="*", type=int)
    i.add_argument("--out", type=str, default="./results/viz/input_dynamics")
    i.add_argument("--recurrent-steps", type=int, default=3)
    i.add_argument("--batch-size", type=int, default=128)
    i.add_argument("--hidden-units", type=int, default=384)
    i.add_argument("--num-experts", type=int, default=8)

    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd == "gating":
        # delegate to visualize_gating.main to keep one code path
        gating_args = [
            "--history",
            args.history,
            "--out",
            args.out,
            "--fps",
            str(args.fps),
        ]
        if args.static:
            gating_args.append("--static")
        if args.graph:
            gating_args.append("--graph")
        if args.show:
            gating_args.append("--show")
        _vg.main(gating_args)
    elif args.cmd == "metrics":
        metrics_args = ["--input", args.input, "--output", args.output]
        if args.show:
            metrics_args.append("--show")
        _vm.main(metrics_args)
    elif args.cmd == "auto":
        # call auto_visualize's main wrapper
        auto_args = ["--summary", args.summary]
        if args.out is not None:
            auto_args += ["--out", args.out]
        if args.show:
            auto_args.append("--show")
        if args.psograph:
            auto_args.append("--psograph")
        auto_args += ["--threshold", str(args.threshold)]
        _auto_module.main(auto_args)
    elif args.cmd == "input":
        # build args for the input dynamics CLI
        input_args = [
            "--experts",
            args.experts,
            "--gating",
            args.gating,
            "--dataset",
            args.dataset,
            "--out",
            args.out,
            "--recurrent-steps",
            str(args.recurrent_steps),
            "--batch-size",
            str(args.batch_size),
            "--hidden-units",
            str(args.hidden_units),
            "--num-experts",
            str(args.num_experts),
        ]
        if args.per_class:
            input_args += ["--per-class", str(args.per_class)]
        if args.sample_ids:
            input_args += [*(str(x) for x in args.sample_ids)]
        _vid.main(input_args)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
