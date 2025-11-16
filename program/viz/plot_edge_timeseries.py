from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser("plot_edge_timeseries")
    p.add_argument("--top_edges_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("./results/viz/edge_timeseries.png"))
    p.add_argument("--edge", type=str, default=None, help="Edge to plot in format src,dst (default: top 5 by frequency)" )
    return p.parse_args(argv)


def load_top_edges(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append(r)
    return rows


def plot_edges(rows: List[dict], out: Path, edge: str | None = None):
    # group by edge (src->dst)
    groups = {}
    iters = sorted(list({int(r["iteration"]) for r in rows}))
    for r in rows:
        k = (int(r["edge_src"]), int(r["edge_dst"]))
        groups.setdefault(k, []).append((int(r["iteration"]), float(r["weight"])))

    # choose top edges by overall max weight if not specified
    if edge is None:
        edge_keys = sorted(groups.keys(), key=lambda k: max([w for (_, w) in groups[k]]), reverse=True)[:5]
    else:
        src, dst = map(int, edge.split(","))
        edge_keys = [(src, dst)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in edge_keys:
        arr = groups.get(k, [])
        if not arr:
            continue
        arr.sort(key=lambda x: x[0])
        xs = [i for (i, _) in arr]
        ys = [w for (_, w) in arr]
        ax.plot(xs, ys, marker="o", label=f"{k[0]}→{k[1]}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Edge weight")
    ax.grid(alpha=0.3)
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved edge timeseries to {out}")


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    rows = load_top_edges(args.top_edges_csv)
    plot_edges(rows, args.out, edge=args.edge)


if __name__ == "__main__":
    main()
