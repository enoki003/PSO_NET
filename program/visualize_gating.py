"""Visualise gating matrix evolution as a network animation.

Reads `pso_history.json` written by `program.pso_train` and animates the
average gating matrix per iteration as a weighted directed graph. Edges
with small weight are dimmed.

Note: requires `matplotlib` and `networkx` installed. Save GIF/MP4 via
matplotlib's writers (ffmpeg/imageio).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate PSO gating matrices")
    parser.add_argument("--history", type=Path, default=Path("./models/pso_gating/pso_history.json"))
    parser.add_argument("--out", type=Path, default=Path("./models/pso_gating/gating_anim.mp4"))
    parser.add_argument("--threshold", type=float, default=0.01, help="Edges below this weight are hidden")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--show", action="store_true", help="Display the animation interactively via matplotlib")
    return parser.parse_args(argv)


def build_graph_from_matrix(mat: np.ndarray, threshold: float = 0.01) -> nx.DiGraph:
    n = mat.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            w = float(mat[i, j])
            if w >= threshold:
                G.add_edge(i, j, weight=w)
    return G


def draw_frame(ax, G: nx.DiGraph, pos, labels=True):
    ax.clear()
    weights = [d.get("weight", 0.0) for _, _, d in G.edges(data=True)]
    if len(weights) == 0:
        widths = []
    else:
        widths = [max(0.5, 4.0 * w) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_size=300, ax=ax, node_color="#1f78b4")
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.9, edge_color="#333333", arrowsize=12, ax=ax)
    if labels:
        lbls = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=lbls, font_size=8, ax=ax)
    ax.set_axis_off()


def animate_matrices(
    matrices: List[np.ndarray],
    out_path: Path,
    threshold: float = 0.01,
    fps: int = 6,
    show: bool = False,
) -> None:
    n = matrices[0].shape[0]
    # fixed node positions on a circle for stable layout
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx):
        mat = matrices[frame_idx]
        G = build_graph_from_matrix(mat, threshold=threshold)
        draw_frame(ax, G, pos)
        ax.set_title(f"Iteration {frame_idx}")

    anim = FuncAnimation(fig, update, frames=len(matrices), interval=max(1, int(1000 / max(1, fps))))

    writer = None
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        try:
            writer = PillowWriter(fps=fps)
        except Exception as exc:  # pillow may be missing
            print(f"GIF writer unavailable ({exc}); falling back to matplotlib default.")
            writer = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if writer is not None:
            anim.save(out_path, writer=writer, dpi=150)
        else:
            anim.save(out_path, fps=fps, dpi=150)
        print(f"Saved animation to {out_path}")
    except Exception as exc:
        print(f"Failed to save animation ({exc}).")
        if not show:
            print("Re-run with --show after installing ffmpeg/pillow or choose a different format.")

    if show:
        plt.show()
    else:
        plt.close(fig)


def load_matrices_from_history(path: Path) -> List[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mats = []
    for entry in data:
        mat = entry.get("avg_gating")
        if mat is None:
            continue
        mats.append(np.asarray(mat, dtype=np.float32))
    return mats


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    mats = load_matrices_from_history(args.history)
    if len(mats) == 0:
        raise SystemExit("No gating matrices found in history file")
    animate_matrices(mats, args.out, threshold=args.threshold, fps=args.fps, show=args.show)


if __name__ == "__main__":
    main()
