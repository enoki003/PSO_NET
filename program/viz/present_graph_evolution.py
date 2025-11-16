from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("present_graph_evolution")
    p.add_argument("--history", type=Path, required=True, help="Path to pso_history.json")
    p.add_argument("--out", type=Path, required=True, help="Output path (mp4/gif)")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--top-k", type=int, default=8, help="Show at most top-k edges")
    return p.parse_args(argv)


def build_graph(mat: np.ndarray, threshold: float, top_k: int) -> nx.DiGraph:
    n = mat.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    flat = []
    for i in range(n):
        for j in range(n):
            w = float(mat[i, j])
            if w >= threshold:
                flat.append((w, i, j))
    flat.sort(reverse=True, key=lambda x: x[0])
    for rank, (w, i, j) in enumerate(flat[:top_k]):
        G.add_edge(i, j, weight=w)
    return G


def draw_graph(ax, G: nx.DiGraph, pos, title: str, weights: List[float]):
    ax.clear()
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#1f78b4", ax=ax)
    weights_list = [d.get("weight", 0) for _, _, d in G.edges(data=True)]
    widths = [max(0.8, 6.0 * float(w)) for w in weights_list]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#333333", arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=8, ax=ax)
    ax.set_title(title)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    history = json.loads(args.history.read_text(encoding="utf-8"))
    matrices = [np.asarray(h.get("avg_gating"), dtype=np.float32) for h in history if h.get("avg_gating") is not None]

    n = matrices[0].shape[0]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}

    fig, (ax_graph, ax_curve) = plt.subplots(1, 2, figsize=(10, 5))

    # prepare fitness curve
    iterations = np.array([h.get("iteration", i) for i, h in enumerate(history)])
    scores = np.array([h.get("best_score", np.nan) for h in history], dtype=np.float32)

    def update(idx: int):
        mat = matrices[idx]
        G = build_graph(mat, args.threshold, args.top_k)
        draw_graph(ax_graph, G, pos, f"Iter {idx}", [])
        ax_curve.clear()
        ax_curve.plot(iterations[: idx + 1], scores[: idx + 1], color="#1f78b4")
        ax_curve.set_xlabel("Iteration")
        ax_curve.set_ylabel("Best score")
        ax_curve.grid(alpha=0.3)

    # Save frames
    frames = []
    for i in range(len(matrices)):
        update(i)
        buf = Path("/tmp") / f"_frame_{i}.png"
        buf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(buf, dpi=150)
        frames.append(str(buf))

    # Convert to GIF (imageio)
    try:
        import imageio

        out_dir = args.out.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = [imageio.imread(f) for f in frames]
        imageio.mimsave(args.out, imgs, fps=args.fps)
        print(f"Saved animation to {args.out}")
    except Exception as exc:
        print(f"Failed to save animation: {exc}")


if __name__ == "__main__":
    main()
