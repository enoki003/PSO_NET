"""Visualise PSO gating matrix evolution and fitness.

Two modes:
1. Animation (default) of average gating matrix per iteration rendered
     as a directed weighted graph.
2. Static plots (``--static``) writing one or more PNG files:
     - ``gating_fitness.png``: left = fitness curve (best score over
         iterations), right = final average gating matrix heatmap.
     - optional edge graph snapshot ``gating_graph.png`` if ``--graph``.

Input artefacts produced by `program.pso_train`:
    * pso_history.json  (contains per-iteration best_score + avg_gating)
    * fitness.csv       (optional; if present used for curve speed)

Dependencies: matplotlib, networkx (graph animation), pillow/imageio
for GIF. Install extras: ``pip install .[viz]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PSO gating optimisation artefacts")
    parser.add_argument("--history", type=Path, default=Path("./models/pso_gating/pso_history.json"), help="Path to pso_history.json")
    parser.add_argument("--fitness-csv", type=Path, default=Path("./models/pso_gating/fitness.csv"), help="Optional fitness.csv path (faster for curve)")
    parser.add_argument("--out", type=Path, default=Path("./models/pso_gating/gating_anim.mp4"), help="Animation output path (.mp4/.gif)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Edges below this weight are hidden in graph")
    parser.add_argument("--fps", type=int, default=6, help="FPS for animation")
    parser.add_argument("--show", action="store_true", help="Display animation/plots interactively")
    parser.add_argument("--static", action="store_true", help="Generate static PNG plots instead of animation")
    parser.add_argument("--graph", action="store_true", help="When using --static also export graph snapshot")
    parser.add_argument("--static-dir", type=Path, default=Path("./results/viz"), help="Directory for static outputs")
    parser.add_argument("--writer", choices=["ffmpeg", "pillow", "auto"], default="auto", help="Animation writer backend preference")
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
    *,
    writer_pref: str = "auto",
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

    suffix = out_path.suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def try_ffmpeg() -> bool:
        from matplotlib.animation import writers
        return "ffmpeg" in writers.list()

    writer_obj = None
    if suffix == ".gif":
        # always prefer Pillow for GIF
        try:
            writer_obj = PillowWriter(fps=fps)
        except Exception as exc:
            print(f"GIF writer unavailable ({exc}); will attempt default backend.")
    elif suffix == ".mp4":
        # choose backend based on availability / user preference
        if writer_pref in ("auto", "ffmpeg") and try_ffmpeg():
            # matplotlib will use ffmpeg automatically when available
            writer_obj = None  # allow anim.save to pick ffmpeg
        else:
            # fallback: auto-convert to GIF if Pillow available
            try:
                alt_path = out_path.with_suffix(".gif")
                writer_obj = PillowWriter(fps=fps)
                print(f"[viz] ffmpeg unavailable; switching output to {alt_path.name} (GIF)")
                out_path = alt_path
            except Exception:
                print("[viz] No ffmpeg or Pillow writer; cannot save animation as mp4/gif.")
                print("Install system ffmpeg or pip install imageio-ffmpeg pillow.")
                return
    else:
        print(f"[viz] Unknown extension '{suffix}'. Use .mp4 or .gif")
        return

    try:
        if writer_obj is not None:
            anim.save(out_path, writer=writer_obj, dpi=150)
        else:
            anim.save(out_path, fps=fps, dpi=150)
        print(f"Saved animation to {out_path}")
    except KeyboardInterrupt:
        print("Interrupted while saving animation; partial file may be unusable.")
    except Exception as exc:
        print(f"Failed to save animation ({exc}).")
        print("Install ffmpeg (Linux: sudo apt install ffmpeg) or use --out with .gif plus pillow.")

    if show:
        plt.show()
    else:
        plt.close(fig)


def load_history(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_matrices(history: List[dict]) -> List[np.ndarray]:
    mats: List[np.ndarray] = []
    for entry in history:
        mat = entry.get("avg_gating")
        if mat is None:
            continue
        mats.append(np.asarray(mat, dtype=np.float32))
    return mats


def load_fitness_curve(history: List[dict], fitness_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    # prefer CSV for speed if it exists
    if fitness_csv.exists():
        try:
            data = np.genfromtxt(fitness_csv, delimiter=",", skip_header=1)
            iterations = data[:, 0]
            scores = data[:, 2]
            return iterations, scores
        except Exception:
            pass  # fall back to JSON parsing
    iterations = np.array([h.get("iteration", i) for i, h in enumerate(history)], dtype=np.int32)
    scores = np.array([h.get("best_score", np.nan) for h in history], dtype=np.float32)
    return iterations, scores


def plot_static(history: List[dict], matrices: List[np.ndarray], out_dir: Path, export_graph: bool, threshold: float, show: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations, scores = load_fitness_curve(history, out_dir / "dummy.csv")  # placeholder path when csv absent

    final_mat: Optional[np.ndarray] = matrices[-1] if matrices else None

    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(10, 4))
    ax_curve.plot(iterations, scores, color="#1f78b4", linewidth=2)
    ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("Best fitness score")
    ax_curve.grid(alpha=0.3)
    ax_curve.set_title("Fitness progression")

    if final_mat is not None:
        im = ax_heat.imshow(final_mat, cmap="viridis")
        ax_heat.set_title("Final avg gating matrix")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    else:
        ax_heat.text(0.5, 0.5, "No gating matrices", ha="center", va="center")
        ax_heat.set_axis_off()

    fig.tight_layout()
    fig_path = out_dir / "gating_fitness.png"
    fig.savefig(fig_path, dpi=160)
    print(f"Saved {fig_path}")
    if not show:
        plt.close(fig)

    if export_graph and final_mat is not None:
        # build graph of final matrix
        G = build_graph_from_matrix(final_mat, threshold=threshold)
        n = final_mat.shape[0]
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        draw_frame(ax2, G, pos)
        ax2.set_title("Final gating graph")
        graph_path = out_dir / "gating_graph.png"
        fig2.savefig(graph_path, dpi=160)
        print(f"Saved {graph_path}")
        if not show:
            plt.close(fig2)
    if show:
        plt.show()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.history.exists():
        raise SystemExit(f"History file not found: {args.history}")
    history = load_history(args.history)
    mats = extract_matrices(history)
    if args.static:
        if args.fitness_csv.exists():
            _ = load_fitness_curve(history, args.fitness_csv)  # just to verify readable
        plot_static(history, mats, args.static_dir, args.graph, args.threshold, args.show)
    else:
        if len(mats) == 0:
            raise SystemExit("No gating matrices found in history file for animation")
        animate_matrices(
            mats,
            args.out,
            threshold=args.threshold,
            fps=args.fps,
            show=args.show,
            writer_pref=args.writer,
        )


if __name__ == "__main__":
    main()
