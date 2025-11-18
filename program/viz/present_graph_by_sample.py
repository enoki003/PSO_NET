from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..gating import build_gating_model
from ..pso_train import normalize_images
from .present_graph_evolution import build_graph, draw_graph


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("present_graph_by_sample")
    p.add_argument("--gating", type=Path, required=True, help="Path to PSO output folder containing gating_weights.npy and pso_history.json")
    p.add_argument("--experts", type=Path, required=False, help="Not required — included for interface parity", default=None)
    p.add_argument("--dataset", choices=["cifar10","cifar100"], default="cifar10")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--sample-ids", nargs="*", type=int, default=None)
    p.add_argument("--per-class", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("./results/viz/pso_sample_graphs"))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--fps", type=int, default=2)
    return p.parse_args(argv)


def get_dataset(name: str):
    if name == "cifar100":
        from keras.datasets import cifar100

        (_, _), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    else:
        from keras.datasets import cifar10

        (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.squeeze().astype(np.int32)
    return x_test, y_test


def select_samples(x_test: np.ndarray, y_test: np.ndarray, ids: List[int] | None, per_class: int) -> List[int]:
    if ids:
        return ids
    if per_class > 0:
        selected = []
        rng = np.random.default_rng(0)
        for c in np.unique(y_test):
            idxs = np.where(y_test == c)[0]
            rng.shuffle(idxs)
            selected.extend(idxs[:per_class].tolist())
        return selected
    return list(range(min(8, x_test.shape[0])))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    x_test, y_test = get_dataset(args.dataset)
    sample_ids = select_samples(x_test, y_test, args.sample_ids, args.per_class)

    gating_model = build_gating_model(args.num_experts, hidden_units=384, img_shape=(32,32,3))
    gating_vec_path = args.gating / "gating_weights.npy"
    if not gating_vec_path.exists():
        raise SystemExit(f"gating_weights.npy not found at {gating_vec_path}")
    gv = np.load(gating_vec_path)
    from ..pso_utils import WeightAdapter as WeightAdapterMetrics

    adapter = WeightAdapterMetrics(gating_model)
    adapter.assign_from_vector(gv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # compute gating for selected samples
    images = x_test[sample_ids]
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    norm_images = normalize_images(images, mean, std)
    gating_logits = gating_model.predict(norm_images, batch_size=32, verbose=0)
    gating_logits = gating_logits.reshape(-1, args.num_experts, args.num_experts)
    gating_mats = softmax(gating_logits, axis=-1)

    # layout for nodes
    n = args.num_experts
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = {i: (np.cos(t), np.sin(t)) for i, t in enumerate(theta)}

    frames = []
    for i, idx in enumerate(sample_ids):
        mat = gating_mats[i]
        # build graph
        G = build_graph(mat, args.threshold, args.top_k)
        # plot image + graph + heatmap
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        ax_image, ax_graph, ax_heat = axes
        ax_image.imshow(images[i].astype('uint8'))
        ax_image.axis('off')
        ax_image.set_title(f"Index {idx} - label {int(y_test[idx])}")
        draw_graph(ax_graph, G, pos, title=f"Sample {idx} gating", weights=[])
        im = ax_heat.imshow(mat, cmap='viridis', vmin=0.0, vmax=1.0)
        ax_heat.set_title("Gating matrix (softmax)")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        fig.tight_layout()

        buf = Path('/tmp') / f'_sample_graph_{i}.png'
        fig.savefig(buf, dpi=150)
        plt.close(fig)
        frames.append(str(buf))

    # Save frames as GIF to out
    imgs = [imageio.imread(f) for f in frames]
    out_gif = out / "gating_by_sample.gif"
    imageio.mimsave(out_gif, imgs, fps=args.fps)
    print(f"Saved sample gated graph gif to {out_gif}")


if __name__ == '__main__':
    main()
