from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .pso_train import load_expert_models, precompute_expert_logits, normalize_images
from .gating import build_gating_model


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize input-dependent gating dynamics using best PSO gating weights")
    parser.add_argument("--experts", type=Path, required=True, help="Path to experts dir")
    parser.add_argument("--gating", type=Path, required=True, help="Gating output dir with gating_weights.npy")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None, help="Test set indices to visualize. If not set, choose first few samples")
    parser.add_argument("--per-class", type=int, default=0, help="Pick N images per class instead of sample ids (disabled when sample-ids provided)")
    parser.add_argument("--out", type=Path, default=Path("./results/viz/input_dynamics"))
    parser.add_argument("--recurrent-steps", type=int, default=3)
    parser.add_argument("--hidden-units", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def get_dataset(name: str):
    if name == "cifar100":
        from keras.datasets import cifar100

        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        from keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    y_test = y_test.squeeze().astype(np.int32)
    return x_test, y_test, mean, std


def select_samples(x_test: np.ndarray, y_test: np.ndarray, ids: List[int] | None, per_class: int) -> List[int]:
    if ids:
        return ids
    if per_class > 0:
        selected = []
        classes = np.unique(y_test)
        rng = np.random.default_rng(0)
        for c in classes:
            idxs = np.where(y_test == c)[0]
            rng.shuffle(idxs)
            select = idxs[:per_class]
            selected.extend(select.tolist())
        return selected
    # default: first min(8, len(test))
    return list(range(min(8, x_test.shape[0])))


def visualize_for_samples(
    out: Path,
    gating_dir: Path,
    experts_dir: Path,
    dataset: str,
    num_experts: int,
    sample_ids: List[int],
    recurrent_steps: int,
    batch_size: int,
    hidden_units: int,
):
    x_test, y_test, mean, std = get_dataset(dataset)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    # load experts and precompute logits for selected samples
    print("Loading expert models...")
    expert_models = load_expert_models(experts_dir, num_experts, learning_rate=1e-3, img_shape=(32, 32, 3), num_classes=10 if dataset=="cifar10" else 100)
    samples = x_test[sample_ids]
    sample_labels = y_test[sample_ids]
    samples_norm = normalize_images(samples, mean, std)
    expert_logits = precompute_expert_logits(expert_models, samples_norm, batch_size=batch_size)
    # shape (n_samples, num_experts, num_classes)

    # build gating model and load weights
    print("Building gating model and loading weights...")
    gating_model = build_gating_model(num_experts, hidden_units, img_shape=(32, 32, 3))
    weights_path = gating_dir / "gating_weights.npy"
    if not weights_path.exists():
        raise SystemExit(f"gating_weights.npy not found at {weights_path}")
    vec = np.load(weights_path)
    from .pso_utils import WeightAdapter as WeightAdapterMetrics
    try:
        from scipy.linalg import eigvals
    except Exception:
        from numpy.linalg import eigvals
    import json

    adapter = WeightAdapterMetrics(gating_model)
    adapter.assign_from_vector(vec)

    out.mkdir(parents=True, exist_ok=True)

    # For each sample, compute gating matrix and animate recurrent mixing
    for i, idx in enumerate(sample_ids):
        img = samples[i]
        lbl = int(sample_labels[i])
        logits = expert_logits[i]  # (num_experts, classes)

        # gating logits from gating_model
        inputs = np.expand_dims(samples_norm[i], axis=0)
        # Reduce retracing warnings by predicting for the whole batch at once
        gating_logits_all = gating_model.predict(samples_norm, batch_size=batch_size, verbose=0)
        gating_logits_all = gating_logits_all.reshape(-1, num_experts, num_experts)
        gating_matrix = softmax(gating_logits_all[i], axis=-1)

        # set up figure: image + heatmap + bar chart
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax_img, ax_heat, ax_bar = axes
        ax_img.imshow(img.astype('uint8'))
        ax_img.axis('off')
        ax_img.set_title(f"Index {idx} Label {lbl}")

        # gating heatmap function
        im = ax_heat.imshow(gating_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
        ax_heat.set_title("Gating matrix")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # mixture progression across recurrent steps
        mixture = logits.copy()  # shape N x C
        max_steps = max(1, recurrent_steps)
        bars = None
        top_k = min(10, logits.shape[1])

        out_gif = out / f"input_{idx}_dynamics.gif"
        frames = []
        prev_mixture = None
        deltas = []
        for step in range(max_steps + 1):
            # compute combined logits and probs
            combined = mixture.mean(axis=0)
            probs = softmax(combined)
            ax_bar.clear()
            ax_bar.bar(np.arange(len(probs)), probs, color="#1f78b4")
            ax_bar.set_ylim(0, 1)
            pred = int(np.argmax(probs))
            pred_prob = float(probs[pred])
            ax_bar.set_title(f"Step {step} - Pred {pred} P={pred_prob:.3f}")

            # also plot expert contributions for predicted class
            contribs = mixture[:, pred]
            if np.max(contribs) > 0:
                contribs_norm = contribs / np.sum(contribs)
            else:
                contribs_norm = contribs
            # small extra subplot to the right? re-use heatmap (overlay) - we will print to console
            # compute delta metric from previous mixture
            if prev_mixture is None:
                delta = 0.0
            else:
                delta = float(np.linalg.norm(mixture - prev_mixture))
            deltas.append(delta)
            prev_mixture = mixture.copy()
            # annotate delta and top experts
            top_experts = np.argsort(-contribs_norm)[:3]
            ax_bar.text(0.99, 0.95, f"Δ: {delta:.4f}\nTop experts: {top_experts.tolist()}", ha="right", va="top", transform=ax_bar.transAxes, fontsize=8)
            ax_bar.set_xlabel("class")

            # update gating heatmap if we wanted per-step gating (static here)
            ax_heat.imshow(gating_matrix, cmap='viridis', vmin=0.0, vmax=1.0)

            fig.canvas.draw()
            # capture figure as image
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            import imageio
            frames.append(imageio.imread(buf))
            buf.close()

            # update mixture
            mixture = np.einsum('ij,jk->ik', gating_matrix, mixture)

        imageio.mimsave(out_gif, frames, fps=1)
        plt.close(fig)
        print(f"Saved input dynamics gif: {out_gif}")
        # print simple delta diagnostics for this sample
        print(f"Sample {idx}: deltas per step: {deltas}")
        # gating diagnostics
        diag = {}
        # row entropies
        entropies = -np.sum(gating_matrix * np.log(gating_matrix + 1e-12), axis=1)
        sparsity = np.mean((gating_matrix < 1e-3).astype(float))
        # spectral radius of gating
        try:
            eigs = eigvals(gating_matrix)
            spectral_radius = float(np.max(np.abs(eigs)))
        except Exception:
            spectral_radius = float(np.nan)
        diag["row_entropies"] = entropies.tolist()
        diag["row_entropy_mean"] = float(np.mean(entropies))
        diag["sparsity_frac"] = float(sparsity)
        diag["spectral_radius"] = spectral_radius
        diag["deltas"] = deltas
        diag["predicted_label"] = int(np.argmax(np.mean(mixture, axis=0)))
        # save gating matrix and mixture images
        try:
            out_sample = out / f"input_{idx}"
            out_sample.mkdir(parents=True, exist_ok=True)
            np.save(out_sample / "gating_matrix.npy", gating_matrix)
            np.save(out_sample / "final_mixture.npy", mixture)
            with open(out_sample / "diagnostics.json", "w", encoding="utf-8") as fp:
                json.dump(diag, fp, indent=2)
            # save gating heatmap
            import matplotlib

            matplotlib.use("Agg")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            im2 = ax2.imshow(gating_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax2.set_title("Gating matrix heatmap")
            fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            fig2.tight_layout()
            fig2.savefig(out_sample / "gating_matrix.png", dpi=150)
            plt.close(fig2)
        except Exception as e:
            print(f"Failed to write diagnostics for sample {idx}: {e}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    x_test, y_test, _, _ = get_dataset(args.dataset)
    sample_ids = select_samples(x_test, y_test, args.sample_ids, args.per_class)
    visualize_for_samples(args.out, args.gating, args.experts, args.dataset, args.num_experts, sample_ids, args.recurrent_steps, args.batch_size, args.hidden_units)


if __name__ == '__main__':
    main()
