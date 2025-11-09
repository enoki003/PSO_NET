"""PSO-driven gating optimisation over trained CIFAR sub experts (CIFAR-100 / CIFAR-10)."""

from __future__ import annotations

import argparse
import time
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tensorflow as tf
from keras.datasets import cifar100, cifar10
from tensorflow import keras

from . import config
from .sub_expert import build_sub_expert_model
from .gating import build_gating_model


@dataclass
class EvalBatch:
    images: np.ndarray
    labels: np.ndarray
    expert_logits: np.ndarray


class EvaluationSet:
    """In-memory evaluation dataset reused across particle evaluations."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        expert_logits: np.ndarray,
        batch_size: int,
    ) -> None:
        self.images = images
        self.labels = labels
        self.expert_logits = expert_logits
        self.batch_size = batch_size

    def __len__(self) -> int:  # pragma: no cover - trivial
        return math.ceil(self.images.shape[0] / self.batch_size)

    def iterate(self) -> Iterable[EvalBatch]:
        total = self.images.shape[0]
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            yield EvalBatch(
                images=self.images[start:end],
                labels=self.labels[start:end],
                expert_logits=self.expert_logits[start:end],
            )


class WeightAdapter:
    """Utility for flattening and restoring Keras model weights."""

    def __init__(self, model: keras.Model) -> None:
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.counts = [int(np.prod(shape)) for shape in self.shapes]
        self.dimension = int(sum(self.counts))

    def assign_from_vector(self, vector: np.ndarray) -> None:
        if vector.size != self.dimension:
            raise ValueError(
                f"Vector length {vector.size} != expected {self.dimension}"
            )
        weights = []
        offset = 0
        for shape, count in zip(self.shapes, self.counts):
            segment = vector[offset : offset + count]
            weights.append(segment.reshape(shape))
            offset += count
        self.model.set_weights(weights)

    def sample_initial(self, rng: np.random.Generator) -> np.ndarray:
        weights = self.model.get_weights()
        flat = np.concatenate([w.reshape(-1) for w in weights])
        noise = rng.normal(scale=0.05, size=flat.shape)
        return flat + noise


def normalize_images(
    images: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    images = images.astype("float32") / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    return (images - mean_arr) / std_arr


def load_expert_models(
    expert_root: Path,
    num_experts: int,
    learning_rate: float,
    *,
    img_shape: tuple[int, int, int],
    num_classes: int,
) -> list[keras.Model]:
    models = []
    for expert_id in range(num_experts):
        expert_dir = expert_root / f"expert_{expert_id:02d}"
        weights_path = expert_dir / "logits.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found for expert {expert_id}: {weights_path}"
            )
        model = build_sub_expert_model(
            use_softmax=False,
            smoothing=0.0,
            learning_rate=learning_rate,
            noise_std=0.0,
            img_shape=img_shape,
            num_classes=num_classes,
        )
        model.load_weights(weights_path)
        models.append(model)
    return models


def precompute_expert_logits(
    models: list[keras.Model], images: np.ndarray, batch_size: int
) -> np.ndarray:
    logits_per_expert = []
    for model in models:
        logits = model.predict(images, batch_size=batch_size, verbose=0)
        logits_per_expert.append(logits)
    stacked = np.stack(logits_per_expert, axis=1)
    return stacked


@dataclass
class FitnessResult:
    score: float
    accuracy: float
    redundancy: float
    complexity: float
    smoothness: float


class ParticleSwarmOptimizer:
    def __init__(
        self,
        *,
        weight_adapter: WeightAdapter,
        evaluate_fn: Callable[[np.ndarray], FitnessResult],
        num_particles: int,
        inertia_max: float,
        inertia_min: float,
        cognitive: float,
        social: float,
        iterations: int,
        rng: np.random.Generator,
        v_max: float | None = None,
        early_stop_patience: int | None = None,
    ) -> None:
        self.weight_adapter = weight_adapter
        self.dimension = weight_adapter.dimension
        self.evaluate_fn = evaluate_fn
        self.num_particles = num_particles
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.cognitive = cognitive
        self.social = social
        self.iterations = iterations
        self.rng = rng

        self.positions = np.stack(
            [weight_adapter.sample_initial(rng) for _ in range(num_particles)]
        )
        self.velocities = np.zeros_like(self.positions)
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(num_particles, -np.inf)
        self.global_best_position = self.positions[0].copy()
        self.global_best_score = -np.inf
        self.history: list[dict[str, float]] = []
        self.v_max = v_max if v_max and v_max > 0 else None
        self.early_stop_patience = max(0, early_stop_patience or 0)
        self._no_improve = 0

    def optimize(self) -> tuple[np.ndarray, list[dict[str, float]]]:
        for iteration in range(self.iterations):
            inertia = self._interpolate_inertia(iteration)
            prev_best = self.global_best_score
            for idx in range(self.num_particles):
                position = self.positions[idx]
                result = self.evaluate_fn(position)

                if result.score > self.personal_best_scores[idx]:
                    self.personal_best_scores[idx] = result.score
                    self.personal_best_positions[idx] = position.copy()

                if result.score > self.global_best_score:
                    self.global_best_score = result.score
                    self.global_best_position = position.copy()

                r1 = self.rng.random(self.dimension)
                r2 = self.rng.random(self.dimension)
                cognitive_term = (
                    self.cognitive * r1 * (self.personal_best_positions[idx] - position)
                )
                social_term = self.social * r2 * (self.global_best_position - position)
                self.velocities[idx] = (
                    inertia * self.velocities[idx] + cognitive_term + social_term
                )
                if self.v_max is not None and self.v_max > 0:
                    np.clip(
                        self.velocities[idx],
                        -self.v_max,
                        self.v_max,
                        out=self.velocities[idx],
                    )
                self.positions[idx] = position + self.velocities[idx]

            # snapshot average gating matrix for the current global best
            avg_gating = None
            try:
                # evaluate_fn is the FitnessEvaluator instance
                avg = self.evaluate_fn.average_gating_matrix(self.global_best_position)
                # convert to nested python lists for JSON serialisation
                avg_gating = np.asarray(avg).tolist()
            except Exception:
                avg_gating = None

            self.history.append(
                {
                    "iteration": iteration,
                    "inertia": inertia,
                    "best_score": self.global_best_score,
                    "avg_gating": avg_gating,
                }
            )

            # early stopping check
            improved = self.global_best_score > prev_best + 1e-12
            if improved:
                self._no_improve = 0
            else:
                self._no_improve += 1
                if (
                    self.early_stop_patience
                    and self._no_improve >= self.early_stop_patience
                ):
                    break
        return self.global_best_position, self.history

    def _interpolate_inertia(self, iteration: int) -> float:
        if self.iterations <= 1:
            return self.inertia_min
        progress = iteration / (self.iterations - 1)
        return self.inertia_max - (self.inertia_max - self.inertia_min) * progress


class FitnessEvaluator:
    def __init__(
        self,
        *,
        weight_adapter: WeightAdapter,
        gating_model: keras.Model,
        eval_set: EvaluationSet,
        num_experts: int,
        recurrent_steps: int,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
    ) -> None:
        self.weight_adapter = weight_adapter
        self.gating_model = gating_model
        self.eval_set = eval_set
        self.num_experts = num_experts
        self.recurrent_steps = recurrent_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        mask = np.ones((num_experts, num_experts), dtype=np.float32) - np.eye(
            num_experts, dtype=np.float32
        )
        self.redundancy_mask = tf.constant(mask[None, ...])

    def __call__(self, vector: np.ndarray) -> FitnessResult:
        self.weight_adapter.assign_from_vector(vector)
        total_samples = 0
        total_correct = 0
        redundancy_accum = 0.0
        complexity_accum = 0.0
        smoothness_accum = 0.0
        batch_count = 0

        for batch in self.eval_set.iterate():
            images = tf.convert_to_tensor(batch.images)
            labels = tf.convert_to_tensor(batch.labels)
            expert_logits = tf.convert_to_tensor(batch.expert_logits)

            gating_logits = self.gating_model(images, training=False)
            gating_logits = tf.reshape(
                gating_logits, (-1, self.num_experts, self.num_experts)
            )
            gating_matrix = tf.nn.softmax(gating_logits, axis=-1)

            mixture = expert_logits
            smooth_batch = 0.0
            for _ in range(self.recurrent_steps):
                updated = tf.einsum("bij,bjk->bik", gating_matrix, mixture)
                delta_norm = tf.norm(updated - mixture, axis=[1, 2])
                smooth_batch += tf.reduce_mean(delta_norm)
                mixture = updated

            combined_logits = tf.reduce_mean(mixture, axis=1)
            predictions = tf.argmax(combined_logits, axis=-1, output_type=tf.int32)
            correct = tf.cast(predictions == labels, tf.int32)

            redundancy_matrix = tf.matmul(
                tf.math.l2_normalize(mixture, axis=2),
                tf.math.l2_normalize(mixture, axis=2),
                transpose_b=True,
            )
            redundancy_matrix = tf.abs(redundancy_matrix * self.redundancy_mask)
            redundancy_accum += float(tf.reduce_mean(redundancy_matrix).numpy())
            complexity_accum += float(
                tf.reduce_mean(
                    tf.reduce_sum(tf.abs(gating_matrix), axis=[1, 2])
                ).numpy()
            )
            smoothness_accum += float(
                (smooth_batch / max(1, self.recurrent_steps)).numpy()
            )

            total_correct += int(tf.reduce_sum(correct).numpy())
            total_samples += labels.shape[0]
            batch_count += 1

        accuracy = total_correct / max(1, total_samples)
        redundancy = redundancy_accum / max(1, batch_count)
        complexity = complexity_accum / max(1, batch_count)
        smoothness = smoothness_accum / max(1, batch_count)
        score = (
            self.alpha * accuracy
            - self.beta * redundancy
            - self.gamma * complexity
            - self.delta * smoothness
        )
        return FitnessResult(score, accuracy, redundancy, complexity, smoothness)

    def average_gating_matrix(self, vector: np.ndarray) -> np.ndarray:
        """Return the mean gating matrix Cx averaged over the evaluation set.

        This is used for logging/visualisation; it does a forward pass of the
        gating model with the provided weight vector and averages the row-
        normalized softmax matrices over the evaluation images.
        """
        self.weight_adapter.assign_from_vector(vector)
        accum = None
        count = 0
        for batch in self.eval_set.iterate():
            images = tf.convert_to_tensor(batch.images)
            gating_logits = self.gating_model(images, training=False)
            gating_logits = tf.reshape(
                gating_logits, (-1, self.num_experts, self.num_experts)
            )
            gating_matrix = tf.nn.softmax(gating_logits, axis=-1)
            mean_matrix = tf.reduce_mean(gating_matrix, axis=0)  # shape (N, N)
            arr = mean_matrix.numpy()
            if accum is None:
                accum = arr
            else:
                accum += arr
            count += 1
        if accum is None:
            return np.zeros((self.num_experts, self.num_experts), dtype=np.float32)
        return accum / max(1, count)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise gating network with PSO")
    parser.add_argument(
        "--experts", type=Path, default=Path("./models/cifar_sub_experts")
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("./models/pso_gating"))
    parser.add_argument("--iterations", type=int, default=config.PSO_DEFAULT_ITERATIONS)
    parser.add_argument("--particles", type=int, default=config.PSO_DEFAULT_PARTICLES)
    parser.add_argument(
        "--dataset", choices=["cifar100", "cifar10"], default="cifar100"
    )
    parser.add_argument("--recurrent-steps", type=int, default=config.PSO_RECURRENT_STEPS)
    parser.add_argument("--log-interval", type=int, default=5, help="Print progress every N iterations")
    parser.add_argument("--profile", action="store_true", help="Print per-iteration timing for debug")
    parser.add_argument(
        "--optimize-split",
        choices=["train", "test", "both"],
        default="train",
        help="Data split used to optimise PSO weights: training set, test set, or both (legacy)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        mean, std = config.CIFAR_CHANNEL_MEAN, config.CIFAR_CHANNEL_STD
        img_shape = config.CIFAR_IMG_SHAPE
        num_classes = config.CIFAR_NUM_CLASSES
    else:  # cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        mean, std = config.CIFAR10_CHANNEL_MEAN, config.CIFAR10_CHANNEL_STD
        img_shape = config.CIFAR10_IMG_SHAPE
        num_classes = config.CIFAR10_NUM_CLASSES

    # Normalise all splits up-front
    x_train = normalize_images(x_train, mean, std)
    x_test = normalize_images(x_test, mean, std)
    y_train = y_train.squeeze().astype(np.int32)
    y_test = y_test.squeeze().astype(np.int32)

    # Select optimisation split
    if args.optimize_split == "train":
        opt_images, opt_labels = x_train, y_train
    elif args.optimize_split == "test":
        opt_images, opt_labels = x_test, y_test
    else:  # both (legacy behaviour)
        opt_images = np.concatenate([x_train, x_test], axis=0)
        opt_labels = np.concatenate([y_train, y_test], axis=0)

    if args.sample_count and args.sample_count < opt_images.shape[0]:
        indices = rng.choice(opt_images.shape[0], size=args.sample_count, replace=False)
        opt_images = opt_images[indices]
        opt_labels = opt_labels[indices]

    expert_models = load_expert_models(
        args.experts,
        args.num_experts,
        args.lr,
        img_shape=img_shape,
        num_classes=num_classes,
    )
    expert_logits = precompute_expert_logits(
        expert_models, opt_images, batch_size=args.batch_size
    )

    eval_set = EvaluationSet(opt_images, opt_labels, expert_logits, batch_size=args.batch_size)
    gating_model = build_gating_model(
        args.num_experts, args.hidden_units, img_shape=img_shape
    )
    weight_adapter = WeightAdapter(gating_model)

    evaluator = FitnessEvaluator(
        weight_adapter=weight_adapter,
        gating_model=gating_model,
        eval_set=eval_set,
        num_experts=args.num_experts,
        recurrent_steps=max(1, args.recurrent_steps),
        alpha=config.FITNESS_ALPHA,
        beta=config.FITNESS_BETA,
        gamma=config.FITNESS_GAMMA,
        delta=config.FITNESS_DELTA,
    )

    optimizer = ParticleSwarmOptimizer(
        weight_adapter=weight_adapter,
        evaluate_fn=evaluator,
        num_particles=args.particles,
        inertia_max=config.PSO_INERTIA_MAX,
        inertia_min=config.PSO_INERTIA_MIN,
        cognitive=config.PSO_COGNITIVE_COEFF,
        social=config.PSO_SOCIAL_COEFF,
        iterations=args.iterations,
        rng=rng,
        v_max=config.PSO_VELOCITY_MAX,
        early_stop_patience=config.PSO_EARLY_STOP_PATIENCE,
    )

    start_time = time.time()
    iteration_times = []
    best_vector = None
    history = []
    # manual optimisation loop with logging to expose progress
    for iteration in range(optimizer.iterations):
        t0 = time.time()
        inertia = optimizer._interpolate_inertia(iteration)
        prev_best = optimizer.global_best_score
        for idx in range(optimizer.num_particles):
            position = optimizer.positions[idx]
            result = evaluator(position)
            if result.score > optimizer.personal_best_scores[idx]:
                optimizer.personal_best_scores[idx] = result.score
                optimizer.personal_best_positions[idx] = position.copy()
            if result.score > optimizer.global_best_score:
                optimizer.global_best_score = result.score
                optimizer.global_best_position = position.copy()
            r1 = optimizer.rng.random(optimizer.dimension)
            r2 = optimizer.rng.random(optimizer.dimension)
            cognitive_term = optimizer.cognitive * r1 * (optimizer.personal_best_positions[idx] - position)
            social_term = optimizer.social * r2 * (optimizer.global_best_position - position)
            optimizer.velocities[idx] = inertia * optimizer.velocities[idx] + cognitive_term + social_term
            if optimizer.v_max is not None and optimizer.v_max > 0:
                np.clip(optimizer.velocities[idx], -optimizer.v_max, optimizer.v_max, out=optimizer.velocities[idx])
            optimizer.positions[idx] = position + optimizer.velocities[idx]
        # snapshot gating matrix
        try:
            avg = evaluator.average_gating_matrix(optimizer.global_best_position)
            avg_gating = np.asarray(avg).tolist()
        except Exception:
            avg_gating = None
        improved = optimizer.global_best_score > prev_best + 1e-12
        if improved:
            optimizer._no_improve = 0
        else:
            optimizer._no_improve += 1
        row = {
            "iteration": iteration,
            "inertia": inertia,
            "best_score": optimizer.global_best_score,
            "avg_gating": avg_gating,
        }
        history.append(row)
        iter_time = time.time() - t0
        iteration_times.append(iter_time)
        if args.profile:
            print(f"[pso] iter {iteration} time={iter_time:.3f}s best={optimizer.global_best_score:.5f}")
        if getattr(args, "log_interval", None) and iteration % max(1, args.log_interval) == 0:
            avg_t = sum(iteration_times) / len(iteration_times)
            remaining = optimizer.iterations - (iteration + 1)
            eta = remaining * avg_t
            print(
                f"[pso] iter {iteration}/{optimizer.iterations - 1} best={optimizer.global_best_score:.5f} "
                f"improved={'Y' if improved else 'N'} avg_iter={avg_t:.3f}s ETA~{eta/60:.1f}m"
            )
        if optimizer.early_stop_patience and optimizer._no_improve >= optimizer.early_stop_patience:
            print(f"[pso] early stop at iteration {iteration}")
            break
    best_vector = optimizer.global_best_position
    total_time = time.time() - start_time
    print(f"[pso] optimisation finished in {total_time/60:.2f} minutes; iterations={len(history)}")
    best_result_train = evaluator(best_vector)

    # Evaluate on held-out test set for an unbiased estimate
    test_logits = precompute_expert_logits(expert_models, x_test, batch_size=args.batch_size)
    test_eval_set = EvaluationSet(x_test, y_test, test_logits, batch_size=args.batch_size)
    test_evaluator = FitnessEvaluator(
        weight_adapter=weight_adapter,
        gating_model=gating_model,
        eval_set=test_eval_set,
        num_experts=args.num_experts,
        recurrent_steps=max(1, args.recurrent_steps),
        alpha=config.FITNESS_ALPHA,
        beta=config.FITNESS_BETA,
        gamma=config.FITNESS_GAMMA,
        delta=config.FITNESS_DELTA,
    )
    best_result_test = test_evaluator(best_vector)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "gating_weights.npy", best_vector)
    with open(args.output / "pso_history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    # also write CSV & numpy history artifacts
    try:
        import csv

        with open(
            args.output / "fitness.csv", "w", newline="", encoding="utf-8"
        ) as csvfp:
            writer = csv.writer(csvfp)
            writer.writerow(["iteration", "inertia", "best_score"])
            for row in history:
                writer.writerow(
                    [row.get("iteration"), row.get("inertia"), row.get("best_score")]
                )
    except Exception as e:  # pragma: no cover - best effort
        print("Failed to write fitness.csv:", e)

    try:
        # extract avg gating matrices into ndarray (iterations, N, N)
        matrices = [
            np.asarray(h["avg_gating"], dtype=np.float32)
            for h in history
            if h.get("avg_gating") is not None
        ]
        if matrices:
            arr = np.stack(matrices, axis=0)
            np.save(args.output / "C_history.npy", arr)
    except Exception as e:  # pragma: no cover
        print("Failed to write C_history.npy:", e)
    with open(args.output / "fitness.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                # Keep top-level metrics mapped to test (unbiased) for downstream tools
                "score": best_result_test.score,
                "accuracy": best_result_test.accuracy,
                "redundancy": best_result_test.redundancy,
                "complexity": best_result_test.complexity,
                "smoothness": best_result_test.smoothness,
                # Include both splits for auditing
                "train": {
                    "score": best_result_train.score,
                    "accuracy": best_result_train.accuracy,
                    "redundancy": best_result_train.redundancy,
                    "complexity": best_result_train.complexity,
                    "smoothness": best_result_train.smoothness,
                    "samples": int(opt_labels.shape[0]),
                },
                "test": {
                    "score": best_result_test.score,
                    "accuracy": best_result_test.accuracy,
                    "redundancy": best_result_test.redundancy,
                    "complexity": best_result_test.complexity,
                    "smoothness": best_result_test.smoothness,
                    "samples": int(y_test.shape[0]),
                },
                "optimize_split": args.optimize_split,
            },
            fp,
            indent=2,
        )

    print("PSO optimisation complete.")
    print("Best fitness (train):", best_result_train.score)
    print("Accuracy (train):", best_result_train.accuracy)
    print("Best fitness (test):", best_result_test.score)
    print("Accuracy (test):", best_result_test.accuracy)
    print("Redundancy (test):", best_result_test.redundancy)
    print("Complexity (test):", best_result_test.complexity)
    print("Smoothness (test):", best_result_test.smoothness)


if __name__ == "__main__":
    main()
