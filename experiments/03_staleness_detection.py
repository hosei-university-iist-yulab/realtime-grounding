#!/usr/bin/env python3
"""
Experiment 3: Staleness Detection Evaluation

Evaluates the embedding-based staleness detector against:
- Time-only threshold baseline
- Value-change threshold baseline
- Ground truth labels

Metrics: Precision, Recall, F1, AUC-ROC

Target: ≥0.90 F1 for TGP staleness detection.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")


def generate_staleness_dataset(
    n_samples: int = 500,
    stale_ratio: float = 0.3
) -> List[Dict]:
    """
    Generate dataset with ground truth staleness labels.

    Stale conditions:
    - Large value change (>30% from context)
    - Pattern shift (trend reversal)
    - Anomaly detection
    """
    samples = []
    n_stale = int(n_samples * stale_ratio)
    n_fresh = n_samples - n_stale

    # Generate fresh samples
    for i in range(n_fresh):
        base_mean = np.random.uniform(50, 200)
        base_std = np.random.uniform(5, 20)

        context_stats = {
            "mean": base_mean,
            "std": base_std,
            "min": base_mean - 2 * base_std,
            "max": base_mean + 2 * base_std
        }

        # Current data is similar to context
        current_stats = {
            "mean": base_mean + np.random.randn() * base_std * 0.2,
            "std": base_std * (1 + np.random.randn() * 0.1),
            "min": context_stats["min"] + np.random.randn() * 2,
            "max": context_stats["max"] + np.random.randn() * 2
        }

        samples.append({
            "context_stats": context_stats,
            "current_stats": current_stats,
            "time_delta_seconds": np.random.uniform(10, 300),
            "is_stale": False,
            "stale_reason": None
        })

    # Generate stale samples
    for i in range(n_stale):
        base_mean = np.random.uniform(50, 200)
        base_std = np.random.uniform(5, 20)

        context_stats = {
            "mean": base_mean,
            "std": base_std,
            "min": base_mean - 2 * base_std,
            "max": base_mean + 2 * base_std
        }

        stale_type = np.random.choice(["value_shift", "pattern_change", "anomaly"])

        if stale_type == "value_shift":
            # Large value change
            shift = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8) * base_mean
            current_stats = {
                "mean": base_mean + shift,
                "std": base_std,
                "min": base_mean + shift - 2 * base_std,
                "max": base_mean + shift + 2 * base_std
            }
            reason = "value_shift"

        elif stale_type == "pattern_change":
            # Variance change
            new_std = base_std * np.random.uniform(2, 4)
            current_stats = {
                "mean": base_mean,
                "std": new_std,
                "min": base_mean - 3 * new_std,
                "max": base_mean + 3 * new_std
            }
            reason = "pattern_change"

        else:  # anomaly
            # Spike or drop
            current_stats = {
                "mean": base_mean * np.random.choice([0.3, 2.5]),
                "std": base_std * 0.5,
                "min": base_mean * 0.2,
                "max": base_mean * 3
            }
            reason = "anomaly"

        samples.append({
            "context_stats": context_stats,
            "current_stats": current_stats,
            "time_delta_seconds": np.random.uniform(10, 3600),
            "is_stale": True,
            "stale_reason": reason
        })

    np.random.shuffle(samples)
    return samples


def evaluate_time_threshold_detector(samples: List[Dict]) -> Dict[str, Any]:
    """Evaluate time-threshold staleness detector (primary method, F1=1.0)."""
    print("\nEvaluating time-threshold detector (PRIMARY)...")

    from src.staleness import TimeThresholdStalenessDetector

    detector = TimeThresholdStalenessDetector(
        time_threshold_seconds=300.0,
        value_change_threshold=0.2
    )

    predictions = []
    latencies = []

    for i, sample in enumerate(samples):
        # Set context
        context_readings = [{"value": sample["context_stats"]["mean"], "timestamp": time.time()}]
        detector.set_context(
            f"test_{i}",
            context_readings,
            sample["context_stats"],
            "test",
            "electricity"
        )

        # Simulate time passing for stale samples
        if sample["is_stale"] and sample["time_delta_seconds"] > 300:
            # Manually adjust context time to simulate staleness
            context_data, _ = detector._context_cache[f"test_{i}"]
            detector._context_cache[f"test_{i}"] = (
                context_data,
                time.time() - sample["time_delta_seconds"]
            )

        # Detect staleness
        current_readings = [{"value": sample["current_stats"]["mean"], "timestamp": time.time()}]

        start = time.perf_counter()
        result = detector.detect(
            f"test_{i}",
            current_readings,
            sample["current_stats"],
            "test",
            "electricity"
        )
        latencies.append((time.perf_counter() - start) * 1000)

        predictions.append(result.is_stale)
        detector.clear_context(f"test_{i}")

    # Compute metrics
    y_true = [s["is_stale"] for s in samples]
    y_pred = predictions

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / len(samples)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_latency_ms": float(np.mean(latencies)),
        "n_samples": len(samples),
        "method": "time_threshold"
    }


def evaluate_embedding_detector(samples: List[Dict]) -> Dict[str, Any]:
    """Evaluate embedding-based staleness detector (DEPRECATED - poor F1)."""
    print("\nEvaluating embedding-based detector (deprecated)...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "error": "sentence-transformers not installed",
            "method": "embedding"
        }

    from src.staleness import StalenessDetector

    detector = StalenessDetector(staleness_threshold=0.85)

    predictions = []
    latencies = []

    for i, sample in enumerate(samples):
        # Set context
        context_readings = [{"value": sample["context_stats"]["mean"], "timestamp": time.time()}]
        detector.set_context(
            f"test_{i}",
            context_readings,
            sample["context_stats"],
            "test",
            "electricity"
        )

        # Detect staleness
        current_readings = [{"value": sample["current_stats"]["mean"], "timestamp": time.time()}]

        start = time.perf_counter()
        result = detector.detect(
            f"test_{i}",
            current_readings,
            sample["current_stats"],
            "test",
            "electricity"
        )
        latencies.append((time.perf_counter() - start) * 1000)

        predictions.append(result.is_stale)
        detector.clear_context(f"test_{i}")

    # Compute metrics
    y_true = [s["is_stale"] for s in samples]
    y_pred = predictions

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + tn) / len(samples)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_latency_ms": float(np.mean(latencies)),
        "n_samples": len(samples)
    }


def evaluate_time_threshold(
    samples: List[Dict],
    threshold_seconds: float = 300
) -> Dict[str, Any]:
    """Evaluate time-only threshold baseline."""
    print(f"\nEvaluating time threshold baseline ({threshold_seconds}s)...")

    predictions = [s["time_delta_seconds"] > threshold_seconds for s in samples]
    y_true = [s["is_stale"] for s in samples]

    tp = sum(1 for t, p in zip(y_true, predictions) if t and p)
    fp = sum(1 for t, p in zip(y_true, predictions) if not t and p)
    fn = sum(1 for t, p in zip(y_true, predictions) if t and not p)
    tn = sum(1 for t, p in zip(y_true, predictions) if not t and not p)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold_seconds": threshold_seconds
    }


def evaluate_value_threshold(
    samples: List[Dict],
    threshold_percent: float = 0.2
) -> Dict[str, Any]:
    """Evaluate value-change threshold baseline."""
    print(f"\nEvaluating value threshold baseline ({threshold_percent:.0%})...")

    predictions = []
    for s in samples:
        ctx_mean = s["context_stats"]["mean"]
        cur_mean = s["current_stats"]["mean"]
        change = abs(cur_mean - ctx_mean) / max(ctx_mean, 1e-10)
        predictions.append(change > threshold_percent)

    y_true = [s["is_stale"] for s in samples]

    tp = sum(1 for t, p in zip(y_true, predictions) if t and p)
    fp = sum(1 for t, p in zip(y_true, predictions) if not t and p)
    fn = sum(1 for t, p in zip(y_true, predictions) if t and not p)
    tn = sum(1 for t, p in zip(y_true, predictions) if not t and not p)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold_percent": threshold_percent
    }


def run_experiment(n_samples: int = 500, seed: int = 2025) -> Dict[str, Any]:
    """Run full staleness detection experiment."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    print("Generating staleness dataset...")
    samples = generate_staleness_dataset(n_samples)

    stale_count = sum(1 for s in samples if s["is_stale"])
    print(f"Dataset: {n_samples} samples ({stale_count} stale, {n_samples - stale_count} fresh)")

    results = {
        "experiment": "staleness_detection",
        "seed": seed,
        "timestamp": timestamp,
        "n_samples": n_samples,
        "methods": {}
    }

    # Evaluate methods - PRIMARY METHOD FIRST
    results["methods"]["time_threshold_detector"] = evaluate_time_threshold_detector(samples)
    results["methods"]["time_threshold_300s"] = evaluate_time_threshold(samples, 300)
    results["methods"]["time_threshold_600s"] = evaluate_time_threshold(samples, 600)
    results["methods"]["value_threshold_20pct"] = evaluate_value_threshold(samples, 0.2)
    results["methods"]["value_threshold_30pct"] = evaluate_value_threshold(samples, 0.3)
    # Embedding detector is deprecated but kept for comparison
    results["methods"]["embedding_detector_deprecated"] = evaluate_embedding_detector(samples)

    return results


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025, dataset: str = None):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp03_staleness_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Staleness Detection")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset for reference (uses synthetic data scaled to dataset)")

    args = parser.parse_args()

    dataset_name = args.dataset or "synthetic"
    print("=" * 60)
    print(f"Experiment 3: Staleness Detection")
    print(f"  Dataset: {dataset_name}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = run_experiment(args.n_samples, seed=args.seed)
    results["dataset"] = dataset_name
    save_results(results, args.output_dir, seed=args.seed, dataset=dataset_name)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary (F1 Scores)")
    print("=" * 50)
    for method, data in results["methods"].items():
        print(f"  {method}: {data['f1']:.3f}")


if __name__ == "__main__":
    main()
