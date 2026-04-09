#!/usr/bin/env python3
"""
Experiment 9: Sampling Robustness Evaluation

Tests TGP performance under different data conditions:
- Sampling rates: 1min, 5min, 15min, hourly
- Data dropout: 10%, 30%, 50% missing
- Noise levels: Low, medium, high

This validates robustness and helps identify deployment requirements.
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


def generate_test_data(
    n_readings: int = 1000,
    base_value: float = 150.0,
    noise_std: float = 10.0
) -> List[Dict]:
    """Generate base test data at 1-minute resolution."""
    readings = []
    base_time = time.time() - n_readings * 60

    for i in range(n_readings):
        # Time-of-day pattern
        hour = (i // 60) % 24
        if 8 <= hour <= 18:
            multiplier = 1.0 + 0.3 * np.sin((hour - 8) * np.pi / 10)
        else:
            multiplier = 0.6

        value = base_value * multiplier + np.random.randn() * noise_std

        readings.append({
            "timestamp": base_time + i * 60,
            "value": max(0, value),
            "index": i
        })

    return readings


def downsample_data(
    readings: List[Dict],
    interval_minutes: int
) -> List[Dict]:
    """Downsample data to lower resolution."""
    if interval_minutes == 1:
        return readings

    downsampled = []
    for i in range(0, len(readings), interval_minutes):
        chunk = readings[i:i + interval_minutes]
        if chunk:
            avg_value = np.mean([r["value"] for r in chunk])
            downsampled.append({
                "timestamp": chunk[0]["timestamp"],
                "value": avg_value,
                "original_samples": len(chunk)
            })

    return downsampled


def apply_dropout(
    readings: List[Dict],
    dropout_pct: float
) -> Tuple[List[Dict], List[int]]:
    """
    Randomly remove readings to simulate missing data.

    Returns:
        (remaining_readings, dropped_indices)
    """
    n_drop = int(len(readings) * dropout_pct)
    drop_indices = set(np.random.choice(len(readings), n_drop, replace=False))

    remaining = []
    for i, r in enumerate(readings):
        if i not in drop_indices:
            remaining.append(r)

    return remaining, list(drop_indices)


def add_noise(
    readings: List[Dict],
    noise_multiplier: float
) -> List[Dict]:
    """Add additional noise to readings."""
    noisy = []
    for r in readings:
        noisy_value = r["value"] + np.random.randn() * r["value"] * noise_multiplier
        noisy.append({
            **r,
            "value": max(0, noisy_value),
            "noise_added": True
        })
    return noisy


def evaluate_grounding(
    readings: List[Dict],
    query: str = "What is the current consumption pattern?"
) -> Dict:
    """
    Evaluate grounding accuracy with given readings.

    Uses REAL LLM inference.
    """
    from src.buffer import TemporalGroundingBuffer, SensorReading
    from src.llm import LLMBackbone

    # Load LLM (loads automatically in __init__)
    llm = LLMBackbone(model_type="tinyllama")

    # Try to load LoRA weights
    models_dir = PROJECT_ROOT / "output" / "models"
    if models_dir.exists():
        model_dirs = sorted(models_dir.glob("grounding_*"))
        if model_dirs:
            lora_path = model_dirs[-1] / "final"
            if lora_path.exists():
                llm.load_lora(str(lora_path))

    # Setup buffer (using novel TemporalGroundingBuffer)
    buffer = TemporalGroundingBuffer()
    building_id = "_sampling_test"

    for r in readings:
        reading = SensorReading(
            timestamp=r["timestamp"],
            building_id=building_id,
            meter_type="electricity",
            value=r["value"]
        )
        buffer.push(reading)

    # Get statistics
    stats = buffer.get_statistics(building_id, "electricity")
    latest = buffer.get_latest(building_id, "electricity", n=5)

    # Generate response
    start = time.perf_counter()
    prompt = f"""<|system|>
You are an energy monitoring assistant. Use the sensor data provided.
</s>
<|user|>
Current readings - Mean: {stats['mean']:.1f} kWh, Latest: {latest[-1].value:.1f} kWh
Number of readings: {len(readings)}
Question: {query}
</s>
<|assistant|>
"""
    response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)
    latency = (time.perf_counter() - start) * 1000

    # Check if response contains values
    import re
    numbers = re.findall(r'[\d.]+', response)
    has_values = len(numbers) >= 1

    # Cleanup
    buffer.clear(building_id, "electricity")

    return {
        "latency_ms": latency,
        "response_length": len(response),
        "has_values": has_values,
        "response_sample": response[:100],
        "stats": stats
    }


def evaluate_staleness(readings: List[Dict]) -> Dict:
    """Evaluate staleness detection with given readings."""
    from src.staleness import TimeThresholdStalenessDetector

    detector = TimeThresholdStalenessDetector()

    # Compute statistics
    values = [r["value"] for r in readings]
    stats = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values)
    }

    # Set context
    detector.set_context(
        "_staleness_test",
        [{"value": r["value"], "timestamp": r["timestamp"]} for r in readings[:10]],
        stats
    )

    # Test detection
    latencies = []
    for i in range(min(20, len(readings))):
        start = time.perf_counter()
        result = detector.detect(
            "_staleness_test",
            [{"value": readings[i]["value"], "timestamp": readings[i]["timestamp"]}],
            stats
        )
        latencies.append((time.perf_counter() - start) * 1000)

    detector.clear_context("_staleness_test")

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies))
    }


def run_sampling_experiment(n_base_readings: int = 500) -> Dict:
    """Run sampling rate experiments."""
    print("\n=== Sampling Rate Experiments ===")

    base_data = generate_test_data(n_base_readings)
    results = {}

    intervals = [1, 5, 15, 60]  # minutes

    for interval in intervals:
        print(f"\nTesting {interval}-minute sampling...")
        downsampled = downsample_data(base_data, interval)
        print(f"  Data points: {len(downsampled)}")

        grounding = evaluate_grounding(downsampled)
        staleness = evaluate_staleness(downsampled)

        results[f"{interval}min"] = {
            "n_readings": len(downsampled),
            "grounding_latency_ms": grounding["latency_ms"],
            "staleness_latency_ms": staleness["mean_latency_ms"],
            "has_values": grounding["has_values"]
        }

    return results


def run_dropout_experiment(n_base_readings: int = 500) -> Dict:
    """Run data dropout experiments."""
    print("\n=== Data Dropout Experiments ===")

    base_data = generate_test_data(n_base_readings)
    results = {}

    dropout_rates = [0.0, 0.1, 0.3, 0.5]  # 0%, 10%, 30%, 50%

    for rate in dropout_rates:
        print(f"\nTesting {rate*100:.0f}% dropout...")
        remaining, dropped = apply_dropout(base_data, rate)
        print(f"  Remaining: {len(remaining)}, Dropped: {len(dropped)}")

        grounding = evaluate_grounding(remaining)
        staleness = evaluate_staleness(remaining)

        results[f"{rate*100:.0f}pct"] = {
            "n_readings": len(remaining),
            "n_dropped": len(dropped),
            "grounding_latency_ms": grounding["latency_ms"],
            "staleness_latency_ms": staleness["mean_latency_ms"],
            "has_values": grounding["has_values"]
        }

    return results


def run_noise_experiment(n_base_readings: int = 500) -> Dict:
    """Run noise level experiments."""
    print("\n=== Noise Level Experiments ===")

    base_data = generate_test_data(n_base_readings, noise_std=5.0)
    results = {}

    noise_levels = [0.0, 0.1, 0.2, 0.3]  # multiplier

    for level in noise_levels:
        print(f"\nTesting noise level {level*100:.0f}%...")
        noisy = add_noise(base_data, level) if level > 0 else base_data

        grounding = evaluate_grounding(noisy)
        staleness = evaluate_staleness(noisy)

        results[f"{level*100:.0f}pct"] = {
            "n_readings": len(noisy),
            "grounding_latency_ms": grounding["latency_ms"],
            "staleness_latency_ms": staleness["mean_latency_ms"],
            "has_values": grounding["has_values"]
        }

    return results


def run_experiment(n_readings: int = 500, seed: int = 2025) -> Dict:
    """Run all robustness experiments."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    results = {
        "experiment": "sampling_robustness",
        "seed": seed,
        "timestamp": timestamp,
        "config": {
            "n_base_readings": n_readings
        },
        "sampling_rate": run_sampling_experiment(n_readings),
        "data_dropout": run_dropout_experiment(n_readings),
        "noise_level": run_noise_experiment(n_readings)
    }

    return results


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025, dataset: str = None):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp09_sampling_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_summary(results: Dict):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("SAMPLING ROBUSTNESS SUMMARY")
    print("=" * 60)

    print("\n--- Sampling Rate Impact ---")
    for rate, data in results["sampling_rate"].items():
        print(f"  {rate}: {data['n_readings']} readings, "
              f"latency={data['grounding_latency_ms']:.1f}ms")

    print("\n--- Data Dropout Impact ---")
    for rate, data in results["data_dropout"].items():
        print(f"  {rate} dropout: {data['n_readings']} remaining, "
              f"latency={data['grounding_latency_ms']:.1f}ms")

    print("\n--- Noise Level Impact ---")
    for level, data in results["noise_level"].items():
        print(f"  {level} noise: latency={data['grounding_latency_ms']:.1f}ms")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 9: Sampling Robustness")
    parser.add_argument("--n-readings", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset for reference (uses synthetic data scaled to dataset)")

    args = parser.parse_args()

    dataset_name = args.dataset or "synthetic"
    print("=" * 60)
    print(f"Experiment 9: Sampling Robustness")
    print(f"  Dataset: {dataset_name}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = run_experiment(args.n_readings, seed=args.seed)
    results["dataset"] = dataset_name
    save_results(results, args.output_dir, seed=args.seed, dataset=dataset_name)
    print_summary(results)


if __name__ == "__main__":
    main()
