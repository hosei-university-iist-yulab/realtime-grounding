#!/usr/bin/env python3
"""
Experiment 6: Scalability Analysis

Tests TGP performance as number of sensors increases.

Metrics:
- Latency vs. sensor count
- Memory usage vs. sensor count
- Throughput (queries/second)

Target: Sub-linear latency growth with sensor count.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")


SENSOR_COUNTS = [10, 50, 100, 500, 1000, 5000]


def benchmark_sensor_scale(
    n_sensors: int,
    n_queries: int = 100,
    readings_per_sensor: int = 100
) -> Dict[str, float]:
    """Benchmark performance at given sensor count."""
    print(f"\nBenchmarking {n_sensors} sensors...")

    from src.buffer import TemporalGroundingBuffer, SensorReading

    buffer = TemporalGroundingBuffer()

    # Populate sensors
    print(f"  Populating {n_sensors} sensors with {readings_per_sensor} readings each...")
    populate_start = time.time()

    for sensor_id in range(n_sensors):
        readings = [
            SensorReading(
                timestamp=time.time() - (readings_per_sensor - i) * 60,
                building_id=f"building_{sensor_id:05d}",
                meter_type="electricity",
                value=100.0 + np.random.randn() * 20
            )
            for i in range(readings_per_sensor)
        ]
        buffer.push_batch(readings)

    populate_time = time.time() - populate_start
    print(f"  Populated in {populate_time:.1f}s")

    # Benchmark queries
    print(f"  Running {n_queries} queries...")
    latencies = []

    for i in range(n_queries):
        sensor_id = np.random.randint(0, n_sensors)
        building = f"building_{sensor_id:05d}"

        start = time.perf_counter()
        _ = buffer.get_latest(building, "electricity", n=10)
        _ = buffer.get_statistics(building, "electricity")
        latencies.append((time.perf_counter() - start) * 1000)

    # Cleanup
    print("  Cleaning up...")
    for sensor_id in range(n_sensors):
        buffer.clear(f"building_{sensor_id:05d}", "electricity")

    return {
        "n_sensors": n_sensors,
        "n_queries": n_queries,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput_qps": n_queries / (sum(latencies) / 1000),
        "populate_time_s": populate_time
    }


def run_experiment(
    sensor_counts: List[int] = None,
    n_queries: int = 100,
    seed: int = 2025
) -> Dict[str, Any]:
    """Run scalability experiment."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    if sensor_counts is None:
        sensor_counts = SENSOR_COUNTS

    results = {
        "experiment": "scalability",
        "seed": seed,
        "timestamp": timestamp,
        "sensor_counts": sensor_counts,
        "results": []
    }

    for n_sensors in sensor_counts:
        try:
            metrics = benchmark_sensor_scale(n_sensors, n_queries)
            results["results"].append(metrics)
        except Exception as e:
            print(f"  Error at {n_sensors} sensors: {e}")
            results["results"].append({
                "n_sensors": n_sensors,
                "error": str(e)
            })

    return results


def analyze_scaling(results: Dict) -> Dict[str, Any]:
    """Analyze scaling behavior."""
    valid_results = [r for r in results["results"] if "mean_latency_ms" in r]

    if len(valid_results) < 2:
        return {"error": "Not enough data points"}

    sensors = [r["n_sensors"] for r in valid_results]
    latencies = [r["mean_latency_ms"] for r in valid_results]

    # Fit log-linear model: latency = a + b * log(sensors)
    log_sensors = np.log(sensors)
    coeffs = np.polyfit(log_sensors, latencies, 1)

    return {
        "scaling_coefficient": float(coeffs[0]),  # b
        "base_latency": float(coeffs[1]),  # a
        "scaling_type": "sub-linear" if coeffs[0] < 1.0 else "linear or worse",
        "max_sensors_tested": max(sensors),
        "max_latency_ms": max(latencies)
    }


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp06_scalability_seed{seed}_{timestamp}.json"

    # Add scaling analysis
    results["analysis"] = analyze_scaling(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 6: Scalability")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--max-sensors", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")

    args = parser.parse_args()

    # Adjust sensor counts based on max
    sensor_counts = [s for s in SENSOR_COUNTS if s <= args.max_sensors]

    print("=" * 50)
    print(f"Experiment 6: Scalability Analysis (seed={args.seed})")
    print("=" * 50)
    print(f"Testing sensor counts: {sensor_counts}")

    results = run_experiment(sensor_counts, args.n_queries, seed=args.seed)
    save_results(results, args.output_dir, seed=args.seed)

    # Print summary
    print("\n" + "=" * 50)
    print("Scalability Summary")
    print("=" * 50)
    print(f"{'Sensors':<10} {'Latency (ms)':<15} {'Throughput (qps)':<15}")
    print("-" * 40)
    for r in results["results"]:
        if "mean_latency_ms" in r:
            print(f"{r['n_sensors']:<10} {r['mean_latency_ms']:<15.2f} {r['throughput_qps']:<15.0f}")

    if "analysis" in results and "scaling_type" in results["analysis"]:
        print(f"\nScaling behavior: {results['analysis']['scaling_type']}")


if __name__ == "__main__":
    main()
