#!/usr/bin/env python3
"""
Experiment 1: Latency Benchmarking

Compares buffer and inference latency across:
- TGP (Ours): TemporalGroundingBuffer (Novel) + TinyLLaMA
- Redis Baseline: CircularBuffer (Redis-backed)
- Cloud API: Claude
- Local LLM: Raw TinyLLaMA (no fine-tuning)

Target: <1ms buffer latency for TGP vs ~0.2ms for Redis.

Output:
- output/results/exp01_latency_{timestamp}.json
- output/figures/fig_latency.pdf
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use GPU 4
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")


def benchmark_tgp(n_queries: int = 100) -> Dict[str, float]:
    """Benchmark TGP pipeline with TemporalGroundingBuffer (Novel)."""
    print("\n[TGP] Benchmarking with TemporalGroundingBuffer (Novel)...")

    from src.buffer import TemporalGroundingBuffer, SensorReading
    from src.staleness import TimeThresholdStalenessDetector

    # Initialize components
    buffer = TemporalGroundingBuffer()
    detector = TimeThresholdStalenessDetector()
    print(f"  Buffer: TemporalGroundingBuffer (in-process, O(1) stats)")

    # Add test data
    for i in range(100):
        reading = SensorReading(
            timestamp=time.time() - (100 - i) * 60,
            building_id="_benchmark",
            meter_type="electricity",
            value=150.0 + np.random.randn() * 10
        )
        buffer.push(reading)
    print(f"  Loaded 100 test readings")

    # Set context
    readings = buffer.get_latest("_benchmark", "electricity", n=10)
    stats = buffer.get_statistics("_benchmark", "electricity")
    detector.set_context(
        "_benchmark",
        [r.to_dict() for r in readings],
        stats,
        "_benchmark",
        "electricity"
    )

    # Benchmark loop
    latencies = []

    for i in range(n_queries):
        start = time.perf_counter()

        # Get sensor data
        latest = buffer.get_latest("_benchmark", "electricity", n=5)
        current_stats = buffer.get_statistics("_benchmark", "electricity")

        # Check staleness
        result = detector.detect(
            "_benchmark",
            [r.to_dict() for r in latest],
            current_stats,
            "_benchmark",
            "electricity"
        )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    # Cleanup
    buffer.clear("_benchmark", "electricity")
    detector.clear_context("_benchmark")

    result = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "n_queries": n_queries
    }
    print(f"  Result: {result['mean_ms']:.4f}ms mean, {result['p99_ms']:.4f}ms p99")
    return result


def benchmark_redis(n_queries: int = 100) -> Dict[str, float]:
    """Benchmark Redis CircularBuffer (baseline)."""
    print("\n[Redis] Benchmarking with CircularBuffer (Redis-backed)...")

    try:
        from src.buffer import CircularBuffer, SensorReading
        from src.staleness import TimeThresholdStalenessDetector

        # Initialize components
        buffer = CircularBuffer()
        detector = TimeThresholdStalenessDetector()
        print(f"  Buffer: CircularBuffer (Redis-backed)")

        # Add test data
        for i in range(100):
            reading = SensorReading(
                timestamp=time.time() - (100 - i) * 60,
                building_id="_redis_benchmark",
                meter_type="electricity",
                value=150.0 + np.random.randn() * 10
            )
            buffer.push(reading)
        print(f"  Loaded 100 test readings")

        # Set context
        readings = buffer.get_latest("_redis_benchmark", "electricity", n=10)
        stats = buffer.get_statistics("_redis_benchmark", "electricity")
        detector.set_context(
            "_redis_benchmark",
            [r.to_dict() for r in readings],
            stats,
            "_redis_benchmark",
            "electricity"
        )

        # Benchmark loop
        latencies = []

        for i in range(n_queries):
            start = time.perf_counter()

            # Get sensor data
            latest = buffer.get_latest("_redis_benchmark", "electricity", n=5)
            current_stats = buffer.get_statistics("_redis_benchmark", "electricity")

            # Check staleness
            result = detector.detect(
                "_redis_benchmark",
                [r.to_dict() for r in latest],
                current_stats,
                "_redis_benchmark",
                "electricity"
            )

            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        # Cleanup
        buffer.clear("_redis_benchmark", "electricity")
        detector.clear_context("_redis_benchmark")

        result = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "n_queries": n_queries
        }
        print(f"  Result: {result['mean_ms']:.4f}ms mean, {result['p99_ms']:.4f}ms p99")
        return result

    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def benchmark_claude(n_queries: int = 10) -> Dict[str, float]:
    """Benchmark Claude API latency."""
    print("\nBenchmarking Claude API...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set, skipping")
        return {"error": "API key not set"}

    try:
        from src.baselines import ClaudeBaseline

        baseline = ClaudeBaseline()
        context = {
            "building_id": "test_building",
            "meter_type": "electricity",
            "statistics": {"mean": 150.0, "std": 10.0}
        }

        latencies = []
        for i in range(n_queries):
            start = time.perf_counter()
            result = baseline.generate(
                "What is the current energy consumption?",
                context,
                max_tokens=50
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            print(f"  Query {i+1}/{n_queries}: {elapsed:.0f}ms")

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "n_queries": n_queries
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def benchmark_local_inference(n_queries: int = 20) -> Dict[str, float]:
    """Benchmark local LLM inference (without fine-tuning)."""
    print("\nBenchmarking local LLM inference...")

    try:
        from src.llm import LLMBackbone, ModelConfig

        config = ModelConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            use_4bit=True,
            use_lora=False  # No fine-tuning for baseline
        )

        llm = LLMBackbone(config=config)

        prompt = llm.format_grounding_prompt(
            {
                "building_id": "test",
                "meter_type": "electricity",
                "statistics": {"mean": 150.0, "std": 10.0}
            },
            "What is the current consumption?"
        )

        # Warmup
        _ = llm.generate(prompt, max_new_tokens=30)

        # Benchmark
        latencies = []
        for i in range(n_queries):
            start = time.perf_counter()
            _ = llm.generate(prompt, max_new_tokens=50)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "n_queries": n_queries
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


def run_experiment(
    n_tgp_queries: int = 100,
    n_api_queries: int = 10,
    n_local_queries: int = 20,
    skip_api: bool = False,
    skip_local: bool = False,
    skip_redis: bool = False,
    seed: int = 2025
) -> Dict[str, Any]:
    """Run full latency benchmark experiment."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    results = {
        "experiment": "latency_benchmark",
        "seed": seed,
        "timestamp": timestamp,
        "config": {
            "n_tgp_queries": n_tgp_queries,
            "n_api_queries": n_api_queries,
            "n_local_queries": n_local_queries
        },
        "methods": {}
    }

    print("\n" + "=" * 60)
    print("BUFFER BENCHMARKS")
    print("=" * 60)

    # TGP benchmark (Novel TemporalGroundingBuffer)
    results["methods"]["tgp_temporal"] = benchmark_tgp(n_tgp_queries)

    # Redis baseline
    if not skip_redis:
        results["methods"]["redis_baseline"] = benchmark_redis(n_tgp_queries)

    # Compute speedup if both available
    if "tgp_temporal" in results["methods"] and "redis_baseline" in results["methods"]:
        if "mean_ms" in results["methods"]["redis_baseline"]:
            tgp_mean = results["methods"]["tgp_temporal"]["mean_ms"]
            redis_mean = results["methods"]["redis_baseline"]["mean_ms"]
            speedup = redis_mean / tgp_mean if tgp_mean > 0 else 0
            results["speedup_vs_redis"] = speedup
            print(f"\n  => TGP is {speedup:.1f}x faster than Redis!")

    print("\n" + "=" * 60)
    print("API/LLM BENCHMARKS")
    print("=" * 60)

    # Claude API benchmark
    if not skip_api:
        results["methods"]["claude"] = benchmark_claude(n_api_queries)

    # Local LLM benchmark
    if not skip_local:
        results["methods"]["local_llm"] = benchmark_local_inference(n_local_queries)

    return results


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp01_latency_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return output_path


def generate_figure(results: Dict, output_dir: str = "output/figures"):
    """Generate latency comparison figure."""
    try:
        from src.utils.visualization import plot_latency_comparison

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for plotting
        plot_data = {}
        for method, data in results["methods"].items():
            if "mean_ms" in data:
                plot_data[method] = {
                    "mean_ms": data["mean_ms"],
                    "std_ms": data.get("std_ms", 0)
                }

        if plot_data:
            output_path = output_dir / "fig_latency.pdf"
            plot_latency_comparison(
                plot_data,
                output_path=str(output_path),
                title="Latency Comparison"
            )
            print(f"Figure saved to {output_path}")

    except Exception as e:
        print(f"Could not generate figure: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: Latency Benchmark")
    parser.add_argument("--n-tgp", type=int, default=100,
                        help="Number of TGP queries")
    parser.add_argument("--n-api", type=int, default=10,
                        help="Number of API queries")
    parser.add_argument("--n-local", type=int, default=20,
                        help="Number of local LLM queries")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API benchmarks")
    parser.add_argument("--skip-local", action="store_true",
                        help="Skip local LLM benchmarks")
    parser.add_argument("--skip-redis", action="store_true",
                        help="Skip Redis baseline benchmark")
    parser.add_argument("--output-dir", type=str, default="output/v2/results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print("=" * 60)
    print(f"Experiment 1: Latency Benchmark (seed={args.seed})")
    print("=" * 60)
    print(f"  TGP queries: {args.n_tgp}")
    print(f"  API queries: {args.n_api}")
    print(f"  Local queries: {args.n_local}")

    # Run experiment
    results = run_experiment(
        n_tgp_queries=args.n_tgp,
        n_api_queries=args.n_api,
        n_local_queries=args.n_local,
        skip_api=args.skip_api,
        skip_local=args.skip_local,
        skip_redis=args.skip_redis,
        seed=args.seed
    )

    # Save results
    save_results(results, args.output_dir, seed=args.seed)

    # Generate figure
    generate_figure(results)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, data in results["methods"].items():
        if "mean_ms" in data:
            print(f"  {method}: {data['mean_ms']:.4f}ms ± {data.get('std_ms', 0):.4f}ms")
        elif "error" in data:
            print(f"  {method}: ERROR - {data['error']}")

    if "speedup_vs_redis" in results:
        print(f"\n  TGP Speedup vs Redis: {results['speedup_vs_redis']:.1f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
