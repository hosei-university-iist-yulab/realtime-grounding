#!/usr/bin/env python3
"""
Experiment 10: Cross-Dataset Validation

Tests generalization of TGP across REAL energy datasets:
- BDG2 (Building Data Genome 2) - Commercial buildings (Kaggle)
- REDD (Reference Energy Disaggregation Dataset) - US residential (Kaggle)

This validates that the approach generalizes across building types.
NO SIMULATED DATA - all results from real datasets only.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")


def check_dataset_availability() -> Dict[str, bool]:
    """Check which REAL datasets are available (BDG2 and REDD only)."""
    data_dir = PROJECT_ROOT / "data"

    availability = {
        "bdg2": False,
        "redd": False
    }

    # Check BDG2 (original or Kaggle format)
    bdg2_paths = [
        data_dir / "raw" / "bdg2",
        data_dir / "raw" / "bdg2_kaggle"  # Kaggle symlink
    ]
    for bdg2_path in bdg2_paths:
        if bdg2_path.exists():
            # Kaggle format: electricity.csv
            if (bdg2_path / "electricity.csv").exists():
                availability["bdg2"] = True
                break
            # Original format: meters.csv or readings/
            if (bdg2_path / "meters.csv").exists() or (bdg2_path / "readings").exists():
                availability["bdg2"] = True
                break

    # Check REDD (original or Kaggle format)
    redd_path = data_dir / "raw" / "redd"
    if redd_path.exists():
        # Kaggle format: dev*.csv files
        dev_files = list(redd_path.glob("dev*.csv"))
        if dev_files:
            availability["redd"] = True
        # Original format: house_* directories
        house_dirs = list(redd_path.glob("house_*"))
        if house_dirs:
            availability["redd"] = True

    return availability


def load_dataset_samples(
    dataset: str,
    n_samples: int = 100,
    seed: int = 42
) -> List[Dict]:
    """
    Load samples from a dataset (supports both original and Kaggle formats).

    Returns list of dicts with:
    - building_id: str
    - readings: List[float]
    - timestamps: List[float]
    - stats: Dict with mean, std, min, max
    """
    np.random.seed(seed)
    data_dir = PROJECT_ROOT / "data"
    samples = []

    if dataset == "bdg2":
        from src.data.loaders import BDG2Loader

        # Try Kaggle path first, then original
        bdg2_path = data_dir / "raw" / "bdg2_kaggle"
        if not bdg2_path.exists():
            bdg2_path = data_dir / "raw" / "bdg2"

        loader = BDG2Loader(str(bdg2_path))
        buildings = loader.list_buildings()[:10]  # Limit to 10 buildings

        for building_id in buildings:
            try:
                df = loader.get_meter_data(building_id, meter_type="electricity")
                if df is not None and len(df) >= n_samples:
                    # Sample readings
                    idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
                    idx = np.sort(idx)

                    readings = df.iloc[idx]["value"].tolist()
                    timestamps = [t.timestamp() for t in df.iloc[idx]["timestamp"]]

                    samples.append({
                        "building_id": building_id,
                        "dataset": "bdg2",
                        "readings": readings,
                        "timestamps": timestamps,
                        "stats": {
                            "mean": float(np.mean(readings)),
                            "std": float(np.std(readings)),
                            "min": float(np.min(readings)),
                            "max": float(np.max(readings))
                        }
                    })
            except Exception as e:
                print(f"  Warning: Could not load {building_id}: {e}")
                continue

    elif dataset == "redd":
        from src.data.loaders import REDDLoader
        loader = REDDLoader(str(data_dir / "raw" / "redd"))
        buildings = loader.list_buildings()

        for building_id in buildings:
            try:
                df = loader.get_meter_data(building_id, meter_type="electricity")
                if len(df) >= n_samples:
                    idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
                    idx = np.sort(idx)

                    readings = df.iloc[idx]["value"].tolist()
                    timestamps = [t.timestamp() for t in df.iloc[idx]["timestamp"]]

                    samples.append({
                        "building_id": building_id,
                        "dataset": "redd",
                        "readings": readings,
                        "timestamps": timestamps,
                        "stats": {
                            "mean": float(np.mean(readings)),
                            "std": float(np.std(readings)),
                            "min": float(np.min(readings)),
                            "max": float(np.max(readings))
                        }
                    })
            except Exception as e:
                print(f"  Warning: Could not load {building_id}: {e}")
                continue

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Only 'bdg2' and 'redd' are supported.")

    return samples


def evaluate_grounding_on_samples(
    samples: List[Dict],
    llm: Any,
    query_templates: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate grounding accuracy on dataset samples.

    Uses REAL LLM inference.
    """
    from src.buffer import TemporalGroundingBuffer, SensorReading

    if query_templates is None:
        query_templates = [
            "What is the current energy consumption?",
            "Is consumption higher or lower than average?",
            "Describe the consumption pattern."
        ]

    buffer = TemporalGroundingBuffer()
    results = []
    latencies = []

    for sample in samples:
        building_id = sample["building_id"]

        # Push readings to buffer
        for ts, val in zip(sample["timestamps"], sample["readings"]):
            reading = SensorReading(
                timestamp=ts,
                building_id=building_id,
                meter_type="electricity",
                value=val
            )
            buffer.push(reading)

        # Test each query
        for query in query_templates:
            stats = sample["stats"]
            latest_val = sample["readings"][-1] if sample["readings"] else 0

            prompt = f"""<|system|>
You are an energy monitoring assistant. Use the sensor data provided.
</s>
<|user|>
Building: {building_id}
Current readings - Mean: {stats['mean']:.1f} kWh, Latest: {latest_val:.1f} kWh
Min: {stats['min']:.1f}, Max: {stats['max']:.1f}
Question: {query}
</s>
<|assistant|>
"""

            start = time.perf_counter()
            response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Check response quality
            import re
            has_numbers = len(re.findall(r'[\d.]+', response)) > 0
            response_length = len(response)

            results.append({
                "building_id": building_id,
                "query": query,
                "has_numbers": has_numbers,
                "response_length": response_length,
                "latency_ms": latency
            })

        # Clear buffer for next building
        buffer.clear(building_id, "electricity")

    # Aggregate results
    valid_responses = sum(1 for r in results if r["has_numbers"])

    return {
        "n_samples": len(samples),
        "n_queries": len(results),
        "valid_response_rate": valid_responses / max(1, len(results)),
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0
    }


def run_cross_validation(
    available_datasets: Dict[str, bool],
    n_samples_per_building: int = 50
) -> Dict[str, Any]:
    """
    Run cross-dataset validation.

    Tests model trained on BDG2 against other datasets.
    """
    from src.llm import LLMBackbone

    print("\n=== Cross-Dataset Validation ===\n")

    # Load LLM with trained weights (loads automatically in __init__)
    print("Loading LLM backbone...")
    llm = LLMBackbone(model_type="tinyllama")

    # Try to load LoRA weights
    models_dir = PROJECT_ROOT / "output" / "models"
    lora_loaded = False
    if models_dir.exists():
        model_dirs = sorted(models_dir.glob("grounding_*"))
        if model_dirs:
            lora_path = model_dirs[-1] / "final"
            if lora_path.exists():
                print(f"Loading LoRA from {lora_path}")
                llm.load_lora(str(lora_path))
                lora_loaded = True

    if not lora_loaded:
        print("WARNING: No trained LoRA weights found. Using base model.")

    results = {
        "lora_loaded": lora_loaded,
        "datasets": {}
    }

    # Test on REAL datasets only (BDG2 and REDD)
    datasets_to_test = ["bdg2", "redd"]

    for dataset in datasets_to_test:
        print(f"\nTesting on {dataset.upper()}...")

        if not available_datasets.get(dataset, False):
            print(f"  ERROR: {dataset.upper()} not available!")
            print(f"  Please download the dataset first using scripts/download_datasets.py")
            results["datasets"][dataset] = {
                "error": f"{dataset} dataset not found",
                "n_buildings": 0
            }
            continue

        print(f"  Loading REAL {dataset} data...")
        try:
            samples = load_dataset_samples(dataset, n_samples=n_samples_per_building)
            if not samples:
                raise ValueError("No samples loaded from dataset")
            print(f"  Loaded {len(samples)} buildings from REAL data")
        except Exception as e:
            print(f"  ERROR loading {dataset}: {e}")
            results["datasets"][dataset] = {
                "error": str(e),
                "n_buildings": 0
            }
            continue

        # Evaluate
        print(f"  Evaluating grounding on {len(samples)} buildings...")
        eval_results = evaluate_grounding_on_samples(samples, llm)

        results["datasets"][dataset] = {
            "n_buildings": len(samples),
            "real_data": True,
            **eval_results
        }

        print(f"  Valid response rate: {eval_results['valid_response_rate']:.2%}")
        print(f"  Mean latency: {eval_results['mean_latency_ms']:.1f}ms")

    return results


def compute_generalization_gap(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute generalization gap between BDG2 (commercial) and REDD (residential).

    Gap = performance difference between dataset types.
    Positive gap means BDG2 performed better.
    """
    datasets = results.get("datasets", {})

    if "bdg2" not in datasets or "error" in datasets.get("bdg2", {}):
        return {"error": "BDG2 baseline not available"}

    if "redd" not in datasets or "error" in datasets.get("redd", {}):
        return {"error": "REDD data not available"}

    bdg2_rate = datasets["bdg2"].get("valid_response_rate", 0)
    redd_rate = datasets["redd"].get("valid_response_rate", 0)

    # Compute gap based on valid response rate
    gaps = {
        "bdg2_to_redd": bdg2_rate - redd_rate,
        "commercial_vs_residential": abs(bdg2_rate - redd_rate)
    }

    # Also compute latency gap
    bdg2_latency = datasets["bdg2"].get("mean_latency_ms", 0)
    redd_latency = datasets["redd"].get("mean_latency_ms", 0)
    if bdg2_latency > 0 and redd_latency > 0:
        gaps["latency_gap_pct"] = (redd_latency - bdg2_latency) / bdg2_latency

    return gaps


def run_experiment(n_samples: int = 50, seed: int = 2025) -> Dict:
    """Run full cross-dataset validation experiment."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    # Check dataset availability
    print("Checking dataset availability...")
    availability = check_dataset_availability()

    for name, available in availability.items():
        status = "Available (REAL data)" if available else "NOT FOUND - download first!"
        print(f"  {name.upper()}: {status}")

    # Run validation
    results = run_cross_validation(availability, n_samples_per_building=n_samples)

    # Compute generalization gaps
    gaps = compute_generalization_gap(results)

    return {
        "experiment": "cross_dataset_validation",
        "seed": seed,
        "timestamp": timestamp,
        "config": {
            "n_samples_per_building": n_samples
        },
        "dataset_availability": availability,
        "results": results,
        "generalization_gaps": gaps
    }


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp10_cross_dataset_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_summary(results: Dict):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("CROSS-DATASET VALIDATION SUMMARY (REAL DATA ONLY)")
    print("=" * 60)

    print("\n--- Dataset Availability ---")
    for name, available in results["dataset_availability"].items():
        status = "REAL data loaded" if available else "NOT AVAILABLE"
        print(f"  {name.upper()}: {status}")

    print("\n--- Performance by Dataset ---")
    for name, data in results["results"]["datasets"].items():
        if "error" in data:
            print(f"  {name.upper()}: ERROR - {data['error']}")
            continue
        print(f"  {name.upper()} (REAL):")
        print(f"    Buildings: {data['n_buildings']}")
        print(f"    Valid responses: {data.get('valid_response_rate', 0):.2%}")
        print(f"    Latency: {data.get('mean_latency_ms', 0):.1f}ms (±{data.get('std_latency_ms', 0):.1f})")

    print("\n--- Generalization Gaps (BDG2 vs REDD) ---")
    gaps = results["generalization_gaps"]
    if "error" in gaps:
        print(f"  ERROR: {gaps['error']}")
    else:
        if "bdg2_to_redd" in gaps:
            gap = gaps["bdg2_to_redd"]
            direction = "better on BDG2" if gap > 0 else "better on REDD"
            print(f"  Response rate gap: {abs(gap):.2%} ({direction})")
        if "commercial_vs_residential" in gaps:
            print(f"  Commercial vs Residential: {gaps['commercial_vs_residential']:.2%} difference")
        if "latency_gap_pct" in gaps:
            lat_gap = gaps["latency_gap_pct"]
            print(f"  Latency gap: {abs(lat_gap):.1%} ({'REDD slower' if lat_gap > 0 else 'BDG2 slower'})")

    # Overall assessment
    print("\n--- Assessment ---")
    if "error" not in gaps:
        comm_gap = gaps.get("commercial_vs_residential", 0)
        if comm_gap < 0.1:
            print("  GOOD: Similar performance on commercial and residential (gap < 10%)")
        elif comm_gap < 0.2:
            print("  MODERATE: Some difference between building types (10% < gap < 20%)")
        else:
            print("  POOR: Large gap between building types (gap > 20%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 10: Cross-Dataset Validation")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Samples per building")
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print(f"Experiment 10: Cross-Dataset Validation (seed={args.seed})")
    print("=" * 60)

    results = run_experiment(args.n_samples, seed=args.seed)
    save_results(results, args.output_dir, seed=args.seed)
    print_summary(results)


if __name__ == "__main__":
    main()
