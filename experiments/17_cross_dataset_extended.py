"""
Experiment 17: Cross-Dataset Validation (Extended)

Tests the fine-tuned multi-task model on multiple datasets:
- BDG2: Commercial buildings (training domain)
- UCR ElectricDevices: Electrical device signatures (transfer domain)

Goal: Validate generalization with <15% accuracy drop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.data.loaders import BDG2Loader, UCRLoader
from src.buffer.trend_analyzer import TrendAnalyzer

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_multitask_model():
    """Find the latest multi-task fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("tinyllama_multitask_")
    ])
    if not model_dirs:
        model_dirs = sorted([
            d for d in models_dir.iterdir()
            if d.is_dir() and d.name.startswith("tinyllama_trend_")
        ])
    if not model_dirs:
        raise FileNotFoundError("No fine-tuned models found")
    return model_dirs[-1] / "final"


def load_finetuned_model():
    """Load the fine-tuned TinyLLaMA model."""
    model_path = find_latest_multitask_model()
    print(f"Loading model from: {model_path}")

    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> Tuple[str, float]:
    """Generate response with latency measurement."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    latency = (time.perf_counter() - start) * 1000

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip(), latency


def extract_trend(response: str) -> str:
    """Extract trend from response."""
    response_lower = response.lower()
    for trend in ["increasing", "decreasing", "stable", "volatile"]:
        if trend in response_lower:
            return trend
    return "unknown"


def compute_ground_truth_trend(values: List[float]) -> str:
    """Compute ground truth trend from values."""
    if len(values) < 10:
        return "unknown"

    # Compute slope
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)

    # Normalize slope by mean
    mean_val = np.mean(values)
    if mean_val > 0:
        slope_normalized = slope / mean_val
    else:
        slope_normalized = slope

    # Compute volatility (coefficient of variation)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val > 0 else 0

    # Classify
    if cv > 0.3:  # High volatility
        return "volatile"
    elif abs(slope_normalized) < 0.001:  # Flat
        return "stable"
    elif slope_normalized > 0:
        return "increasing"
    else:
        return "decreasing"


def test_on_bdg2(model, tokenizer, analyzer: TrendAnalyzer, n_samples: int = 50, seed: int = 2025) -> Dict:
    """Test on BDG2 dataset (training domain)."""
    print("\n  Loading BDG2 samples...")

    loader = BDG2Loader(str(PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"))
    buildings = loader.list_buildings()[:10]  # Use first 10 buildings

    results = []
    np.random.seed(seed)

    for building in buildings:
        try:
            df = loader.get_meter_data(building)
            if df is None or len(df) < 100:
                continue

            # Sample windows
            for _ in range(n_samples // len(buildings)):
                start_idx = np.random.randint(0, max(1, len(df) - 100))
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                # Get ground truth
                gt_trend = compute_ground_truth_trend(values)
                if gt_trend == "unknown":
                    continue

                # Analyze with TrendAnalyzer
                features = analyzer.analyze(values, timestamps)
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

                # Create prompt
                slope_per_hour = features.slope * 3600
                prompt = f"""You are analyzing energy consumption data for a building.

Recent statistics (last {len(values)} readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh

Enhanced trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

                response, latency = generate(model, tokenizer, prompt)
                pred_trend = extract_trend(response)

                results.append({
                    "dataset": "bdg2",
                    "building": building,
                    "gt_trend": gt_trend,
                    "pred_trend": pred_trend,
                    "correct": gt_trend == pred_trend,
                    "latency_ms": latency
                })

        except Exception as e:
            print(f"    Warning: {building} - {e}")
            continue

    # Compute metrics
    if not results:
        return {"accuracy": 0, "n_samples": 0, "latency_ms": 0}

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = np.mean([r["latency_ms"] for r in results])

    return {
        "dataset": "BDG2",
        "accuracy": accuracy * 100,
        "n_samples": len(results),
        "avg_latency_ms": avg_latency,
        "results": results
    }


def test_on_ucr(model, tokenizer, analyzer: TrendAnalyzer, n_samples: int = 50, seed: int = 2025) -> Dict:
    """Test on UCR ElectricDevices dataset (transfer domain)."""
    print("\n  Loading UCR ElectricDevices samples...")

    loader = UCRLoader("ElectricDevices")
    samples = loader.get_random_samples(n_samples=n_samples, seed=seed)

    results = []

    for sample in samples:
        values = sample["values"]
        timestamps = sample["timestamps"]

        # Get ground truth
        gt_trend = compute_ground_truth_trend(values)
        if gt_trend == "unknown":
            continue

        # Analyze with TrendAnalyzer
        features = analyzer.analyze(values, timestamps)
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

        # Create prompt
        slope_per_hour = features.slope * 3600
        prompt = f"""You are analyzing energy consumption data for a building.

Recent statistics (last {len(values)} readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh

Enhanced trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

        response, latency = generate(model, tokenizer, prompt)
        pred_trend = extract_trend(response)

        results.append({
            "dataset": "ucr",
            "sample_id": sample["building_id"],
            "class_label": sample["class_label"],
            "gt_trend": gt_trend,
            "pred_trend": pred_trend,
            "correct": gt_trend == pred_trend,
            "latency_ms": latency
        })

    # Compute metrics
    if not results:
        return {"accuracy": 0, "n_samples": 0, "latency_ms": 0}

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = np.mean([r["latency_ms"] for r in results])

    return {
        "dataset": "UCR-ElectricDevices",
        "accuracy": accuracy * 100,
        "n_samples": len(results),
        "avg_latency_ms": avg_latency,
        "results": results
    }


def run_cross_dataset_experiment(n_samples_per_dataset: int = 50, seed: int = 2025):
    """Run cross-dataset validation experiment."""

    print("=" * 80)
    print(f"EXPERIMENT 17: Cross-Dataset Validation (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    print("  ✓ Model loaded")

    # Initialize analyzer
    analyzer = TrendAnalyzer()

    # Test on BDG2 (training domain)
    print("\n[2/4] Testing on BDG2 (training domain)...")
    bdg2_results = test_on_bdg2(model, tokenizer, analyzer, n_samples_per_dataset, seed)
    print(f"  ✓ BDG2: {bdg2_results['accuracy']:.1f}% ({bdg2_results['n_samples']} samples)")

    # Test on UCR (transfer domain)
    print("\n[3/4] Testing on UCR ElectricDevices (transfer domain)...")
    ucr_results = test_on_ucr(model, tokenizer, analyzer, n_samples_per_dataset, seed)
    print(f"  ✓ UCR: {ucr_results['accuracy']:.1f}% ({ucr_results['n_samples']} samples)")

    # Compute cross-dataset gap
    gap = bdg2_results["accuracy"] - ucr_results["accuracy"]

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp17_crossdataset_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "cross_dataset_validation",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "datasets": {
                "bdg2": {
                    "accuracy": bdg2_results["accuracy"],
                    "n_samples": bdg2_results["n_samples"],
                    "avg_latency_ms": bdg2_results["avg_latency_ms"]
                },
                "ucr": {
                    "accuracy": ucr_results["accuracy"],
                    "n_samples": ucr_results["n_samples"],
                    "avg_latency_ms": ucr_results["avg_latency_ms"]
                }
            },
            "cross_dataset_gap": gap,
            "target_gap": 15.0
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("CROSS-DATASET VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Dataset':<25} {'Accuracy':<15} {'Samples':<10} {'Latency'}")
    print("-" * 60)
    print(f"{'BDG2 (training)':<25} {bdg2_results['accuracy']:<15.1f}% {bdg2_results['n_samples']:<10} {bdg2_results['avg_latency_ms']:.0f}ms")
    print(f"{'UCR (transfer)':<25} {ucr_results['accuracy']:<15.1f}% {ucr_results['n_samples']:<10} {ucr_results['avg_latency_ms']:.0f}ms")
    print("-" * 60)
    print(f"\nCross-dataset gap: {gap:.1f}%")
    print(f"Target: <15%")

    if abs(gap) < 15:
        print("\n✓ SUCCESS: Cross-dataset gap is within 15%!")
    else:
        print(f"\n⚠ Gap exceeds target by {abs(gap) - 15:.1f}%")

    return {
        "bdg2": bdg2_results,
        "ucr": ucr_results,
        "gap": gap
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    results = run_cross_dataset_experiment(n_samples_per_dataset=50, seed=args.seed)
    print(f"\n✓ Experiment 17 completed (seed={args.seed})!")
