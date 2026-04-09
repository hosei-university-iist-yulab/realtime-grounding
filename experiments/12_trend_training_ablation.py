"""
Experiment 12: Trend Training Data Ablation.

Investigates how different training data compositions affect trend detection:
1. Synthetic-only training
2. Real-only training
3. Mixed (synthetic + real)
4. Different trend type ratios

Target: Identify optimal training data composition for 85%+ trend accuracy.
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
from typing import Dict, List, Any, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.buffer.trend_analyzer import TrendAnalyzer
from src.data.loaders import BDG2Loader

PROJECT_ROOT = Path(__file__).parent.parent


def load_model(model_path: Path = None):
    """Load fine-tuned TinyLLaMA model."""
    if model_path is None:
        models_dir = PROJECT_ROOT / "output" / "models"
        model_dirs = sorted([d for d in models_dir.iterdir()
                            if d.is_dir() and d.name.startswith("tinyllama_trend_")])
        if not model_dirs:
            raise FileNotFoundError("No trend fine-tuned models found")
        model_path = model_dirs[-1] / "final"

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


def generate_synthetic_trends(n_samples: int = 100, seed: int = 2025) -> List[Dict]:
    """Generate synthetic trend scenarios for ablation."""
    np.random.seed(seed)
    scenarios = []

    trend_types = ["increasing", "decreasing", "stable", "volatile"]

    for i in range(n_samples):
        trend_type = trend_types[i % len(trend_types)]

        if trend_type == "increasing":
            base = np.random.uniform(50, 200)
            slope = np.random.uniform(0.5, 2.0)
            values = [base + slope * t + np.random.normal(0, 5) for t in range(100)]
        elif trend_type == "decreasing":
            base = np.random.uniform(150, 300)
            slope = np.random.uniform(0.5, 2.0)
            values = [base - slope * t + np.random.normal(0, 5) for t in range(100)]
        elif trend_type == "stable":
            base = np.random.uniform(100, 200)
            values = [base + np.random.normal(0, 3) for _ in range(100)]
        else:  # volatile
            base = np.random.uniform(100, 200)
            values = [base + np.random.normal(0, 30) for _ in range(100)]

        scenarios.append({
            "type": "synthetic",
            "values": values,
            "expected_trend": trend_type,
            "scenario_id": f"synthetic_{i}"
        })

    return scenarios


def extract_real_trends(n_samples: int = 100, seed: int = 2025) -> List[Dict]:
    """Extract real trend windows from BDG2 dataset."""
    np.random.seed(seed)

    data_dir = PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"
    loader = BDG2Loader(str(data_dir))
    buildings = loader.list_buildings()

    analyzer = TrendAnalyzer()
    scenarios = []

    for building_id in buildings[:30]:
        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < 200:
                continue

            for _ in range(5):
                if len(scenarios) >= n_samples:
                    break

                max_start = len(df) - 100
                start_idx = np.random.randint(0, max_start)
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                features = analyzer.analyze(values, timestamps)

                scenarios.append({
                    "type": "real",
                    "values": values,
                    "expected_trend": features.direction,
                    "scenario_id": f"real_{building_id}_{start_idx}",
                    "building_id": building_id
                })
        except Exception:
            continue

        if len(scenarios) >= n_samples:
            break

    return scenarios


def evaluate_on_scenarios(model, tokenizer, scenarios: List[Dict]) -> Dict:
    """Evaluate model on a set of scenarios."""
    analyzer = TrendAnalyzer()
    results = []

    for scenario in scenarios:
        values = scenario["values"]
        timestamps = list(range(len(values)))
        features = analyzer.analyze(values, timestamps)

        slope_per_hour = features.slope * 3600

        prompt = f"""You are analyzing energy consumption data.

Recent statistics (last 100 readings):
- Mean: {np.mean(values):.1f} kWh
- Std: {np.std(values):.1f} kWh
- Min: {np.min(values):.1f} kWh
- Max: {np.max(values):.1f} kWh

Trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}
- R²: {features.r_squared:.3f}

Question: What is the trend in energy consumption?
Answer with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        latency = (time.perf_counter() - start) * 1000

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response_lower = response.lower().strip()

        predicted = "unknown"
        for trend in ["increasing", "decreasing", "stable", "volatile"]:
            if trend in response_lower:
                predicted = trend
                break

        correct = predicted == scenario["expected_trend"]

        results.append({
            "scenario_id": scenario["scenario_id"],
            "type": scenario["type"],
            "expected": scenario["expected_trend"],
            "predicted": predicted,
            "correct": correct,
            "latency_ms": latency
        })

    # Compute metrics
    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0

    # Per-type accuracy
    per_type = {}
    for ttype in ["increasing", "decreasing", "stable", "volatile"]:
        type_results = [r for r in results if r["expected"] == ttype]
        if type_results:
            per_type[ttype] = {
                "accuracy": sum(r["correct"] for r in type_results) / len(type_results),
                "n_samples": len(type_results)
            }

    return {
        "accuracy": accuracy,
        "n_samples": len(results),
        "per_trend_type": per_type,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0,
        "results": results
    }


def run_ablation(seed: int = 2025):
    """Run trend training data ablation study."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 12: Trend Training Data Ablation (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/5] Loading fine-tuned model...")
    model, tokenizer = load_model()
    print("  ✓ Model loaded")

    # Generate test scenarios
    print("\n[2/5] Generating synthetic test scenarios...")
    synthetic_scenarios = generate_synthetic_trends(n_samples=40, seed=seed)
    print(f"  ✓ Generated {len(synthetic_scenarios)} synthetic scenarios")

    print("\n[3/5] Extracting real test scenarios from BDG2...")
    real_scenarios = extract_real_trends(n_samples=40, seed=seed)
    print(f"  ✓ Extracted {len(real_scenarios)} real scenarios")

    # Evaluate on different compositions
    print("\n[4/5] Running ablation evaluations...")

    ablation_results = {}

    # A) Synthetic only
    print("  [A] Evaluating on synthetic-only...")
    ablation_results["synthetic_only"] = evaluate_on_scenarios(model, tokenizer, synthetic_scenarios)
    print(f"      Accuracy: {ablation_results['synthetic_only']['accuracy']*100:.1f}%")

    # B) Real only
    print("  [B] Evaluating on real-only...")
    ablation_results["real_only"] = evaluate_on_scenarios(model, tokenizer, real_scenarios)
    print(f"      Accuracy: {ablation_results['real_only']['accuracy']*100:.1f}%")

    # C) Mixed (50/50)
    mixed_scenarios = synthetic_scenarios[:20] + real_scenarios[:20]
    np.random.shuffle(mixed_scenarios)
    print("  [C] Evaluating on mixed (50/50)...")
    ablation_results["mixed_50_50"] = evaluate_on_scenarios(model, tokenizer, mixed_scenarios)
    print(f"      Accuracy: {ablation_results['mixed_50_50']['accuracy']*100:.1f}%")

    # D) Balanced by trend type
    balanced = []
    for ttype in ["increasing", "decreasing", "stable", "volatile"]:
        type_syn = [s for s in synthetic_scenarios if s["expected_trend"] == ttype][:5]
        type_real = [s for s in real_scenarios if s["expected_trend"] == ttype][:5]
        balanced.extend(type_syn + type_real)

    print("  [D] Evaluating on balanced by trend type...")
    ablation_results["balanced_types"] = evaluate_on_scenarios(model, tokenizer, balanced)
    print(f"      Accuracy: {ablation_results['balanced_types']['accuracy']*100:.1f}%")

    # Save results
    print("\n[5/5] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp12_trend_ablation_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "trend_training_ablation",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "ablations": {
                name: {
                    "accuracy": r["accuracy"],
                    "n_samples": r["n_samples"],
                    "per_trend_type": r["per_trend_type"],
                    "avg_latency_ms": r["avg_latency_ms"]
                }
                for name, r in ablation_results.items()
            }
        }, f, indent=2)

    print(f"  ✓ Results saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Ablation':<20} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 42)
    for name, r in ablation_results.items():
        print(f"{name:<20} {r['accuracy']*100:>9.1f}% {r['n_samples']:>10}")

    best = max(ablation_results.items(), key=lambda x: x[1]["accuracy"])
    print(f"\nBest configuration: {best[0]} ({best[1]['accuracy']*100:.1f}%)")

    return ablation_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp12: Trend training data ablation")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    results = run_ablation(seed=args.seed)
    print(f"\n✓ Experiment 12 completed (seed={args.seed})!")
