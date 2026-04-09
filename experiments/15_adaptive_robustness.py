"""
Experiment 15: Adaptive Robustness Testing

Tests TGP robustness under data degradation conditions:
- 50% dropout (target: <10× latency degradation)
- Compares fixed vs adaptive sampling strategies

Uses the multi-task fine-tuned model for evaluation.
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
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.buffer.temporal_buffer import TemporalGroundingBuffer, SensorReading
from src.buffer.trend_analyzer import TrendAnalyzer

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class RobustnessResult:
    """Result from a single robustness test."""
    condition: str
    dropout_rate: float
    noise_level: float
    n_readings: int
    latency_ms: float
    response_valid: bool
    trend_correct: bool
    expected_trend: str
    predicted_trend: str


class AdaptiveSampler:
    """Adaptive sampling strategy for degraded data."""

    def __init__(self):
        self.base_window = 60  # Default readings to consider

    def get_window_size(self, dropout_rate: float, noise_level: float) -> int:
        """Compute adaptive window size based on data quality."""
        # More dropout → need more readings to compensate
        if dropout_rate >= 0.5:
            return 100  # Use longer history
        elif dropout_rate >= 0.3:
            return 80
        elif noise_level >= 0.2:
            return 80  # More averaging for noise
        else:
            return self.base_window

    def get_robust_statistics(self, values: List[float], use_robust: bool = False) -> Dict:
        """Compute statistics, optionally using robust estimators."""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        if use_robust:
            # Use median and MAD (robust to outliers)
            median = float(np.median(values))
            mad = float(np.median(np.abs(np.array(values) - median)))
            return {
                "mean": median,  # Use median as center
                "std": mad * 1.4826,  # Convert MAD to std estimate
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "robust": True
            }
        else:
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "robust": False
            }


def find_latest_multitask_model():
    """Find the latest multi-task fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_multitask_")])
    if not model_dirs:
        # Fallback to trend model
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_trend_")])
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_test_data(n_readings: int, trend: str = "stable", seed: int = 2025) -> Tuple[List[float], List[float]]:
    """Generate test data with specific trend."""
    np.random.seed(seed)

    timestamps = [time.time() - (n_readings - i) * 60 for i in range(n_readings)]

    if trend == "increasing":
        base = 100 + np.linspace(0, 50, n_readings)
        noise = np.random.randn(n_readings) * 5
    elif trend == "decreasing":
        base = 150 - np.linspace(0, 50, n_readings)
        noise = np.random.randn(n_readings) * 5
    elif trend == "volatile":
        base = 100 + 30 * np.sin(np.linspace(0, 10 * np.pi, n_readings))
        noise = np.random.randn(n_readings) * 15
    else:  # stable
        base = np.full(n_readings, 120)
        noise = np.random.randn(n_readings) * 5

    values = (base + noise).tolist()
    return values, timestamps


def apply_degradation(
    values: List[float],
    timestamps: List[float],
    dropout_rate: float,
    noise_level: float,
    seed: int = 2025
) -> Tuple[List[float], List[float]]:
    """Apply dropout and noise to data."""
    np.random.seed(seed + 1000)  # Offset seed for degradation

    # Apply dropout
    if dropout_rate > 0:
        keep_mask = np.random.rand(len(values)) > dropout_rate
        values = [v for v, k in zip(values, keep_mask) if k]
        timestamps = [t for t, k in zip(timestamps, keep_mask) if k]

    # Apply noise
    if noise_level > 0:
        noise = np.random.randn(len(values)) * np.array(values) * noise_level
        values = [max(0, v + n) for v, n in zip(values, noise)]

    return values, timestamps


def run_single_test(
    model, tokenizer, analyzer: TrendAnalyzer, sampler: AdaptiveSampler,
    values: List[float], timestamps: List[float],
    expected_trend: str, dropout_rate: float, noise_level: float,
    use_adaptive: bool
) -> RobustnessResult:
    """Run a single robustness test."""

    # Get window size
    if use_adaptive:
        window_size = sampler.get_window_size(dropout_rate, noise_level)
        use_robust = noise_level >= 0.2
    else:
        window_size = 60  # Fixed
        use_robust = False

    # Use only recent window
    if len(values) > window_size:
        values = values[-window_size:]
        timestamps = timestamps[-window_size:]

    # Compute features
    if len(values) < 10:
        # Not enough data
        return RobustnessResult(
            condition="fixed" if not use_adaptive else "adaptive",
            dropout_rate=dropout_rate,
            noise_level=noise_level,
            n_readings=len(values),
            latency_ms=0,
            response_valid=False,
            trend_correct=False,
            expected_trend=expected_trend,
            predicted_trend="unknown"
        )

    features = analyzer.analyze(values, timestamps)
    stats = sampler.get_robust_statistics(values, use_robust)

    # Create prompt
    slope_per_hour = features.slope * 3600
    prompt = f"""You are analyzing energy consumption data for a building.

Recent statistics (last {len(values)} readings):
- Mean: {stats['mean']:.1f} kWh
- Std: {stats['std']:.1f} kWh
- Min: {stats['min']:.1f} kWh
- Max: {stats['max']:.1f} kWh

Enhanced trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}
- R²: {features.r_squared:.3f}
- Volatility: {features.volatility:.2f}
- Has change point: {features.has_change_point}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

    # Generate
    start = time.perf_counter()
    response = generate(model, tokenizer, prompt)
    latency = (time.perf_counter() - start) * 1000

    # Extract trend from response
    response_lower = response.lower()
    predicted = "unknown"
    for trend in ["increasing", "decreasing", "stable", "volatile"]:
        if trend in response_lower:
            predicted = trend
            break

    return RobustnessResult(
        condition="fixed" if not use_adaptive else "adaptive",
        dropout_rate=dropout_rate,
        noise_level=noise_level,
        n_readings=len(values),
        latency_ms=latency,
        response_valid=predicted != "unknown",
        trend_correct=predicted == expected_trend,
        expected_trend=expected_trend,
        predicted_trend=predicted
    )


def run_robustness_experiment(n_trials: int = 5, seed: int = 2025):
    """Run full robustness experiment."""

    print("=" * 80)
    print(f"EXPERIMENT 15: Adaptive Robustness Testing (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    print("  ✓ Model loaded")

    # Initialize
    analyzer = TrendAnalyzer()
    sampler = AdaptiveSampler()

    # Test conditions
    conditions = [
        {"dropout": 0.0, "noise": 0.0, "name": "clean"},
        {"dropout": 0.1, "noise": 0.0, "name": "10% dropout"},
        {"dropout": 0.3, "noise": 0.0, "name": "30% dropout"},
        {"dropout": 0.5, "noise": 0.0, "name": "50% dropout"},
        {"dropout": 0.0, "noise": 0.1, "name": "10% noise"},
        {"dropout": 0.0, "noise": 0.2, "name": "20% noise"},
        {"dropout": 0.3, "noise": 0.1, "name": "30% dropout + 10% noise"},
    ]

    trends = ["stable", "increasing", "decreasing", "volatile"]

    print(f"\n[2/4] Running tests ({len(conditions)} conditions × {len(trends)} trends × 2 strategies)...")

    results = []
    baseline_latencies = {}  # Store clean condition latencies

    for cond in conditions:
        print(f"\n  Testing: {cond['name']}...")

        for trend in trends:
            # Generate clean data
            values, timestamps = generate_test_data(200, trend, seed)

            # Apply degradation
            deg_values, deg_timestamps = apply_degradation(
                values.copy(), timestamps.copy(),
                cond["dropout"], cond["noise"], seed
            )

            # Test fixed strategy
            result_fixed = run_single_test(
                model, tokenizer, analyzer, sampler,
                deg_values.copy(), deg_timestamps.copy(),
                trend, cond["dropout"], cond["noise"],
                use_adaptive=False
            )
            results.append(result_fixed)

            # Store baseline
            if cond["name"] == "clean":
                baseline_latencies[trend] = result_fixed.latency_ms

            # Test adaptive strategy
            result_adaptive = run_single_test(
                model, tokenizer, analyzer, sampler,
                deg_values.copy(), deg_timestamps.copy(),
                trend, cond["dropout"], cond["noise"],
                use_adaptive=True
            )
            results.append(result_adaptive)

    # Compute metrics
    print("\n[3/4] Computing metrics...")

    # Group by condition and strategy
    metrics = {}
    for cond in conditions:
        cond_name = cond["name"]
        metrics[cond_name] = {"fixed": [], "adaptive": []}

        for r in results:
            if r.dropout_rate == cond["dropout"] and r.noise_level == cond["noise"]:
                metrics[cond_name][r.condition].append(r)

    # Compute degradation ratios
    summary = {}
    for cond_name, strategies in metrics.items():
        summary[cond_name] = {}
        for strategy, res_list in strategies.items():
            if not res_list:
                continue

            latencies = [r.latency_ms for r in res_list]
            accuracies = [r.trend_correct for r in res_list]

            avg_latency = np.mean(latencies)
            accuracy = np.mean(accuracies) * 100

            # Compute degradation ratio
            baseline_avg = np.mean(list(baseline_latencies.values())) if baseline_latencies else avg_latency
            degradation_ratio = avg_latency / baseline_avg if baseline_avg > 0 else 1.0

            summary[cond_name][strategy] = {
                "avg_latency_ms": avg_latency,
                "accuracy": accuracy,
                "degradation_ratio": degradation_ratio,
                "n_samples": len(res_list)
            }

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp15_robustness_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "adaptive_robustness",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "baseline_latencies": baseline_latencies,
            "results": [
                {
                    "condition": r.condition,
                    "dropout_rate": r.dropout_rate,
                    "noise_level": r.noise_level,
                    "n_readings": r.n_readings,
                    "latency_ms": r.latency_ms,
                    "trend_correct": r.trend_correct,
                    "expected": r.expected_trend,
                    "predicted": r.predicted_trend
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"\n{'Condition':<25} {'Strategy':<10} {'Latency':<12} {'Accuracy':<10} {'Degradation'}")
    print("-" * 80)

    for cond_name, strategies in summary.items():
        for strategy, data in strategies.items():
            degradation = data["degradation_ratio"]
            status = "✓" if degradation < 10 else "⚠"
            print(f"{cond_name:<25} {strategy:<10} {data['avg_latency_ms']:<12.1f} {data['accuracy']:<10.1f}% {degradation:.1f}× {status}")

    # Check 50% dropout target
    print("\n" + "-" * 80)
    if "50% dropout" in summary:
        fixed_deg = summary["50% dropout"].get("fixed", {}).get("degradation_ratio", 999)
        adaptive_deg = summary["50% dropout"].get("adaptive", {}).get("degradation_ratio", 999)

        print(f"\n50% Dropout Target: <10× degradation")
        print(f"  Fixed strategy:    {fixed_deg:.1f}×")
        print(f"  Adaptive strategy: {adaptive_deg:.1f}×")

        if adaptive_deg < 10:
            print("\n✓ SUCCESS: Adaptive strategy achieves <10× degradation!")
        else:
            print(f"\n⚠ Gap: Need {adaptive_deg - 10:.1f}× improvement")

    return summary, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    summary, results = run_robustness_experiment(seed=args.seed)
    print(f"\n✓ Experiment 15 completed (seed={args.seed})!")
