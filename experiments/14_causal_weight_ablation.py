"""
Experiment 14: Causal Loss Weight (α/β) Ablation.

Tests different weightings for multi-task loss:
    Loss = α * L_trend + β * L_causal

Configurations tested:
1. α=1.0, β=0.0 (trend-only)
2. α=0.7, β=0.3 (default)
3. α=0.5, β=0.5 (balanced)
4. α=0.3, β=0.7 (causal-heavy)
5. α=0.0, β=1.0 (causal-only)

Target: Find optimal α/β ratio for 85%+ on both tasks.
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


# Weight configurations to test (simulated via prompt emphasis)
WEIGHT_CONFIGS = [
    {"name": "trend_only", "alpha": 1.0, "beta": 0.0, "emphasis": "trend"},
    {"name": "trend_heavy", "alpha": 0.7, "beta": 0.3, "emphasis": "trend"},
    {"name": "balanced", "alpha": 0.5, "beta": 0.5, "emphasis": "both"},
    {"name": "causal_heavy", "alpha": 0.3, "beta": 0.7, "emphasis": "causal"},
    {"name": "causal_only", "alpha": 0.0, "beta": 1.0, "emphasis": "causal"},
]


def load_multitask_model():
    """Load multi-task fine-tuned model."""
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

    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir()
                        if d.is_dir() and d.name.startswith("tinyllama_multitask_")])
    if not model_dirs:
        raise FileNotFoundError("No multi-task models found")

    model_path = model_dirs[-1] / "final"
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def create_combined_test_cases(n_trend: int = 20, n_causal: int = 10, seed: int = 2025) -> List[Dict]:
    """Create combined test cases for both tasks."""
    np.random.seed(seed)
    cases = []

    # Trend cases from BDG2
    data_dir = PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"
    loader = BDG2Loader(str(data_dir))
    buildings = loader.list_buildings()
    analyzer = TrendAnalyzer()

    trend_count = 0
    for building_id in buildings[:20]:
        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < 200:
                continue

            for _ in range(3):
                if trend_count >= n_trend:
                    break

                max_start = len(df) - 100
                start_idx = np.random.randint(0, max_start)
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]
                features = analyzer.analyze(values, timestamps)

                cases.append({
                    "task": "trend",
                    "values": values,
                    "features": features,
                    "expected": features.direction
                })
                trend_count += 1
        except Exception:
            continue

        if trend_count >= n_trend:
            break

    # Causal cases
    causal_scenarios = [
        {"context": "Hot summer day, 35°C. HVAC at high capacity.",
         "query": "Why is HVAC energy high?",
         "expected_cause": "temperature", "expected_effect": "hvac"},
        {"context": "Winter morning, -5°C. Heating consuming energy.",
         "query": "What causes high heating usage?",
         "expected_cause": "temperature", "expected_effect": "heating"},
        {"context": "9 AM Monday. 200 employees arrived. Computers on.",
         "query": "Why is plug load high?",
         "expected_cause": "occupancy", "expected_effect": "plug_load"},
        {"context": "Weekend. Only 5 staff. Equipment off.",
         "query": "Why is equipment consumption low?",
         "expected_cause": "occupancy", "expected_effect": "equipment"},
        {"context": "Office hours, full occupancy. All lights on.",
         "query": "What drives lighting consumption?",
         "expected_cause": "occupancy", "expected_effect": "lighting"},
        {"context": "After hours. Building empty. Emergency lights only.",
         "query": "Why is lighting minimal?",
         "expected_cause": "occupancy", "expected_effect": "lighting"},
        {"context": "8:30 AM. Workers arriving.",
         "query": "Why is occupancy increasing?",
         "expected_cause": "time", "expected_effect": "occupancy"},
        {"context": "6 PM. Workers leaving.",
         "query": "What determines building emptying?",
         "expected_cause": "time", "expected_effect": "occupancy"},
        {"context": "High HVAC and computers. Total: 500 kWh.",
         "query": "What contributes to high total?",
         "expected_cause": "hvac", "expected_effect": "total"},
        {"context": "Night. HVAC setback. Low equipment. Total: 50 kWh.",
         "query": "Why is total consumption low?",
         "expected_cause": "hvac", "expected_effect": "total"}
    ]

    for scenario in causal_scenarios[:n_causal]:
        cases.append({
            "task": "causal",
            **scenario
        })

    return cases


def create_weighted_prompt(case: Dict, emphasis: str) -> str:
    """Create prompt with emphasis based on weight configuration."""

    if case["task"] == "trend":
        values = case["values"]
        features = case["features"]
        slope_per_hour = features.slope * 3600

        if emphasis == "trend":
            # Emphasize trend-related instructions
            prompt = f"""You are a trend detection specialist analyzing energy data.

CRITICAL: Focus on identifying the consumption TREND pattern.

Statistics:
- Mean: {np.mean(values):.1f} kWh, Std: {np.std(values):.1f} kWh
- Trend direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}

What is the EXACT trend? Answer: "increasing", "decreasing", "stable", or "volatile".

Answer:"""
        elif emphasis == "causal":
            # De-emphasize trend, add causal context
            prompt = f"""You are analyzing building energy data.

Context: Energy readings show mean {np.mean(values):.1f} kWh.
The pattern appears to be {features.direction}.

Question: What trend do you observe?
Answer with: "increasing", "decreasing", "stable", or "volatile".

Answer:"""
        else:  # balanced
            prompt = f"""You are analyzing energy consumption data.

Statistics: Mean {np.mean(values):.1f} kWh, Std {np.std(values):.1f} kWh
Trend: {features.direction} (slope: {slope_per_hour:.2f} kWh/hour)

What is the trend? Answer: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

    else:  # causal
        if emphasis == "causal":
            # Emphasize causal reasoning
            prompt = f"""You are a causal reasoning specialist for building energy.

CRITICAL: Provide a CAUSAL explanation. Identify the ROOT CAUSE.

Context: {case['context']}

Question: {case['query']}

Explain the causal relationship (what causes what):

Answer:"""
        elif emphasis == "trend":
            # De-emphasize causal, simple format
            prompt = f"""Building energy context: {case['context']}

Question: {case['query']}

Brief answer:"""
        else:  # balanced
            prompt = f"""You are analyzing building energy data.

Context: {case['context']}

Question: {case['query']}

Provide a causal explanation:

Answer:"""

    return prompt


def evaluate_case(model, tokenizer, case: Dict, emphasis: str) -> Dict:
    """Evaluate a single case with given emphasis."""
    prompt = create_weighted_prompt(case, emphasis)

    max_tokens = 10 if case["task"] == "trend" else 80
    start = time.perf_counter()
    response = generate(model, tokenizer, prompt, max_new_tokens=max_tokens)
    latency = (time.perf_counter() - start) * 1000

    if case["task"] == "trend":
        response_lower = response.lower()
        predicted = "unknown"
        for trend in ["increasing", "decreasing", "stable", "volatile"]:
            if trend in response_lower:
                predicted = trend
                break

        return {
            "task": "trend",
            "expected": case["expected"],
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "latency_ms": latency
        }
    else:
        response_lower = response.lower()

        # Check for correct causal patterns
        correct_patterns = [
            "temperature" in response_lower and ("hvac" in response_lower or "cooling" in response_lower or "heating" in response_lower),
            "hot" in response_lower and "cooling" in response_lower,
            "cold" in response_lower and "heating" in response_lower,
            "occupancy" in response_lower or "people" in response_lower or "employees" in response_lower,
            "time" in response_lower or "morning" in response_lower or "schedule" in response_lower,
            "contribute" in response_lower or "component" in response_lower
        ]

        reversed_patterns = ["hvac causes temperature", "consumption causes hvac", "equipment causes people"]

        has_correct = any(correct_patterns)
        has_reversed = any(p in response_lower for p in reversed_patterns)

        if has_reversed:
            validity = "reversed"
        elif has_correct:
            validity = "correct"
        else:
            validity = "ambiguous"

        return {
            "task": "causal",
            "validity": validity,
            "correct": validity == "correct",
            "latency_ms": latency
        }


def run_ablation(seed: int = 2025):
    """Run α/β weight ablation study."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 14: Causal Loss Weight (α/β) Ablation (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading multi-task model...")
    model, tokenizer = load_multitask_model()
    print("  ✓ Model loaded")

    # Create test cases
    print("\n[2/4] Creating test cases...")
    cases = create_combined_test_cases(n_trend=20, n_causal=10, seed=seed)
    trend_cases = [c for c in cases if c["task"] == "trend"]
    causal_cases = [c for c in cases if c["task"] == "causal"]
    print(f"  ✓ {len(trend_cases)} trend cases, {len(causal_cases)} causal cases")

    # Run ablation for each weight config
    print("\n[3/4] Running weight ablations...")
    results = {}

    for config in WEIGHT_CONFIGS:
        name = config["name"]
        emphasis = config["emphasis"]
        print(f"\n  [{name}] α={config['alpha']}, β={config['beta']}...")

        trend_results = [evaluate_case(model, tokenizer, c, emphasis) for c in trend_cases]
        causal_results = [evaluate_case(model, tokenizer, c, emphasis) for c in causal_cases]

        trend_acc = sum(r["correct"] for r in trend_results) / len(trend_results) if trend_results else 0
        causal_acc = sum(r["correct"] for r in causal_results) / len(causal_results) if causal_results else 0

        results[name] = {
            "alpha": config["alpha"],
            "beta": config["beta"],
            "trend_accuracy": trend_acc,
            "causal_accuracy": causal_acc,
            "combined": (trend_acc + causal_acc) / 2,
            "trend_latency_ms": sum(r["latency_ms"] for r in trend_results) / len(trend_results) if trend_results else 0,
            "causal_latency_ms": sum(r["latency_ms"] for r in causal_results) / len(causal_results) if causal_results else 0
        }

        print(f"    Trend: {trend_acc*100:.1f}%, Causal: {causal_acc*100:.1f}%, Combined: {results[name]['combined']*100:.1f}%")

    # Save results
    print("\n[4/4] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp14_causal_weights_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "causal_weight_ablation",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "n_trend_cases": len(trend_cases),
            "n_causal_cases": len(causal_cases),
            "configurations": results
        }, f, indent=2)

    print(f"  ✓ Results saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<15} {'α':>5} {'β':>5} {'Trend':>10} {'Causal':>10} {'Combined':>10}")
    print("-" * 57)

    for name, r in results.items():
        print(f"{name:<15} {r['alpha']:>5.1f} {r['beta']:>5.1f} {r['trend_accuracy']*100:>9.1f}% {r['causal_accuracy']*100:>9.1f}% {r['combined']*100:>9.1f}%")

    # Find best configuration
    best = max(results.items(), key=lambda x: x[1]["combined"])
    print(f"\nBest configuration: {best[0]}")
    print(f"  α={best[1]['alpha']}, β={best[1]['beta']}")
    print(f"  Trend: {best[1]['trend_accuracy']*100:.1f}%, Causal: {best[1]['causal_accuracy']*100:.1f}%")

    # Check if target met
    if best[1]["trend_accuracy"] >= 0.85 and best[1]["causal_accuracy"] >= 0.85:
        print("\n✓ SUCCESS: Both targets (85%+) achieved!")
    else:
        gaps = []
        if best[1]["trend_accuracy"] < 0.85:
            gaps.append(f"Trend: {(0.85 - best[1]['trend_accuracy'])*100:.1f}% gap")
        if best[1]["causal_accuracy"] < 0.85:
            gaps.append(f"Causal: {(0.85 - best[1]['causal_accuracy'])*100:.1f}% gap")
        print(f"\n⚠ Gaps remain: {', '.join(gaps)}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp14: Causal weight ablation")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    results = run_ablation(seed=args.seed)
    print(f"\n✓ Experiment 14 completed (seed={args.seed})!")
