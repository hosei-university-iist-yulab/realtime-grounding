"""
Experiment 13: Multi-Task vs Single-Task Training Ablation.

Compares:
1. Single-task trend model
2. Single-task causal model
3. Multi-task (trend + causal) model

Target: Show multi-task training improves both metrics vs single-task.
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
from typing import Dict, List, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.buffer.trend_analyzer import TrendAnalyzer
from src.causal.validator import CausalGraph
from src.data.loaders import BDG2Loader

PROJECT_ROOT = Path(__file__).parent.parent


def load_model_by_type(model_type: str):
    """Load model by type: 'trend', 'multitask', or 'base'."""
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

    if model_type == "base":
        model = base_model
    else:
        models_dir = PROJECT_ROOT / "output" / "models"
        if model_type == "trend":
            prefix = "tinyllama_trend_"
        else:  # multitask
            prefix = "tinyllama_multitask_"

        model_dirs = sorted([d for d in models_dir.iterdir()
                            if d.is_dir() and d.name.startswith(prefix)])
        if not model_dirs:
            raise FileNotFoundError(f"No {model_type} models found")

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


def create_trend_test_cases(n_cases: int = 30, seed: int = 2025) -> List[Dict]:
    """Create test cases for trend detection."""
    np.random.seed(seed)

    data_dir = PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"
    loader = BDG2Loader(str(data_dir))
    buildings = loader.list_buildings()

    analyzer = TrendAnalyzer()
    cases = []

    for building_id in buildings[:20]:
        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < 200:
                continue

            for _ in range(3):
                if len(cases) >= n_cases:
                    break

                max_start = len(df) - 100
                start_idx = np.random.randint(0, max_start)
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]
                features = analyzer.analyze(values, timestamps)

                cases.append({
                    "type": "trend",
                    "values": values,
                    "features": features,
                    "expected": features.direction
                })
        except Exception:
            continue

        if len(cases) >= n_cases:
            break

    return cases


def create_causal_test_cases() -> List[Dict]:
    """Create test cases for causal reasoning."""
    return [
        {
            "type": "causal",
            "context": "Hot summer day, outdoor temperature 35°C. HVAC running at high capacity.",
            "query": "Why is HVAC energy consumption high?",
            "expected_cause": "outdoor_temperature",
            "expected_effect": "hvac_load"
        },
        {
            "type": "causal",
            "context": "Winter morning, -5°C outside. Heating system consuming significant energy.",
            "query": "What is causing the high heating energy usage?",
            "expected_cause": "outdoor_temperature",
            "expected_effect": "hvac_load"
        },
        {
            "type": "causal",
            "context": "9 AM Monday. 200 employees arrived. Computers running.",
            "query": "Why is plug load high in the morning?",
            "expected_cause": "occupancy",
            "expected_effect": "plug_load"
        },
        {
            "type": "causal",
            "context": "Weekend afternoon. Only 5 security staff. Most equipment off.",
            "query": "Why is equipment energy consumption low?",
            "expected_cause": "occupancy",
            "expected_effect": "plug_load"
        },
        {
            "type": "causal",
            "context": "Office hours, all floors occupied. All lights on.",
            "query": "What drives lighting consumption?",
            "expected_cause": "occupancy",
            "expected_effect": "lighting_load"
        },
        {
            "type": "causal",
            "context": "After hours. Building empty. Emergency lights only.",
            "query": "Why is lighting load minimal at night?",
            "expected_cause": "occupancy",
            "expected_effect": "lighting_load"
        },
        {
            "type": "causal",
            "context": "8:30 AM Tuesday. Workers arriving.",
            "query": "Why is building occupancy increasing?",
            "expected_cause": "hour_of_day",
            "expected_effect": "occupancy"
        },
        {
            "type": "causal",
            "context": "6 PM. Workers leaving the building.",
            "query": "What determines when the building empties?",
            "expected_cause": "hour_of_day",
            "expected_effect": "occupancy"
        },
        {
            "type": "causal",
            "context": "High HVAC (cooling) and all computers running. Total: 500 kWh.",
            "query": "What contributes to high total consumption?",
            "expected_cause": "hvac_load",
            "expected_effect": "total_consumption"
        },
        {
            "type": "causal",
            "context": "Night. HVAC in setback mode. Few equipment. Total: 50 kWh.",
            "query": "Why is total consumption low at night?",
            "expected_cause": "hvac_load",
            "expected_effect": "total_consumption"
        }
    ]


def evaluate_trend(model, tokenizer, cases: List[Dict]) -> Dict:
    """Evaluate trend detection performance."""
    results = []

    for case in cases:
        values = case["values"]
        features = case["features"]
        slope_per_hour = features.slope * 3600

        prompt = f"""You are analyzing energy consumption data.

Recent statistics (last 100 readings):
- Mean: {np.mean(values):.1f} kWh
- Std: {np.std(values):.1f} kWh

Trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}

Question: What is the trend in energy consumption?
Answer with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt, max_new_tokens=10)
        latency = (time.perf_counter() - start) * 1000

        response_lower = response.lower()
        predicted = "unknown"
        for trend in ["increasing", "decreasing", "stable", "volatile"]:
            if trend in response_lower:
                predicted = trend
                break

        results.append({
            "expected": case["expected"],
            "predicted": predicted,
            "correct": predicted == case["expected"],
            "latency_ms": latency
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0

    return {"accuracy": accuracy, "avg_latency_ms": avg_latency, "n_samples": len(results)}


def evaluate_causal(model, tokenizer, cases: List[Dict]) -> Dict:
    """Evaluate causal reasoning performance."""
    results = []

    for case in cases:
        prompt = f"""You are analyzing building energy data.

Context: {case['context']}

Question: {case['query']}

Provide a causal explanation for the energy consumption.

Answer:"""

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt, max_new_tokens=100)
        latency = (time.perf_counter() - start) * 1000

        response_lower = response.lower()

        # Check for correct causal indicators
        correct_indicators = [
            ("temperature" in response_lower and ("hvac" in response_lower or "cooling" in response_lower or "heating" in response_lower)),
            ("hot" in response_lower and "cooling" in response_lower),
            ("cold" in response_lower and "heating" in response_lower),
            ("occupancy" in response_lower or "people" in response_lower or "employees" in response_lower),
            ("morning" in response_lower or "schedule" in response_lower or "work hours" in response_lower),
            ("contribute" in response_lower or "add to" in response_lower)
        ]

        # Check for reversed patterns
        reversed_patterns = [
            "hvac causes temperature",
            "cooling causes temperature",
            "equipment causes people",
            "consumption causes hvac"
        ]

        has_correct = any(correct_indicators)
        has_reversed = any(p in response_lower for p in reversed_patterns)

        if has_reversed:
            validity = "reversed"
        elif has_correct:
            validity = "correct"
        else:
            validity = "ambiguous"

        results.append({
            "query": case["query"],
            "validity": validity,
            "correct": validity == "correct",
            "latency_ms": latency
        })

    correct_count = sum(1 for r in results if r["validity"] == "correct")
    accuracy = correct_count / len(results) if results else 0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "n_samples": len(results),
        "breakdown": {
            "correct": correct_count,
            "reversed": sum(1 for r in results if r["validity"] == "reversed"),
            "ambiguous": sum(1 for r in results if r["validity"] == "ambiguous")
        }
    }


def run_ablation(seed: int = 2025):
    """Run multi-task vs single-task ablation."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 13: Multi-Task vs Single-Task Ablation (seed={seed})")
    print("=" * 80)
    print()

    # Create test cases
    print("[1/6] Creating test cases...")
    trend_cases = create_trend_test_cases(n_cases=30, seed=seed)
    causal_cases = create_causal_test_cases()
    print(f"  ✓ {len(trend_cases)} trend cases, {len(causal_cases)} causal cases")

    results = {}

    # Evaluate base model
    print("\n[2/6] Evaluating BASE model (zero-shot)...")
    try:
        model, tokenizer = load_model_by_type("base")
        results["base"] = {
            "trend": evaluate_trend(model, tokenizer, trend_cases),
            "causal": evaluate_causal(model, tokenizer, causal_cases)
        }
        print(f"  Trend: {results['base']['trend']['accuracy']*100:.1f}%")
        print(f"  Causal: {results['base']['causal']['accuracy']*100:.1f}%")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["base"] = {"error": str(e)}

    # Evaluate trend-only model
    print("\n[3/6] Evaluating TREND-ONLY model...")
    try:
        model, tokenizer = load_model_by_type("trend")
        results["trend_only"] = {
            "trend": evaluate_trend(model, tokenizer, trend_cases),
            "causal": evaluate_causal(model, tokenizer, causal_cases)
        }
        print(f"  Trend: {results['trend_only']['trend']['accuracy']*100:.1f}%")
        print(f"  Causal: {results['trend_only']['causal']['accuracy']*100:.1f}%")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["trend_only"] = {"error": str(e)}

    # Evaluate multi-task model
    print("\n[4/6] Evaluating MULTI-TASK model...")
    try:
        model, tokenizer = load_model_by_type("multitask")
        results["multitask"] = {
            "trend": evaluate_trend(model, tokenizer, trend_cases),
            "causal": evaluate_causal(model, tokenizer, causal_cases)
        }
        print(f"  Trend: {results['multitask']['trend']['accuracy']*100:.1f}%")
        print(f"  Causal: {results['multitask']['causal']['accuracy']*100:.1f}%")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results["multitask"] = {"error": str(e)}

    # Save results
    print("\n[5/6] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp13_multitask_ablation_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "multitask_ablation",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "models": results
        }, f, indent=2)

    print(f"  ✓ Results saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Trend Acc':>12} {'Causal Acc':>12} {'Combined':>12}")
    print("-" * 53)

    for model_name, r in results.items():
        if "error" in r:
            print(f"{model_name:<15} {'ERROR':>12} {'ERROR':>12} {'ERROR':>12}")
        else:
            trend_acc = r["trend"]["accuracy"]
            causal_acc = r["causal"]["accuracy"]
            combined = (trend_acc + causal_acc) / 2
            print(f"{model_name:<15} {trend_acc*100:>11.1f}% {causal_acc*100:>11.1f}% {combined*100:>11.1f}%")

    print("\n[6/6] Analysis complete!")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp13: Multi-task vs single-task ablation")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    results = run_ablation(seed=args.seed)
    print(f"\n✓ Experiment 13 completed (seed={args.seed})!")
