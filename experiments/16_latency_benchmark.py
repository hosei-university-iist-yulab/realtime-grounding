"""
Experiment 16: Latency Benchmark

Comprehensive latency testing for V2 with optimized settings.
Target: <2000ms total latency.
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
from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.buffer.trend_analyzer import TrendAnalyzer

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_model():
    """Find latest fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_multitask_")])
    if not model_dirs:
        model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_trend_")])
    return model_dirs[-1] / "final" if model_dirs else None


def load_model():
    """Load model with optimized settings."""
    model_path = find_latest_model()
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> tuple:
    """Generate and return response with latency."""
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


def run_latency_benchmark(n_trials: int = 20, seed: int = 2025):
    """Run latency benchmark."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 16: Latency Benchmark (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/3] Loading model...")
    model, tokenizer = load_model()
    print("  ✓ Model loaded")

    # Warmup
    print("\n[2/3] Warmup runs...")
    for _ in range(3):
        _, _ = generate(model, tokenizer, "Test warmup", max_new_tokens=5)
    print("  ✓ Warmup complete")

    # Benchmark different scenarios
    print(f"\n[3/3] Running {n_trials} trials per scenario...")

    analyzer = TrendAnalyzer()

    scenarios = {
        "trend_short": {
            "max_tokens": 10,
            "prompt": """You are analyzing energy consumption data.
Mean: 120.5 kWh, Std: 15.2 kWh
Direction: stable, Confidence: 0.85
Question: What is the trend?
Answer with one word: increasing, decreasing, stable, or volatile.
Answer:"""
        },
        "trend_standard": {
            "max_tokens": 20,
            "prompt": """You are analyzing energy consumption data for a building.

Recent statistics (last 100 readings):
- Mean: 145.2 kWh
- Std: 18.3 kWh
- Min: 98.5 kWh
- Max: 201.3 kWh

Enhanced trend analysis:
- Direction: stable
- Slope: 0.02 kWh/hour
- Confidence: 0.92
- R²: 0.012
- Volatility: 0.15
- Has change point: False

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile".

Answer:"""
        },
        "causal_short": {
            "max_tokens": 30,
            "prompt": """You are analyzing building energy data.

Context: Hot summer day (35°C). HVAC running high.

Question: Why is HVAC consumption high?
Answer briefly:"""
        },
        "causal_standard": {
            "max_tokens": 50,
            "prompt": """You are analyzing building energy data.

Context: It's a hot summer day with outdoor temperature at 35°C. HVAC is running at high capacity.

Question: Why is HVAC energy consumption high?

Provide a causal explanation for the energy consumption.

Answer:"""
        },
    }

    results = {}

    for name, config in scenarios.items():
        print(f"\n  {name}...")
        latencies = []

        for i in range(n_trials):
            _, latency = generate(model, tokenizer, config["prompt"], config["max_tokens"])
            latencies.append(latency)

        results[name] = {
            "max_tokens": config["max_tokens"],
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        print(f"    Mean: {results[name]['mean_ms']:.0f}ms, P95: {results[name]['p95_ms']:.0f}ms")

    # Save results
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp16_latency_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "latency_benchmark",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "n_trials": n_trials,
            "results": results
        }, f, indent=2)

    print(f"\n\nResults saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("LATENCY SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<20} {'Tokens':<8} {'Mean':<10} {'P95':<10} {'Status'}")
    print("-" * 60)

    for name, data in results.items():
        mean = data["mean_ms"]
        p95 = data["p95_ms"]
        status = "✓" if mean < 2000 else "⚠"
        print(f"{name:<20} {data['max_tokens']:<8} {mean:<10.0f} {p95:<10.0f} {status}")

    # Overall assessment
    print("\n" + "-" * 60)
    trend_latency = results["trend_standard"]["mean_ms"]
    causal_latency = results["causal_standard"]["mean_ms"]

    print(f"\nTarget: <2000ms")
    print(f"Trend detection: {trend_latency:.0f}ms {'✓' if trend_latency < 2000 else '⚠'}")
    print(f"Causal reasoning: {causal_latency:.0f}ms {'✓' if causal_latency < 2000 else '⚠'}")

    if trend_latency < 2000:
        print("\n✓ SUCCESS: Trend detection meets <2000ms target!")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    results = run_latency_benchmark(n_trials=20, seed=args.seed)
    print(f"\n✓ Experiment 16 completed (seed={args.seed})!")
