"""
Experiment 11c: Test Fine-Tuned TinyLLaMA on Trend Detection.

Uses the LoRA fine-tuned model from the fine-tuning pipeline.
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

from src.buffer.temporal_buffer import TemporalGroundingBuffer, SensorReading
from src.buffer.trend_analyzer import TrendAnalyzer
from src.data.loaders import BDG2Loader

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_model():
    """Find the latest fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_trend_")])
    if not model_dirs:
        raise FileNotFoundError("No fine-tuned models found in output/models/")
    return model_dirs[-1] / "final"


def load_finetuned_model(model_path: Path = None):
    """Load the fine-tuned TinyLLaMA model."""
    if model_path is None:
        model_path = find_latest_model()

    print(f"Loading fine-tuned model from: {model_path}")

    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 10) -> str:
    """Generate response from fine-tuned model."""
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


def extract_trend_from_response(response: str) -> str:
    """Extract trend label from model response."""
    response_lower = response.lower()
    for trend in ["increasing", "decreasing", "stable", "volatile"]:
        if trend in response_lower:
            return trend
    return "unknown"


def run_finetuned_evaluation(n_windows: int = 50, seed: int = 2025):
    """Run evaluation with fine-tuned model."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 11c: Fine-Tuned TinyLLaMA Evaluation (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    print("  ✓ Model loaded")

    # Load test data (use new windows, not training data)
    print("\n[2/4] Extracting fresh test windows from BDG2...")
    data_dir = PROJECT_ROOT / "data" / "raw"
    bdg2_path = data_dir / "bdg2" / "data" / "meters" / "cleaned"
    loader = BDG2Loader(str(bdg2_path))
    buildings = loader.list_buildings()

    analyzer = TrendAnalyzer()
    test_windows = []

    # Seed already set at start of function

    for building_id in buildings[:20]:  # Use different buildings
        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < 200:
                continue

            # Extract 3 windows per building
            for _ in range(3):
                if len(test_windows) >= n_windows:
                    break

                max_start = len(df) - 100
                start_idx = np.random.randint(0, max_start)
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                features = analyzer.analyze(values, timestamps)

                test_windows.append({
                    "building_id": building_id,
                    "values": values,
                    "timestamps": timestamps,
                    "features": features,
                    "expected_trend": features.direction
                })

        except Exception as e:
            continue

        if len(test_windows) >= n_windows:
            break

    print(f"  ✓ Extracted {len(test_windows)} test windows")

    # Run evaluation
    print("\n[3/4] Evaluating fine-tuned model...")
    results = []

    for i, window in enumerate(test_windows):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_windows)}...")

        features = window["features"]
        slope_per_hour = features.slope * 3600

        # Create prompt (same format as training)
        prompt = f"""You are analyzing energy consumption data for a building.

Recent statistics (last 100 readings):
- Mean: {np.mean(window['values']):.1f} kWh
- Std: {np.std(window['values']):.1f} kWh
- Min: {np.min(window['values']):.1f} kWh
- Max: {np.max(window['values']):.1f} kWh

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

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt)
        latency = (time.perf_counter() - start) * 1000

        predicted = extract_trend_from_response(response)
        correct = predicted == window["expected_trend"]

        results.append({
            "building_id": window["building_id"],
            "expected": window["expected_trend"],
            "predicted": predicted,
            "correct": correct,
            "response": response[:100],
            "latency_ms": latency
        })

    # Compute accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    # Per-trend breakdown
    print("\n[4/4] Computing results...")
    trend_types = set(r["expected"] for r in results)
    per_type_acc = {}

    for ttype in sorted(trend_types):
        type_results = [r for r in results if r["expected"] == ttype]
        type_acc = sum(r["correct"] for r in type_results) / len(type_results) if type_results else 0
        per_type_acc[ttype] = {
            "accuracy": type_acc,
            "n_samples": len(type_results)
        }
        print(f"  {ttype:15s}: {type_acc * 100:5.1f}% [{len(type_results)} samples]")

    # Save results
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp11c_finetuned_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "trend_features_finetuned",
            "timestamp": datetime.now().isoformat(),
            "n_windows": len(test_windows),
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "per_type": per_type_acc,
            "results": results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {result_file}")
    print(f"{'='*80}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Accuracy:        {accuracy * 100:.1f}%")
    print(f"Avg latency:     {avg_latency:.1f} ms")
    print(f"Target:          85%+")

    if accuracy >= 0.85:
        print("\n✓ SUCCESS: Achieved 85%+ target!")
    else:
        print(f"\n⚠ Gap to target: {(0.85 - accuracy) * 100:.1f} percentage points")

    return accuracy, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test fine-tuned TinyLLaMA on trend detection")
    parser.add_argument("--n-windows", type=int, default=50, help="Number of test windows")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    accuracy, results = run_finetuned_evaluation(n_windows=args.n_windows, seed=args.seed)
    print(f"\n✓ Experiment 11c completed (seed={args.seed})!")
