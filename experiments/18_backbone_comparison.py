"""
Experiment 18: LLM Backbone Comparison

Compares different LLM backbones on trend detection:
- TinyLLaMA 1.1B (fine-tuned)
- Phi-2 2.7B (zero-shot)
- Qwen-2.5 3B (zero-shot)

Also tests prompt engineering strategies:
- Zero-shot
- Zero-shot CoT
- Few-shot
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

from src.data.loaders import BDG2Loader
from src.buffer.trend_analyzer import TrendAnalyzer
from src.baselines.prompt_baselines import PromptFormatter, extract_trend_from_response

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class BackboneResult:
    """Result for a single backbone test."""
    backbone: str
    strategy: str
    accuracy: float
    avg_latency_ms: float
    memory_mb: float
    n_samples: int


def find_latest_multitask_model():
    """Find the latest multi-task fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([
        d for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("tinyllama_multitask_")
    ])
    if model_dirs:
        return model_dirs[-1] / "final"
    return None


def load_backbone(backbone_type: str, use_lora: bool = False):
    """Load a specific backbone model."""
    print(f"  Loading {backbone_type}...")

    configs = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "phi2": "microsoft/phi-2",
        "qwen": "Qwen/Qwen2.5-3B-Instruct"
    }

    if backbone_type not in configs:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    model_name = configs[backbone_type]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA if requested (for TinyLLaMA fine-tuned)
    if use_lora and backbone_type == "tinyllama":
        lora_path = find_latest_multitask_model()
        if lora_path and lora_path.exists():
            model = PeftModel.from_pretrained(model, str(lora_path))
            print(f"    ✓ Loaded LoRA from {lora_path.parent.name}")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> Tuple[str, float]:
    """Generate response with latency."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
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


def compute_ground_truth(values: List[float]) -> str:
    """Compute ground truth trend."""
    if len(values) < 10:
        return "unknown"

    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)

    mean_val = np.mean(values)
    slope_norm = slope / mean_val if mean_val > 0 else slope

    std_val = np.std(values)
    cv = std_val / mean_val if mean_val > 0 else 0

    if cv > 0.3:
        return "volatile"
    elif abs(slope_norm) < 0.001:
        return "stable"
    elif slope_norm > 0:
        return "increasing"
    else:
        return "decreasing"


def prepare_test_samples(n_samples: int = 30, seed: int = 2025) -> List[Dict]:
    """Prepare test samples from BDG2."""
    print("  Preparing test samples from BDG2...")

    loader = BDG2Loader(str(PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"))
    buildings = loader.list_buildings()[:5]
    analyzer = TrendAnalyzer()

    samples = []
    np.random.seed(seed)

    for building in buildings:
        try:
            df = loader.get_meter_data(building)
            if df is None or len(df) < 100:
                continue

            for _ in range(n_samples // len(buildings)):
                start_idx = np.random.randint(0, max(1, len(df) - 100))
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                gt = compute_ground_truth(values)
                if gt == "unknown":
                    continue

                features = analyzer.analyze(values, timestamps)
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

                trend_info = {
                    "direction": features.direction,
                    "slope_per_hour": features.slope * 3600,
                    "confidence": features.confidence,
                    "r_squared": features.r_squared,
                    "volatility": features.volatility
                }

                samples.append({
                    "building": building,
                    "stats": stats,
                    "trend_info": trend_info,
                    "gt_trend": gt
                })

        except Exception as e:
            continue

    print(f"    ✓ Prepared {len(samples)} samples")
    return samples


def test_backbone(
    model, tokenizer, samples: List[Dict],
    strategy: str = "zero_shot"
) -> Dict:
    """Test a backbone with specific prompting strategy."""
    formatter = PromptFormatter()

    results = []
    for sample in samples:
        stats = sample["stats"]
        trend_info = sample["trend_info"]
        gt = sample["gt_trend"]

        # Format prompt based on strategy
        if strategy == "zero_shot":
            prompt = formatter.format_zero_shot(stats, trend_info)
        elif strategy == "zero_shot_cot":
            prompt = formatter.format_zero_shot_cot(stats, trend_info)
        elif strategy == "few_shot":
            prompt = formatter.format_few_shot(stats, trend_info)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        response, latency = generate(model, tokenizer, prompt)
        pred = extract_trend_from_response(response)

        results.append({
            "gt": gt,
            "pred": pred,
            "correct": gt == pred,
            "latency_ms": latency
        })

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    avg_latency = np.mean([r["latency_ms"] for r in results]) if results else 0

    # Get memory usage
    memory_mb = 0
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024**2

    return {
        "accuracy": accuracy * 100,
        "avg_latency_ms": avg_latency,
        "memory_mb": memory_mb,
        "n_samples": len(results)
    }


def run_backbone_comparison(n_samples: int = 30, seed: int = 2025):
    """Run backbone comparison experiment."""

    print("=" * 80)
    print(f"EXPERIMENT 18: LLM Backbone Comparison (seed={seed})")
    print("=" * 80)
    print()

    # Prepare test samples
    print("[1/5] Preparing test data...")
    samples = prepare_test_samples(n_samples, seed)

    results = []

    # Test TinyLLaMA (fine-tuned)
    print("\n[2/5] Testing TinyLLaMA (fine-tuned)...")
    model, tokenizer = load_backbone("tinyllama", use_lora=True)

    for strategy in ["zero_shot"]:  # Fine-tuned only needs zero-shot
        print(f"    Strategy: {strategy}")
        res = test_backbone(model, tokenizer, samples, strategy)
        results.append(BackboneResult(
            backbone="TinyLLaMA-FT",
            strategy=strategy,
            accuracy=res["accuracy"],
            avg_latency_ms=res["avg_latency_ms"],
            memory_mb=res["memory_mb"],
            n_samples=res["n_samples"]
        ))

    del model, tokenizer
    torch.cuda.empty_cache()

    # Test Phi-2 (zero-shot)
    print("\n[3/5] Testing Phi-2 (zero-shot)...")
    model, tokenizer = load_backbone("phi2", use_lora=False)

    for strategy in ["zero_shot", "zero_shot_cot", "few_shot"]:
        print(f"    Strategy: {strategy}")
        res = test_backbone(model, tokenizer, samples, strategy)
        results.append(BackboneResult(
            backbone="Phi-2",
            strategy=strategy,
            accuracy=res["accuracy"],
            avg_latency_ms=res["avg_latency_ms"],
            memory_mb=res["memory_mb"],
            n_samples=res["n_samples"]
        ))

    del model, tokenizer
    torch.cuda.empty_cache()

    # Test Qwen-2.5 (zero-shot)
    print("\n[4/5] Testing Qwen-2.5 (zero-shot)...")
    try:
        model, tokenizer = load_backbone("qwen", use_lora=False)

        for strategy in ["zero_shot", "zero_shot_cot", "few_shot"]:
            print(f"    Strategy: {strategy}")
            res = test_backbone(model, tokenizer, samples, strategy)
            results.append(BackboneResult(
                backbone="Qwen-2.5",
                strategy=strategy,
                accuracy=res["accuracy"],
                avg_latency_ms=res["avg_latency_ms"],
                memory_mb=res["memory_mb"],
                n_samples=res["n_samples"]
            ))

        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"    ⚠ Qwen-2.5 failed: {e}")

    # Save results
    print("\n[5/5] Saving results...")
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp18_backbone_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "backbone_comparison",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "results": [
                {
                    "backbone": r.backbone,
                    "strategy": r.strategy,
                    "accuracy": r.accuracy,
                    "avg_latency_ms": r.avg_latency_ms,
                    "memory_mb": r.memory_mb,
                    "n_samples": r.n_samples
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKBONE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Backbone':<15} {'Strategy':<15} {'Accuracy':<12} {'Latency':<12} {'Memory'}")
    print("-" * 70)

    for r in results:
        print(f"{r.backbone:<15} {r.strategy:<15} {r.accuracy:<12.1f}% {r.avg_latency_ms:<12.0f}ms {r.memory_mb:.0f}MB")

    # Find best
    best = max(results, key=lambda x: x.accuracy)
    print("\n" + "-" * 70)
    print(f"Best: {best.backbone} ({best.strategy}) - {best.accuracy:.1f}% accuracy")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    results = run_backbone_comparison(n_samples=30, seed=args.seed)
    print(f"\n✓ Experiment 18 completed (seed={args.seed})!")
