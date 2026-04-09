"""
Experiment 19: Edge vs Cloud API Comparison

Compares TinyLLaMA (edge) vs Claude API (cloud) on:
- Latency
- Cost per query
- Offline capability
- Privacy (data stays local vs sent to cloud)

This is the KEY experiment justifying edge deployment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load env before other imports
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.baselines.api_baselines import ClaudeBaseline
from src.data.loaders import BDG2Loader
from src.buffer.trend_analyzer import TrendAnalyzer

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_model():
    """Find latest fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_multitask_")])
    if model_dirs:
        return model_dirs[-1] / "final"
    return None


def load_edge_model():
    """Load fine-tuned TinyLLaMA for edge inference."""
    model_path = find_latest_model()
    print(f"Loading edge model from: {model_path}")

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


def generate_edge(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> Tuple[str, float]:
    """Generate with edge model, return response and latency."""
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


def prepare_test_queries(n_queries: int = 10, seed: int = 2025) -> List[Dict]:
    """Prepare test queries from BDG2."""
    print("  Preparing test queries...")

    loader = BDG2Loader(str(PROJECT_ROOT / "data" / "raw" / "bdg2" / "data" / "meters" / "cleaned"))
    buildings = loader.list_buildings()[:3]
    analyzer = TrendAnalyzer()

    queries = []
    np.random.seed(seed)

    for building in buildings:
        try:
            df = loader.get_meter_data(building)
            if df is None or len(df) < 100:
                continue

            for _ in range(n_queries // len(buildings)):
                start_idx = np.random.randint(0, max(1, len(df) - 100))
                window = df.iloc[start_idx:start_idx + 100]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                features = analyzer.analyze(values, timestamps)
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }

                # Create edge prompt
                slope_per_hour = features.slope * 3600
                edge_prompt = f"""You are analyzing energy consumption data for a building.

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

                # Create API context
                api_context = {
                    "building_id": building,
                    "meter_type": "electricity",
                    "statistics": stats,
                    "readings": [
                        {"timestamp": str(window.iloc[-i]["timestamp"]), "value": float(window.iloc[-i]["value"])}
                        for i in range(1, min(6, len(window)))
                    ]
                }

                queries.append({
                    "building": building,
                    "edge_prompt": edge_prompt,
                    "api_context": api_context,
                    "api_query": "What is the trend in energy consumption? Answer with one word: increasing, decreasing, stable, or volatile."
                })

        except Exception as e:
            continue

    print(f"    ✓ Prepared {len(queries)} queries")
    return queries


def run_api_comparison(n_queries: int = 10, seed: int = 2025):
    """Run edge vs cloud API comparison."""

    print("=" * 80)
    print(f"EXPERIMENT 19: Edge vs Cloud API Comparison (seed={seed})")
    print("=" * 80)
    print()

    # Prepare queries
    print("[1/4] Preparing test queries...")
    queries = prepare_test_queries(n_queries, seed)

    # Load edge model
    print("\n[2/4] Loading edge model (TinyLLaMA fine-tuned)...")
    edge_model, edge_tokenizer = load_edge_model()
    print("  ✓ Edge model loaded")

    # Memory usage
    edge_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    # Initialize Claude baseline
    print("\n[3/4] Initializing Claude API baseline...")
    try:
        claude = ClaudeBaseline()
        print("  ✓ Claude API ready")
    except Exception as e:
        print(f"  ✗ Claude API failed: {e}")
        return

    # Run comparison
    print(f"\n[4/4] Running {len(queries)} queries on both methods...")

    edge_results = []
    api_results = []

    for i, query in enumerate(queries):
        print(f"  Query {i+1}/{len(queries)}...", end=" ")

        # Edge inference
        edge_response, edge_latency = generate_edge(edge_model, edge_tokenizer, query["edge_prompt"])
        edge_results.append({
            "latency_ms": edge_latency,
            "response": edge_response[:100]
        })

        # API inference
        api_result = claude.generate(query["api_query"], query["api_context"], max_tokens=50)
        api_results.append({
            "latency_ms": api_result.latency_ms,
            "response": api_result.response[:100],
            "tokens": api_result.tokens_used,
            "cost_usd": api_result.cost_usd,
            "error": api_result.error
        })

        print(f"Edge: {edge_latency:.0f}ms, API: {api_result.latency_ms:.0f}ms")

    # Compute metrics
    edge_latencies = [r["latency_ms"] for r in edge_results]
    api_latencies = [r["latency_ms"] for r in api_results if r["error"] is None]
    api_costs = [r["cost_usd"] for r in api_results if r["error"] is None]
    api_tokens = [r["tokens"] for r in api_results if r["error"] is None]

    edge_avg = np.mean(edge_latencies)
    edge_p95 = np.percentile(edge_latencies, 95)
    api_avg = np.mean(api_latencies) if api_latencies else 0
    api_p95 = np.percentile(api_latencies, 95) if api_latencies else 0

    total_api_cost = sum(api_costs)
    avg_api_cost = np.mean(api_costs) if api_costs else 0

    # Save results
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp19_api_seed{seed}_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "edge_vs_cloud_api",
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "n_queries": len(queries),
            "edge": {
                "model": "TinyLLaMA-1.1B (fine-tuned)",
                "avg_latency_ms": edge_avg,
                "p95_latency_ms": edge_p95,
                "memory_mb": edge_memory,
                "cost_per_query": 0,
                "offline_capable": True,
                "data_privacy": "Local (no data sent)"
            },
            "cloud_api": {
                "model": "claude-sonnet-4-20250514",
                "avg_latency_ms": api_avg,
                "p95_latency_ms": api_p95,
                "avg_tokens": np.mean(api_tokens) if api_tokens else 0,
                "avg_cost_per_query": avg_api_cost,
                "total_cost": total_api_cost,
                "offline_capable": False,
                "data_privacy": "Sent to cloud"
            },
            "speedup": api_avg / edge_avg if edge_avg > 0 else 0,
            "edge_results": edge_results,
            "api_results": api_results
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")

    # Summary
    print("\n" + "=" * 80)
    print("EDGE vs CLOUD API COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Edge (TinyLLaMA)':<20} {'Cloud (Claude API)'}")
    print("-" * 70)
    print(f"{'Avg Latency':<25} {edge_avg:<20.0f}ms {api_avg:.0f}ms")
    print(f"{'P95 Latency':<25} {edge_p95:<20.0f}ms {api_p95:.0f}ms")
    print(f"{'Memory':<25} {edge_memory:<20.0f}MB N/A (cloud)")
    print(f"{'Cost per query':<25} {'$0.00':<20} ${avg_api_cost:.6f}")
    print(f"{'Offline capable':<25} {'Yes':<20} No")
    print(f"{'Data privacy':<25} {'Local':<20} Sent to cloud")
    print("-" * 70)
    print(f"\nSpeedup: {api_avg / edge_avg:.1f}x faster with edge deployment")
    print(f"Cost savings: 100% (edge is free after deployment)")

    return {
        "edge_avg": edge_avg,
        "api_avg": api_avg,
        "speedup": api_avg / edge_avg if edge_avg > 0 else 0
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    args = parser.parse_args()

    results = run_api_comparison(n_queries=10, seed=args.seed)
    print(f"\n✓ Experiment 19 completed (seed={args.seed})!")
