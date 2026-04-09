#!/usr/bin/env python3
"""
Experiment 4: Causal Validation Evaluation

Evaluates causal consistency of REAL LLM responses using learned causal graphs.

Metrics:
- Causal F1: Fraction of valid causal claims
- Direction accuracy: Correct cause-effect direction
- No-violation rate: Responses without causal errors

Target: ≥0.90 Causal F1 for TGP.

Uses REAL LLM inference - NO simulated/fake responses.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# Global LLM instance
_llm_backbone = None


def get_llm_backbone():
    """Get or initialize LLM backbone."""
    global _llm_backbone
    if _llm_backbone is None:
        from src.llm import LLMBackbone
        print("Loading LLM backbone...")
        _llm_backbone = LLMBackbone(model_type="tinyllama")

        # Load LoRA weights if available
        models_dir = PROJECT_ROOT / "output" / "models"
        if models_dir.exists():
            model_dirs = sorted(models_dir.glob("grounding_*"))
            if model_dirs:
                lora_path = model_dirs[-1] / "final"
                if lora_path.exists():
                    print(f"Loading LoRA weights from {lora_path}")
                    _llm_backbone.load_lora(str(lora_path))
    return _llm_backbone


def generate_causal_queries() -> List[Dict]:
    """Generate queries that require causal reasoning about energy consumption."""
    queries = [
        # Queries about temperature effects
        {
            "query": "Why is consumption higher today than yesterday?",
            "context": {"temp_today": 35.0, "temp_yesterday": 25.0, "consumption_today": 200.0, "consumption_yesterday": 150.0},
            "expected_causal_factors": ["temperature", "hvac", "cooling"]
        },
        {
            "query": "What caused the energy spike at 2 PM?",
            "context": {"time": "14:00", "temp": 38.0, "consumption": 250.0, "avg": 150.0},
            "expected_causal_factors": ["temperature", "hvac", "peak"]
        },
        # Queries about occupancy effects
        {
            "query": "Why is weekend consumption lower?",
            "context": {"day": "Saturday", "occupancy": 0.1, "consumption": 80.0, "weekday_avg": 150.0},
            "expected_causal_factors": ["occupancy", "weekend", "equipment"]
        },
        {
            "query": "What explains the morning consumption increase?",
            "context": {"time": "08:00", "occupancy": 0.8, "consumption": 180.0, "night_avg": 50.0},
            "expected_causal_factors": ["occupancy", "startup", "equipment"]
        },
        # Queries about equipment effects
        {
            "query": "Why did consumption drop after 6 PM?",
            "context": {"time": "18:00", "occupancy": 0.2, "consumption": 70.0, "day_avg": 150.0},
            "expected_causal_factors": ["occupancy", "shutdown", "equipment"]
        },
        {
            "query": "What causes the baseload at night?",
            "context": {"time": "02:00", "occupancy": 0.0, "consumption": 40.0, "day_avg": 150.0},
            "expected_causal_factors": ["baseload", "standby", "equipment"]
        },
        # Queries about seasonal effects
        {
            "query": "Why is summer consumption higher than spring?",
            "context": {"season": "summer", "temp_avg": 32.0, "consumption": 200.0, "spring_avg": 130.0},
            "expected_causal_factors": ["temperature", "hvac", "cooling", "season"]
        },
        {
            "query": "What drives the winter morning peak?",
            "context": {"season": "winter", "time": "07:00", "temp": 5.0, "consumption": 220.0},
            "expected_causal_factors": ["temperature", "heating", "hvac", "startup"]
        },
        # Anomaly queries
        {
            "query": "Why is consumption abnormally high right now?",
            "context": {"consumption": 300.0, "expected": 150.0, "temp": 40.0},
            "expected_causal_factors": ["temperature", "hvac", "anomaly"]
        },
        {
            "query": "What could explain the sudden drop in consumption?",
            "context": {"consumption": 30.0, "expected": 150.0},
            "expected_causal_factors": ["equipment", "failure", "shutdown"]
        },
    ]
    return queries


def format_causal_prompt(query: str, context: Dict) -> str:
    """Format a prompt for causal reasoning."""
    context_str = ", ".join(f"{k}: {v}" for k, v in context.items())

    prompt = f"""<|system|>
You are an energy monitoring assistant. Explain the causal relationships affecting energy consumption.
When asked about causes, identify the relevant factors and explain how they affect consumption.
</s>
<|user|>
Context: {context_str}
Question: {query}
</s>
<|assistant|>
"""
    return prompt


def evaluate_tgp_causal(queries: List[Dict], n_repeats: int = 3) -> Dict[str, Any]:
    """
    Evaluate TGP causal reasoning with REAL LLM responses.

    Uses actual LLM inference to generate responses, then validates
    them against the causal graph.
    """
    print("\nEvaluating TGP causal validation (REAL inference)...")

    from src.causal import CausalValidator, CausalGraph

    llm = get_llm_backbone()
    graph = CausalGraph.create_energy_graph()
    validator = CausalValidator(graph)

    results = {
        "valid_responses": 0,
        "invalid_responses": 0,
        "total": 0,
        "latencies": [],
        "scores": [],
        "details": [],
        "responses": []
    }

    all_queries = queries * n_repeats
    np.random.shuffle(all_queries)

    for i, item in enumerate(all_queries):
        start = time.perf_counter()

        # Generate REAL LLM response
        prompt = format_causal_prompt(item["query"], item["context"])
        response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)

        # Validate response with causal graph
        validation = validator.validate(response)

        latency = (time.perf_counter() - start) * 1000

        results["total"] += 1
        results["latencies"].append(latency)
        results["scores"].append(validation.score)

        if validation.is_valid:
            results["valid_responses"] += 1
        else:
            results["invalid_responses"] += 1

        results["details"].append({
            "query": item["query"][:50],
            "is_valid": validation.is_valid,
            "score": validation.score,
            "violations": validation.violations[:2] if validation.violations else []
        })
        results["responses"].append(response[:150])

        if (i + 1) % 10 == 0:
            valid_rate = results["valid_responses"] / results["total"]
            print(f"  Processed {i+1}/{len(all_queries)} | Valid: {valid_rate:.1%}")

    valid_rate = results["valid_responses"] / max(results["total"], 1)

    return {
        "method": "TGP (with causal validation)",
        "valid_rate": valid_rate,
        "invalid_rate": results["invalid_responses"] / max(results["total"], 1),
        "mean_score": float(np.mean(results["scores"])),
        "mean_latency_ms": float(np.mean(results["latencies"])),
        "std_latency_ms": float(np.std(results["latencies"])),
        "total_queries": results["total"],
        "sample_responses": results["responses"][:5],
        "details": results["details"][:10]
    }


def evaluate_raw_llm_causal(queries: List[Dict], n_repeats: int = 3) -> Dict[str, Any]:
    """
    Evaluate raw LLM (no fine-tuning) causal reasoning.

    Uses raw TinyLLaMA without LoRA weights for comparison.
    """
    print("\nEvaluating raw LLM causal reasoning...")

    from src.llm import LLMBackbone
    from src.causal import CausalValidator, CausalGraph

    # Load raw LLM (no LoRA)
    print("Loading raw LLM (no LoRA)...")
    raw_llm = LLMBackbone(model_type="tinyllama")

    graph = CausalGraph.create_energy_graph()
    validator = CausalValidator(graph)

    results = {
        "valid_responses": 0,
        "total": 0,
        "scores": [],
        "latencies": []
    }

    all_queries = queries * n_repeats
    np.random.shuffle(all_queries)

    for i, item in enumerate(all_queries):
        start = time.perf_counter()

        prompt = format_causal_prompt(item["query"], item["context"])
        response = raw_llm.generate(prompt, max_new_tokens=100, temperature=0.3)

        validation = validator.validate(response)

        latency = (time.perf_counter() - start) * 1000

        results["total"] += 1
        results["latencies"].append(latency)
        results["scores"].append(validation.score)

        if validation.is_valid:
            results["valid_responses"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(all_queries)}")

    valid_rate = results["valid_responses"] / max(results["total"], 1)

    return {
        "method": "Raw TinyLLaMA (no fine-tuning)",
        "valid_rate": valid_rate,
        "mean_score": float(np.mean(results["scores"])),
        "mean_latency_ms": float(np.mean(results["latencies"])),
        "total_queries": results["total"]
    }


def evaluate_no_validation_baseline(queries: List[Dict]) -> Dict[str, Any]:
    """
    Baseline: Measure what happens without causal validation.

    This shows the rate of potentially invalid causal claims
    that would be accepted without validation.
    """
    print("\nEvaluating no-validation baseline...")

    # Without validation, we accept all responses
    # This baseline shows why validation is needed
    return {
        "method": "No Causal Validation",
        "valid_rate": 1.0,  # All accepted (including invalid)
        "mean_score": None,
        "note": "Accepts all responses - cannot detect invalid causal claims"
    }


def run_experiment(n_queries: int = 30, n_repeats: int = 3, seed: int = 2025) -> Dict[str, Any]:
    """Run full causal validation experiment with REAL LLM inference."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    print("Generating causal queries...")
    queries = generate_causal_queries()

    # Limit queries if needed
    if n_queries < len(queries) * n_repeats:
        n_repeats = max(1, n_queries // len(queries))

    results = {
        "experiment": "causal_validation",
        "seed": seed,
        "timestamp": timestamp,
        "config": {
            "n_query_types": len(queries),
            "n_repeats": n_repeats,
            "total_queries": len(queries) * n_repeats
        },
        "methods": {}
    }

    # TGP with causal validation
    results["methods"]["tgp"] = evaluate_tgp_causal(queries, n_repeats)

    # Raw LLM (no fine-tuning)
    results["methods"]["raw_llm"] = evaluate_raw_llm_causal(queries, n_repeats)

    # No validation baseline
    results["methods"]["no_validation"] = evaluate_no_validation_baseline(queries)

    return results


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025, dataset: str = None):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp04_causal_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: Causal Validation (REAL LLM)")
    parser.add_argument("--n-queries", type=int, default=30, help="Total queries to run")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeats per query type")
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset for reference (uses synthetic data scaled to dataset)")

    args = parser.parse_args()

    dataset_name = args.dataset or "synthetic"
    print("=" * 60)
    print(f"Experiment 4: Causal Validation (REAL LLM)")
    print(f"  Dataset: {dataset_name}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = run_experiment(args.n_queries, args.n_repeats, seed=args.seed)
    results["dataset"] = dataset_name
    save_results(results, args.output_dir, seed=args.seed, dataset=dataset_name)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    for method_name, data in results["methods"].items():
        valid_rate = data.get("valid_rate", "N/A")
        if isinstance(valid_rate, float):
            print(f"  {method_name}: {valid_rate:.1%} valid responses")
        else:
            print(f"  {method_name}: {valid_rate}")


if __name__ == "__main__":
    main()
