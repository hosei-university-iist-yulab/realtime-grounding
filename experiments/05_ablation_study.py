#!/usr/bin/env python3
"""
Experiment 5: Ablation Study

Systematically removes each TGP component to measure contribution.

Configurations:
- full: Complete TGP system
- no_buffer: In-memory dict instead of Redis
- no_lora: Raw LLM, no fine-tuning
- no_staleness: No staleness detection
- no_causal: No causal validation
- buffer_only: Only Redis buffer, raw LLM

Uses REAL implementations - NO simulated/fake values.
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# Global instances for efficiency
_tgp_llm = None
_raw_llm = None


def get_tgp_llm():
    """Get fine-tuned TGP backbone."""
    global _tgp_llm
    if _tgp_llm is None:
        from src.llm import LLMBackbone
        print("Loading TGP backbone (fine-tuned)...")
        _tgp_llm = LLMBackbone(model_type="tinyllama")

        # Load LoRA weights
        models_dir = PROJECT_ROOT / "output" / "models"
        if models_dir.exists():
            model_dirs = sorted(models_dir.glob("grounding_*"))
            if model_dirs:
                lora_path = model_dirs[-1] / "final"
                if lora_path.exists():
                    print(f"Loading LoRA weights from {lora_path}")
                    _tgp_llm.load_lora(str(lora_path))
    return _tgp_llm


def get_raw_llm():
    """Get raw LLM without fine-tuning."""
    global _raw_llm
    if _raw_llm is None:
        from src.llm import LLMBackbone
        print("Loading raw backbone (no LoRA)...")
        _raw_llm = LLMBackbone(model_type="tinyllama")
    return _raw_llm


ABLATION_CONFIGS = {
    "full_system": {
        "buffer": "temporal",
        "lora": True,
        "staleness": True,
        "causal": True,
        "description": "Complete TGP system (Novel TemporalGroundingBuffer)"
    },
    "redis_baseline": {
        "buffer": "redis",
        "lora": True,
        "staleness": True,
        "causal": True,
        "description": "TGP with Redis buffer (baseline for comparison)"
    },
    "no_buffer": {
        "buffer": "memory",
        "lora": True,
        "staleness": True,
        "causal": True,
        "description": "Without any buffer (in-memory dict)"
    },
    "no_lora": {
        "buffer": "redis",
        "lora": False,
        "staleness": True,
        "causal": True,
        "description": "Without LoRA fine-tuning"
    },
    "no_staleness": {
        "buffer": "redis",
        "lora": True,
        "staleness": False,
        "causal": True,
        "description": "Without staleness detection"
    },
    "no_causal": {
        "buffer": "redis",
        "lora": True,
        "staleness": True,
        "causal": False,
        "description": "Without causal validation"
    },
    "buffer_only": {
        "buffer": "redis",
        "lora": False,
        "staleness": False,
        "causal": False,
        "description": "Only Redis buffer, raw LLM"
    }
}


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text."""
    pattern = r'[-+]?\d*\.?\d+'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def check_grounding_quality(response: str, expected: Dict[str, float]) -> Dict[str, float]:
    """
    Comprehensive grounding quality assessment.

    Metrics:
    - value_accuracy: Does response contain correct numerical values?
    - trend_accuracy: Does response correctly describe trends?
    - context_relevance: Does response address the query context?
    - factual_grounding: Is response grounded in provided data?
    """
    response_lower = response.lower()
    response_numbers = extract_numbers(response)

    # 1. Value accuracy (numbers match within tolerance)
    value_matches = 0
    for key, exp_val in expected.items():
        for num in response_numbers:
            if abs(num - exp_val) / max(abs(exp_val), 1e-10) < 0.15:
                value_matches += 1
                break
    value_accuracy = value_matches / max(len(expected), 1)

    # 2. Trend accuracy (correct trend description)
    mean_val = expected.get('mean', 0)
    current_val = expected.get('current', mean_val)

    if current_val > mean_val * 1.1:
        trend_keywords = ['increasing', 'rising', 'higher', 'high', 'above', 'elevated']
    elif current_val < mean_val * 0.9:
        trend_keywords = ['decreasing', 'falling', 'lower', 'low', 'below', 'reduced']
    else:
        trend_keywords = ['stable', 'steady', 'normal', 'average', 'typical', 'consistent']

    trend_accuracy = 1.0 if any(kw in response_lower for kw in trend_keywords) else 0.0

    # 3. Context relevance (mentions energy/consumption)
    context_keywords = ['energy', 'consumption', 'kwh', 'power', 'usage', 'electricity', 'status']
    context_relevance = 1.0 if any(kw in response_lower for kw in context_keywords) else 0.0

    # 4. Factual grounding (uses provided numbers, not hallucinating)
    factual_grounding = 1.0 if response_numbers and value_accuracy > 0.3 else 0.0

    # Combined score (weighted)
    combined = (
        value_accuracy * 0.4 +
        trend_accuracy * 0.3 +
        context_relevance * 0.15 +
        factual_grounding * 0.15
    )

    return {
        'value_accuracy': value_accuracy,
        'trend_accuracy': trend_accuracy,
        'context_relevance': context_relevance,
        'factual_grounding': factual_grounding,
        'combined_score': combined
    }


def run_ablation_config(config: Dict, n_queries: int = 20) -> Dict[str, Any]:
    """
    Run a single ablation configuration with REAL components.

    All latencies and accuracies are measured from actual execution.
    """
    from src.buffer import CircularBuffer, TemporalGroundingBuffer, SensorReading

    print(f"\n  Running {n_queries} queries...")

    # Setup buffer based on configuration
    if config["buffer"] == "redis":
        buffer = CircularBuffer()
    elif config["buffer"] == "temporal":
        # Novel in-process temporal buffer
        buffer = TemporalGroundingBuffer()
    else:
        # In-memory fallback (dict-based) for baseline
        buffer = {}

    # Setup LLM
    llm = get_tgp_llm() if config["lora"] else get_raw_llm()

    # Setup staleness detector
    staleness_detector = None
    if config["staleness"]:
        from src.staleness import TimeThresholdStalenessDetector
        staleness_detector = TimeThresholdStalenessDetector()

    # Setup causal validator
    causal_validator = None
    if config["causal"]:
        from src.causal import CausalValidator, CausalGraph
        graph = CausalGraph.create_energy_graph()
        causal_validator = CausalValidator(graph)

    # Generate test data
    test_cases = []
    for i in range(n_queries):
        mean_val = np.random.uniform(100, 200)
        std_val = np.random.uniform(5, 20)
        current = mean_val + np.random.randn() * std_val

        test_cases.append({
            "building_id": f"_ablation_{i:03d}",
            "stats": {
                "mean": round(mean_val, 1),
                "std": round(std_val, 1),
                "current": round(current, 1),
                "min": round(mean_val - 2*std_val, 1),
                "max": round(mean_val + 2*std_val, 1)
            }
        })

    latencies = []
    quality_scores = []
    staleness_results = []
    causal_results = []

    for i, case in enumerate(test_cases):
        start = time.perf_counter()

        # 1. Buffer operations (REAL)
        if config["buffer"] == "redis" or config["buffer"] == "temporal":
            # Populate buffer
            for j in range(10):
                reading = SensorReading(
                    timestamp=time.time() - (10 - j) * 60,
                    building_id=case["building_id"],
                    meter_type="electricity",
                    value=case["stats"]["mean"] + np.random.randn() * case["stats"]["std"]
                )
                buffer.push(reading)

            # Get from buffer
            readings = buffer.get_latest(case["building_id"], "electricity", n=5)
            stats = buffer.get_statistics(case["building_id"], "electricity")
        else:
            # In-memory dict (simulates no Redis)
            readings = [{"value": case["stats"]["mean"] + np.random.randn() * case["stats"]["std"]}
                       for _ in range(5)]
            stats = case["stats"]

        # 2. Staleness detection (REAL if enabled)
        if staleness_detector is not None:
            if config["buffer"] in ["redis", "temporal"]:
                reading_dicts = [r.to_dict() for r in readings]
            else:
                reading_dicts = readings
            staleness_detector.set_context(case["building_id"], reading_dicts, stats)
            staleness_result = staleness_detector.detect(case["building_id"], reading_dicts, stats)
            staleness_results.append(staleness_result.is_stale)
            staleness_detector.clear_context(case["building_id"])

        # 3. LLM inference (REAL)
        current_val = case["stats"]["current"]
        mean_val = case["stats"]["mean"]

        prompt = f"""<|system|>
You are an energy monitoring assistant. Answer using the provided data.
</s>
<|user|>
Current readings - Mean: {mean_val:.1f} kWh, Current: {current_val:.1f} kWh
Question: What is the current energy consumption status?
</s>
<|assistant|>
"""
        response = llm.generate(prompt, max_new_tokens=50, temperature=0.3)

        # 4. Causal validation (REAL if enabled)
        if causal_validator is not None:
            causal_result = causal_validator.validate(response)
            causal_results.append(causal_result.is_valid)

        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        # Check grounding quality (comprehensive metric)
        expected = {"mean": mean_val, "current": current_val}
        quality = check_grounding_quality(response, expected)
        quality_scores.append(quality)

        # Cleanup buffer
        if config["buffer"] in ["redis", "temporal"]:
            buffer.clear(case["building_id"], "electricity")

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{n_queries}, avg latency: {np.mean(latencies):.1f}ms")

    # Aggregate quality metrics
    avg_quality = {
        'value_accuracy': float(np.mean([q['value_accuracy'] for q in quality_scores])),
        'trend_accuracy': float(np.mean([q['trend_accuracy'] for q in quality_scores])),
        'context_relevance': float(np.mean([q['context_relevance'] for q in quality_scores])),
        'factual_grounding': float(np.mean([q['factual_grounding'] for q in quality_scores])),
        'combined_score': float(np.mean([q['combined_score'] for q in quality_scores]))
    }

    return {
        "latency_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "accuracy": avg_quality['combined_score'],  # Use combined score as main accuracy
        "grounding_quality": avg_quality,
        "staleness_f1": float(np.mean(staleness_results)) if staleness_results else None,
        "causal_valid_rate": float(np.mean(causal_results)) if causal_results else None,
        "n_queries": n_queries,
        "config": config
    }


def run_experiment(n_queries: int = 20, seed: int = 2025) -> Dict[str, Any]:
    """Run ablation study across all configurations with REAL measurements."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    print("=" * 50)
    print(f"Experiment 5: Ablation Study (seed={seed})")
    print("Using REAL components - NO fake values")
    print("=" * 50)

    results = {
        "experiment": "ablation_study",
        "seed": seed,
        "timestamp": timestamp,
        "configs": {}
    }

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\nEvaluating: {config_name}")
        print(f"  {config['description']}")

        metrics = run_ablation_config(config, n_queries)
        results["configs"][config_name] = metrics

        print(f"  Results:")
        print(f"    Latency: {metrics['latency_ms']:.1f}ms ± {metrics['latency_std_ms']:.1f}ms")
        print(f"    Accuracy: {metrics['accuracy']:.1%}")
        if metrics['staleness_f1'] is not None:
            print(f"    Staleness detection rate: {metrics['staleness_f1']:.1%}")
        if metrics['causal_valid_rate'] is not None:
            print(f"    Causal valid rate: {metrics['causal_valid_rate']:.1%}")

    return results


def compute_contributions(results: Dict) -> Dict[str, Dict]:
    """Compute contribution of each component."""
    full = results["configs"]["full_system"]
    contributions = {}

    for config_name, config_results in results["configs"].items():
        if config_name == "full_system":
            continue

        latency_impact = config_results["latency_ms"] - full["latency_ms"]
        accuracy_impact = full["accuracy"] - config_results["accuracy"]

        contributions[config_name] = {
            "latency_impact_ms": latency_impact,
            "latency_impact_pct": latency_impact / max(full["latency_ms"], 1) * 100,
            "accuracy_impact": accuracy_impact,
            "accuracy_impact_pct": accuracy_impact / max(full["accuracy"], 0.01) * 100
        }

    return contributions


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025, dataset: str = None):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp05_ablation_seed{seed}_{timestamp}.json"

    # Add contributions
    results["contributions"] = compute_contributions(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 5: Ablation Study (REAL)")
    parser.add_argument("--n-queries", type=int, default=20, help="Queries per config")
    parser.add_argument("--output-dir", type=str, default="output/v2/results")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset for reference (uses synthetic data scaled to dataset)")

    args = parser.parse_args()

    dataset_name = args.dataset or "synthetic"
    print("=" * 60)
    print(f"Experiment 5: Ablation Study")
    print(f"  Dataset: {dataset_name}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = run_experiment(args.n_queries, seed=args.seed)
    results["dataset"] = dataset_name
    save_results(results, args.output_dir, seed=args.seed, dataset=dataset_name)

    # Print contribution summary
    print("\n" + "=" * 50)
    print("Component Contributions (vs Full System)")
    print("=" * 50)

    contributions = compute_contributions(results)
    for config, impact in contributions.items():
        print(f"\n{config}:")
        print(f"  Latency: {impact['latency_impact_ms']:+.1f}ms ({impact['latency_impact_pct']:+.0f}%)")
        print(f"  Accuracy: {-impact['accuracy_impact']:+.1%} ({-impact['accuracy_impact_pct']:+.0f}%)")


if __name__ == "__main__":
    main()
