#!/usr/bin/env python3
"""
Experiment 2: Grounding Accuracy Evaluation

Measures how accurately the LLM references real sensor values.

Metrics:
- Value accuracy: Does response contain correct sensor values?
- Trend accuracy: Does response correctly identify trends?
- Context relevance: Is response grounded in provided data?

Target: ≥95% grounding accuracy for TGP.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# Global LLM instance (lazy loaded)
_llm_backbone = None


def get_llm_backbone(model_path: Optional[str] = None):
    """Get or initialize LLM backbone (singleton pattern for efficiency)."""
    global _llm_backbone

    if _llm_backbone is None:
        from src.llm import LLMBackbone

        print("Loading LLM backbone...")
        _llm_backbone = LLMBackbone(model_type="tinyllama")

        # Try to load trained LoRA weights if available
        if model_path is None:
            # Find most recent trained model
            models_dir = PROJECT_ROOT / "output" / "models"
            if models_dir.exists():
                model_dirs = sorted(models_dir.glob("grounding_*"))
                if model_dirs:
                    lora_path = model_dirs[-1] / "final"
                    if lora_path.exists():
                        print(f"Loading LoRA weights from {lora_path}")
                        _llm_backbone.load_lora(str(lora_path))
        elif model_path and Path(model_path).exists():
            print(f"Loading LoRA weights from {model_path}")
            _llm_backbone.load_lora(model_path)

    return _llm_backbone


def format_grounding_prompt(query: str, stats: Dict[str, float]) -> str:
    """Format prompt with sensor context for grounding."""
    prompt = f"""<|system|>
You are an energy monitoring assistant. Answer questions using the provided sensor data.
Always include specific values from the data in your response.
</s>
<|user|>
Current sensor readings:
- Mean consumption: {stats['mean']:.1f} kWh
- Current reading: {stats.get('current', stats['mean']):.1f} kWh
- Min: {stats['min']:.1f} kWh, Max: {stats['max']:.1f} kWh
- Standard deviation: {stats['std']:.1f} kWh

Question: {query}
</s>
<|assistant|>
"""
    return prompt


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    pattern = r'[-+]?\d*\.?\d+'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def check_value_accuracy(
    response: str,
    expected_values: Dict[str, float],
    tolerance: float = 0.15
) -> Tuple[bool, float]:
    """
    Check if response contains expected values within tolerance.

    Returns:
        (is_accurate, accuracy_score)
    """
    response_numbers = extract_numbers(response)
    if not response_numbers or not expected_values:
        return False, 0.0

    matches = 0
    for key, expected in expected_values.items():
        for num in response_numbers:
            if abs(num - expected) / max(abs(expected), 1e-10) < tolerance:
                matches += 1
                break

    accuracy = matches / len(expected_values)
    return accuracy >= 0.5, accuracy


def check_trend_accuracy(
    response: str,
    actual_trend: str  # "increasing", "decreasing", "stable"
) -> bool:
    """Check if response correctly identifies trend."""
    response_lower = response.lower()

    trend_keywords = {
        "increasing": ["increasing", "rising", "higher", "up", "grew", "spike"],
        "decreasing": ["decreasing", "falling", "lower", "down", "dropped", "reduced"],
        "stable": ["stable", "constant", "steady", "normal", "typical", "consistent"]
    }

    for keyword in trend_keywords.get(actual_trend, []):
        if keyword in response_lower:
            return True
    return False


def generate_test_cases(n_cases: int = 100, dataset: str = None, seed: int = 2025) -> List[Dict]:
    """Generate test cases with known ground truth.

    Args:
        n_cases: Number of test cases
        dataset: Dataset key (e.g., 'bdg2', 'ukdale'). If None, uses synthetic data.
        seed: Random seed
    """
    np.random.seed(seed)
    test_cases = []

    # Use real data if dataset specified
    if dataset and dataset != 'synthetic':
        try:
            from src.config.datasets import get_loader, get_samples_from_loader, DATASETS
            config = DATASETS.get(dataset)
            if config:
                loader = get_loader(dataset)
                samples = get_samples_from_loader(loader, config['type'], n_cases, seed)

                for i, sample in enumerate(samples):
                    values = sample['values']
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    current = values[-1] if values else mean_value

                    # Determine trend from actual data
                    if len(values) >= 10:
                        recent_mean = np.mean(values[-10:])
                        earlier_mean = np.mean(values[:10])
                        if recent_mean > earlier_mean + std_value:
                            trend = "increasing"
                        elif recent_mean < earlier_mean - std_value:
                            trend = "decreasing"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"

                    test_cases.append({
                        "building_id": sample.get('building_id', f"sample_{i}"),
                        "query": "What is the current energy consumption pattern?",
                        "context": {
                            "statistics": {
                                "mean": round(mean_value, 1),
                                "std": round(std_value, 1),
                                "min": round(min(values), 1),
                                "max": round(max(values), 1),
                                "current": round(current, 1)
                            }
                        },
                        "expected_values": {
                            "mean": round(mean_value, 1),
                            "current": round(current, 1)
                        },
                        "expected_trend": trend,
                        "values": values  # Store for buffer population
                    })

                if test_cases:
                    print(f"  Loaded {len(test_cases)} real samples from {dataset}")
                    return test_cases

        except Exception as e:
            print(f"  Warning: Could not load {dataset}: {e}. Using synthetic data.")

    # Fallback to synthetic data
    print("  Using synthetic test data")
    for i in range(n_cases):
        mean_value = np.random.uniform(50, 200)
        std_value = np.random.uniform(5, 30)
        current = mean_value + np.random.randn() * std_value

        if current > mean_value + std_value:
            trend = "increasing"
        elif current < mean_value - std_value:
            trend = "decreasing"
        else:
            trend = "stable"

        test_cases.append({
            "building_id": f"building_{i:03d}",
            "query": "What is the current energy consumption pattern?",
            "context": {
                "statistics": {
                    "mean": round(mean_value, 1),
                    "std": round(std_value, 1),
                    "min": round(mean_value - 2*std_value, 1),
                    "max": round(mean_value + 2*std_value, 1),
                    "current": round(current, 1)
                }
            },
            "expected_values": {
                "mean": round(mean_value, 1),
                "current": round(current, 1)
            },
            "expected_trend": trend
        })

    return test_cases


def evaluate_tgp(test_cases: List[Dict], use_llm: bool = True) -> Dict[str, Any]:
    """
    Evaluate TGP grounding accuracy using REAL LLM inference.

    Args:
        test_cases: List of test cases with ground truth
        use_llm: Whether to use LLM (True) or skip for quick buffer-only test (False)

    Returns:
        Dictionary with accuracy metrics
    """
    print("\nEvaluating TGP grounding accuracy...")

    from src.buffer import TemporalGroundingBuffer, SensorReading
    from src.staleness import TimeThresholdStalenessDetector

    buffer = TemporalGroundingBuffer()
    detector = TimeThresholdStalenessDetector()

    # Load LLM backbone for REAL inference
    llm = None
    if use_llm:
        llm = get_llm_backbone()
        print(f"Using LLM: {llm.config.model_name}")

    results = {
        "value_accuracies": [],
        "trend_correct": [],
        "latencies": [],
        "responses": []  # Store actual responses for debugging
    }

    for i, case in enumerate(test_cases):
        # Populate buffer with test data
        stats = case["context"]["statistics"]
        for j in range(60):
            reading = SensorReading(
                timestamp=time.time() - (60 - j) * 60,
                building_id=case["building_id"],
                meter_type="electricity",
                value=stats["mean"] + np.random.randn() * stats["std"]
            )
            buffer.push(reading)

        # Get data from buffer
        start = time.perf_counter()
        latest = buffer.get_latest(case["building_id"], "electricity", n=5)
        current_stats = buffer.get_statistics(case["building_id"], "electricity")

        # Add current reading to stats for prompt
        current_stats["current"] = stats.get("current", current_stats["mean"])

        # Generate REAL LLM response
        if llm is not None:
            prompt = format_grounding_prompt(case["query"], current_stats)
            response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)
        else:
            # Fallback only for quick testing without GPU
            raise RuntimeError(
                "LLM not loaded. This experiment requires actual LLM inference. "
                "Set use_llm=True or ensure GPU is available."
            )

        latency = (time.perf_counter() - start) * 1000

        # Evaluate response against ground truth
        _, value_acc = check_value_accuracy(response, case["expected_values"])
        trend_ok = check_trend_accuracy(response, case["expected_trend"])

        results["value_accuracies"].append(value_acc)
        results["trend_correct"].append(trend_ok)
        results["latencies"].append(latency)
        results["responses"].append(response[:200])  # Truncate for storage

        # Cleanup
        buffer.clear(case["building_id"], "electricity")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_cases)} | "
                  f"Value Acc: {np.mean(results['value_accuracies']):.1%} | "
                  f"Trend Acc: {np.mean(results['trend_correct']):.1%}")

    return {
        "mean_value_accuracy": float(np.mean(results["value_accuracies"])),
        "trend_accuracy": float(np.mean(results["trend_correct"])),
        "mean_latency_ms": float(np.mean(results["latencies"])),
        "std_latency_ms": float(np.std(results["latencies"])),
        "n_cases": len(test_cases),
        "sample_responses": results["responses"][:5]  # Store first 5 for inspection
    }


def evaluate_baseline(test_cases: List[Dict], use_llm: bool = True) -> Dict[str, Any]:
    """
    Evaluate baseline (no grounding) accuracy using REAL LLM.

    This baseline calls the LLM WITHOUT providing sensor context,
    demonstrating the value of real-time grounding.
    """
    print("\nEvaluating baseline (LLM without real-time sensor data)...")

    llm = None
    if use_llm:
        llm = get_llm_backbone()

    results = {
        "value_accuracies": [],
        "trend_correct": [],
        "latencies": [],
        "responses": []
    }

    for i, case in enumerate(test_cases):
        start = time.perf_counter()

        if llm is not None:
            # Baseline: Ask LLM WITHOUT providing sensor values
            prompt = f"""<|system|>
You are an energy monitoring assistant.
</s>
<|user|>
Question about building {case['building_id']}: {case['query']}
</s>
<|assistant|>
"""
            response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)
        else:
            raise RuntimeError(
                "LLM not loaded. This experiment requires actual LLM inference."
            )

        latency = (time.perf_counter() - start) * 1000

        _, value_acc = check_value_accuracy(response, case["expected_values"])
        trend_ok = check_trend_accuracy(response, case["expected_trend"])

        results["value_accuracies"].append(value_acc)
        results["trend_correct"].append(trend_ok)
        results["latencies"].append(latency)
        results["responses"].append(response[:200])

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_cases)}")

    return {
        "mean_value_accuracy": float(np.mean(results["value_accuracies"])),
        "trend_accuracy": float(np.mean(results["trend_correct"])),
        "mean_latency_ms": float(np.mean(results["latencies"])),
        "n_cases": len(test_cases),
        "sample_responses": results["responses"][:5]
    }


def run_experiment(n_cases: int = 100, seed: int = 2025, dataset: str = None) -> Dict[str, Any]:
    """Run full grounding accuracy experiment."""
    np.random.seed(seed)
    timestamp = datetime.now().isoformat()

    # Generate test cases
    print("Generating test cases...")
    test_cases = generate_test_cases(n_cases, dataset=dataset, seed=seed)

    results = {
        "experiment": "grounding_accuracy",
        "dataset": dataset or "synthetic",
        "seed": seed,
        "timestamp": timestamp,
        "n_cases": len(test_cases),
        "methods": {}
    }

    # Evaluate TGP
    results["methods"]["tgp"] = evaluate_tgp(test_cases)

    # Evaluate baseline
    results["methods"]["no_grounding"] = evaluate_baseline(test_cases)

    return results


def save_results(results: Dict, output_dir: str = "output/v2/results", seed: int = 2025):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"exp02_grounding_seed{seed}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Grounding Accuracy")
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to use (bdg2, ukdale, uci_household, uci_steel, uci_tetouan)")

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.dataset:
        output_dir = f"output/v2/{args.dataset}/seed{args.seed}"
    else:
        output_dir = "output/v2/results"

    print("=" * 60)
    print(f"Experiment 2: Grounding Accuracy")
    print(f"  Dataset: {args.dataset or 'synthetic'}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    results = run_experiment(args.n_cases, seed=args.seed, dataset=args.dataset)
    save_results(results, output_dir, seed=args.seed)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for method, data in results["methods"].items():
        print(f"\n{method}:")
        print(f"  Value Accuracy: {data['mean_value_accuracy']:.1%}")
        print(f"  Trend Accuracy: {data['trend_accuracy']:.1%}")


if __name__ == "__main__":
    main()
