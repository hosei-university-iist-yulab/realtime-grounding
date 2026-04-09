"""
Experiment 11b: Test Phi-2 (2.7B) for Trend Detection.

This is a quick test to see if a larger model (Phi-2) can handle
zero-shot trend detection better than TinyLLaMA.

If Phi-2 achieves >30% accuracy, we'll use it.
If not, we'll fine-tune TinyLLaMA for faster inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pathlib import Path
import importlib.util
import numpy as np

# Import extract and evaluate functions from 11_trend_features
spec = importlib.util.spec_from_file_location("exp11", Path(__file__).parent / "11_trend_features.py")
exp11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp11)

from src.llm.backbone import ModelConfig

# Override model config to use Phi-2
def run_phi2_experiment(
    dataset_name: str = "bdg2",
    n_windows: int = 50,
    output_dir: str = "output/v2/results",
    seed: int = 2025
):
    """Run Exp11 with Phi-2 instead of TinyLLaMA."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 11b: Phi-2 (2.7B) Zero-Shot Test (seed={seed})")
    print("=" * 80)
    print("\nTesting if larger model improves zero-shot trend detection...")
    print()

    # Create modified version
    def run_with_phi2(*args, **kwargs):
        # Call original but inject Phi-2 config
        from src.llm.backbone import LLMBackbone, ModelConfig

        print("\n[1/4] Initializing Phi-2 (2.7B)...")
        config = ModelConfig(
            model_name="microsoft/phi-2",
            use_4bit=True,
            max_length=1024
        )
        llm = LLMBackbone(config=config)
        print(f"  ✓ Loaded {config.model_name}")

        # Rest of experiment unchanged - just swap the model
        # Extract scenarios
        print(f"\n[2/4] Extracting trend windows from REAL {kwargs.get('dataset_name', 'bdg2').upper()} data...")
        all_scenarios = exp11.extract_trend_windows_from_dataset(
            dataset_name=kwargs.get('dataset_name', 'bdg2'),
            window_size=100,
            n_windows_per_building=5,
            max_buildings=kwargs.get('n_windows', 50) // 5
        )

        if len(all_scenarios) == 0:
            raise RuntimeError(f"No trend windows extracted. Check dataset availability.")

        print(f"  ✓ Extracted {len(all_scenarios)} real trend windows")

        # Baseline
        print(f"\n[3/4] Evaluating baseline (V1 - basic stats only)...")
        baseline_results = []
        for i, scenario in enumerate(all_scenarios):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(all_scenarios)}...")
            result = exp11.evaluate_trend_detection(llm, scenario, use_enhanced_features=False)
            baseline_results.append(result)

        baseline_accuracy = sum(r["correct"] for r in baseline_results) / len(baseline_results)
        print(f"  ✓ Baseline accuracy: {baseline_accuracy * 100:.1f}%")

        # Enhanced
        print(f"\n[4/4] Evaluating enhanced (V2 - with TrendAnalyzer)...")
        enhanced_results = []
        for i, scenario in enumerate(all_scenarios):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(all_scenarios)}...")
            result = exp11.evaluate_trend_detection(llm, scenario, use_enhanced_features=True)
            enhanced_results.append(result)

        enhanced_accuracy = sum(r["correct"] for r in enhanced_results) / len(enhanced_results)
        print(f"  ✓ Enhanced accuracy: {enhanced_accuracy * 100:.1f}%")
        print(f"  ✓ Improvement: +{(enhanced_accuracy - baseline_accuracy) * 100:.1f} percentage points")

        # Per-trend analysis
        print("\n[Analysis] Computing per-trend-type accuracy...")
        trend_types = set(r["expected"] for r in baseline_results)
        per_type_results = {}

        for ttype in sorted(trend_types):
            baseline_for_type = [r for r in baseline_results if r["expected"] == ttype]
            enhanced_for_type = [r for r in enhanced_results if r["expected"] == ttype]

            baseline_acc = sum(r["correct"] for r in baseline_for_type) / len(baseline_for_type) if baseline_for_type else 0
            enhanced_acc = sum(r["correct"] for r in enhanced_for_type) / len(enhanced_for_type) if enhanced_for_type else 0

            per_type_results[ttype] = {
                "baseline_accuracy": baseline_acc,
                "enhanced_accuracy": enhanced_acc,
                "improvement": enhanced_acc - baseline_acc,
                "n_samples": len(baseline_for_type)
            }

            print(f"  {ttype:15s}: {baseline_acc * 100:5.1f}% → {enhanced_acc * 100:5.1f}% (+{(enhanced_acc - baseline_acc) * 100:+.1f}%) [{len(baseline_for_type)} samples]")

        # Package and save
        from datetime import datetime
        results = {
            "experiment": "trend_features_phi2_test",
            "timestamp": datetime.now().isoformat(),
            "dataset": kwargs.get('dataset_name', 'bdg2'),
            "n_scenarios": len(all_scenarios),
            "n_trend_types": len(trend_types),
            "baseline": {"accuracy": baseline_accuracy, "results": baseline_results},
            "enhanced": {"accuracy": enhanced_accuracy, "results": enhanced_results},
            "improvement": {
                "absolute": enhanced_accuracy - baseline_accuracy,
                "relative": (enhanced_accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
            },
            "per_type": per_type_results,
            "config": {
                "window_size": 100,
                "llm_model": "microsoft/phi-2",
                "device": "cuda:0 (via CUDA_VISIBLE_DEVICES)"
            }
        }

        output_path = Path(kwargs.get('output_dir', 'output/v2/results'))
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"exp11b_phi2_{timestamp}.json"

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"RESULTS SAVED: {result_file}")
        print(f"{'='*80}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Model: Phi-2 (2.7B)")
        print(f"Baseline accuracy:  {baseline_accuracy * 100:.1f}%")
        print(f"Enhanced accuracy:  {enhanced_accuracy * 100:.1f}%")
        print(f"Improvement:        +{(enhanced_accuracy - baseline_accuracy) * 100:.1f} percentage points")

        # Decision
        print("\n" + "=" * 80)
        print("DECISION")
        print("=" * 80)
        if enhanced_accuracy >= 0.85:
            print("✓ SUCCESS: Achieved 85%+ target with Phi-2 zero-shot!")
        elif enhanced_accuracy >= 0.50:
            print(f"⚠ PARTIAL: {enhanced_accuracy*100:.1f}% achieved. Phi-2 works but needs prompt optimization.")
        elif enhanced_accuracy >= 0.30:
            print(f"⚠ MARGINAL: {enhanced_accuracy*100:.1f}% achieved. Consider fine-tuning Phi-2.")
        else:
            print(f"✗ FAILED: {enhanced_accuracy*100:.1f}% is too low. Must fine-tune TinyLLaMA for faster inference.")

        return results

    return run_with_phi2(
        dataset_name=dataset_name,
        n_windows=n_windows,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp11b: Test Phi-2 for trend detection")
    parser.add_argument("--dataset", type=str, default="bdg2", help="Dataset: 'bdg2' or 'redd'")
    parser.add_argument("--n-windows", type=int, default=50, help="Number of trend windows")
    parser.add_argument("--output-dir", type=str, default="output/v2/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")

    args = parser.parse_args()

    results = run_phi2_experiment(
        dataset_name=args.dataset,
        n_windows=args.n_windows,
        output_dir=args.output_dir,
        seed=args.seed
    )

    print(f"\n✓ Phi-2 test completed (seed={args.seed})!")
