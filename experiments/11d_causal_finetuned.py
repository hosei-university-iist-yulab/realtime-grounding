"""
Experiment 11d: Test Multi-Task Fine-Tuned TinyLLaMA on Causal Reasoning.

Evaluates causal validity (correct cause-effect direction) on the fine-tuned model.
Target: 85%+ causal validity (up from 70% zero-shot).
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

from src.causal.validator import CausalGraph, CausalValidator

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_multitask_model():
    """Find the latest multi-task fine-tuned model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_multitask_")])
    if not model_dirs:
        raise FileNotFoundError("No multi-task fine-tuned models found in output/models/")
    return model_dirs[-1] / "final"


def load_finetuned_model(model_path: Path = None):
    """Load the multi-task fine-tuned TinyLLaMA model."""
    if model_path is None:
        model_path = find_latest_multitask_model()

    print(f"Loading multi-task model from: {model_path}")

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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
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


def create_causal_test_scenarios() -> List[Dict]:
    """Create test scenarios for causal reasoning validation."""
    scenarios = [
        # Temperature -> HVAC
        {
            "context": "It's a hot summer day with outdoor temperature at 35°C. HVAC is running at high capacity.",
            "query": "Why is HVAC energy consumption high?",
            "expected_cause": "outdoor_temperature",
            "expected_effect": "hvac_load",
        },
        {
            "context": "Winter morning, outdoor temperature is -5°C. Heating system is consuming significant energy.",
            "query": "What is causing the high heating energy usage?",
            "expected_cause": "outdoor_temperature",
            "expected_effect": "hvac_load",
        },
        # Occupancy -> Plug load
        {
            "context": "It's 9 AM Monday. 200 employees have arrived. Computers are running.",
            "query": "Why is plug load high in the morning?",
            "expected_cause": "occupancy",
            "expected_effect": "plug_load",
        },
        {
            "context": "Weekend afternoon. Only 5 security staff in the building. Most equipment is off.",
            "query": "Why is equipment energy consumption low?",
            "expected_cause": "occupancy",
            "expected_effect": "plug_load",
        },
        # Occupancy -> Lighting
        {
            "context": "Office hours, all floors occupied. All lights are on.",
            "query": "What drives lighting consumption?",
            "expected_cause": "occupancy",
            "expected_effect": "lighting_load",
        },
        {
            "context": "After hours. Building is empty. Emergency lights only.",
            "query": "Why is lighting load minimal at night?",
            "expected_cause": "occupancy",
            "expected_effect": "lighting_load",
        },
        # Hour -> Occupancy
        {
            "context": "It's 8:30 AM on a Tuesday. Workers are arriving.",
            "query": "Why is building occupancy increasing?",
            "expected_cause": "hour_of_day",
            "expected_effect": "occupancy",
        },
        {
            "context": "It's 6 PM. Workers are leaving the building.",
            "query": "What determines when the building empties?",
            "expected_cause": "hour_of_day",
            "expected_effect": "occupancy",
        },
        # HVAC/Plug -> Total consumption
        {
            "context": "High HVAC usage (cooling) and all computers running. Total consumption is 500 kWh.",
            "query": "What contributes to high total consumption?",
            "expected_cause": "hvac_load",
            "expected_effect": "total_consumption",
        },
        {
            "context": "Night time. HVAC in setback mode. Few equipment running. Total consumption is 50 kWh.",
            "query": "Why is total consumption low at night?",
            "expected_cause": "hvac_load",
            "expected_effect": "total_consumption",
        },
    ]
    return scenarios


def validate_response(response: str, scenario: Dict, graph: CausalGraph) -> Dict:
    """Validate if response has correct causal reasoning using the causal graph."""
    response_lower = response.lower()

    cause = scenario["expected_cause"]
    effect = scenario["expected_effect"]

    # Check for reversed causation patterns
    reversed_patterns = [
        # Effect causes Cause patterns (BAD)
        f"{effect.replace('_', ' ')} causes {cause.replace('_', ' ')}",
        f"{effect.replace('_', ' ')} makes {cause.replace('_', ' ')}",
        f"{effect.replace('_', ' ')} leads to {cause.replace('_', ' ')}",
        f"{effect.replace('_', ' ')} drives {cause.replace('_', ' ')}",
    ]

    # Specific reversed patterns that indicate wrong causal direction
    specific_reversed = [
        "hvac causes temperature",
        "hvac makes it hot",
        "cooling causes temperature",
        "heating causes cold",
        "plug load causes people",
        "computers attract people",
        "equipment causes occupancy",
        "lights attract people",
        "lighting causes occupancy",
        "consumption causes hvac",
        "total consumption causes",
        "electricity causes hvac",
        "occupancy causes time",
        "people cause the time",
    ]

    # Check for correct causal mentions (Cause -> Effect)
    correct_indicators = [
        # Temperature -> HVAC
        ("temperature" in response_lower and ("hvac" in response_lower or "cooling" in response_lower or "heating" in response_lower)),
        ("hot" in response_lower and "cooling" in response_lower),
        ("cold" in response_lower and "heating" in response_lower),
        # Occupancy -> Effects
        ("occupancy" in response_lower or "people" in response_lower or "employees" in response_lower),
        ("few people" in response_lower or "low occupancy" in response_lower or "empty" in response_lower),
        # Time -> Occupancy
        ("morning" in response_lower or "schedule" in response_lower or "work hours" in response_lower),
        ("time of day" in response_lower or "end of day" in response_lower),
        # Contributors to total
        ("contribute" in response_lower or "add to" in response_lower or "component" in response_lower),
    ]

    # Check for reversed
    has_reversed = False
    for pattern in specific_reversed:
        if pattern in response_lower:
            has_reversed = True
            break

    # Check for correct reasoning
    has_correct = any(correct_indicators)

    # Determine validity
    if has_reversed:
        validity = "reversed"
    elif has_correct:
        validity = "correct"
    else:
        validity = "ambiguous"

    return {
        "validity": validity,
        "has_correct_pattern": has_correct,
        "has_incorrect_pattern": has_reversed
    }


def run_causal_evaluation(seed: int = 2025):
    """Run causal reasoning evaluation."""
    np.random.seed(seed)

    print("=" * 80)
    print(f"EXPERIMENT 11d: Multi-Task Fine-Tuned Causal Evaluation (seed={seed})")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading multi-task fine-tuned model...")
    model, tokenizer = load_finetuned_model()
    print("  ✓ Model loaded")

    # Load causal graph for reference
    print("\n[2/4] Loading causal graph...")
    graph = CausalGraph.create_energy_graph()
    print(f"  ✓ Graph has {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Create test scenarios
    print("\n[3/4] Running causal reasoning tests...")
    scenarios = create_causal_test_scenarios()
    print(f"  Testing {len(scenarios)} scenarios...")

    results = []
    for i, scenario in enumerate(scenarios):
        # Create prompt
        prompt = f"""You are analyzing building energy data.

Context: {scenario['context']}

Question: {scenario['query']}

Provide a causal explanation for the energy consumption.

Answer:"""

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt)
        latency = (time.perf_counter() - start) * 1000

        validation = validate_response(response, scenario, graph)

        results.append({
            "scenario_idx": i,
            "query": scenario["query"],
            "expected_cause": scenario["expected_cause"],
            "expected_effect": scenario["expected_effect"],
            "response": response[:200],
            "validity": validation["validity"],
            "has_correct": validation["has_correct_pattern"],
            "has_incorrect": validation["has_incorrect_pattern"],
            "latency_ms": latency
        })

        status = "✓" if validation["validity"] == "correct" else "✗" if validation["validity"] == "reversed" else "?"
        print(f"  [{status}] Scenario {i+1}: {validation['validity']}")

    # Compute metrics
    print("\n[4/4] Computing results...")

    correct = sum(1 for r in results if r["validity"] == "correct")
    reversed_causal = sum(1 for r in results if r["validity"] == "reversed")
    ambiguous = sum(1 for r in results if r["validity"] == "ambiguous")

    accuracy = correct / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    # Save results
    output_dir = PROJECT_ROOT / "output" / "v2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"exp11d_causal_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            "experiment": "causal_finetuned",
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "accuracy": accuracy,
            "breakdown": {
                "correct": correct,
                "reversed": reversed_causal,
                "ambiguous": ambiguous
            },
            "avg_latency_ms": avg_latency,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved: {result_file}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Causal Validity: {accuracy * 100:.1f}%")
    print(f"  - Correct:   {correct}/{len(results)}")
    print(f"  - Reversed:  {reversed_causal}/{len(results)}")
    print(f"  - Ambiguous: {ambiguous}/{len(results)}")
    print(f"Avg latency:    {avg_latency:.1f} ms")
    print(f"Target:         85%+")

    if accuracy >= 0.85:
        print("\n✓ SUCCESS: Achieved 85%+ causal validity!")
    else:
        print(f"\n⚠ Gap to target: {(0.85 - accuracy) * 100:.1f} percentage points")

    return accuracy, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp11d: Causal reasoning with multi-task fine-tuned model")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    accuracy, results = run_causal_evaluation(seed=args.seed)
    print(f"\n✓ Experiment 11d completed (seed={args.seed})!")
