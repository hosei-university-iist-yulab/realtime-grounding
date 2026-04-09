"""
Prepare training data for causal reasoning fine-tuning.

Creates Q&A pairs that teach:
1. Correct cause-effect directions
2. Which factors affect energy consumption
3. How to avoid reversed causation errors

Combined with trend data for multi-task training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from src.causal.validator import CausalGraph

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class CausalSample:
    """Training sample for causal reasoning."""
    query: str
    context: str
    correct_answer: str
    incorrect_answer: str  # For contrastive learning
    causal_chain: List[str]  # e.g., ["temperature", "hvac_load", "total_consumption"]
    task_type: str  # "causal" or "trend"


def generate_causal_qa_pairs(graph: CausalGraph, n_per_edge: int = 5) -> List[CausalSample]:
    """
    Generate Q&A pairs for each causal edge.

    Teaches:
    - Correct direction: "temperature causes HVAC load"
    - Not reversed: "HVAC load does NOT cause temperature"
    """
    samples = []

    # Templates for different causal relationships
    templates = {
        ("outdoor_temperature", "hvac_load"): {
            "queries": [
                "Why is HVAC load higher today?",
                "What causes increased cooling/heating?",
                "Why did HVAC energy spike?",
            ],
            "correct": [
                "Higher outdoor temperature causes increased HVAC load as the system works harder to maintain comfort.",
                "The outdoor temperature increased, requiring more HVAC energy for cooling.",
                "Hot weather outside drives up HVAC consumption.",
            ],
            "incorrect": [
                "High HVAC load causes the outdoor temperature to rise.",  # WRONG
                "The cooling system is making it hotter outside.",  # WRONG
            ],
        },
        ("occupancy", "plug_load"): {
            "queries": [
                "Why is plug load higher in the morning?",
                "What drives equipment energy use?",
                "Why did office equipment consumption increase?",
            ],
            "correct": [
                "Higher occupancy leads to more plug load as people use computers and equipment.",
                "More people arrived, increasing equipment and device usage.",
                "Occupancy increased, driving up plug loads from computers and devices.",
            ],
            "incorrect": [
                "High electricity use caused more people to come to work.",  # WRONG
                "The computers attracted more workers.",  # WRONG
            ],
        },
        ("hour_of_day", "occupancy"): {
            "queries": [
                "Why is occupancy higher at 9 AM?",
                "What determines how many people are in the building?",
                "Why is the building empty at night?",
            ],
            "correct": [
                "Time of day determines occupancy - people arrive for work in the morning.",
                "Occupancy follows daily schedules, with peak at business hours.",
                "At night (late hours), scheduled occupancy is low as the building closes.",
            ],
            "incorrect": [
                "High occupancy causes the time to change.",  # WRONG
                "People presence controls what hour it is.",  # WRONG
            ],
        },
        ("hvac_load", "total_consumption"): {
            "queries": [
                "Why is total consumption so high?",
                "What contributes to energy use?",
                "Why did electricity bill increase?",
            ],
            "correct": [
                "High HVAC load is a major contributor to total consumption.",
                "HVAC systems consume significant energy, driving up total consumption.",
                "Heating/cooling energy adds to the total building consumption.",
            ],
            "incorrect": [
                "High total consumption causes HVAC to use more energy.",  # WRONG
                "The electricity bill made HVAC work harder.",  # WRONG
            ],
        },
        ("occupancy", "lighting_load"): {
            "queries": [
                "Why are lights using more energy?",
                "What causes lighting consumption to increase?",
                "Why is lighting load higher during the day?",
            ],
            "correct": [
                "Higher occupancy requires more lighting for the occupied spaces.",
                "More people in the building means more lights are turned on.",
                "Occupancy drives lighting demand - empty rooms don't need lights.",
            ],
            "incorrect": [
                "Bright lights attract more people to the building.",  # WRONG
                "High lighting consumption causes people to arrive.",  # WRONG
            ],
        },
        ("plug_load", "total_consumption"): {
            "queries": [
                "How do computers affect energy use?",
                "Why did total consumption go up when equipment was added?",
                "What's the impact of office equipment on energy?",
            ],
            "correct": [
                "Plug loads from computers and equipment contribute to total consumption.",
                "New equipment added plug load, increasing total energy consumption.",
                "Office equipment is a component of total building consumption.",
            ],
            "incorrect": [
                "Total consumption makes computers use more power.",  # WRONG
                "High electricity use caused more equipment to be installed.",  # WRONG
            ],
        },
    }

    for (cause, effect), data in templates.items():
        if not graph.has_edge(cause, effect):
            continue

        for i in range(min(n_per_edge, len(data["queries"]))):
            query = data["queries"][i % len(data["queries"])]
            correct = data["correct"][i % len(data["correct"])]
            incorrect = data["incorrect"][i % len(data["incorrect"])]

            # Create context with some values
            context = f"Building energy data shows {cause.replace('_', ' ')} is elevated."

            samples.append(CausalSample(
                query=query,
                context=context,
                correct_answer=correct,
                incorrect_answer=incorrect,
                causal_chain=[cause, effect],
                task_type="causal"
            ))

    return samples


def generate_causal_explanation_pairs(n_samples: int = 100) -> List[CausalSample]:
    """
    Generate explanation pairs with correct causal reasoning.
    """
    samples = []

    # Scenario templates
    scenarios = [
        {
            "context": "It's a hot summer day (35°C outside). HVAC is running at maximum capacity. Total consumption is 250 kWh.",
            "query": "Why is energy consumption so high?",
            "correct": "The high outdoor temperature (35°C) causes the HVAC system to work harder for cooling, which increases total consumption.",
            "incorrect": "High energy consumption is making the building hot, which requires more cooling.",
            "chain": ["outdoor_temperature", "hvac_load", "total_consumption"]
        },
        {
            "context": "It's 8 AM Monday. 200 employees just arrived. Computers and lights are turning on.",
            "query": "What's driving the morning energy spike?",
            "correct": "The increase in occupancy (200 employees arriving) drives up plug load from computers and lighting load, increasing total consumption.",
            "incorrect": "The energy spike is causing employees to come to work.",
            "chain": ["occupancy", "plug_load", "total_consumption"]
        },
        {
            "context": "It's Saturday. Only security staff (5 people) are in the building. Most areas are dark.",
            "query": "Why is weekend consumption lower than weekdays?",
            "correct": "Low occupancy on weekends (only 5 people) results in minimal plug load and lighting load, reducing total consumption.",
            "incorrect": "Low energy consumption causes people to stay home on weekends.",
            "chain": ["occupancy", "plug_load", "total_consumption"]
        },
        {
            "context": "Solar panels are generating 50 kW during peak sun. Interior lights are dimmed automatically.",
            "query": "How does solar radiation affect lighting?",
            "correct": "High solar radiation provides natural daylight, reducing the need for artificial lighting load.",
            "incorrect": "Low lighting load causes the sun to shine brighter.",
            "chain": ["solar_radiation", "lighting_load"]
        },
        {
            "context": "Building is 50 years old with poor insulation. HVAC runs constantly to maintain temperature.",
            "query": "Why does this old building use so much energy?",
            "correct": "The building's age results in poor efficiency, which causes HVAC systems to work harder, increasing total consumption.",
            "incorrect": "High HVAC usage is making the building older.",
            "chain": ["building_age", "efficiency", "hvac_load"]
        },
    ]

    np.random.seed(42)
    for i in range(n_samples):
        scenario = scenarios[i % len(scenarios)]

        # Add some variation
        variation = np.random.choice(["", " The trend appears stable.", " This is typical for this time."])

        samples.append(CausalSample(
            query=scenario["query"],
            context=scenario["context"] + variation,
            correct_answer=scenario["correct"],
            incorrect_answer=scenario["incorrect"],
            causal_chain=scenario["chain"],
            task_type="causal"
        ))

    return samples


def format_training_sample(sample: CausalSample) -> Dict:
    """Format sample for training."""
    prompt = f"""You are analyzing building energy data.

Context: {sample.context}

Question: {sample.query}

Provide a causal explanation for the energy consumption."""

    return {
        "prompt": prompt,
        "completion": sample.correct_answer,
        "task_type": sample.task_type,
        "causal_chain": sample.causal_chain,
        "metadata": {
            "incorrect_answer": sample.incorrect_answer,
        }
    }


def load_trend_data(path: Path) -> List[Dict]:
    """Load existing trend training data."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("PREPARING CAUSAL REASONING TRAINING DATA")
    print("=" * 80)
    print()

    # Load causal graph
    print("[1/4] Loading causal graph...")
    graph = CausalGraph.create_energy_graph()
    print(f"  ✓ Graph has {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Generate Q&A pairs
    print("\n[2/4] Generating causal Q&A pairs...")
    qa_samples = generate_causal_qa_pairs(graph, n_per_edge=10)
    print(f"  ✓ Generated {len(qa_samples)} Q&A samples")

    # Generate explanation pairs
    print("\n[3/4] Generating causal explanation pairs...")
    explanation_samples = generate_causal_explanation_pairs(n_samples=150)
    print(f"  ✓ Generated {len(explanation_samples)} explanation samples")

    # Combine and format
    all_causal = qa_samples + explanation_samples
    formatted = [format_training_sample(s) for s in all_causal]

    # Load existing trend data
    print("\n[4/4] Combining with trend training data...")
    trend_path = PROJECT_ROOT / "data" / "processed" / "trend_detection" / "train.json"
    trend_data = load_trend_data(trend_path)

    # Add task_type to trend data
    for item in trend_data:
        item["task_type"] = "trend"
        item["causal_chain"] = []

    # Combine datasets
    combined = trend_data + formatted
    np.random.seed(42)
    np.random.shuffle(combined)

    # Split train/val
    split_idx = int(len(combined) * 0.8)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    print(f"  ✓ Combined dataset: {len(combined)} samples")
    print(f"    - Trend samples: {len(trend_data)}")
    print(f"    - Causal samples: {len(formatted)}")
    print(f"    - Train: {len(train_data)}, Val: {len(val_data)}")

    # Save
    output_dir = PROJECT_ROOT / "data" / "processed" / "multitask"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)

    # Stats
    stats = {
        "total_samples": len(combined),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "trend_samples": len(trend_data),
        "causal_samples": len(formatted),
        "task_distribution": {
            "trend": sum(1 for x in combined if x.get("task_type") == "trend"),
            "causal": sum(1 for x in combined if x.get("task_type") == "causal"),
        }
    }

    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ MULTI-TASK TRAINING DATA PREPARED!")
    print("=" * 80)
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - train.json ({len(train_data)} samples)")
    print(f"  - val.json ({len(val_data)} samples)")
    print(f"  - stats.json")


if __name__ == "__main__":
    main()
