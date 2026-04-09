"""
Training Data Generator for Temporal Grounding Pipeline.

Processes BDG2 dataset to create training examples for:
1. Sensor-to-text grounding
2. Staleness detection
3. Causal explanation generation

Output format compatible with LoRA fine-tuning.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingExample:
    """Single training example for LLM fine-tuning."""
    input_text: str
    output_text: str
    building_id: str
    meter_type: str
    timestamp: str
    task_type: str  # 'grounding', 'staleness', 'causal'


class BDG2Loader:
    """Load and process Building Data Genome 2 dataset."""

    BDG2_PATH = PROJECT_ROOT / "data" / "raw" / "bdg2"

    # Meter type columns in BDG2
    METER_TYPES = {
        0: "electricity",
        1: "chilledwater",
        2: "steam",
        3: "hotwater",
        4: "gas",
        5: "water",
        6: "irrigation",
        7: "solar"
    }

    def __init__(self):
        """Initialize loader."""
        self.metadata = None
        self.weather = None
        self._loaded_meters: Dict[str, pd.DataFrame] = {}

    def load_metadata(self) -> pd.DataFrame:
        """Load building metadata."""
        if self.metadata is None:
            meta_path = self.BDG2_PATH / "data" / "metadata" / "metadata.csv"
            if meta_path.exists():
                self.metadata = pd.read_csv(meta_path)
            else:
                raise FileNotFoundError(f"Metadata not found at {meta_path}")
        return self.metadata

    def load_weather(self) -> pd.DataFrame:
        """Load weather data."""
        if self.weather is None:
            weather_path = self.BDG2_PATH / "data" / "weather" / "weather.csv"
            if weather_path.exists():
                self.weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
            else:
                print("Warning: Weather data not found")
                self.weather = pd.DataFrame()
        return self.weather

    def load_meter_data(self, meter_type: str = "electricity") -> pd.DataFrame:
        """
        Load meter reading data.

        Args:
            meter_type: Type of meter to load

        Returns:
            DataFrame with meter readings
        """
        if meter_type in self._loaded_meters:
            return self._loaded_meters[meter_type]

        # Look for meter data files
        data_paths = [
            self.BDG2_PATH / "data" / "meters" / "cleaned" / f"{meter_type}.csv",
            self.BDG2_PATH / "data" / "meters" / "raw" / f"{meter_type}.csv",
            self.BDG2_PATH / "data" / f"{meter_type}_cleaned.csv",
        ]

        for path in data_paths:
            if path.exists():
                df = pd.read_csv(path, parse_dates=["timestamp"])
                self._loaded_meters[meter_type] = df
                return df

        raise FileNotFoundError(f"Meter data not found for {meter_type}")

    def get_building_data(
        self,
        building_id: str,
        meter_type: str = "electricity",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for a specific building.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with building's meter readings
        """
        df = self.load_meter_data(meter_type)

        if building_id not in df.columns:
            raise ValueError(f"Building {building_id} not found in {meter_type} data")

        result = df[["timestamp", building_id]].copy()
        result.columns = ["timestamp", "value"]
        result = result.dropna()

        if start_date:
            result = result[result["timestamp"] >= start_date]
        if end_date:
            result = result[result["timestamp"] <= end_date]

        return result

    def list_buildings(self) -> List[str]:
        """List available building IDs."""
        try:
            df = self.load_meter_data("electricity")
            return [c for c in df.columns if c != "timestamp"]
        except FileNotFoundError:
            return []

    def get_building_info(self, building_id: str) -> Dict[str, Any]:
        """Get metadata for a building."""
        meta = self.load_metadata()
        row = meta[meta["building_id"] == building_id]
        if len(row) == 0:
            return {}
        return row.iloc[0].to_dict()


class TrainingDataGenerator:
    """Generate training data for temporal grounding."""

    # Question templates for grounding task
    GROUNDING_TEMPLATES = [
        "What is the current energy consumption pattern for {building}?",
        "Is {building}'s electricity usage normal right now?",
        "Describe the recent energy consumption of {building}.",
        "What's happening with {building}'s power consumption?",
        "How much electricity is {building} currently using?",
        "Analyze the energy usage pattern for {building}.",
        "What is the power demand like at {building}?",
        "Compare {building}'s current consumption to its typical usage.",
    ]

    # Response templates
    RESPONSE_TEMPLATES = {
        "normal": [
            "The current consumption of {value:.1f} kWh is within normal range "
            "(mean: {mean:.1f} kWh, std: {std:.1f} kWh). The building is operating "
            "as expected for this time of day.",
            "{building} is consuming {value:.1f} kWh, which is {deviation} the average "
            "of {mean:.1f} kWh. This is typical for {time_period}.",
        ],
        "high": [
            "Alert: {building} shows elevated consumption at {value:.1f} kWh, which is "
            "{percent_diff:.0f}% above the normal average of {mean:.1f} kWh. "
            "This may indicate increased HVAC load or equipment usage.",
            "The current reading of {value:.1f} kWh is significantly higher than typical "
            "({mean:.1f} kWh average). Possible causes include weather conditions or "
            "occupancy changes.",
        ],
        "low": [
            "Energy consumption is below normal at {value:.1f} kWh "
            "(typical: {mean:.1f} kWh). This could indicate reduced occupancy or "
            "equipment shutdown.",
            "{building} is using only {value:.1f} kWh compared to the average of "
            "{mean:.1f} kWh. This {percent_diff:.0f}% reduction may be due to "
            "{time_period} patterns.",
        ],
    }

    def __init__(self, loader: Optional[BDG2Loader] = None):
        """
        Initialize generator.

        Args:
            loader: BDG2 data loader
        """
        self.loader = loader or BDG2Loader()
        self.examples: List[TrainingExample] = []

    def _compute_statistics(
        self,
        values: np.ndarray,
        window_size: int = 24
    ) -> Dict[str, float]:
        """Compute statistics for a window of values."""
        if len(values) < window_size:
            window = values
        else:
            window = values[-window_size:]

        return {
            "mean": float(np.mean(window)),
            "std": float(np.std(window)),
            "min": float(np.min(window)),
            "max": float(np.max(window)),
            "current": float(values[-1]) if len(values) > 0 else 0.0
        }

    def _classify_consumption(
        self,
        current: float,
        mean: float,
        std: float
    ) -> Tuple[str, float]:
        """
        Classify consumption level.

        Returns:
            (category, z_score)
        """
        if std == 0:
            return "normal", 0.0

        z_score = (current - mean) / std

        if z_score > 2.0:
            return "high", z_score
        elif z_score < -2.0:
            return "low", z_score
        else:
            return "normal", z_score

    def _get_time_period(self, hour: int) -> str:
        """Get time period description."""
        if 6 <= hour < 12:
            return "morning hours"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "nighttime"

    def generate_grounding_examples(
        self,
        building_id: str,
        n_examples: int = 100,
        meter_type: str = "electricity"
    ) -> List[TrainingExample]:
        """
        Generate grounding task examples for a building.

        Args:
            building_id: Building to generate examples for
            n_examples: Number of examples to generate
            meter_type: Meter type to use

        Returns:
            List of training examples
        """
        try:
            df = self.loader.get_building_data(building_id, meter_type)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {building_id}: {e}")
            return []

        if len(df) < 100:
            return []

        examples = []
        values = df["value"].values
        timestamps = df["timestamp"].values

        # Sample random windows
        for _ in range(n_examples):
            idx = random.randint(48, len(df) - 1)  # Need history
            window = values[max(0, idx - 48):idx + 1]
            stats = self._compute_statistics(window)

            category, z_score = self._classify_consumption(
                stats["current"], stats["mean"], stats["std"]
            )

            ts = pd.Timestamp(timestamps[idx])
            time_period = self._get_time_period(ts.hour)

            # Select templates
            question = random.choice(self.GROUNDING_TEMPLATES).format(
                building=building_id
            )

            response_template = random.choice(self.RESPONSE_TEMPLATES[category])
            response = response_template.format(
                building=building_id,
                value=stats["current"],
                mean=stats["mean"],
                std=stats["std"],
                deviation="above" if z_score > 0 else "below",
                percent_diff=abs(z_score * 100),
                time_period=time_period
            )

            example = TrainingExample(
                input_text=question,
                output_text=response,
                building_id=building_id,
                meter_type=meter_type,
                timestamp=str(ts),
                task_type="grounding"
            )
            examples.append(example)

        return examples

    def generate_staleness_examples(
        self,
        building_id: str,
        n_examples: int = 50,
        meter_type: str = "electricity"
    ) -> List[TrainingExample]:
        """
        Generate staleness detection examples.

        Creates pairs of contexts that are:
        - Fresh (similar data)
        - Stale (significantly different data)
        """
        try:
            df = self.loader.get_building_data(building_id, meter_type)
        except (FileNotFoundError, ValueError):
            return []

        if len(df) < 200:
            return []

        examples = []
        values = df["value"].values

        for _ in range(n_examples):
            # Sample two windows
            idx1 = random.randint(48, len(df) - 100)
            window1 = values[idx1 - 48:idx1]
            stats1 = self._compute_statistics(window1)

            # Fresh: nearby window
            if random.random() < 0.5:
                idx2 = idx1 + random.randint(1, 10)
                window2 = values[max(0, idx2 - 48):idx2]
                is_stale = False
            else:
                # Stale: distant window with different patterns
                idx2 = random.randint(48, len(df) - 1)
                while abs(idx2 - idx1) < 100:
                    idx2 = random.randint(48, len(df) - 1)
                window2 = values[max(0, idx2 - 48):idx2]
                is_stale = True

            stats2 = self._compute_statistics(window2)

            input_text = (
                f"Context: Building {building_id}, mean consumption {stats1['mean']:.1f} kWh, "
                f"current reading {stats1['current']:.1f} kWh.\n"
                f"Current data: mean {stats2['mean']:.1f} kWh, reading {stats2['current']:.1f} kWh.\n"
                f"Is the context stale?"
            )

            if is_stale:
                output_text = (
                    f"Yes, the context is stale. The consumption pattern has changed "
                    f"significantly from {stats1['mean']:.1f} kWh to {stats2['mean']:.1f} kWh "
                    f"(change of {abs(stats2['mean'] - stats1['mean']):.1f} kWh). "
                    f"The context should be refreshed."
                )
            else:
                output_text = (
                    f"No, the context is still fresh. The current consumption of "
                    f"{stats2['current']:.1f} kWh is consistent with the context mean of "
                    f"{stats1['mean']:.1f} kWh."
                )

            example = TrainingExample(
                input_text=input_text,
                output_text=output_text,
                building_id=building_id,
                meter_type=meter_type,
                timestamp="",
                task_type="staleness"
            )
            examples.append(example)

        return examples

    def generate_causal_examples(
        self,
        building_id: str,
        n_examples: int = 30
    ) -> List[TrainingExample]:
        """
        Generate causal explanation examples.

        Uses known causal relationships to create
        explanation training data.
        """
        # Causal explanations for energy patterns
        causal_templates = [
            {
                "input": "Why is {building}'s consumption high this afternoon?",
                "output": "The elevated consumption is likely due to increased HVAC load. "
                          "Higher outdoor temperatures during afternoon hours cause the "
                          "cooling system to work harder, resulting in higher electricity usage. "
                          "Additionally, peak occupancy during business hours increases plug loads "
                          "and lighting demands."
            },
            {
                "input": "What's causing the low energy usage at {building} on Saturday?",
                "output": "Weekend energy consumption is typically lower due to reduced occupancy. "
                          "With fewer people in the building, plug loads from computers and equipment "
                          "decrease significantly. HVAC setpoints may also be adjusted for energy savings "
                          "during non-business days."
            },
            {
                "input": "Explain the energy spike at {building} this morning.",
                "output": "Morning energy spikes are commonly caused by HVAC startup. After nighttime "
                          "setback, the building systems work harder to bring temperatures back to "
                          "comfort levels before occupants arrive. This temporary increase is normal "
                          "and typically subsides within 1-2 hours."
            },
            {
                "input": "Why did {building}'s consumption increase compared to last month?",
                "output": "Seasonal temperature changes are the primary driver of consumption increases. "
                          "As outdoor temperatures become more extreme (hotter or colder), HVAC systems "
                          "must work harder to maintain indoor comfort, leading to higher electricity usage. "
                          "Building efficiency and occupancy patterns may also contribute."
            },
        ]

        examples = []
        for _ in range(n_examples):
            template = random.choice(causal_templates)
            example = TrainingExample(
                input_text=template["input"].format(building=building_id),
                output_text=template["output"],
                building_id=building_id,
                meter_type="electricity",
                timestamp="",
                task_type="causal"
            )
            examples.append(example)

        return examples

    def generate_all(
        self,
        n_buildings: int = 10,
        examples_per_building: int = 100
    ) -> List[TrainingExample]:
        """
        Generate training data for multiple buildings.

        Args:
            n_buildings: Number of buildings to use
            examples_per_building: Examples per building per task

        Returns:
            All generated examples
        """
        buildings = self.loader.list_buildings()[:n_buildings]

        if not buildings:
            # CRITICAL: Do not silently fall back to synthetic data
            # This violates the project's "NO FAKE DATA" policy
            raise RuntimeError(
                "No buildings found in BDG2 dataset. "
                "Please download BDG2 first:\n"
                "  git clone https://github.com/buds-lab/building-data-genome-project-2.git data/raw/bdg2\n"
                "\n"
                "If you intentionally want synthetic data for testing, use:\n"
                "  generator.generate_synthetic(allow_synthetic=True)"
            )

        all_examples = []

        for building in tqdm(buildings, desc="Generating examples"):
            # Grounding examples
            grounding = self.generate_grounding_examples(
                building,
                n_examples=examples_per_building
            )
            all_examples.extend(grounding)

            # Staleness examples
            staleness = self.generate_staleness_examples(
                building,
                n_examples=examples_per_building // 2
            )
            all_examples.extend(staleness)

            # Causal examples
            causal = self.generate_causal_examples(
                building,
                n_examples=examples_per_building // 3
            )
            all_examples.extend(causal)

        self.examples = all_examples
        return all_examples

    def generate_synthetic(self, allow_synthetic: bool = False) -> List[TrainingExample]:
        """
        Generate synthetic training data (TESTING ONLY).

        WARNING: This generates FAKE data and should ONLY be used for:
        - Unit testing
        - Quick development iteration
        - When explicitly requested by the user

        For actual research experiments, use real datasets (BDG2, REDD, UK-DALE).

        Args:
            allow_synthetic: Must be True to confirm you understand this is fake data

        Returns:
            List of synthetic training examples
        """
        if not allow_synthetic:
            raise ValueError(
                "Synthetic data generation requires explicit opt-in.\n"
                "Call generate_synthetic(allow_synthetic=True) if you understand "
                "this produces FAKE data unsuitable for research."
            )

        import warnings
        warnings.warn(
            "GENERATING SYNTHETIC DATA - Results are NOT valid for research publications!",
            UserWarning
        )
        print("⚠️  WARNING: Generating SYNTHETIC training data (NOT for research)...")

        buildings = [f"Building_{i:03d}" for i in range(10)]
        examples = []

        for building in buildings:
            for _ in range(50):
                mean = random.uniform(50, 200)
                std = random.uniform(5, 30)
                current = mean + random.gauss(0, std)

                question = random.choice(self.GROUNDING_TEMPLATES).format(
                    building=building
                )

                category = "normal" if abs(current - mean) < 2 * std else (
                    "high" if current > mean else "low"
                )

                response = random.choice(self.RESPONSE_TEMPLATES[category]).format(
                    building=building,
                    value=current,
                    mean=mean,
                    std=std,
                    deviation="above" if current > mean else "below",
                    percent_diff=abs((current - mean) / mean * 100),
                    time_period="this time period"
                )

                examples.append(TrainingExample(
                    input_text=question,
                    output_text=response,
                    building_id=building,
                    meter_type="electricity",
                    timestamp=str(datetime.now()),
                    task_type="grounding"
                ))

        self.examples = examples
        return examples

    def save(self, output_path: str, format: str = "jsonl"):
        """
        Save training data to file.

        Args:
            output_path: Output file path
            format: Output format ('jsonl', 'json', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for ex in self.examples:
                    f.write(json.dumps(asdict(ex)) + "\n")

        elif format == "json":
            with open(output_path, "w") as f:
                json.dump([asdict(ex) for ex in self.examples], f, indent=2)

        elif format == "csv":
            df = pd.DataFrame([asdict(ex) for ex in self.examples])
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved {len(self.examples)} examples to {output_path}")

    def to_chat_format(self) -> List[Dict]:
        """
        Convert to chat format for fine-tuning.

        Returns:
            List of chat-formatted examples
        """
        chat_examples = []

        for ex in self.examples:
            chat = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an energy monitoring assistant. Analyze sensor data "
                                   "and provide accurate, real-time insights about building "
                                   "energy consumption."
                    },
                    {
                        "role": "user",
                        "content": ex.input_text
                    },
                    {
                        "role": "assistant",
                        "content": ex.output_text
                    }
                ],
                "metadata": {
                    "building_id": ex.building_id,
                    "meter_type": ex.meter_type,
                    "task_type": ex.task_type
                }
            }
            chat_examples.append(chat)

        return chat_examples

    def save_chat_format(self, output_path: str):
        """Save in chat format for fine-tuning."""
        chat_data = self.to_chat_format()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for ex in chat_data:
                f.write(json.dumps(ex) + "\n")

        print(f"Saved {len(chat_data)} chat examples to {output_path}")


def main():
    """Main entry point for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate TGP training data")
    parser.add_argument("--n-buildings", type=int, default=10,
                        help="Number of buildings to use")
    parser.add_argument("--examples-per-building", type=int, default=100,
                        help="Examples per building")
    parser.add_argument("--output", type=str,
                        default="data/training/tgp_training.jsonl",
                        help="Output file path")
    parser.add_argument("--format", choices=["jsonl", "json", "csv"],
                        default="jsonl", help="Output format")
    parser.add_argument("--chat-format", action="store_true",
                        help="Also save in chat format")

    args = parser.parse_args()

    # Generate data
    generator = TrainingDataGenerator()
    examples = generator.generate_all(
        n_buildings=args.n_buildings,
        examples_per_building=args.examples_per_building
    )

    print(f"\nGenerated {len(examples)} training examples:")
    task_counts = {}
    for ex in examples:
        task_counts[ex.task_type] = task_counts.get(ex.task_type, 0) + 1
    for task, count in task_counts.items():
        print(f"  - {task}: {count}")

    # Save
    generator.save(args.output, args.format)

    if args.chat_format:
        chat_path = args.output.replace(".jsonl", "_chat.jsonl")
        generator.save_chat_format(chat_path)


if __name__ == "__main__":
    main()
