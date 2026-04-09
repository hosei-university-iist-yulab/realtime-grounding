"""
Prepare training data for trend detection fine-tuning.

Extracts 600 trend windows from BDG2 dataset:
- 480 training samples (80%)
- 120 validation samples (20%)

Auto-labeled using TrendAnalyzer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from src.buffer.trend_analyzer import TrendAnalyzer
from src.data.loaders import BDG2Loader

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class TrainingSample:
    """Training sample for trend detection."""
    building_id: str
    window_id: str
    mean: float
    std: float
    min_val: float
    max_val: float
    # Enhanced features
    slope_per_hour: float
    confidence: float
    r_squared: float
    volatility: float
    has_change_point: bool
    # Label
    trend: str  # "increasing", "decreasing", "stable", "volatile"
    # Prompt and completion
    prompt: str
    completion: str


def extract_training_samples(
    n_samples: int = 600,
    window_size: int = 100,
    seed: int = 42
) -> List[TrainingSample]:
    """
    Extract training samples from BDG2 dataset.

    Args:
        n_samples: Total samples to extract
        window_size: Readings per window
        seed: Random seed

    Returns:
        List of TrainingSample objects
    """
    np.random.seed(seed)

    data_dir = PROJECT_ROOT / "data" / "raw"
    bdg2_path = data_dir / "bdg2" / "data" / "meters" / "cleaned"

    loader = BDG2Loader(str(bdg2_path))
    buildings = loader.list_buildings()

    print(f"Loading BDG2: {len(buildings)} buildings available")

    analyzer = TrendAnalyzer()
    samples = []

    # Distribute samples across buildings
    samples_per_building = max(1, n_samples // len(buildings))

    for building_id in buildings:
        if len(samples) >= n_samples:
            break

        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < window_size * 2:
                continue

            # Extract multiple windows from this building
            for _ in range(samples_per_building):
                if len(samples) >= n_samples:
                    break

                # Random start position
                max_start = len(df) - window_size
                if max_start < 0:
                    break

                start_idx = np.random.randint(0, max_start)
                window = df.iloc[start_idx:start_idx + window_size]

                values = window["value"].tolist()
                timestamps = [t.timestamp() for t in window["timestamp"]]

                # Compute features
                features = analyzer.analyze(values, timestamps)

                # Basic statistics
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                min_val = float(np.min(values))
                max_val = float(np.max(values))

                # Compute slope per hour from raw slope (units/second)
                slope_per_hour = features.slope * 3600

                # Create training prompt (with enhanced features)
                prompt = f"""You are analyzing energy consumption data for a building.

Recent statistics (last {window_size} readings):
- Mean: {mean_val:.1f} kWh
- Std: {std_val:.1f} kWh
- Min: {min_val:.1f} kWh
- Max: {max_val:.1f} kWh

Enhanced trend analysis:
- Direction: {features.direction}
- Slope: {slope_per_hour:.2f} kWh/hour
- Confidence: {features.confidence:.2f}
- R²: {features.r_squared:.3f}
- Volatility: {features.volatility:.2f}
- Has change point: {features.has_change_point}

Question: What is the trend in energy consumption?
Answer ONLY with one word: "increasing", "decreasing", "stable", or "volatile"."""

                # Expected completion
                completion = features.direction

                sample = TrainingSample(
                    building_id=building_id,
                    window_id=f"{building_id}_w{len(samples)}",
                    mean=mean_val,
                    std=std_val,
                    min_val=min_val,
                    max_val=max_val,
                    slope_per_hour=slope_per_hour,
                    confidence=features.confidence,
                    r_squared=features.r_squared,
                    volatility=features.volatility,
                    has_change_point=features.has_change_point,
                    trend=features.direction,
                    prompt=prompt,
                    completion=completion
                )

                samples.append(sample)

        except Exception as e:
            print(f"  Warning: Could not process {building_id}: {e}")
            continue

    print(f"✓ Extracted {len(samples)} training samples")
    return samples


def split_train_val(samples: List[TrainingSample], val_ratio: float = 0.2, seed: int = 42):
    """Split samples into train and validation sets."""
    np.random.seed(seed)

    # Shuffle
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * (1 - val_ratio))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    return train_samples, val_samples


def save_dataset(samples: List[TrainingSample], output_path: Path, name: str):
    """Save dataset in JSON format for fine-tuning."""

    # Convert to training format
    training_data = []
    for sample in samples:
        training_data.append({
            "prompt": sample.prompt,
            "completion": sample.completion,
            "metadata": {
                "building_id": sample.building_id,
                "window_id": sample.window_id,
                "trend": sample.trend,
                "slope": sample.slope_per_hour,
                "confidence": sample.confidence
            }
        })

    output_file = output_path / f"{name}.json"
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"  Saved {len(training_data)} samples to {output_file}")

    # Also save statistics
    trend_counts = {}
    for sample in samples:
        trend_counts[sample.trend] = trend_counts.get(sample.trend, 0) + 1

    stats_file = output_path / f"{name}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "n_samples": len(samples),
            "trend_distribution": trend_counts,
            "window_size": 100
        }, f, indent=2)

    print(f"  Distribution: {trend_counts}")


if __name__ == "__main__":
    print("=" * 80)
    print("PREPARING TREND DETECTION TRAINING DATA")
    print("=" * 80)
    print()

    # Extract samples
    print("[1/3] Extracting 600 trend windows from BDG2...")
    samples = extract_training_samples(n_samples=600, window_size=100, seed=42)

    # Split train/val
    print("\n[2/3] Splitting into train (80%) and val (20%)...")
    train_samples, val_samples = split_train_val(samples, val_ratio=0.2, seed=42)
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")

    # Save datasets
    print("\n[3/3] Saving datasets...")
    output_dir = PROJECT_ROOT / "data" / "processed" / "trend_detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset(train_samples, output_dir, "train")
    save_dataset(val_samples, output_dir, "val")

    print("\n" + "=" * 80)
    print("✓ Training data prepared successfully!")
    print("=" * 80)
    print(f"\nFiles saved to: {output_dir}")
    print("  - train.json (training set)")
    print("  - val.json (validation set)")
    print("  - train_stats.json (statistics)")
    print("  - val_stats.json (statistics)")
