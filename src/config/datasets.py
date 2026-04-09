"""
Shared Dataset Configuration for Multi-Dataset Experiments.

Defines all available datasets with their loaders and configurations.
Used by all experiments for consistent multi-dataset evaluation.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent

# All available datasets for experiments
DATASETS = {
    'bdg2': {
        'name': 'BDG2',
        'type': 'building',
        'description': 'Commercial Buildings',
        'loader_class': 'BDG2Loader',
        'loader_args': {'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'bdg2' / 'data' / 'meters' / 'cleaned')},
    },
    'ukdale': {
        'name': 'UK-DALE',
        'type': 'building',
        'description': 'Residential (UK)',
        'loader_class': 'UKDALELoader',
        'loader_args': {'data_dir': str(PROJECT_ROOT / 'data' / 'raw' / 'ukdale')},
    },
    'uci_household': {
        'name': 'UCI-Household',
        'type': 'uci',
        'description': 'Household Power (1-min)',
        'loader_class': 'UCILoader',
        'loader_args': {'dataset_name': 'household_power'},
    },
    'uci_steel': {
        'name': 'UCI-Steel',
        'type': 'uci',
        'description': 'Steel Industry (15-min)',
        'loader_class': 'UCILoader',
        'loader_args': {'dataset_name': 'steel_industry'},
    },
    'uci_tetouan': {
        'name': 'UCI-Tetouan',
        'type': 'uci',
        'description': 'Tetouan City (10-min)',
        'loader_class': 'UCILoader',
        'loader_args': {'dataset_name': 'tetouan_power'},
    },
}

# Default seeds for reproducibility
DEFAULT_SEEDS = [2025, 2026]

# Dataset order for tables
DATASET_ORDER = ['bdg2', 'ukdale', 'uci_household', 'uci_steel', 'uci_tetouan']


def get_loader(dataset_key: str):
    """Get data loader for a dataset."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_key]

    if config['loader_class'] == 'BDG2Loader':
        from src.data.loaders import BDG2Loader
        return BDG2Loader(config['loader_args']['data_dir'])
    elif config['loader_class'] == 'UKDALELoader':
        from src.data.loaders import UKDALELoader
        return UKDALELoader(config['loader_args']['data_dir'])
    elif config['loader_class'] == 'UCILoader':
        from src.data.loaders import UCILoader
        return UCILoader(config['loader_args']['dataset_name'])
    else:
        raise ValueError(f"Unknown loader: {config['loader_class']}")


def get_samples_from_loader(loader, dataset_type: str, n_samples: int = 50, seed: int = 2025) -> List[Dict]:
    """Extract samples from a data loader."""
    np.random.seed(seed)

    if dataset_type == 'uci':
        return loader.get_random_samples(n_samples, seed)

    # Building-type loaders (BDG2, UK-DALE)
    samples = []
    buildings = loader.list_buildings()
    if not buildings:
        return []

    np.random.shuffle(buildings)

    for building_id in buildings[:min(len(buildings), n_samples * 2)]:
        if len(samples) >= n_samples:
            break

        try:
            df = loader.get_meter_data(building_id, meter_type="electricity")
            if df is None or len(df) < 100:
                continue

            max_start = len(df) - 100
            if max_start <= 0:
                continue

            start_idx = np.random.randint(0, max_start)
            window = df.iloc[start_idx:start_idx + 100]

            values = window["value"].tolist()
            if "timestamp" in window.columns:
                timestamps = [t.timestamp() if hasattr(t, 'timestamp') else float(t)
                             for t in window["timestamp"]]
            else:
                timestamps = list(range(len(values)))

            samples.append({
                "building_id": building_id,
                "values": values,
                "timestamps": timestamps,
            })

        except Exception:
            continue

    return samples


def get_output_dir(dataset_key: str, seed: int) -> Path:
    """Get output directory for a dataset and seed."""
    output_dir = PROJECT_ROOT / "output" / "v2" / dataset_key / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_all_datasets() -> List[str]:
    """List all available dataset keys."""
    return list(DATASETS.keys())
