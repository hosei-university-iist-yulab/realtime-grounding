"""
UCR Time Series Archive Loader.

Loads energy-related datasets from UCR Time Series Classification Archive.
Uses the `aeon` library for auto-download.

Supported datasets:
- ElectricDevices: 8,926 samples, 7 classes of electrical device signatures
- RefrigerationDevices: 750 samples, 3 classes of refrigeration appliances
- PowerCons: 360 samples, power consumption patterns (2 classes)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import time

from . import DatasetLoader


class UCRLoader(DatasetLoader):
    """
    Loader for UCR Time Series Archive datasets.

    Provides access to energy-related time series from UCR archive
    for cross-dataset validation experiments.
    """

    SUPPORTED_DATASETS = {
        "ElectricDevices": {
            "description": "Electrical device signatures",
            "n_classes": 7,
            "length": 96,
        },
        "RefrigerationDevices": {
            "description": "Refrigeration appliance patterns",
            "n_classes": 3,
            "length": 720,
        },
        "PowerCons": {
            "description": "Power consumption (Italy vs France)",
            "n_classes": 2,
            "length": 144,
        },
    }

    def __init__(self, dataset_name: str = "ElectricDevices"):
        """
        Initialize UCR loader.

        Args:
            dataset_name: Name of UCR dataset to load
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self._dataset_name = dataset_name
        self._data = None
        self._labels = None
        self._loaded = False

    def _load_data(self):
        """Load data from UCR archive (downloads if needed)."""
        if self._loaded:
            return

        try:
            from aeon.datasets import load_classification
            X, y = load_classification(self._dataset_name)

            # X shape: (n_samples, n_channels, length)
            # Convert to 2D array (n_samples, length) for univariate
            if X.ndim == 3:
                X = X[:, 0, :]  # Take first channel

            self._data = X
            self._labels = y
            self._loaded = True

        except ImportError:
            raise ImportError(
                "aeon library required. Install with: pip install aeon"
            )

    def list_buildings(self) -> List[str]:
        """
        List all available 'buildings' (samples) in the dataset.

        Returns sample indices as building IDs.
        """
        self._load_data()
        return [f"sample_{i}" for i in range(len(self._data))]

    def get_meter_data(
        self,
        building_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        meter_type: str = "electricity"
    ) -> pd.DataFrame:
        """
        Get time series data for a specific sample.

        Args:
            building_id: Sample identifier (e.g., "sample_0")
            start: Not used (UCR data has no timestamps)
            end: Not used
            meter_type: Ignored (all UCR energy data is electricity)

        Returns:
            DataFrame with columns: timestamp, value
        """
        self._load_data()

        # Parse sample index
        idx = int(building_id.replace("sample_", ""))

        if idx < 0 or idx >= len(self._data):
            raise ValueError(f"Sample index {idx} out of range")

        values = self._data[idx]
        n_points = len(values)

        # Create synthetic timestamps (1-minute intervals)
        base_time = datetime(2020, 1, 1)
        timestamps = [base_time + timedelta(minutes=i) for i in range(n_points)]

        return pd.DataFrame({
            "timestamp": timestamps,
            "value": values
        })

    def get_metadata(self, building_id: str) -> Dict:
        """
        Get metadata for a specific sample.

        Returns class label and dataset info.
        """
        self._load_data()

        idx = int(building_id.replace("sample_", ""))
        label = self._labels[idx]

        return {
            "building_id": building_id,
            "dataset": self._dataset_name,
            "class_label": str(label),
            "description": self.SUPPORTED_DATASETS[self._dataset_name]["description"],
            "series_length": len(self._data[idx]),
        }

    def get_statistics(
        self,
        building_id: str,
        meter_type: str = "electricity"
    ) -> Dict:
        """
        Get summary statistics for a sample.
        """
        self._load_data()

        idx = int(building_id.replace("sample_", ""))
        values = self._data[idx]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def get_samples_by_class(self, class_label: str, n_samples: int = 10) -> List[str]:
        """
        Get sample IDs for a specific class.

        Args:
            class_label: Class label to filter by
            n_samples: Maximum number of samples to return

        Returns:
            List of building_ids matching the class
        """
        self._load_data()

        matching = [
            f"sample_{i}"
            for i, label in enumerate(self._labels)
            if str(label) == str(class_label)
        ]

        return matching[:n_samples]

    def get_random_samples(self, n_samples: int = 50, seed: int = 42) -> List[Dict]:
        """
        Get random samples for trend detection testing.

        Returns list of dicts with values and timestamps.
        """
        self._load_data()

        np.random.seed(seed)
        indices = np.random.choice(len(self._data), min(n_samples, len(self._data)), replace=False)

        samples = []
        for idx in indices:
            values = self._data[idx].tolist()
            base_time = time.time() - len(values) * 60

            samples.append({
                "building_id": f"sample_{idx}",
                "dataset": self._dataset_name,
                "values": values,
                "timestamps": [base_time + i * 60 for i in range(len(values))],
                "class_label": str(self._labels[idx]),
            })

        return samples

    @property
    def name(self) -> str:
        """Dataset name."""
        return f"UCR-{self._dataset_name}"

    @property
    def license(self) -> str:
        """Dataset license."""
        return "UCR Time Series Archive (Academic Use)"

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        self._load_data()
        return len(self._data)

    @property
    def series_length(self) -> int:
        """Length of each time series."""
        return self.SUPPORTED_DATASETS[self._dataset_name]["length"]
