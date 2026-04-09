"""
UCI Machine Learning Repository Loader for Energy/Power Datasets.

Loads regression time-series datasets suitable for real-time grounding:
- Individual Household Electric Power Consumption (2M+ samples, 1-min)
- Steel Industry Energy Consumption (35K samples)
- Power Consumption of Tetouan City (52K samples, 3 networks)

All datasets have continuous values suitable for grounding accuracy evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import os

from . import DatasetLoader


class UCILoader(DatasetLoader):
    """
    Loader for UCI ML Repository energy/power datasets.

    Provides access to regression time-series datasets with continuous
    values for grounding accuracy evaluation.
    """

    SUPPORTED_DATASETS = {
        "household_power": {
            "name": "Individual Household Electric Power Consumption",
            "url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
            "description": "Household power (1-min)",
            "filename": "household_power_consumption.txt",
            "sep": ";",
            "value_column": "Global_active_power",
            "date_columns": ["Date", "Time"],
            "n_samples": 2075259,
        },
        "steel_industry": {
            "name": "Steel Industry Energy Consumption",
            "url": "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
            "description": "Steel factory (15-min)",
            "filename": "Steel_industry_data.csv",
            "sep": ",",
            "value_column": "Usage_kWh",
            "date_columns": ["date"],
            "n_samples": 35040,
        },
        "tetouan_power": {
            "name": "Power Consumption of Tetouan City",
            "url": "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip",
            "description": "Tetouan city (10-min)",
            "filename": "Tetuan City power consumption.csv",
            "sep": ",",
            "value_column": "Zone 1 Power Consumption",
            "date_columns": ["DateTime"],
            "n_samples": 52417,
        },
    }

    def __init__(self, dataset_name: str = "household_power", data_dir: Optional[str] = None):
        """
        Initialize UCI loader.

        Args:
            dataset_name: Name of dataset to load
            data_dir: Directory containing downloaded data (optional)
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        self._dataset_name = dataset_name
        self._config = self.SUPPORTED_DATASETS[dataset_name]

        # Set data directory
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "uci"

        self._data = None
        self._loaded = False

    def _download_dataset(self):
        """Download dataset from UCI if not present."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = self._data_dir / self._dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        zip_path = dataset_dir / f"{self._dataset_name}.zip"
        data_file = dataset_dir / self._config["filename"]

        if data_file.exists():
            return data_file

        # Download
        print(f"  Downloading {self._config['name']}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(self._config["url"], zip_path)

            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)

            # Find the data file (may be in subdirectory)
            for root, dirs, files in os.walk(dataset_dir):
                for f in files:
                    if f == self._config["filename"] or f.endswith('.txt') or f.endswith('.csv'):
                        found_path = Path(root) / f
                        if found_path != data_file and found_path.exists():
                            # Move to expected location
                            import shutil
                            shutil.move(str(found_path), str(data_file))
                            break

            print(f"  Downloaded to {dataset_dir}")

        except Exception as e:
            raise RuntimeError(f"Failed to download {self._dataset_name}: {e}")

        return data_file

    def _load_data(self):
        """Load data from file."""
        if self._loaded:
            return

        data_file = self._download_dataset()

        if not data_file.exists():
            # Try alternative filenames
            dataset_dir = self._data_dir / self._dataset_name
            for f in dataset_dir.glob("*"):
                if f.suffix in ['.txt', '.csv']:
                    data_file = f
                    break

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"  Loading {self._config['name']}...")

        # Read data
        try:
            df = pd.read_csv(
                data_file,
                sep=self._config["sep"],
                low_memory=False,
                na_values=['?', '']
            )
        except Exception as e:
            # Try with different encoding
            df = pd.read_csv(
                data_file,
                sep=self._config["sep"],
                low_memory=False,
                na_values=['?', ''],
                encoding='latin-1'
            )

        # Parse dates
        date_cols = self._config["date_columns"]
        if len(date_cols) == 2:
            # Date and Time in separate columns
            df["timestamp"] = pd.to_datetime(
                df[date_cols[0]] + " " + df[date_cols[1]],
                format="%d/%m/%Y %H:%M:%S",
                errors='coerce'
            )
        else:
            df["timestamp"] = pd.to_datetime(df[date_cols[0]], errors='coerce')

        # Get value column
        value_col = self._config["value_column"]
        df["value"] = pd.to_numeric(df[value_col], errors='coerce')

        # Clean data
        df = df.dropna(subset=["timestamp", "value"])
        df = df[df["value"] > 0]  # Remove zero/negative values
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._data = df[["timestamp", "value"]]
        self._loaded = True
        print(f"  Loaded {len(self._data):,} records")

    def list_buildings(self) -> List[str]:
        """
        List available 'buildings' (time segments).

        For single-source datasets, returns segment IDs.
        """
        self._load_data()

        # Create segments based on data size
        segment_size = 10000  # Points per segment
        n_segments = min(100, len(self._data) // segment_size)

        return [f"segment_{i}" for i in range(n_segments)]

    def get_meter_data(
        self,
        building_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        meter_type: str = "electricity"
    ) -> pd.DataFrame:
        """
        Get time series data for a specific segment.

        Args:
            building_id: Segment identifier (e.g., "segment_0")
            start: Not used
            end: Not used
            meter_type: Ignored (all data is electricity)

        Returns:
            DataFrame with columns: timestamp, value
        """
        self._load_data()

        # Parse segment index
        idx = int(building_id.replace("segment_", ""))
        segment_size = 10000

        start_idx = idx * segment_size
        end_idx = min(start_idx + segment_size, len(self._data))

        return self._data.iloc[start_idx:end_idx].copy()

    def get_metadata(self, building_id: str) -> Dict:
        """Get metadata for a specific segment."""
        self._load_data()

        idx = int(building_id.replace("segment_", ""))
        segment_size = 10000

        start_idx = idx * segment_size
        end_idx = min(start_idx + segment_size, len(self._data))

        segment = self._data.iloc[start_idx:end_idx]

        return {
            "building_id": building_id,
            "dataset": self._dataset_name,
            "description": self._config["description"],
            "n_points": len(segment),
            "time_range": {
                "start": str(segment["timestamp"].iloc[0]) if len(segment) > 0 else None,
                "end": str(segment["timestamp"].iloc[-1]) if len(segment) > 0 else None,
            }
        }

    def get_statistics(self, building_id: str, meter_type: str = "electricity") -> Dict:
        """Get summary statistics for a segment."""
        df = self.get_meter_data(building_id)

        return {
            "mean": float(df["value"].mean()),
            "std": float(df["value"].std()),
            "min": float(df["value"].min()),
            "max": float(df["value"].max()),
            "count": len(df),
        }

    def get_random_samples(self, n_samples: int = 50, seed: int = 42) -> List[Dict]:
        """
        Get random samples for trend detection testing.

        Returns list of dicts with values and timestamps.
        """
        self._load_data()

        np.random.seed(seed)
        window_size = 100  # Points per sample

        max_start = len(self._data) - window_size
        if max_start <= 0:
            return []

        start_indices = np.random.choice(max_start, min(n_samples, max_start // window_size), replace=False)

        samples = []
        for start_idx in start_indices:
            window = self._data.iloc[start_idx:start_idx + window_size]

            samples.append({
                "building_id": f"sample_{start_idx}",
                "dataset": self._dataset_name,
                "values": window["value"].tolist(),
                "timestamps": [t.timestamp() for t in window["timestamp"]],
            })

        return samples

    @property
    def name(self) -> str:
        """Dataset name."""
        return f"UCI-{self._config['name']}"

    @property
    def license(self) -> str:
        """Dataset license."""
        return "UCI Machine Learning Repository (CC BY 4.0)"

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        self._load_data()
        return len(self._data)
