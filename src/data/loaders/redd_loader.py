"""
REDD Dataset Loader.

Reference Energy Disaggregation Dataset (REDD)
- Source: MIT CSAIL
- URL: http://redd.csail.mit.edu/
- License: Academic use only
- Data: 6 houses, ~2 weeks each, 1Hz sampling for some channels

This loader handles both the original REDD format and the Kaggle version.
Kaggle version: https://www.kaggle.com/datasets/pawelkauf/redd-part

Original format: house_*/channel_*.dat
Kaggle format: dev1.csv through dev6.csv
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from . import DatasetLoader


class REDDLoader(DatasetLoader):
    """
    Loader for REDD (Reference Energy Disaggregation Dataset).

    REDD contains data from 6 houses in the US, with both
    aggregate and circuit-level measurements.

    Supports both original format (house_*/channel_*.dat) and
    Kaggle format (dev*.csv).

    Example:
        loader = REDDLoader("/path/to/redd")
        buildings = loader.list_buildings()
        data = loader.get_meter_data("house_1")
    """

    def __init__(self, data_path: str):
        """
        Initialize REDD loader.

        Args:
            data_path: Path to REDD directory
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"REDD data not found at {data_path}. "
                "Download from Kaggle: pawelkauf/redd-part"
            )

        # Detect format
        self._kaggle_format = self._detect_kaggle_format()
        self._buildings = self._discover_buildings()
        self._cache: Dict[str, pd.DataFrame] = {}

    def _detect_kaggle_format(self) -> bool:
        """Detect if this is Kaggle format (dev*.csv files)."""
        dev_files = list(self.data_path.glob("dev*.csv"))
        return len(dev_files) > 0

    def _discover_buildings(self) -> List[str]:
        """Discover available houses in the dataset."""
        if self._kaggle_format:
            # Kaggle format: dev1.csv through dev6.csv represent devices
            # Treat each device file as a "house" for compatibility
            buildings = []
            for f in sorted(self.data_path.glob("dev*.csv")):
                # dev1.csv -> house_1
                dev_num = f.stem.replace("dev", "")
                buildings.append(f"house_{dev_num}")
            return buildings
        else:
            # Original format: house_* directories
            buildings = []
            for item in self.data_path.iterdir():
                if item.is_dir() and item.name.startswith("house_"):
                    buildings.append(item.name)
            return sorted(buildings)

    @property
    def name(self) -> str:
        return "REDD (Reference Energy Disaggregation Dataset)"

    @property
    def license(self) -> str:
        return "Academic use only - MIT CSAIL"

    def list_buildings(self) -> List[str]:
        """List all houses in the dataset."""
        return self._buildings

    def get_meter_data(
        self,
        building_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        meter_type: str = "electricity"
    ) -> pd.DataFrame:
        """
        Get meter data for a house.

        REDD uses channel numbers, with channels 1 and 2 typically
        being the two mains phases (aggregate consumption).

        Args:
            building_id: House ID (e.g., "house_1")
            start: Start datetime
            end: End datetime
            meter_type: "electricity" uses mains, other values use circuits

        Returns:
            DataFrame with timestamp and value columns
        """
        if building_id not in self._buildings:
            raise ValueError(f"Building {building_id} not found. "
                           f"Available: {self._buildings}")

        # Check cache
        cache_key = f"{building_id}_{meter_type}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            df = self._load_house_data(building_id, meter_type)
            self._cache[cache_key] = df

        # Apply time filters
        if start:
            df = df[df["timestamp"] >= start]
        if end:
            df = df[df["timestamp"] <= end]

        return df

    def _load_house_data(
        self,
        building_id: str,
        meter_type: str
    ) -> pd.DataFrame:
        """Load data from REDD files (original or Kaggle format)."""
        if self._kaggle_format:
            return self._load_kaggle_data(building_id)
        else:
            return self._load_original_data(building_id, meter_type)

    def _load_kaggle_data(self, building_id: str) -> pd.DataFrame:
        """Load data from Kaggle REDD format (dev*.csv)."""
        # building_id is house_N, convert to devN.csv
        dev_num = building_id.replace("house_", "")
        dev_file = self.data_path / f"dev{dev_num}.csv"

        if not dev_file.exists():
            raise FileNotFoundError(f"Device file not found: {dev_file}")

        # Kaggle format: time, dev (power in watts)
        df = pd.read_csv(dev_file)

        # Handle column names
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            # Assume first column is timestamp
            df["timestamp"] = pd.to_datetime(df.iloc[:, 0], unit="s")

        # Get power value
        if "dev" in df.columns:
            power = df["dev"]
        elif "power" in df.columns:
            power = df["power"]
        else:
            # Assume second column is power
            power = df.iloc[:, 1]

        # Convert power (W) to energy (kWh)
        df["value"] = power / 1000.0  # W to kW, treat as instantaneous

        return df[["timestamp", "value"]].dropna()

    def _load_original_data(
        self,
        building_id: str,
        meter_type: str
    ) -> pd.DataFrame:
        """Load data from original REDD format (house_*/channel_*.dat)."""
        house_path = self.data_path / building_id

        if meter_type == "electricity":
            # Load mains (channels 1 and 2)
            channels = [1, 2]
        else:
            # Try to load specific channel
            channels = [int(meter_type.replace("channel_", ""))]

        all_data = []
        for channel in channels:
            channel_file = house_path / f"channel_{channel}.dat"
            if channel_file.exists():
                # REDD format: timestamp (Unix epoch) and power (Watts)
                df = pd.read_csv(
                    channel_file,
                    sep=" ",
                    names=["timestamp", "power"],
                    header=None
                )
                df["channel"] = channel
                all_data.append(df)

        if not all_data:
            raise FileNotFoundError(f"No data files found for {building_id}")

        # Combine channels (sum mains phases)
        combined = pd.concat(all_data, ignore_index=True)

        # Group by timestamp and sum power
        result = combined.groupby("timestamp").agg({
            "power": "sum"
        }).reset_index()

        # Convert timestamp to datetime
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="s")

        # Convert power (W) to energy (kWh) assuming 1-second samples
        # P (W) * 1 second / 3600 = Wh / 1000 = kWh
        result["value"] = result["power"] / 3600 / 1000

        return result[["timestamp", "value"]]

    def get_metadata(self, building_id: str) -> Dict:
        """Get metadata for a house."""
        if building_id not in self._buildings:
            raise ValueError(f"Building {building_id} not found")

        house_path = self.data_path / building_id

        # Count channels
        channels = list(house_path.glob("channel_*.dat"))

        # Try to load labels
        labels_file = house_path / "labels.dat"
        labels = {}
        if labels_file.exists():
            with open(labels_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        labels[int(parts[0])] = " ".join(parts[1:])

        return {
            "building_id": building_id,
            "dataset": "REDD",
            "building_type": "residential",
            "location": "USA",
            "n_channels": len(channels),
            "channel_labels": labels
        }

    def get_statistics(
        self,
        building_id: str,
        meter_type: str = "electricity"
    ) -> Dict:
        """Get statistics for a house."""
        df = self.get_meter_data(building_id, meter_type=meter_type)

        if df.empty:
            return {"error": "No data available"}

        values = df["value"].dropna()

        return {
            "building_id": building_id,
            "meter_type": meter_type,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "n_samples": len(values),
            "start_date": str(df["timestamp"].min()),
            "end_date": str(df["timestamp"].max())
        }

    def get_all_channels(self, building_id: str) -> List[int]:
        """Get list of available channels for a house."""
        house_path = self.data_path / building_id
        channels = []
        for f in house_path.glob("channel_*.dat"):
            try:
                channel = int(f.stem.replace("channel_", ""))
                channels.append(channel)
            except ValueError:
                continue
        return sorted(channels)
