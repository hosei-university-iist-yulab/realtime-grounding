"""
UK-DALE Dataset Loader.

UK Domestic Appliance-Level Electricity Dataset
- Source: Jack Kelly, Imperial College London
- URL: https://jack-kelly.com/data/
- License: CC-BY-4.0
- Data: 5 UK houses, up to 4 years, 6-second aggregate sampling

This loader handles both original UK-DALE format and Kaggle version.
Kaggle version: https://www.kaggle.com/datasets/abdelmdz/uk-dale

Original format: house_*/channel_*.dat or house_*.h5
Kaggle format: ukdale.h5 (single HDF5 file with all buildings)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# hdf5plugin must be imported BEFORE h5py to enable compression filters
try:
    import hdf5plugin
except ImportError:
    pass  # Optional, only needed for compressed HDF5 files

from . import DatasetLoader


class UKDALELoader(DatasetLoader):
    """
    Loader for UK-DALE (UK Domestic Appliance-Level Electricity).

    UK-DALE contains long-term data from 5 UK houses,
    with both aggregate and appliance-level measurements.

    Supports both original format (house_*/channel_*.dat) and
    Kaggle format (ukdale.h5).

    Example:
        loader = UKDALELoader("/path/to/ukdale")
        buildings = loader.list_buildings()
        data = loader.get_meter_data("house_1")
    """

    def __init__(self, data_path: str):
        """
        Initialize UK-DALE loader.

        Args:
            data_path: Path to UK-DALE directory or single HDF5 file
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"UK-DALE data not found at {data_path}. "
                "Download from Kaggle: abdelmdz/uk-dale"
            )

        # Detect format
        self._kaggle_format = self._detect_kaggle_format()
        self._h5_file = None
        if self._kaggle_format:
            self._h5_file = self._find_h5_file()

        self._buildings = self._discover_buildings()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._metadata_cache: Dict[str, Dict] = {}

    def _detect_kaggle_format(self) -> bool:
        """Detect if this is Kaggle format (single ukdale.h5 file)."""
        h5_file = self.data_path / "ukdale.h5"
        if h5_file.exists():
            return True
        # Check if data_path itself is an h5 file
        if self.data_path.suffix == ".h5":
            return True
        return False

    def _find_h5_file(self) -> Path:
        """Find the HDF5 file."""
        if self.data_path.suffix == ".h5":
            return self.data_path
        h5_file = self.data_path / "ukdale.h5"
        if h5_file.exists():
            return h5_file
        raise FileNotFoundError("Could not find ukdale.h5 file")

    def _discover_buildings(self) -> List[str]:
        """Discover available houses in the dataset."""
        if self._kaggle_format:
            # Read building keys from HDF5
            try:
                import h5py
                with h5py.File(self._h5_file, "r") as f:
                    buildings = []
                    for key in f.keys():
                        if key.startswith("building"):
                            # buildingN -> house_N
                            num = key.replace("building", "")
                            buildings.append(f"house_{num}")
                    return sorted(buildings)
            except Exception:
                return []
        else:
            # Original format: house_* directories
            buildings = []
            for item in self.data_path.iterdir():
                if item.is_dir() and item.name.startswith("house_"):
                    buildings.append(item.name)
            return sorted(buildings)

    @property
    def name(self) -> str:
        return "UK-DALE (UK Domestic Appliance-Level Electricity)"

    @property
    def license(self) -> str:
        return "CC-BY-4.0 (Creative Commons Attribution)"

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

        UK-DALE uses meter numbers, with meter 1 typically
        being the aggregate/mains consumption.

        Args:
            building_id: House ID (e.g., "house_1")
            start: Start datetime
            end: End datetime
            meter_type: "electricity" uses mains, or specify "meter_N"

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
        """Load data from UK-DALE files (original or Kaggle format)."""
        if self._kaggle_format:
            return self._load_kaggle_data(building_id, meter_type)
        else:
            return self._load_original_data(building_id, meter_type)

    def _load_kaggle_data(
        self,
        building_id: str,
        meter_type: str
    ) -> pd.DataFrame:
        """Load data from Kaggle UK-DALE format (ukdale.h5)."""
        import h5py

        if meter_type == "electricity":
            meter_num = 1
        else:
            try:
                meter_num = int(meter_type.replace("meter_", ""))
            except ValueError:
                meter_num = 1

        # house_N -> buildingN
        building_num = building_id.replace("house_", "")

        with h5py.File(self._h5_file, "r") as f:
            building_key = f"building{building_num}"

            if building_key not in f:
                raise ValueError(f"Building {building_key} not found in HDF5")

            building_group = f[building_key]

            # Check for elec/meter structure
            if "elec" in building_group:
                elec_group = building_group["elec"]
                meter_key = f"meter{meter_num}"

                if meter_key in elec_group:
                    meter_data = elec_group[meter_key]

                    # Try to read table dataset directly
                    if "table" in meter_data:
                        table_data = meter_data["table"][:]
                        # table has columns: index (timestamp), values_block_0 (power)
                        timestamps = pd.to_datetime(table_data["index"], unit="ns")
                        values = table_data["values_block_0"].flatten()

                        df = pd.DataFrame({
                            "timestamp": timestamps,
                            "value": values / 1000  # W to kW
                        })
                        return df.dropna()
                    else:
                        # Raw array format
                        data = meter_data[:]
                        n_samples = len(data)
                        df = pd.DataFrame({
                            "timestamp": pd.date_range(
                                start="2012-01-01",
                                periods=n_samples,
                                freq="6S"
                            ),
                            "value": data / 1000 if data.ndim == 1 else data[:, 0] / 1000
                        })
                        return df.dropna()

            # Fallback: try to read any meter data
            raise FileNotFoundError(f"Could not find meter data for {building_id}")

    def _load_original_data(
        self,
        building_id: str,
        meter_type: str
    ) -> pd.DataFrame:
        """Load data from original UK-DALE format."""
        house_path = self.data_path / building_id

        if meter_type == "electricity":
            meter_num = 1
        else:
            try:
                meter_num = int(meter_type.replace("meter_", ""))
            except ValueError:
                meter_num = 1

        # UK-DALE can have different file formats
        h5_file = house_path / f"{building_id}.h5"
        csv_dir = house_path / "mains" if meter_num == 1 else house_path / f"channel_{meter_num}"

        if h5_file.exists():
            return self._load_from_hdf5(h5_file, meter_num)
        elif csv_dir.exists():
            return self._load_from_csv(csv_dir)
        else:
            dat_file = house_path / f"channel_{meter_num}.dat"
            if dat_file.exists():
                return self._load_from_dat(dat_file)
            raise FileNotFoundError(
                f"No data files found for {building_id} meter {meter_num}"
            )

    def _load_from_hdf5(self, h5_file: Path, meter_num: int) -> pd.DataFrame:
        """Load data from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 files. pip install h5py")

        with h5py.File(h5_file, "r") as f:
            # UK-DALE HDF5 structure: /building1/elec/meter1
            building_num = int(h5_file.stem.replace("house_", ""))
            key = f"/building{building_num}/elec/meter{meter_num}"

            if key not in f:
                raise KeyError(f"Meter {meter_num} not found in {h5_file}")

            data = f[key][:]

            # Assuming structure has timestamp and power columns
            df = pd.DataFrame(data)
            if "timestamp" in df.columns and "power" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df["value"] = df["power"] / 1000  # W to kWh (assuming hourly)
            else:
                # Handle raw array format
                df = pd.DataFrame({
                    "timestamp": pd.date_range(
                        start="2012-01-01",
                        periods=len(data),
                        freq="6S"
                    ),
                    "value": data[:, 0] / 1000 if len(data.shape) > 1 else data / 1000
                })

        return df[["timestamp", "value"]]

    def _load_from_csv(self, csv_dir: Path) -> pd.DataFrame:
        """Load data from CSV files."""
        all_data = []

        for csv_file in sorted(csv_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            all_data.append(df)

        if not all_data:
            # Try .dat files
            for dat_file in sorted(csv_dir.glob("*.dat")):
                df = self._load_from_dat(dat_file)
                all_data.append(df)

        if not all_data:
            raise FileNotFoundError(f"No CSV/DAT files in {csv_dir}")

        combined = pd.concat(all_data, ignore_index=True)

        # Normalize column names
        if "timestamp" not in combined.columns:
            if "time" in combined.columns:
                combined["timestamp"] = pd.to_datetime(combined["time"])
            elif combined.index.name == "timestamp":
                combined = combined.reset_index()
                combined["timestamp"] = pd.to_datetime(combined["timestamp"])

        if "value" not in combined.columns:
            if "power" in combined.columns:
                combined["value"] = combined["power"] / 1000
            elif "watts" in combined.columns:
                combined["value"] = combined["watts"] / 1000

        return combined[["timestamp", "value"]]

    def _load_from_dat(self, dat_file: Path) -> pd.DataFrame:
        """Load data from .dat file (space-separated)."""
        df = pd.read_csv(
            dat_file,
            sep=" ",
            names=["timestamp", "power"],
            header=None
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["value"] = df["power"] / 1000  # W to kWh

        return df[["timestamp", "value"]]

    def get_metadata(self, building_id: str) -> Dict:
        """Get metadata for a house."""
        if building_id in self._metadata_cache:
            return self._metadata_cache[building_id]

        if building_id not in self._buildings:
            raise ValueError(f"Building {building_id} not found")

        house_path = self.data_path / building_id

        # Try to load labels file
        labels = {}
        labels_file = house_path / "labels.dat"
        if labels_file.exists():
            with open(labels_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        labels[int(parts[0])] = " ".join(parts[1:])

        # Count available meters
        n_meters = len(list(house_path.glob("channel_*.dat")))
        if n_meters == 0:
            n_meters = len(list(house_path.glob("*.csv")))

        metadata = {
            "building_id": building_id,
            "dataset": "UK-DALE",
            "building_type": "residential",
            "location": "United Kingdom",
            "n_meters": n_meters,
            "meter_labels": labels,
            "sampling_rate": "6 seconds (aggregate)"
        }

        self._metadata_cache[building_id] = metadata
        return metadata

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
            "end_date": str(df["timestamp"].max()),
            "duration_days": (df["timestamp"].max() - df["timestamp"].min()).days
        }

    def get_all_meters(self, building_id: str) -> List[int]:
        """Get list of available meters for a house."""
        house_path = self.data_path / building_id
        meters = set()

        for f in house_path.glob("channel_*.dat"):
            try:
                meter = int(f.stem.replace("channel_", ""))
                meters.add(meter)
            except ValueError:
                continue

        for f in house_path.glob("meter*.csv"):
            try:
                meter = int(f.stem.replace("meter", ""))
                meters.add(meter)
            except ValueError:
                continue

        return sorted(meters) if meters else [1]  # Default to meter 1
