"""
BDG2 Dataset Loader (Kaggle Version).

Building Data Genome Project 2
- Source: Clayton Miller, NUS
- URL: https://www.kaggle.com/datasets/claytonmiller/buildingdatagenomeproject2
- License: CC-BY-4.0
- Data: ~3,000 buildings, 2 years of hourly data

This loader handles the Kaggle version of BDG2 with wide-format CSV files.
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from . import DatasetLoader


class BDG2Loader(DatasetLoader):
    """
    Loader for BDG2 (Building Data Genome Project 2) - Kaggle version.

    BDG2 contains energy data from ~3,000 buildings worldwide,
    with electricity, gas, water, and other meter types.

    Example:
        loader = BDG2Loader("/path/to/bdg2")
        buildings = loader.list_buildings()
        data = loader.get_meter_data("Panther_office_Hannah")
    """

    METER_TYPES = [
        "electricity", "gas", "chilledwater", "hotwater",
        "steam", "water", "irrigation", "solar"
    ]

    def __init__(self, data_path: str):
        """
        Initialize BDG2 loader.

        Args:
            data_path: Path to BDG2 directory (contains electricity.csv, metadata.csv, etc.)
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"BDG2 data not found at {data_path}. "
                "Download from Kaggle: claytonmiller/buildingdatagenomeproject2"
            )

        self._metadata_df = None
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._buildings = None

    def _load_metadata(self) -> pd.DataFrame:
        """Load building metadata."""
        if self._metadata_df is None:
            metadata_file = self.data_path / "metadata.csv"
            if metadata_file.exists():
                self._metadata_df = pd.read_csv(metadata_file)
            else:
                self._metadata_df = pd.DataFrame()
        return self._metadata_df

    def _discover_buildings(self, meter_type: str = "electricity") -> List[str]:
        """Discover available buildings from a meter file."""
        meter_file = self.data_path / f"{meter_type}.csv"
        if not meter_file.exists():
            # Try cleaned version
            meter_file = self.data_path / f"{meter_type}_cleaned.csv"

        if not meter_file.exists():
            return []

        # Read just the header to get building names
        df = pd.read_csv(meter_file, nrows=0)
        buildings = [col for col in df.columns if col != "timestamp"]
        return buildings

    @property
    def name(self) -> str:
        return "BDG2 (Building Data Genome Project 2)"

    @property
    def license(self) -> str:
        return "CC-BY-4.0 (Creative Commons Attribution)"

    def list_buildings(self, meter_type: str = "electricity") -> List[str]:
        """List all buildings with the specified meter type."""
        if self._buildings is None:
            self._buildings = self._discover_buildings(meter_type)
        return self._buildings

    def get_meter_data(
        self,
        building_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        meter_type: str = "electricity"
    ) -> pd.DataFrame:
        """
        Get meter data for a building.

        Args:
            building_id: Building identifier (e.g., "Panther_office_Hannah")
            start: Start datetime
            end: End datetime
            meter_type: Meter type (electricity, gas, etc.)

        Returns:
            DataFrame with timestamp and value columns
        """
        cache_key = f"{meter_type}"

        if cache_key not in self._data_cache:
            # Load the meter data file
            meter_file = self.data_path / f"{meter_type}.csv"
            if not meter_file.exists():
                meter_file = self.data_path / f"{meter_type}_cleaned.csv"

            if not meter_file.exists():
                raise FileNotFoundError(f"Meter file not found for {meter_type}")

            df = pd.read_csv(meter_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self._data_cache[cache_key] = df

        df = self._data_cache[cache_key]

        if building_id not in df.columns:
            raise ValueError(f"Building {building_id} not found in {meter_type} data")

        # Extract single building's data
        result = pd.DataFrame({
            "timestamp": df["timestamp"],
            "value": df[building_id]
        })

        # Remove NaN values
        result = result.dropna()

        # Apply time filters
        if start:
            result = result[result["timestamp"] >= start]
        if end:
            result = result[result["timestamp"] <= end]

        return result.reset_index(drop=True)

    def get_metadata(self, building_id: str) -> Dict:
        """Get metadata for a building."""
        metadata_df = self._load_metadata()

        if metadata_df.empty:
            return {
                "building_id": building_id,
                "dataset": "BDG2",
                "building_type": self._infer_building_type(building_id)
            }

        # Find building in metadata
        row = metadata_df[metadata_df["building_id"] == building_id]

        if row.empty:
            return {
                "building_id": building_id,
                "dataset": "BDG2",
                "building_type": self._infer_building_type(building_id)
            }

        row = row.iloc[0]
        return {
            "building_id": building_id,
            "dataset": "BDG2",
            "site_id": row.get("site_id", "unknown"),
            "primary_use": row.get("primaryspaceusage", "unknown"),
            "sub_use": row.get("sub_primaryspaceusage", "unknown"),
            "sqm": row.get("sqm"),
            "sqft": row.get("sqft"),
            "year_built": row.get("yearbuilt"),
            "timezone": row.get("timezone", "UTC")
        }

    def _infer_building_type(self, building_id: str) -> str:
        """Infer building type from building ID."""
        # BDG2 naming: Site_Type_Name (e.g., Panther_office_Hannah)
        parts = building_id.split("_")
        if len(parts) >= 2:
            return parts[1]  # office, lodging, education, etc.
        return "unknown"

    def get_statistics(
        self,
        building_id: str,
        meter_type: str = "electricity"
    ) -> Dict:
        """Get statistics for a building."""
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

    def get_available_meters(self) -> List[str]:
        """Get list of available meter types."""
        available = []
        for meter_type in self.METER_TYPES:
            meter_file = self.data_path / f"{meter_type}.csv"
            cleaned_file = self.data_path / f"{meter_type}_cleaned.csv"
            if meter_file.exists() or cleaned_file.exists():
                available.append(meter_type)
        return available
