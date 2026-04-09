"""
Dataset Loaders for Energy Consumption Data.

Provides unified interface for loading data from multiple sources:
- BDG2: Building Data Genome 2 (commercial buildings)
- REDD: Reference Energy Disaggregation Dataset (residential)
- UK-DALE: UK Domestic Appliance-Level Electricity (residential)

All loaders implement the DatasetLoader abstract base class
for consistent API across different datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    All dataset loaders must implement this interface
    to ensure consistency across different data sources.
    """

    @abstractmethod
    def list_buildings(self) -> List[str]:
        """
        List all available buildings/houses in the dataset.

        Returns:
            List of building identifiers
        """
        pass

    @abstractmethod
    def get_meter_data(
        self,
        building_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        meter_type: str = "electricity"
    ) -> pd.DataFrame:
        """
        Get meter data for a specific building.

        Args:
            building_id: Building identifier
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            meter_type: Type of meter ('electricity', 'gas', etc.)

        Returns:
            DataFrame with columns: timestamp, value, [additional columns]
        """
        pass

    @abstractmethod
    def get_metadata(self, building_id: str) -> Dict:
        """
        Get metadata for a specific building.

        Args:
            building_id: Building identifier

        Returns:
            Dictionary with building metadata
        """
        pass

    @abstractmethod
    def get_statistics(
        self,
        building_id: str,
        meter_type: str = "electricity"
    ) -> Dict:
        """
        Get summary statistics for a building.

        Args:
            building_id: Building identifier
            meter_type: Type of meter

        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass

    @property
    @abstractmethod
    def license(self) -> str:
        """Dataset license information."""
        pass


from .redd_loader import REDDLoader
from .ukdale_loader import UKDALELoader
from .bdg2_loader import BDG2Loader
from .ucr_loader import UCRLoader
from .uci_loader import UCILoader

__all__ = [
    "DatasetLoader",
    "REDDLoader",
    "UKDALELoader",
    "BDG2Loader",
    "UCRLoader",
    "UCILoader",
]
