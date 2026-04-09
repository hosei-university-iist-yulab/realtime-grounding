"""
Building Consumption Profiles.

Defines realistic energy consumption patterns for different building types:
- Office buildings: 9-5 peak, weekend reduction
- Residential: Morning/evening peaks, variable weekend
- Industrial: Shift-based, continuous operation
- Healthcare: 24/7 with slight day peak
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import numpy as np


@dataclass
class ProfileConfig:
    """Configuration for building profile."""
    base_load_kwh: float = 100.0
    peak_multiplier: float = 2.0
    weekend_factor: float = 0.5
    seasonal_amplitude: float = 0.2  # 20% seasonal variation
    noise_std: float = 5.0


class BuildingProfile(ABC):
    """Abstract base class for building consumption profiles."""

    def __init__(self, config: Optional[ProfileConfig] = None):
        """
        Initialize profile.

        Args:
            config: Profile configuration
        """
        self.config = config or ProfileConfig()
        self._last_value: Optional[float] = None

    @abstractmethod
    def get_consumption(self, dt: datetime) -> float:
        """
        Get expected consumption at given datetime.

        Args:
            dt: Datetime for consumption calculation

        Returns:
            Expected consumption in kWh
        """
        pass

    def _get_seasonal_factor(self, dt: datetime) -> float:
        """Get seasonal adjustment factor."""
        # Day of year (0-365)
        day_of_year = dt.timetuple().tm_yday

        # Sinusoidal pattern: peak in summer (cooling) and winter (heating)
        # Two peaks per year
        annual_position = day_of_year / 365.0
        seasonal = 1.0 + self.config.seasonal_amplitude * np.cos(4 * np.pi * annual_position)

        return seasonal

    def _apply_noise(self, value: float) -> float:
        """Apply Gaussian noise to value."""
        noise = np.random.randn() * self.config.noise_std
        return max(0.0, value + noise)


class OfficeProfile(BuildingProfile):
    """
    Office building consumption profile.

    Characteristics:
    - Peak hours: 9 AM - 6 PM on weekdays
    - Reduced weekend consumption (30-50% of weekday)
    - Morning ramp-up, evening ramp-down
    - HVAC dominates consumption
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        super().__init__(config or ProfileConfig(
            base_load_kwh=120.0,
            peak_multiplier=2.5,
            weekend_factor=0.35,
            noise_std=8.0
        ))

    def get_consumption(self, dt: datetime) -> float:
        """Get office building consumption."""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5

        base = self.config.base_load_kwh

        # Time-of-day pattern
        if is_weekend:
            # Minimal weekend operation
            if 10 <= hour <= 16:
                multiplier = 0.5
            else:
                multiplier = 0.2
            base *= self.config.weekend_factor
        else:
            # Weekday pattern
            if hour < 6:
                # Night (minimal)
                multiplier = 0.15
            elif 6 <= hour < 8:
                # Pre-occupancy ramp-up
                multiplier = 0.3 + 0.2 * (hour - 6)
            elif 8 <= hour < 9:
                # Early morning ramp-up
                multiplier = 0.7 + 0.3 * (hour - 8)
            elif 9 <= hour <= 17:
                # Peak office hours
                # Slight dip at lunch
                if 12 <= hour <= 13:
                    multiplier = 0.9
                else:
                    multiplier = 1.0
            elif 17 < hour <= 19:
                # Evening ramp-down
                multiplier = 1.0 - 0.3 * (hour - 17)
            elif 19 < hour <= 22:
                # Late evening
                multiplier = 0.4 - 0.1 * (hour - 19)
            else:
                # Night
                multiplier = 0.15

        value = base * multiplier * self.config.peak_multiplier
        value *= self._get_seasonal_factor(dt)

        return self._apply_noise(value)


class ResidentialProfile(BuildingProfile):
    """
    Residential building consumption profile.

    Characteristics:
    - Morning peak (6-9 AM): breakfast, showers
    - Evening peak (5-10 PM): cooking, entertainment
    - Weekend: more daytime usage
    - Seasonal HVAC impact
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        super().__init__(config or ProfileConfig(
            base_load_kwh=50.0,
            peak_multiplier=2.0,
            weekend_factor=1.2,  # Higher on weekends
            noise_std=10.0  # More variable
        ))

    def get_consumption(self, dt: datetime) -> float:
        """Get residential consumption."""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5

        base = self.config.base_load_kwh

        # Weekend adjustment
        if is_weekend:
            base *= self.config.weekend_factor

        # Time-of-day pattern
        if hour < 5:
            # Night (minimal)
            multiplier = 0.2
        elif 5 <= hour < 7:
            # Early morning wake-up
            multiplier = 0.3 + 0.2 * (hour - 5)
        elif 7 <= hour <= 9:
            # Morning peak
            multiplier = 0.8 + 0.2 * (1 - abs(hour - 8))
        elif 9 < hour < 17:
            # Daytime (lower if weekday, higher if weekend)
            if is_weekend:
                multiplier = 0.6
            else:
                multiplier = 0.3
        elif 17 <= hour <= 21:
            # Evening peak
            peak_hour = 19
            multiplier = 0.7 + 0.3 * (1 - abs(hour - peak_hour) / 2)
        elif 21 < hour <= 23:
            # Late evening wind-down
            multiplier = 0.5 - 0.15 * (hour - 21)
        else:
            multiplier = 0.2

        value = base * multiplier * self.config.peak_multiplier
        value *= self._get_seasonal_factor(dt)

        return self._apply_noise(value)


class IndustrialProfile(BuildingProfile):
    """
    Industrial facility consumption profile.

    Characteristics:
    - Shift-based operation (3 shifts possible)
    - High base load from machinery
    - Less weekend variation (continuous operation)
    - Demand charges consideration
    """

    def __init__(
        self,
        config: Optional[ProfileConfig] = None,
        shifts: int = 2  # 1, 2, or 3 shifts
    ):
        super().__init__(config or ProfileConfig(
            base_load_kwh=500.0,
            peak_multiplier=1.5,
            weekend_factor=0.8,  # Some weekend operation
            noise_std=20.0
        ))
        self.shifts = shifts

    def get_consumption(self, dt: datetime) -> float:
        """Get industrial consumption."""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5

        base = self.config.base_load_kwh

        # Shift patterns
        if self.shifts == 1:
            # Single shift: 6 AM - 2 PM
            if 6 <= hour < 14:
                multiplier = 1.0
            elif 5 <= hour < 6 or 14 <= hour < 15:
                multiplier = 0.5  # Ramp up/down
            else:
                multiplier = 0.15  # Standby
        elif self.shifts == 2:
            # Two shifts: 6 AM - 10 PM
            if 6 <= hour < 22:
                # Shift change dips
                if hour in [14, 15]:
                    multiplier = 0.8
                else:
                    multiplier = 1.0
            else:
                multiplier = 0.2
        else:
            # Three shifts: 24/7
            # Brief dips at shift changes
            if hour in [6, 14, 22]:
                multiplier = 0.85
            else:
                multiplier = 1.0

        # Weekend reduction
        if is_weekend:
            base *= self.config.weekend_factor

        value = base * multiplier * self.config.peak_multiplier

        return self._apply_noise(value)


class HealthcareProfile(BuildingProfile):
    """
    Healthcare facility consumption profile.

    Characteristics:
    - 24/7 operation
    - Slight daytime peak (more procedures, visitors)
    - Critical systems always on
    - Higher base load
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        super().__init__(config or ProfileConfig(
            base_load_kwh=300.0,
            peak_multiplier=1.4,
            weekend_factor=0.95,  # Almost same
            noise_std=15.0
        ))

    def get_consumption(self, dt: datetime) -> float:
        """Get healthcare consumption."""
        hour = dt.hour

        base = self.config.base_load_kwh

        # 24/7 operation with slight daytime increase
        if 7 <= hour <= 19:
            # Daytime: more procedures, visitors, staff
            multiplier = 1.0 + 0.1 * np.sin((hour - 7) * np.pi / 12)
        elif 19 < hour <= 23:
            # Evening
            multiplier = 0.9
        else:
            # Night
            multiplier = 0.85

        value = base * multiplier * self.config.peak_multiplier
        value *= self._get_seasonal_factor(dt)

        return self._apply_noise(value)


def create_profile(
    building_type: str,
    config: Optional[ProfileConfig] = None,
    **kwargs
) -> BuildingProfile:
    """
    Factory function to create building profiles.

    Args:
        building_type: Type of building ('office', 'residential', 'industrial', 'healthcare')
        config: Optional profile configuration
        **kwargs: Additional arguments for specific profile types

    Returns:
        BuildingProfile instance

    Raises:
        ValueError: If building_type is unknown
    """
    profiles = {
        "office": OfficeProfile,
        "residential": ResidentialProfile,
        "industrial": IndustrialProfile,
        "healthcare": HealthcareProfile
    }

    if building_type.lower() not in profiles:
        raise ValueError(
            f"Unknown building type: {building_type}. "
            f"Options: {list(profiles.keys())}"
        )

    profile_class = profiles[building_type.lower()]
    return profile_class(config, **kwargs) if kwargs else profile_class(config)
