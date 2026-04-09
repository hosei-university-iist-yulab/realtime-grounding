"""
Test Scenarios for Sensor Simulation.

Defines different scenarios for testing the Temporal Grounding Pipeline:
- Normal: Typical building operation
- Anomaly: Equipment failures, unusual patterns
- Peak Demand: High consumption periods
- Seasonal: Summer/winter variations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from .sensor_stream import SensorReading, SensorStream, StreamConfig
from .building_profiles import BuildingProfile, create_profile


@dataclass
class ScenarioConfig:
    """Configuration for test scenarios."""
    duration_hours: float = 24.0
    interval_minutes: float = 1.0
    buildings: List[str] = field(default_factory=lambda: ["building_001"])
    seed: Optional[int] = None  # For reproducibility


class Scenario(ABC):
    """Abstract base class for test scenarios."""

    def __init__(self, config: Optional[ScenarioConfig] = None):
        """
        Initialize scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or ScenarioConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    @abstractmethod
    def generate(self) -> List[SensorReading]:
        """
        Generate sensor readings for this scenario.

        Returns:
            List of sensor readings
        """
        pass

    @abstractmethod
    def get_ground_truth(self) -> Dict:
        """
        Get ground truth labels for evaluation.

        Returns:
            Dictionary with ground truth information
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name for logging."""
        pass


class NormalScenario(Scenario):
    """
    Normal building operation scenario.

    Characteristics:
    - Typical daily patterns
    - Expected seasonal variation
    - Minimal anomalies
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        building_type: str = "office"
    ):
        super().__init__(config)
        self.building_type = building_type
        self.profile = create_profile(building_type)
        self._readings: List[SensorReading] = []

    @property
    def name(self) -> str:
        return f"Normal ({self.building_type})"

    def generate(self) -> List[SensorReading]:
        """Generate normal operation readings."""
        stream = SensorStream(
            config=StreamConfig(
                interval_seconds=self.config.interval_minutes * 60,
                buildings=self.config.buildings,
                enable_anomalies=False  # No anomalies in normal scenario
            ),
            profile=self.profile
        )

        # Generate historical data
        self._readings = []
        for building_id in self.config.buildings:
            readings = stream.generate_historical(
                building_id=building_id,
                duration_hours=self.config.duration_hours,
                interval_minutes=self.config.interval_minutes
            )
            self._readings.extend(readings)

        return self._readings

    def get_ground_truth(self) -> Dict:
        """Get ground truth for normal scenario."""
        return {
            "scenario": "normal",
            "expected_anomalies": 0,
            "expected_pattern": "regular_daily",
            "n_readings": len(self._readings)
        }


class AnomalyScenario(Scenario):
    """
    Anomaly injection scenario.

    Injects various types of anomalies:
    - Equipment failure (sudden drop)
    - Demand spike (sudden increase)
    - Sensor malfunction (erratic readings)
    - Gradual degradation
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        anomaly_type: str = "spike",  # spike, drop, erratic, gradual
        anomaly_start_pct: float = 0.4,  # Start at 40% through
        anomaly_duration_pct: float = 0.1  # Last 10% of duration
    ):
        super().__init__(config)
        self.anomaly_type = anomaly_type
        self.anomaly_start_pct = anomaly_start_pct
        self.anomaly_duration_pct = anomaly_duration_pct
        self._readings: List[SensorReading] = []
        self._anomaly_indices: List[int] = []

    @property
    def name(self) -> str:
        return f"Anomaly ({self.anomaly_type})"

    def generate(self) -> List[SensorReading]:
        """Generate readings with injected anomalies."""
        # First generate normal readings
        normal = NormalScenario(self.config)
        self._readings = normal.generate()

        # Calculate anomaly window
        n_readings = len(self._readings)
        start_idx = int(n_readings * self.anomaly_start_pct)
        duration = int(n_readings * self.anomaly_duration_pct)
        end_idx = min(start_idx + duration, n_readings)

        self._anomaly_indices = list(range(start_idx, end_idx))

        # Inject anomalies
        for idx in self._anomaly_indices:
            reading = self._readings[idx]
            original_value = reading.value

            if self.anomaly_type == "spike":
                # Sudden increase (150-300% of normal)
                reading.value *= np.random.uniform(1.5, 3.0)
            elif self.anomaly_type == "drop":
                # Sudden decrease (10-40% of normal)
                reading.value *= np.random.uniform(0.1, 0.4)
            elif self.anomaly_type == "erratic":
                # Random fluctuations
                reading.value *= np.random.uniform(0.2, 3.0)
            elif self.anomaly_type == "gradual":
                # Gradual increase over time
                progress = (idx - start_idx) / max(1, duration - 1)
                reading.value *= (1.0 + progress * 1.5)

            reading.metadata["anomaly"] = True
            reading.metadata["anomaly_type"] = self.anomaly_type
            reading.metadata["original_value"] = original_value

        return self._readings

    def get_ground_truth(self) -> Dict:
        """Get ground truth for anomaly scenario."""
        return {
            "scenario": "anomaly",
            "anomaly_type": self.anomaly_type,
            "anomaly_start_idx": self._anomaly_indices[0] if self._anomaly_indices else None,
            "anomaly_end_idx": self._anomaly_indices[-1] if self._anomaly_indices else None,
            "anomaly_count": len(self._anomaly_indices),
            "n_readings": len(self._readings)
        }


class PeakDemandScenario(Scenario):
    """
    Peak demand scenario.

    Simulates high-demand periods:
    - Summer afternoon (cooling)
    - Winter morning (heating)
    - Special events
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        peak_type: str = "summer_afternoon"  # summer_afternoon, winter_morning, event
    ):
        super().__init__(config)
        self.peak_type = peak_type
        self._readings: List[SensorReading] = []
        self._peak_indices: List[int] = []

    @property
    def name(self) -> str:
        return f"Peak Demand ({self.peak_type})"

    def generate(self) -> List[SensorReading]:
        """Generate peak demand readings."""
        # Generate base readings
        normal = NormalScenario(self.config)
        self._readings = normal.generate()

        # Apply peak modifications based on type
        for idx, reading in enumerate(self._readings):
            dt = datetime.fromtimestamp(reading.timestamp)
            is_peak = self._is_peak_period(dt)

            if is_peak:
                self._peak_indices.append(idx)
                if self.peak_type == "summer_afternoon":
                    # High cooling load
                    reading.value *= np.random.uniform(1.4, 2.0)
                elif self.peak_type == "winter_morning":
                    # High heating load
                    reading.value *= np.random.uniform(1.3, 1.8)
                elif self.peak_type == "event":
                    # Special event spike
                    reading.value *= np.random.uniform(1.5, 2.5)

                reading.metadata["peak_demand"] = True
                reading.metadata["peak_type"] = self.peak_type

        return self._readings

    def _is_peak_period(self, dt: datetime) -> bool:
        """Check if datetime is in peak period."""
        hour = dt.hour

        if self.peak_type == "summer_afternoon":
            return 14 <= hour <= 18
        elif self.peak_type == "winter_morning":
            return 6 <= hour <= 10
        elif self.peak_type == "event":
            return 10 <= hour <= 22
        return False

    def get_ground_truth(self) -> Dict:
        """Get ground truth for peak demand scenario."""
        return {
            "scenario": "peak_demand",
            "peak_type": self.peak_type,
            "peak_count": len(self._peak_indices),
            "peak_percentage": len(self._peak_indices) / max(1, len(self._readings)) * 100,
            "n_readings": len(self._readings)
        }


class SeasonalScenario(Scenario):
    """
    Seasonal variation scenario.

    Tests model's ability to handle seasonal patterns:
    - Summer: High cooling, longer days
    - Winter: High heating, shorter days
    - Transition: Variable patterns
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        season: str = "summer"  # summer, winter, spring, fall
    ):
        super().__init__(config)
        self.season = season
        self._readings: List[SensorReading] = []

    @property
    def name(self) -> str:
        return f"Seasonal ({self.season})"

    def generate(self) -> List[SensorReading]:
        """Generate seasonal readings."""
        # Map season to approximate day of year
        season_days = {
            "spring": 80,   # Late March
            "summer": 172,  # Late June
            "fall": 265,    # Late September
            "winter": 355   # Late December
        }

        base_day = season_days.get(self.season, 172)

        # Generate readings starting from seasonal date
        stream = SensorStream(
            config=StreamConfig(
                interval_seconds=self.config.interval_minutes * 60,
                buildings=self.config.buildings,
                enable_anomalies=False
            )
        )

        self._readings = []
        for building_id in self.config.buildings:
            readings = stream.generate_historical(
                building_id=building_id,
                duration_hours=self.config.duration_hours,
                interval_minutes=self.config.interval_minutes
            )

            # Apply seasonal adjustments
            for reading in readings:
                factor = self._get_seasonal_factor()
                reading.value *= factor
                reading.metadata["season"] = self.season

            self._readings.extend(readings)

        return self._readings

    def _get_seasonal_factor(self) -> float:
        """Get consumption multiplier for season."""
        factors = {
            "summer": 1.3,  # High cooling
            "winter": 1.4,  # High heating
            "spring": 0.9,  # Mild
            "fall": 0.95    # Mild
        }
        base = factors.get(self.season, 1.0)
        return base * np.random.uniform(0.95, 1.05)

    def get_ground_truth(self) -> Dict:
        """Get ground truth for seasonal scenario."""
        return {
            "scenario": "seasonal",
            "season": self.season,
            "expected_factor": self._get_seasonal_factor(),
            "n_readings": len(self._readings)
        }


class ScenarioManager:
    """
    Manages and runs multiple test scenarios.

    Example:
        manager = ScenarioManager()
        manager.add_scenario(NormalScenario())
        manager.add_scenario(AnomalyScenario(anomaly_type="spike"))

        results = manager.run_all()
    """

    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios: List[Scenario] = []
        self.results: Dict[str, Dict] = {}

    def add_scenario(self, scenario: Scenario):
        """Add scenario to manager."""
        self.scenarios.append(scenario)

    def add_default_scenarios(self, config: Optional[ScenarioConfig] = None):
        """Add standard set of test scenarios."""
        self.scenarios = [
            NormalScenario(config, building_type="office"),
            NormalScenario(config, building_type="residential"),
            AnomalyScenario(config, anomaly_type="spike"),
            AnomalyScenario(config, anomaly_type="drop"),
            PeakDemandScenario(config, peak_type="summer_afternoon"),
            SeasonalScenario(config, season="summer"),
            SeasonalScenario(config, season="winter")
        ]

    def run_all(self) -> Dict[str, Tuple[List[SensorReading], Dict]]:
        """
        Run all scenarios and collect results.

        Returns:
            Dictionary mapping scenario name to (readings, ground_truth)
        """
        self.results = {}

        for scenario in self.scenarios:
            print(f"Running scenario: {scenario.name}")
            readings = scenario.generate()
            ground_truth = scenario.get_ground_truth()

            self.results[scenario.name] = (readings, ground_truth)

            print(f"  Generated {len(readings)} readings")
            if "anomaly_count" in ground_truth:
                print(f"  Anomalies: {ground_truth['anomaly_count']}")

        return self.results

    def get_all_readings(self) -> List[SensorReading]:
        """Get all readings from all scenarios."""
        all_readings = []
        for readings, _ in self.results.values():
            all_readings.extend(readings)
        return all_readings

    def get_summary(self) -> Dict:
        """Get summary of all scenarios."""
        return {
            name: gt for name, (_, gt) in self.results.items()
        }
