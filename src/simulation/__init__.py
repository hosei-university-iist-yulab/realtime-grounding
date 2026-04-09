"""
Real-Time Sensor Simulation Module.

Provides realistic building energy sensor data streams for testing
and validating the Temporal Grounding Pipeline.

Components:
- SensorStream: Real-time data generator with configurable intervals
- BuildingProfile: Realistic consumption patterns (HVAC, occupancy)
- ScenarioManager: Test scenarios (normal, anomaly, peak demand)
"""

from .sensor_stream import (
    SensorStream,
    StreamConfig,
    SensorReading as SimulatedReading
)
from .building_profiles import (
    BuildingProfile,
    OfficeProfile,
    ResidentialProfile,
    IndustrialProfile,
    create_profile
)
from .scenarios import (
    Scenario,
    NormalScenario,
    AnomalyScenario,
    PeakDemandScenario,
    SeasonalScenario,
    ScenarioManager
)

__all__ = [
    # Stream
    "SensorStream",
    "StreamConfig",
    "SimulatedReading",
    # Profiles
    "BuildingProfile",
    "OfficeProfile",
    "ResidentialProfile",
    "IndustrialProfile",
    "create_profile",
    # Scenarios
    "Scenario",
    "NormalScenario",
    "AnomalyScenario",
    "PeakDemandScenario",
    "SeasonalScenario",
    "ScenarioManager"
]
