"""
Real-Time Sensor Stream Generator.

Generates realistic building energy sensor data with configurable:
- Sampling intervals (1min, 5min, 15min, hourly)
- Building profiles (office, residential, industrial)
- Anomaly injection
- Seasonal patterns
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Generator
import numpy as np


@dataclass
class SensorReading:
    """A single sensor reading."""
    timestamp: float
    building_id: str
    meter_type: str
    value: float
    unit: str = "kWh"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "building_id": self.building_id,
            "meter_type": self.meter_type,
            "value": self.value,
            "unit": self.unit,
            "metadata": self.metadata
        }


@dataclass
class StreamConfig:
    """Configuration for sensor stream."""
    interval_seconds: float = 60.0  # 1 minute default
    buildings: List[str] = field(default_factory=lambda: ["building_001"])
    meter_types: List[str] = field(default_factory=lambda: ["electricity"])
    noise_std: float = 5.0
    drift_rate: float = 0.001  # Small drift over time
    enable_anomalies: bool = True
    anomaly_probability: float = 0.02  # 2% chance per reading


class SensorStream:
    """
    Real-time sensor data stream generator.

    Generates realistic building energy data that can be:
    - Pushed to Redis buffer in real-time
    - Used for streaming evaluation
    - Tested with different sampling rates

    Example:
        stream = SensorStream(config)
        stream.start()

        for reading in stream.get_readings(timeout=5.0):
            buffer.push(reading)

        stream.stop()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        profile: Optional["BuildingProfile"] = None
    ):
        """
        Initialize sensor stream.

        Args:
            config: Stream configuration
            profile: Building profile for consumption patterns
        """
        self.config = config or StreamConfig()
        self.profile = profile

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._callbacks: List[Callable[[SensorReading], None]] = []

        # State tracking
        self._last_values: Dict[str, float] = {}
        self._start_time: Optional[float] = None

    def start(self):
        """Start generating sensor readings in background thread."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._generate_loop, daemon=True)
        self._thread.start()
        print(f"Sensor stream started (interval: {self.config.interval_seconds}s)")

    def stop(self):
        """Stop generating sensor readings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("Sensor stream stopped")

    def _generate_loop(self):
        """Main generation loop (runs in background thread)."""
        while self._running:
            try:
                readings = self._generate_batch()
                for reading in readings:
                    # Add to queue
                    try:
                        self._queue.put(reading, block=False)
                    except queue.Full:
                        # Drop oldest if queue full
                        try:
                            self._queue.get_nowait()
                            self._queue.put(reading, block=False)
                        except queue.Empty:
                            pass

                    # Call registered callbacks
                    for callback in self._callbacks:
                        try:
                            callback(reading)
                        except Exception as e:
                            print(f"Callback error: {e}")

                # Wait for next interval
                time.sleep(self.config.interval_seconds)

            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(1.0)

    def _generate_batch(self) -> List[SensorReading]:
        """Generate readings for all buildings and meters."""
        readings = []
        current_time = time.time()
        current_dt = datetime.fromtimestamp(current_time)

        for building_id in self.config.buildings:
            for meter_type in self.config.meter_types:
                value = self._generate_value(
                    building_id, meter_type, current_dt
                )

                reading = SensorReading(
                    timestamp=current_time,
                    building_id=building_id,
                    meter_type=meter_type,
                    value=value,
                    metadata={
                        "hour": current_dt.hour,
                        "day_of_week": current_dt.weekday(),
                        "is_weekend": current_dt.weekday() >= 5
                    }
                )
                readings.append(reading)

        return readings

    def _generate_value(
        self,
        building_id: str,
        meter_type: str,
        current_dt: datetime
    ) -> float:
        """Generate a realistic sensor value."""
        key = f"{building_id}_{meter_type}"

        # Base value from profile or default
        if self.profile:
            base_value = self.profile.get_consumption(current_dt)
        else:
            base_value = self._default_consumption(current_dt, meter_type)

        # Add noise
        noise = np.random.randn() * self.config.noise_std

        # Add drift from last value (momentum)
        if key in self._last_values:
            last = self._last_values[key]
            drift = (base_value - last) * self.config.drift_rate
            value = last + drift + noise
        else:
            value = base_value + noise

        # Inject anomaly
        if self.config.enable_anomalies:
            if np.random.random() < self.config.anomaly_probability:
                anomaly_type = np.random.choice(["spike", "drop", "plateau"])
                if anomaly_type == "spike":
                    value *= np.random.uniform(1.5, 3.0)
                elif anomaly_type == "drop":
                    value *= np.random.uniform(0.1, 0.5)
                # plateau: keep same value

        # Ensure non-negative
        value = max(0.0, value)

        # Store for next iteration
        self._last_values[key] = value

        return round(value, 2)

    def _default_consumption(self, dt: datetime, meter_type: str) -> float:
        """Default consumption pattern (office building)."""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5

        if meter_type == "electricity":
            # Base load
            base = 100.0

            # Time-of-day pattern (office hours)
            if is_weekend:
                base *= 0.4
            elif 8 <= hour <= 18:
                base *= 1.0 + 0.5 * np.sin((hour - 8) * np.pi / 10)
            elif 6 <= hour < 8:
                base *= 0.6 + 0.1 * (hour - 6)
            elif 18 < hour <= 22:
                base *= 0.8 - 0.1 * (hour - 18)
            else:
                base *= 0.3

            return base

        elif meter_type == "gas":
            # Heating load (winter pattern)
            base = 50.0
            if 5 <= hour <= 9 or 16 <= hour <= 21:
                base *= 1.5
            return base

        else:
            return 100.0

    def get_readings(
        self,
        timeout: Optional[float] = None,
        max_readings: Optional[int] = None
    ) -> Generator[SensorReading, None, None]:
        """
        Get readings from the stream.

        Args:
            timeout: Maximum time to wait for readings
            max_readings: Maximum number of readings to return

        Yields:
            SensorReading objects
        """
        count = 0
        start = time.time()

        while True:
            if max_readings and count >= max_readings:
                break

            if timeout and (time.time() - start) > timeout:
                break

            try:
                reading = self._queue.get(timeout=1.0)
                yield reading
                count += 1
            except queue.Empty:
                if not self._running:
                    break
                continue

    def register_callback(self, callback: Callable[[SensorReading], None]):
        """Register callback to be called for each new reading."""
        self._callbacks.append(callback)

    def generate_historical(
        self,
        building_id: str,
        meter_type: str = "electricity",
        duration_hours: float = 24.0,
        interval_minutes: float = 1.0
    ) -> List[SensorReading]:
        """
        Generate historical data (not real-time).

        Useful for populating buffers with historical context.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            duration_hours: How many hours of data to generate
            interval_minutes: Interval between readings

        Returns:
            List of historical readings
        """
        readings = []
        end_time = time.time()
        start_time = end_time - (duration_hours * 3600)
        interval_seconds = interval_minutes * 60

        current_time = start_time
        while current_time <= end_time:
            dt = datetime.fromtimestamp(current_time)
            value = self._generate_value(building_id, meter_type, dt)

            readings.append(SensorReading(
                timestamp=current_time,
                building_id=building_id,
                meter_type=meter_type,
                value=value,
                metadata={
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                    "historical": True
                }
            ))

            current_time += interval_seconds

        return readings

    def simulate_realtime(
        self,
        duration_seconds: float,
        callback: Optional[Callable[[SensorReading], None]] = None
    ) -> List[SensorReading]:
        """
        Simulate real-time data generation for a fixed duration.

        This runs synchronously (blocking) and returns all generated readings.

        Args:
            duration_seconds: How long to run simulation
            callback: Optional callback for each reading

        Returns:
            All generated readings
        """
        all_readings = []
        start = time.time()

        while (time.time() - start) < duration_seconds:
            readings = self._generate_batch()
            all_readings.extend(readings)

            if callback:
                for reading in readings:
                    callback(reading)

            # Simulate real-time delay (scaled down for testing)
            time.sleep(min(self.config.interval_seconds, 0.1))

        return all_readings
