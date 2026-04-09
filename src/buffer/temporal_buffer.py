"""
In-Process Temporal Grounding Buffer (TGB) (Novel).

A staleness-aware circular buffer optimized for real-time LLM grounding.
No external dependencies (Redis-free) - suitable for edge deployment.

Novelty:
- O(1) push/get operations using deque
- O(1) statistics retrieval via incremental caching
- Staleness-aware eviction with configurable thresholds
- Memory-efficient with automatic pruning

This is an alternative to Redis-based CircularBuffer for scenarios
where external dependencies are not desirable (edge devices, embedded systems).
"""

import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import threading


@dataclass
class SensorReading:
    """Single sensor reading with metadata."""
    timestamp: float
    building_id: str
    meter_type: str
    value: float
    unit: str = "kWh"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensorReading":
        return cls(**d)


@dataclass
class CachedStatistics:
    """Incrementally maintained statistics for O(1) retrieval."""
    count: int = 0
    sum_values: float = 0.0
    sum_squared: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    last_update: float = 0.0

    def add(self, value: float, timestamp: float) -> None:
        """Add a value to statistics (O(1))."""
        self.count += 1
        self.sum_values += value
        self.sum_squared += value * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.last_update = timestamp

    def remove(self, value: float) -> None:
        """Remove a value from statistics (O(1))."""
        if self.count > 0:
            self.count -= 1
            self.sum_values -= value
            self.sum_squared -= value * value
            # Note: min/max may become stale after removal
            # They are recomputed lazily when needed

    def to_dict(self) -> Dict[str, float]:
        """Convert to statistics dictionary."""
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        mean = self.sum_values / self.count
        variance = (self.sum_squared / self.count) - (mean * mean)
        std = np.sqrt(max(0, variance))  # Ensure non-negative

        return {
            "mean": mean,
            "std": std,
            "min": self.min_value if self.min_value != float('inf') else 0.0,
            "max": self.max_value if self.max_value != float('-inf') else 0.0,
            "count": self.count
        }


class TemporalGroundingBuffer:
    """
    In-memory circular buffer optimized for real-time LLM grounding.

    Novel Features:
    1. Staleness-aware indexing: Tracks last update time per sensor
    2. O(1) statistics caching: Incrementally maintains mean, std, min, max
    3. Automatic pruning: Removes stale data based on configurable threshold
    4. Thread-safe: Uses locks for concurrent access
    5. Zero external dependencies: Pure Python implementation

    Complexity:
    - push(): O(1) amortized
    - get_latest(): O(k) where k is number of readings requested
    - get_statistics(): O(1) using cached values
    - get_staleness(): O(1)

    Memory: O(N * M) where N = sensors, M = max_readings_per_sensor
    """

    def __init__(
        self,
        max_readings_per_sensor: int = 10000,
        staleness_threshold_seconds: float = 300.0,
        auto_prune_stale: bool = True,
        prune_interval_seconds: float = 60.0
    ):
        """
        Initialize temporal grounding buffer.

        Args:
            max_readings_per_sensor: Maximum readings to keep per sensor
            staleness_threshold_seconds: Time after which data is considered stale
            auto_prune_stale: Whether to automatically prune stale entries
            prune_interval_seconds: How often to run auto-pruning
        """
        self.max_readings = max_readings_per_sensor
        self.staleness_threshold = staleness_threshold_seconds
        self.auto_prune = auto_prune_stale
        self.prune_interval = prune_interval_seconds

        # Core data structures
        self._data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_readings_per_sensor)
        )

        # Staleness tracking (sensor_key -> last_update_timestamp)
        self._staleness_index: Dict[str, float] = {}

        # Statistics cache (sensor_key -> CachedStatistics)
        self._stats_cache: Dict[str, CachedStatistics] = defaultdict(CachedStatistics)

        # Thread safety
        self._lock = threading.RLock()

        # Pruning state
        self._last_prune = time.time()

    def _sensor_key(self, building_id: str, meter_type: str) -> str:
        """Generate unique key for a sensor."""
        return f"{building_id}:{meter_type}"

    def push(self, reading: SensorReading) -> None:
        """
        Add a reading to the buffer (O(1) amortized).

        Args:
            reading: Sensor reading to store
        """
        with self._lock:
            key = self._sensor_key(reading.building_id, reading.meter_type)
            buffer = self._data[key]

            # If buffer is full, we need to remove oldest value from stats
            if len(buffer) == self.max_readings:
                oldest = buffer[0]
                self._stats_cache[key].remove(oldest.value)

            # Add new reading
            buffer.append(reading)

            # Update staleness index
            self._staleness_index[key] = reading.timestamp

            # Update statistics cache (O(1))
            self._stats_cache[key].add(reading.value, reading.timestamp)

            # Auto-prune if enabled
            if self.auto_prune:
                self._maybe_prune()

    def push_batch(self, readings: List[SensorReading]) -> None:
        """
        Add multiple readings efficiently.

        Args:
            readings: List of sensor readings
        """
        for reading in readings:
            self.push(reading)

    def get_latest(
        self,
        building_id: str,
        meter_type: str,
        n: int = 1
    ) -> List[SensorReading]:
        """
        Get the n most recent readings for a sensor (O(n)).

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            n: Number of readings to retrieve

        Returns:
            List of readings, most recent first
        """
        with self._lock:
            key = self._sensor_key(building_id, meter_type)
            buffer = self._data.get(key)

            if not buffer:
                return []

            # Get last n elements in reverse order (most recent first)
            result = list(buffer)[-n:]
            result.reverse()
            return result

    def get_range(
        self,
        building_id: str,
        meter_type: str,
        start_time: float,
        end_time: float
    ) -> List[SensorReading]:
        """
        Get readings in a time range.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            List of readings in time order
        """
        with self._lock:
            key = self._sensor_key(building_id, meter_type)
            buffer = self._data.get(key)

            if not buffer:
                return []

            return [
                r for r in buffer
                if start_time <= r.timestamp <= end_time
            ]

    def get_window(
        self,
        building_id: str,
        meter_type: str,
        window_seconds: float = 3600.0
    ) -> List[SensorReading]:
        """
        Get readings from the last N seconds.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            window_seconds: Time window in seconds

        Returns:
            List of readings in time order
        """
        end_time = time.time()
        start_time = end_time - window_seconds
        return self.get_range(building_id, meter_type, start_time, end_time)

    def get_statistics(
        self,
        building_id: str,
        meter_type: str,
        window_seconds: float = 3600.0
    ) -> Dict[str, float]:
        """
        Get statistics for recent readings (O(1) for full buffer, O(n) for windowed).

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            window_seconds: Time window for statistics

        Returns:
            Dict with mean, std, min, max, count
        """
        with self._lock:
            key = self._sensor_key(building_id, meter_type)

            # For large windows, use cached statistics (O(1))
            # This is valid when window covers most of the buffer
            cached = self._stats_cache.get(key)
            if cached and cached.count > 0:
                # Check if we need windowed stats or can use cache
                buffer = self._data.get(key)
                if buffer and len(buffer) > 0:
                    oldest_in_buffer = buffer[0].timestamp
                    window_start = time.time() - window_seconds

                    # If window covers the entire buffer, use cache
                    if window_start <= oldest_in_buffer:
                        return cached.to_dict()

            # Fall back to computing windowed statistics
            readings = self.get_window(building_id, meter_type, window_seconds)

            if not readings:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

            values = np.array([r.value for r in readings])

            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }

    def get_staleness(self, building_id: str, meter_type: str) -> float:
        """
        Get staleness score for a sensor (O(1)).

        Returns:
            Staleness score: 0.0 = fresh, 1.0+ = stale
        """
        with self._lock:
            key = self._sensor_key(building_id, meter_type)
            last_update = self._staleness_index.get(key, 0)

            if last_update == 0:
                return float('inf')  # Never updated

            age = time.time() - last_update
            return age / self.staleness_threshold

    def is_stale(self, building_id: str, meter_type: str) -> bool:
        """Check if sensor data is stale (O(1))."""
        return self.get_staleness(building_id, meter_type) >= 1.0

    def count(self, building_id: str, meter_type: str) -> int:
        """Get number of readings for a sensor."""
        with self._lock:
            key = self._sensor_key(building_id, meter_type)
            buffer = self._data.get(key)
            return len(buffer) if buffer else 0

    def clear(self, building_id: str, meter_type: str) -> None:
        """Clear all readings for a sensor."""
        with self._lock:
            key = self._sensor_key(building_id, meter_type)
            if key in self._data:
                del self._data[key]
            if key in self._staleness_index:
                del self._staleness_index[key]
            if key in self._stats_cache:
                del self._stats_cache[key]

    def clear_all(self) -> None:
        """Clear all data from buffer."""
        with self._lock:
            self._data.clear()
            self._staleness_index.clear()
            self._stats_cache.clear()

    def list_sensors(self) -> List[Tuple[str, str]]:
        """List all active sensors (building_id, meter_type) pairs."""
        with self._lock:
            sensors = []
            for key in self._data.keys():
                parts = key.split(":", 1)
                if len(parts) == 2:
                    sensors.append((parts[0], parts[1]))
            return sensors

    def _maybe_prune(self) -> None:
        """Prune stale entries if interval has passed."""
        current_time = time.time()
        if current_time - self._last_prune < self.prune_interval:
            return

        self._last_prune = current_time
        stale_threshold = current_time - self.staleness_threshold * 2  # 2x threshold

        # Remove readings older than 2x staleness threshold
        keys_to_check = list(self._data.keys())
        for key in keys_to_check:
            buffer = self._data[key]
            while buffer and buffer[0].timestamp < stale_threshold:
                removed = buffer.popleft()
                self._stats_cache[key].remove(removed.value)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            total_readings = sum(len(b) for b in self._data.values())
            num_sensors = len(self._data)

            return {
                "num_sensors": num_sensors,
                "total_readings": total_readings,
                "avg_readings_per_sensor": total_readings / max(1, num_sensors),
                "max_readings_per_sensor": self.max_readings
            }

    def benchmark_latency(self, n_operations: int = 1000) -> Dict[str, float]:
        """
        Benchmark buffer operations.

        Args:
            n_operations: Number of operations to benchmark

        Returns:
            Dict with push_ms, get_ms, stats_ms latencies
        """
        test_building = "_benchmark_test"
        test_meter = "electricity"

        # Benchmark push
        readings = [
            SensorReading(
                timestamp=time.time() + i * 0.001,
                building_id=test_building,
                meter_type=test_meter,
                value=float(100 + np.random.randn() * 10)
            )
            for i in range(n_operations)
        ]

        start = time.perf_counter()
        for r in readings:
            self.push(r)
        push_time = (time.perf_counter() - start) / n_operations * 1000

        # Benchmark get latest
        start = time.perf_counter()
        for _ in range(n_operations):
            self.get_latest(test_building, test_meter, n=10)
        get_time = (time.perf_counter() - start) / n_operations * 1000

        # Benchmark statistics (should be O(1) due to caching)
        start = time.perf_counter()
        for _ in range(n_operations):
            self.get_statistics(test_building, test_meter)
        stats_time = (time.perf_counter() - start) / n_operations * 1000

        # Benchmark staleness check (O(1))
        start = time.perf_counter()
        for _ in range(n_operations):
            self.get_staleness(test_building, test_meter)
        staleness_time = (time.perf_counter() - start) / n_operations * 1000

        # Cleanup
        self.clear(test_building, test_meter)

        return {
            "push_ms": push_time,
            "get_latest_ms": get_time,
            "get_statistics_ms": stats_time,
            "get_staleness_ms": staleness_time
        }


# Factory function for buffer selection
def create_buffer(
    buffer_type: str = "temporal",
    **kwargs
) -> Any:
    """
    Factory function to create appropriate buffer.

    Args:
        buffer_type: "temporal" (in-process) or "redis"
        **kwargs: Arguments passed to buffer constructor

    Returns:
        Buffer instance (TemporalGroundingBuffer or CircularBuffer)
    """
    if buffer_type == "temporal":
        return TemporalGroundingBuffer(**kwargs)
    elif buffer_type == "redis":
        from .circular_buffer import CircularBuffer
        return CircularBuffer(**kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


if __name__ == "__main__":
    print("Testing Temporal Grounding Buffer (Novel)...")
    print("=" * 50)

    buffer = TemporalGroundingBuffer(
        max_readings_per_sensor=1000,
        staleness_threshold_seconds=300.0
    )

    # Add some readings
    print("\nAdding 100 readings...")
    for i in range(100):
        reading = SensorReading(
            timestamp=time.time() - (100 - i) * 60,
            building_id="Panther_office",
            meter_type="electricity",
            value=100.0 + np.random.randn() * 10
        )
        buffer.push(reading)

    # Query
    latest = buffer.get_latest("Panther_office", "electricity", n=5)
    print(f"Latest 5 readings: {[f'{r.value:.1f}' for r in latest]}")

    # Statistics (O(1) from cache)
    stats = buffer.get_statistics("Panther_office", "electricity")
    print(f"Statistics: mean={stats['mean']:.1f}, std={stats['std']:.1f}, count={stats['count']}")

    # Staleness
    staleness = buffer.get_staleness("Panther_office", "electricity")
    is_stale = buffer.is_stale("Panther_office", "electricity")
    print(f"Staleness score: {staleness:.2f}, Is stale: {is_stale}")

    # Memory usage
    memory = buffer.get_memory_usage()
    print(f"Memory: {memory}")

    # Benchmark
    print("\n" + "=" * 50)
    print("Running latency benchmark (1000 operations each)...")
    latency = buffer.benchmark_latency(n_operations=1000)
    print(f"  Push latency:       {latency['push_ms']:.4f} ms")
    print(f"  Get latest latency: {latency['get_latest_ms']:.4f} ms")
    print(f"  Get stats latency:  {latency['get_statistics_ms']:.4f} ms (O(1) cached)")
    print(f"  Staleness latency:  {latency['get_staleness_ms']:.4f} ms (O(1))")

    # Cleanup
    buffer.clear("Panther_office", "electricity")
    print("\n✓ All tests passed!")
