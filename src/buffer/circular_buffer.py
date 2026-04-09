"""
Redis-based Circular Buffer for Real-Time Sensor Data.

Provides <10ms retrieval latency for temporal grounding.
Uses Redis sorted sets with timestamp scores for efficient range queries.
"""

import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np

try:
    import redis
except ImportError:
    raise ImportError("redis package required: pip install redis")


@dataclass
class SensorReading:
    """Single sensor reading with metadata."""
    timestamp: float
    building_id: str
    meter_type: str  # electricity, chilledwater, steam, hotwater, etc.
    value: float
    unit: str = "kWh"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensorReading":
        return cls(**d)


class CircularBuffer:
    """
    Redis-backed circular buffer for sensor readings.

    Design:
    - Uses Redis sorted sets with timestamp as score
    - Automatic expiry for old data (configurable TTL)
    - Supports range queries by time window
    - O(log N) insertion and O(log N + M) range retrieval

    Key structure:
    - sensor:{building_id}:{meter_type} -> sorted set of readings
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_readings_per_sensor: int = 10000,
        ttl_seconds: int = 86400,  # 24 hours default
        key_prefix: str = "tgp"
    ):
        """
        Initialize buffer connection.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_readings_per_sensor: Max readings to keep per sensor
            ttl_seconds: Time-to-live for readings
            key_prefix: Prefix for all Redis keys
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.max_readings = max_readings_per_sensor
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

        # Verify connection
        try:
            self.client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis at {host}:{port}: {e}")

    def _sensor_key(self, building_id: str, meter_type: str) -> str:
        """Generate Redis key for a sensor."""
        return f"{self.key_prefix}:sensor:{building_id}:{meter_type}"

    def push(self, reading: SensorReading) -> None:
        """
        Add a reading to the buffer.

        Args:
            reading: Sensor reading to store
        """
        key = self._sensor_key(reading.building_id, reading.meter_type)

        # Store as JSON with timestamp as score
        data = json.dumps(reading.to_dict())
        self.client.zadd(key, {data: reading.timestamp})

        # Trim to max size (keep most recent)
        self.client.zremrangebyrank(key, 0, -self.max_readings - 1)

        # Set TTL on key
        self.client.expire(key, self.ttl_seconds)

    def push_batch(self, readings: List[SensorReading]) -> None:
        """
        Add multiple readings efficiently using pipeline.

        Args:
            readings: List of sensor readings
        """
        if not readings:
            return

        pipe = self.client.pipeline()

        # Group by sensor key
        by_key: Dict[str, List[tuple]] = {}
        for r in readings:
            key = self._sensor_key(r.building_id, r.meter_type)
            if key not in by_key:
                by_key[key] = []
            by_key[key].append((json.dumps(r.to_dict()), r.timestamp))

        # Batch operations
        for key, items in by_key.items():
            mapping = {data: score for data, score in items}
            pipe.zadd(key, mapping)
            pipe.zremrangebyrank(key, 0, -self.max_readings - 1)
            pipe.expire(key, self.ttl_seconds)

        pipe.execute()

    def get_latest(
        self,
        building_id: str,
        meter_type: str,
        n: int = 1
    ) -> List[SensorReading]:
        """
        Get the n most recent readings for a sensor.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            n: Number of readings to retrieve

        Returns:
            List of readings, most recent first
        """
        key = self._sensor_key(building_id, meter_type)
        items = self.client.zrevrange(key, 0, n - 1)

        return [SensorReading.from_dict(json.loads(item)) for item in items]

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
        key = self._sensor_key(building_id, meter_type)
        items = self.client.zrangebyscore(key, start_time, end_time)

        return [SensorReading.from_dict(json.loads(item)) for item in items]

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
        Compute statistics for recent readings.

        Args:
            building_id: Building identifier
            meter_type: Type of meter
            window_seconds: Time window for statistics

        Returns:
            Dict with mean, std, min, max, count
        """
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

    def count(self, building_id: str, meter_type: str) -> int:
        """Get number of readings for a sensor."""
        key = self._sensor_key(building_id, meter_type)
        return self.client.zcard(key)

    def clear(self, building_id: str, meter_type: str) -> None:
        """Clear all readings for a sensor."""
        key = self._sensor_key(building_id, meter_type)
        self.client.delete(key)

    def clear_all(self) -> None:
        """Clear all TGP data from Redis."""
        pattern = f"{self.key_prefix}:*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    def list_sensors(self) -> List[tuple]:
        """List all active sensors (building_id, meter_type) pairs."""
        pattern = f"{self.key_prefix}:sensor:*"
        keys = self.client.keys(pattern)

        sensors = []
        prefix_len = len(f"{self.key_prefix}:sensor:")
        for key in keys:
            parts = key[prefix_len:].split(":")
            if len(parts) == 2:
                sensors.append((parts[0], parts[1]))

        return sensors

    def benchmark_latency(self, n_operations: int = 1000) -> Dict[str, float]:
        """
        Benchmark buffer operations.

        Args:
            n_operations: Number of operations to benchmark

        Returns:
            Dict with push_ms, get_ms, range_ms latencies
        """
        test_building = "_benchmark_test"
        test_meter = "electricity"

        # Benchmark push
        readings = [
            SensorReading(
                timestamp=time.time() + i * 0.001,
                building_id=test_building,
                meter_type=test_meter,
                value=float(i)
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

        # Benchmark range query
        start = time.perf_counter()
        for _ in range(n_operations):
            self.get_window(test_building, test_meter, window_seconds=60.0)
        range_time = (time.perf_counter() - start) / n_operations * 1000

        # Cleanup
        self.clear(test_building, test_meter)

        return {
            "push_ms": push_time,
            "get_latest_ms": get_time,
            "get_range_ms": range_time
        }


if __name__ == "__main__":
    # Quick test
    buffer = CircularBuffer()

    # Add some readings
    for i in range(100):
        reading = SensorReading(
            timestamp=time.time() + i * 0.1,
            building_id="Panther_office_Leigh",
            meter_type="electricity",
            value=100.0 + np.random.randn() * 10
        )
        buffer.push(reading)

    # Query
    latest = buffer.get_latest("Panther_office_Leigh", "electricity", n=5)
    print(f"Latest 5 readings: {[r.value for r in latest]}")

    # Statistics
    stats = buffer.get_statistics("Panther_office_Leigh", "electricity", window_seconds=60.0)
    print(f"Statistics: {stats}")

    # Benchmark
    print("\nRunning latency benchmark...")
    latency = buffer.benchmark_latency(n_operations=500)
    print(f"Push latency: {latency['push_ms']:.3f} ms")
    print(f"Get latest latency: {latency['get_latest_ms']:.3f} ms")
    print(f"Range query latency: {latency['get_range_ms']:.3f} ms")

    # Cleanup
    buffer.clear("Panther_office_Leigh", "electricity")
