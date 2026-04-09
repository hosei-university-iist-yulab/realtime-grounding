"""
Unit tests for the circular buffer module.

Run with: pytest tests/test_buffer.py -v
"""

import time
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.buffer import CircularBuffer, SensorReading


@pytest.fixture
def buffer():
    """Create a test buffer instance."""
    buf = CircularBuffer(
        host="localhost",
        port=6379,
        max_readings_per_sensor=100,
        ttl_seconds=60,
        key_prefix="test_tgp"
    )
    yield buf
    # Cleanup after tests
    buf.clear_all()


@pytest.fixture
def sample_reading():
    """Create a sample sensor reading."""
    return SensorReading(
        timestamp=time.time(),
        building_id="test_building",
        meter_type="electricity",
        value=150.0,
        unit="kWh"
    )


class TestSensorReading:
    """Tests for SensorReading dataclass."""

    def test_creation(self, sample_reading):
        """Test reading creation."""
        assert sample_reading.building_id == "test_building"
        assert sample_reading.meter_type == "electricity"
        assert sample_reading.value == 150.0

    def test_to_dict(self, sample_reading):
        """Test conversion to dictionary."""
        d = sample_reading.to_dict()
        assert "timestamp" in d
        assert d["building_id"] == "test_building"
        assert d["value"] == 150.0

    def test_from_dict(self, sample_reading):
        """Test creation from dictionary."""
        d = sample_reading.to_dict()
        reading = SensorReading.from_dict(d)
        assert reading.building_id == sample_reading.building_id
        assert reading.value == sample_reading.value


class TestCircularBuffer:
    """Tests for CircularBuffer."""

    def test_connection(self, buffer):
        """Test Redis connection."""
        # Should not raise
        buffer.client.ping()

    def test_push_single(self, buffer, sample_reading):
        """Test pushing a single reading."""
        buffer.push(sample_reading)

        # Verify it was stored
        latest = buffer.get_latest(
            sample_reading.building_id,
            sample_reading.meter_type,
            n=1
        )
        assert len(latest) == 1
        assert latest[0].value == sample_reading.value

    def test_push_batch(self, buffer):
        """Test batch push."""
        readings = [
            SensorReading(
                timestamp=time.time() + i,
                building_id="batch_test",
                meter_type="electricity",
                value=100.0 + i
            )
            for i in range(10)
        ]

        buffer.push_batch(readings)

        # Verify count
        count = buffer.count("batch_test", "electricity")
        assert count == 10

        # Cleanup
        buffer.clear("batch_test", "electricity")

    def test_get_latest(self, buffer):
        """Test getting latest readings."""
        building = "latest_test"

        # Push readings with increasing timestamps
        for i in range(20):
            reading = SensorReading(
                timestamp=time.time() + i * 0.01,
                building_id=building,
                meter_type="electricity",
                value=float(i)
            )
            buffer.push(reading)

        # Get latest 5
        latest = buffer.get_latest(building, "electricity", n=5)
        assert len(latest) == 5

        # Should be in reverse chronological order
        values = [r.value for r in latest]
        assert values == [19, 18, 17, 16, 15]

        buffer.clear(building, "electricity")

    def test_get_range(self, buffer):
        """Test time range queries."""
        building = "range_test"
        base_time = time.time()

        # Push readings over 10 seconds
        for i in range(10):
            reading = SensorReading(
                timestamp=base_time + i,
                building_id=building,
                meter_type="electricity",
                value=float(i)
            )
            buffer.push(reading)

        # Query middle range
        readings = buffer.get_range(
            building, "electricity",
            start_time=base_time + 3,
            end_time=base_time + 7
        )

        assert len(readings) == 5  # 3, 4, 5, 6, 7
        values = [r.value for r in readings]
        assert values == [3, 4, 5, 6, 7]

        buffer.clear(building, "electricity")

    def test_statistics(self, buffer):
        """Test statistics computation."""
        building = "stats_test"

        # Push readings
        for i in range(100):
            reading = SensorReading(
                timestamp=time.time() + i * 0.01,
                building_id=building,
                meter_type="electricity",
                value=100.0 + np.random.randn() * 10
            )
            buffer.push(reading)

        stats = buffer.get_statistics(building, "electricity", window_seconds=60.0)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == 100
        assert 90 < stats["mean"] < 110  # Should be around 100

        buffer.clear(building, "electricity")

    def test_max_readings_limit(self, buffer):
        """Test that buffer respects max readings limit."""
        building = "limit_test"

        # Buffer is configured for 100 max readings
        for i in range(150):
            reading = SensorReading(
                timestamp=time.time() + i * 0.001,
                building_id=building,
                meter_type="electricity",
                value=float(i)
            )
            buffer.push(reading)

        count = buffer.count(building, "electricity")
        assert count <= 100  # Should be trimmed to max

        buffer.clear(building, "electricity")

    def test_list_sensors(self, buffer):
        """Test listing active sensors."""
        # Add data for multiple sensors
        for building in ["building_a", "building_b"]:
            for meter in ["electricity", "gas"]:
                reading = SensorReading(
                    timestamp=time.time(),
                    building_id=building,
                    meter_type=meter,
                    value=100.0
                )
                buffer.push(reading)

        sensors = buffer.list_sensors()
        assert len(sensors) >= 4

        # Cleanup
        for building in ["building_a", "building_b"]:
            for meter in ["electricity", "gas"]:
                buffer.clear(building, meter)

    def test_clear(self, buffer, sample_reading):
        """Test clearing sensor data."""
        buffer.push(sample_reading)
        assert buffer.count(sample_reading.building_id, sample_reading.meter_type) > 0

        buffer.clear(sample_reading.building_id, sample_reading.meter_type)
        assert buffer.count(sample_reading.building_id, sample_reading.meter_type) == 0


class TestBufferPerformance:
    """Performance tests for buffer."""

    def test_latency_target(self, buffer):
        """Test that latency meets <10ms target."""
        latency = buffer.benchmark_latency(n_operations=100)

        # Should be well under 10ms
        assert latency["push_ms"] < 10.0, f"Push too slow: {latency['push_ms']}ms"
        assert latency["get_latest_ms"] < 10.0, f"Get too slow: {latency['get_latest_ms']}ms"
        assert latency["get_range_ms"] < 10.0, f"Range too slow: {latency['get_range_ms']}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
