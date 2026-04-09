"""
Unit tests for the staleness detection module.

Run with: pytest tests/test_staleness.py -v
"""

import time
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.staleness import StalenessDetector, StalenessResult, AdaptiveStalenessDetector


@pytest.fixture
def detector():
    """Create a test detector instance."""
    return StalenessDetector(
        staleness_threshold=0.85,
        max_age_seconds=3600.0
    )


@pytest.fixture
def sample_context():
    """Create sample context data."""
    return {
        "readings": [{"value": 150.0, "timestamp": time.time()}],
        "statistics": {"mean": 150.0, "std": 10.0, "min": 130.0, "max": 170.0},
        "building_id": "test_building",
        "meter_type": "electricity"
    }


class TestStalenessDetector:
    """Tests for StalenessDetector."""

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.staleness_threshold == 0.85
        assert detector.max_age_seconds == 3600.0
        assert detector.encoder is not None

    def test_set_context(self, detector, sample_context):
        """Test setting context."""
        embedding = detector.set_context(
            "test_key",
            sample_context["readings"],
            sample_context["statistics"],
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        assert embedding is not None
        assert len(embedding) == detector.embedding_dim
        assert "test_key" in detector._context_cache

        detector.clear_context("test_key")

    def test_detect_fresh(self, detector, sample_context):
        """Test detection of fresh (not stale) data."""
        # Set context
        detector.set_context(
            "fresh_test",
            sample_context["readings"],
            sample_context["statistics"],
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        # Detect with similar data
        similar_stats = {"mean": 152.0, "std": 11.0, "min": 132.0, "max": 172.0}
        result = detector.detect(
            "fresh_test",
            [{"value": 152.0, "timestamp": time.time()}],
            similar_stats,
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        assert isinstance(result, StalenessResult)
        assert not result.is_stale
        assert result.similarity > detector.staleness_threshold

        detector.clear_context("fresh_test")

    def test_detect_stale(self, detector, sample_context):
        """Test detection of stale data."""
        # Set context
        detector.set_context(
            "stale_test",
            sample_context["readings"],
            sample_context["statistics"],
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        # Detect with very different data
        different_stats = {"mean": 50.0, "std": 5.0, "min": 40.0, "max": 60.0}
        result = detector.detect(
            "stale_test",
            [{"value": 50.0, "timestamp": time.time()}],
            different_stats,
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        assert result.is_stale
        assert result.similarity < detector.staleness_threshold

        detector.clear_context("stale_test")

    def test_detect_no_context(self, detector, sample_context):
        """Test detection when context doesn't exist."""
        result = detector.detect(
            "nonexistent_key",
            sample_context["readings"],
            sample_context["statistics"],
            sample_context["building_id"],
            sample_context["meter_type"]
        )

        assert result.is_stale
        assert result.staleness_score == 1.0
        assert "No context" in result.reason

    def test_context_age_staleness(self, detector, sample_context):
        """Test age-based staleness."""
        # Create detector with short max age
        short_age_detector = StalenessDetector(
            staleness_threshold=0.85,
            max_age_seconds=0.1  # 100ms
        )

        short_age_detector.set_context(
            "age_test",
            sample_context["readings"],
            sample_context["statistics"]
        )

        # Wait for context to age
        time.sleep(0.2)

        result = short_age_detector.detect(
            "age_test",
            sample_context["readings"],
            sample_context["statistics"]
        )

        assert result.is_stale
        assert "too old" in result.reason

        short_age_detector.clear_context("age_test")

    def test_update_context(self, detector, sample_context):
        """Test context update."""
        # Set initial context
        detector.set_context(
            "update_test",
            sample_context["readings"],
            sample_context["statistics"]
        )

        initial_age = detector.get_context_age("update_test")

        time.sleep(0.1)

        # Update context
        detector.update_context(
            "update_test",
            [{"value": 155.0}],
            {"mean": 155.0, "std": 12.0}
        )

        new_age = detector.get_context_age("update_test")

        assert new_age < initial_age

        detector.clear_context("update_test")

    def test_clear_context(self, detector, sample_context):
        """Test clearing context."""
        detector.set_context("clear_test", sample_context["readings"], sample_context["statistics"])
        assert "clear_test" in [k for k, _ in detector.list_contexts()]

        result = detector.clear_context("clear_test")
        assert result is True
        assert "clear_test" not in [k for k, _ in detector.list_contexts()]

    def test_clear_nonexistent_context(self, detector):
        """Test clearing nonexistent context."""
        result = detector.clear_context("nonexistent")
        assert result is False

    def test_list_contexts(self, detector, sample_context):
        """Test listing contexts."""
        # Add multiple contexts
        for i in range(3):
            detector.set_context(f"list_test_{i}", sample_context["readings"], sample_context["statistics"])

        contexts = detector.list_contexts()
        assert len(contexts) >= 3

        # Check format
        for key, age in contexts:
            assert isinstance(key, str)
            assert isinstance(age, float)
            assert age >= 0

        # Cleanup
        for i in range(3):
            detector.clear_context(f"list_test_{i}")


class TestAdaptiveStalenessDetector:
    """Tests for adaptive staleness detector."""

    def test_initialization(self):
        """Test adaptive detector initialization."""
        detector = AdaptiveStalenessDetector(
            initial_threshold=0.85,
            learning_rate=0.01
        )
        assert detector.staleness_threshold == 0.85
        assert detector.learning_rate == 0.01

    def test_threshold_adaptation(self):
        """Test that threshold adapts based on feedback."""
        detector = AdaptiveStalenessDetector(
            initial_threshold=0.85,
            learning_rate=0.05
        )

        # Set context
        detector.set_context("adapt_test", [{"value": 100}], {"mean": 100})

        initial_threshold = detector.staleness_threshold

        # Provide feedback that should increase threshold
        for _ in range(20):
            result = detector.detect("adapt_test", [{"value": 100}], {"mean": 100})
            # Simulate false negative feedback
            detector.adapt_threshold(feedback_is_stale=True, result=result)

        # Threshold should have increased
        assert detector.staleness_threshold > initial_threshold

        detector.clear_context("adapt_test")


class TestStalenessPerformance:
    """Performance tests for staleness detection."""

    def test_latency_target(self, detector):
        """Test that latency meets target."""
        latency = detector.benchmark_latency(n_runs=50)

        # Should be under 10ms for embedding-based detection
        assert latency["mean_ms"] < 10.0, f"Detection too slow: {latency['mean_ms']}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
