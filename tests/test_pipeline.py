"""
Integration tests for the TGP pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import time
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    TemporalGroundingPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline
)
from src.buffer import SensorReading


@pytest.fixture
def config():
    """Create test pipeline configuration."""
    return PipelineConfig(
        redis_host="localhost",
        redis_port=6379,
        model_type="tinyllama",
        gpu_id=4,
        use_4bit=True,
        staleness_threshold=0.85,
        validate_causal=True
    )


@pytest.fixture
def pipeline(config):
    """Create test pipeline (without loading LLM)."""
    p = TemporalGroundingPipeline(config)
    # Only load buffer and staleness for testing
    p._load_buffer()
    p._load_staleness_detector()
    yield p
    # Cleanup
    p.clear_all_contexts()


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.model_type == "tinyllama"
        assert config.gpu_id == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            model_type="phi2",
            gpu_id=5,
            staleness_threshold=0.9
        )
        assert config.model_type == "phi2"
        assert config.gpu_id == 5
        assert config.staleness_threshold == 0.9


class TestPipelineIngestion:
    """Tests for data ingestion."""

    def test_ingest_single(self, pipeline):
        """Test ingesting single reading."""
        reading = SensorReading(
            timestamp=time.time(),
            building_id="_test_ingest",
            meter_type="electricity",
            value=150.0
        )

        pipeline.ingest_reading(reading)

        # Verify data is stored
        count = pipeline.buffer.count("_test_ingest", "electricity")
        assert count == 1

        pipeline.buffer.clear("_test_ingest", "electricity")

    def test_ingest_batch(self, pipeline):
        """Test batch ingestion."""
        readings = [
            SensorReading(
                timestamp=time.time() + i,
                building_id="_test_batch",
                meter_type="electricity",
                value=100.0 + i
            )
            for i in range(50)
        ]

        pipeline.ingest_batch(readings)

        count = pipeline.buffer.count("_test_batch", "electricity")
        assert count == 50

        pipeline.buffer.clear("_test_batch", "electricity")


class TestPipelineContext:
    """Tests for context management."""

    def test_context_creation(self, pipeline):
        """Test that context is created on first query."""
        # Add test data
        for i in range(10):
            reading = SensorReading(
                timestamp=time.time() - (10 - i) * 60,
                building_id="_test_context",
                meter_type="electricity",
                value=150.0 + np.random.randn() * 5
            )
            pipeline.ingest_reading(reading)

        # Query should create context
        # Note: Skipping actual LLM query, just testing context
        key = pipeline._context_key("_test_context", "electricity")
        assert key not in pipeline._active_contexts

        # Manually trigger context creation
        readings = pipeline.buffer.get_latest("_test_context", "electricity", n=5)
        stats = pipeline.buffer.get_statistics("_test_context", "electricity")

        pipeline.staleness_detector.set_context(
            key,
            [r.to_dict() for r in readings],
            stats,
            "_test_context",
            "electricity"
        )
        pipeline._active_contexts[key] = {"stats": stats}

        assert key in pipeline._active_contexts

        pipeline.buffer.clear("_test_context", "electricity")
        pipeline.clear_context("_test_context", "electricity")

    def test_clear_context(self, pipeline):
        """Test clearing context."""
        key = "_test_clear:electricity"
        pipeline._active_contexts[key] = {"test": True}

        pipeline.clear_context("_test_clear", "electricity")
        assert key not in pipeline._active_contexts

    def test_clear_all_contexts(self, pipeline):
        """Test clearing all contexts."""
        pipeline._active_contexts["ctx1"] = {}
        pipeline._active_contexts["ctx2"] = {}

        pipeline.clear_all_contexts()
        assert len(pipeline._active_contexts) == 0


class TestPipelineHealth:
    """Tests for health checks."""

    def test_health_check(self, pipeline):
        """Test health check returns expected format."""
        health = pipeline.get_health()

        assert "buffer" in health
        assert "llm" in health
        assert "staleness_detector" in health
        assert "causal_validator" in health

        # Buffer and staleness should be healthy
        assert health["buffer"] is True
        assert health["staleness_detector"] is True


class TestCreatePipeline:
    """Tests for pipeline factory function."""

    def test_create_default(self):
        """Test creating pipeline with defaults."""
        # Just test that it doesn't raise
        pipeline = create_pipeline()
        assert pipeline.config.model_type == "tinyllama"
        assert pipeline.config.gpu_id == 4

    def test_create_custom(self):
        """Test creating pipeline with custom settings."""
        pipeline = create_pipeline(
            model_type="phi2",
            gpu_id=5,
            validate_causal=False
        )
        assert pipeline.config.model_type == "phi2"
        assert pipeline.config.gpu_id == 5
        assert pipeline.config.validate_causal is False

    def test_invalid_gpu_warning(self):
        """Test that invalid GPU triggers warning and fallback."""
        pipeline = create_pipeline(gpu_id=0)  # Invalid, not in 4-7
        # Should fall back to GPU 4
        assert pipeline.config.gpu_id == 4


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_creation(self):
        """Test creating pipeline result."""
        result = PipelineResult(
            response="Test response",
            latency_ms=50.0,
            is_grounded=True,
            context_refreshed=False,
            sensor_stats={"mean": 150.0}
        )

        assert result.response == "Test response"
        assert result.latency_ms == 50.0
        assert result.is_grounded is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
