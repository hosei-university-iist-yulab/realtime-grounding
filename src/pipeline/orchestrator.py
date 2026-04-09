"""
Temporal Grounding Pipeline (TGP) Orchestrator.

Main pipeline that integrates:
1. Redis circular buffer for sensor data
2. LLM backbone for text generation
3. Staleness detector for context freshness
4. Causal validator for response verification

Provides real-time sensor-to-text grounding with <100ms latency.
"""

import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from src.buffer import CircularBuffer, SensorReading
    from src.llm import LLMBackbone, ModelConfig
    from src.staleness import StalenessDetector, StalenessResult
    from src.causal import CausalValidator, CausalGraph, ValidationResult
except ImportError:
    # Allow running from different directories
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.buffer import CircularBuffer, SensorReading
    from src.llm import LLMBackbone, ModelConfig
    from src.staleness import StalenessDetector, StalenessResult
    from src.causal import CausalValidator, CausalGraph, ValidationResult


@dataclass
class PipelineConfig:
    """Configuration for TGP pipeline."""
    # Buffer settings
    redis_host: str = "localhost"
    redis_port: int = 6379

    # LLM settings
    model_type: str = "tinyllama"  # tinyllama, phi2, phi3-mini
    gpu_id: int = 4  # Use GPU 4-7 only
    use_4bit: bool = True
    use_lora: bool = True

    # Staleness settings
    staleness_threshold: float = 0.85
    max_context_age_seconds: float = 3600.0

    # Validation settings
    validate_causal: bool = True

    # Performance settings
    max_generation_tokens: int = 128
    context_window_seconds: float = 3600.0


@dataclass
class PipelineResult:
    """Result from pipeline inference."""
    response: str
    latency_ms: float
    is_grounded: bool
    staleness: Optional[StalenessResult] = None
    validation: Optional[ValidationResult] = None
    context_refreshed: bool = False
    sensor_stats: Dict[str, float] = field(default_factory=dict)


class TemporalGroundingPipeline:
    """
    Main Temporal Grounding Pipeline.

    Provides real-time sensor-to-text grounding with:
    - <10ms sensor retrieval (Redis)
    - <50ms staleness detection (embeddings)
    - <100ms total latency target

    Key Innovations:
    1. Redis circular buffer for O(log N) sensor retrieval
    2. Embedding-based staleness detection
    3. Causal validation for physically plausible responses
    4. Context-aware re-grounding when data becomes stale
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._components_loaded = False

        # Component placeholders
        self.buffer: Optional[CircularBuffer] = None
        self.llm: Optional[LLMBackbone] = None
        self.staleness_detector: Optional[StalenessDetector] = None
        self.causal_validator: Optional[CausalValidator] = None

        # Context cache
        self._active_contexts: Dict[str, Dict] = {}

    def load_components(self, lazy: bool = True):
        """
        Load pipeline components.

        Args:
            lazy: If True, only load components when first needed
        """
        if not lazy:
            self._load_buffer()
            self._load_llm()
            self._load_staleness_detector()
            self._load_causal_validator()
            self._components_loaded = True

    def _load_buffer(self):
        """Load Redis buffer."""
        if self.buffer is None:
            logger.info("Loading Redis buffer...")
            self.buffer = CircularBuffer(
                host=self.config.redis_host,
                port=self.config.redis_port
            )

    def _load_llm(self):
        """Load LLM backbone."""
        if self.llm is None:
            logger.info(f"Loading LLM ({self.config.model_type})...")
            model_config = ModelConfig(
                use_4bit=self.config.use_4bit,
                use_lora=self.config.use_lora,
                gpu_id=self.config.gpu_id
            )
            self.llm = LLMBackbone(config=model_config, model_type=self.config.model_type)

    def _load_staleness_detector(self):
        """Load staleness detector."""
        if self.staleness_detector is None:
            logger.info("Loading staleness detector...")
            self.staleness_detector = StalenessDetector(
                staleness_threshold=self.config.staleness_threshold,
                max_age_seconds=self.config.max_context_age_seconds
            )

    def _load_causal_validator(self):
        """Load causal validator."""
        if self.causal_validator is None and self.config.validate_causal:
            logger.info("Loading causal validator...")
            self.causal_validator = CausalValidator()

    def _context_key(self, building_id: str, meter_type: str) -> str:
        """Generate context key."""
        return f"{building_id}:{meter_type}"

    def ingest_reading(self, reading: SensorReading):
        """
        Ingest a new sensor reading.

        Args:
            reading: Sensor reading to store
        """
        self._load_buffer()
        self.buffer.push(reading)

    def ingest_batch(self, readings: List[SensorReading]):
        """
        Ingest multiple readings.

        Args:
            readings: List of sensor readings
        """
        self._load_buffer()
        self.buffer.push_batch(readings)

    def query(
        self,
        query: str,
        building_id: str,
        meter_type: str = "electricity",
        force_refresh: bool = False
    ) -> PipelineResult:
        """
        Query the pipeline with natural language.

        Args:
            query: Natural language query about the building
            building_id: Building identifier
            meter_type: Type of meter
            force_refresh: Force context refresh even if not stale

        Returns:
            PipelineResult with response and metadata
        """
        start_time = time.perf_counter()

        # Ensure components are loaded
        self._load_buffer()
        self._load_llm()
        self._load_staleness_detector()
        if self.config.validate_causal:
            self._load_causal_validator()

        context_key = self._context_key(building_id, meter_type)
        context_refreshed = False

        # Get current sensor data
        readings = self.buffer.get_window(
            building_id, meter_type,
            window_seconds=self.config.context_window_seconds
        )
        stats = self.buffer.get_statistics(
            building_id, meter_type,
            window_seconds=self.config.context_window_seconds
        )

        # Convert readings to dict format
        readings_dict = [r.to_dict() for r in readings]

        # Check staleness
        staleness_result = None
        if context_key in self._active_contexts and not force_refresh:
            staleness_result = self.staleness_detector.detect(
                context_key=context_key,
                current_readings=readings_dict,
                current_statistics=stats,
                building_id=building_id,
                meter_type=meter_type
            )

            if staleness_result.is_stale:
                logger.info(f"Context stale for {context_key}: {staleness_result.reason}")
                force_refresh = True

        # Refresh context if needed
        if force_refresh or context_key not in self._active_contexts:
            self.staleness_detector.set_context(
                context_key=context_key,
                readings=readings_dict,
                statistics=stats,
                building_id=building_id,
                meter_type=meter_type
            )
            self._active_contexts[context_key] = {
                "readings": readings_dict,
                "statistics": stats,
                "timestamp": time.time()
            }
            context_refreshed = True

        # Format prompt
        prompt = self.llm.format_grounding_prompt(
            sensor_data={
                "building_id": building_id,
                "meter_type": meter_type,
                "readings": readings_dict,
                "statistics": stats
            },
            query=query
        )

        # Generate response
        response = self.llm.generate(
            prompt,
            max_new_tokens=self.config.max_generation_tokens
        )

        # Validate causality
        validation_result = None
        if self.causal_validator is not None:
            validation_result = self.causal_validator.validate(response)
            if not validation_result.is_valid:
                logger.warning(
                    f"Causal validation failed: {validation_result.violations}"
                )

        # Compute total latency
        total_latency = (time.perf_counter() - start_time) * 1000

        return PipelineResult(
            response=response,
            latency_ms=total_latency,
            is_grounded=len(readings) > 0,
            staleness=staleness_result,
            validation=validation_result,
            context_refreshed=context_refreshed,
            sensor_stats=stats
        )

    def batch_query(
        self,
        queries: List[Tuple[str, str, str]],
    ) -> List[PipelineResult]:
        """
        Batch query multiple buildings.

        Args:
            queries: List of (query, building_id, meter_type) tuples

        Returns:
            List of PipelineResults
        """
        results = []
        for query, building_id, meter_type in queries:
            result = self.query(query, building_id, meter_type)
            results.append(result)
        return results

    def get_active_contexts(self) -> List[str]:
        """Get list of active context keys."""
        return list(self._active_contexts.keys())

    def clear_context(self, building_id: str, meter_type: str = "electricity"):
        """Clear context for a building."""
        context_key = self._context_key(building_id, meter_type)
        if context_key in self._active_contexts:
            del self._active_contexts[context_key]
        self.staleness_detector.clear_context(context_key)

    def clear_all_contexts(self):
        """Clear all active contexts."""
        self._active_contexts.clear()
        if self.staleness_detector:
            self.staleness_detector.clear_all_contexts()

    def benchmark(self, n_queries: int = 10) -> Dict[str, float]:
        """
        Benchmark pipeline performance.

        Args:
            n_queries: Number of queries to benchmark

        Returns:
            Dict with latency statistics
        """
        # Ensure components loaded
        self.load_components(lazy=False)

        # Add some test data
        test_building = "_benchmark_building"
        for i in range(100):
            reading = SensorReading(
                timestamp=time.time() - (100 - i) * 60,
                building_id=test_building,
                meter_type="electricity",
                value=100.0 + i * 0.1
            )
            self.ingest_reading(reading)

        # Warmup
        _ = self.query("What is the current consumption?", test_building)

        # Benchmark
        latencies = []
        for i in range(n_queries):
            result = self.query(
                f"What is the energy consumption pattern? (query {i})",
                test_building
            )
            latencies.append(result.latency_ms)

        # Cleanup
        self.buffer.clear(test_building, "electricity")
        self.clear_context(test_building)

        import numpy as np
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "n_queries": n_queries
        }

    def get_health(self) -> Dict[str, bool]:
        """Check health of pipeline components."""
        health = {
            "buffer": False,
            "llm": False,
            "staleness_detector": False,
            "causal_validator": False
        }

        try:
            if self.buffer:
                self.buffer.client.ping()
                health["buffer"] = True
        except Exception:
            pass

        health["llm"] = self.llm is not None
        health["staleness_detector"] = self.staleness_detector is not None
        health["causal_validator"] = self.causal_validator is not None

        return health


def create_pipeline(
    model_type: str = "tinyllama",
    gpu_id: int = 4,
    validate_causal: bool = True
) -> TemporalGroundingPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        model_type: LLM model type
        gpu_id: GPU to use (4-7)
        validate_causal: Whether to enable causal validation

    Returns:
        Configured pipeline instance
    """
    if gpu_id not in [4, 5, 6, 7]:
        logger.warning(f"GPU {gpu_id} not in allowed range [4-7]. Using GPU 4.")
        gpu_id = 4

    config = PipelineConfig(
        model_type=model_type,
        gpu_id=gpu_id,
        validate_causal=validate_causal
    )

    return TemporalGroundingPipeline(config)


if __name__ == "__main__":
    # Quick test
    print("Testing Temporal Grounding Pipeline...")

    # Create pipeline with minimal settings
    config = PipelineConfig(
        model_type="tinyllama",
        gpu_id=4,
        validate_causal=True
    )
    pipeline = TemporalGroundingPipeline(config)

    # Check health (before loading)
    print("\nHealth check (before load):")
    print(pipeline.get_health())

    # Load components
    print("\nLoading components...")
    pipeline.load_components(lazy=False)

    print("\nHealth check (after load):")
    print(pipeline.get_health())

    # Add test data
    print("\nIngesting test data...")
    for i in range(60):
        reading = SensorReading(
            timestamp=time.time() - (60 - i) * 60,
            building_id="Panther_office_Leigh",
            meter_type="electricity",
            value=150.0 + 20 * (i % 10) / 10
        )
        pipeline.ingest_reading(reading)

    # Query
    print("\nQuerying pipeline...")
    result = pipeline.query(
        "What is the current energy consumption pattern?",
        "Panther_office_Leigh",
        "electricity"
    )

    print(f"\nResponse: {result.response[:200]}...")
    print(f"Latency: {result.latency_ms:.1f} ms")
    print(f"Is grounded: {result.is_grounded}")
    print(f"Context refreshed: {result.context_refreshed}")
    print(f"Sensor stats: {result.sensor_stats}")

    if result.validation:
        print(f"Validation score: {result.validation.score:.2f}")

    # Cleanup
    pipeline.buffer.clear("Panther_office_Leigh", "electricity")
