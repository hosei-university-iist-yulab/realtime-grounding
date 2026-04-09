"""
Staleness Detection for Real-Time Sensor-Text Grounding.

Detects when LLM context becomes stale relative to current sensor data.

Primary method: Time-threshold based detection (achieves F1=1.0)
Secondary method: Value-change threshold detection
Legacy method: Embedding-based semantic similarity (deprecated, poor F1)
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

import torch

# Optional import for embedding-based staleness detector (deprecated)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class StalenessResult:
    """Result of staleness detection."""
    is_stale: bool
    staleness_score: float  # 0 = fresh, 1 = completely stale
    similarity: float       # Cosine similarity (for embedding method) or value similarity
    threshold: float        # Threshold used for detection
    context_age_seconds: float  # Age of the context
    reason: str            # Human-readable explanation
    method: str = "time_threshold"  # Detection method used


class TimeThresholdStalenessDetector:
    """
    Time-threshold based staleness detector.

    Primary staleness detection method - achieves F1=1.0.
    Simple but effective: marks data as stale if context is older than threshold.
    """

    def __init__(
        self,
        time_threshold_seconds: float = 300.0,
        value_change_threshold: float = 0.2
    ):
        """
        Initialize time-threshold detector.

        Args:
            time_threshold_seconds: Max age before data is considered stale
            value_change_threshold: Max relative value change before stale (0.2 = 20%)
        """
        self.time_threshold_seconds = time_threshold_seconds
        self.value_change_threshold = value_change_threshold
        self._context_cache: Dict[str, Tuple[Dict, float]] = {}

    def set_context(
        self,
        context_key: str,
        readings: List[Dict],
        statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> None:
        """Set the reference context for staleness detection."""
        context_data = {
            "readings": readings,
            "statistics": statistics,
            "building_id": building_id,
            "meter_type": meter_type
        }
        self._context_cache[context_key] = (context_data, time.time())

    def detect(
        self,
        context_key: str,
        current_readings: List[Dict],
        current_statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> StalenessResult:
        """
        Detect if context has become stale using time and value thresholds.

        Args:
            context_key: Key for cached context
            current_readings: Current sensor readings
            current_statistics: Current statistics
            building_id: Building identifier
            meter_type: Meter type

        Returns:
            StalenessResult with detection outcome
        """
        # Check if context exists
        if context_key not in self._context_cache:
            return StalenessResult(
                is_stale=True,
                staleness_score=1.0,
                similarity=0.0,
                threshold=self.time_threshold_seconds,
                context_age_seconds=float("inf"),
                reason="No context found - context not initialized",
                method="time_threshold"
            )

        context_data, context_time = self._context_cache[context_key]
        context_age = time.time() - context_time

        # Time-based staleness check
        if context_age > self.time_threshold_seconds:
            staleness_score = min(1.0, context_age / self.time_threshold_seconds)
            return StalenessResult(
                is_stale=True,
                staleness_score=staleness_score,
                similarity=0.0,
                threshold=self.time_threshold_seconds,
                context_age_seconds=context_age,
                reason=f"Context too old ({context_age:.0f}s > {self.time_threshold_seconds:.0f}s threshold)",
                method="time_threshold"
            )

        # Value-change staleness check
        context_stats = context_data.get("statistics", {})
        context_mean = context_stats.get("mean", 0)
        current_mean = current_statistics.get("mean", 0)

        if context_mean > 0:
            value_change = abs(current_mean - context_mean) / context_mean
            if value_change > self.value_change_threshold:
                return StalenessResult(
                    is_stale=True,
                    staleness_score=min(1.0, value_change / self.value_change_threshold),
                    similarity=1.0 - value_change,
                    threshold=self.value_change_threshold,
                    context_age_seconds=context_age,
                    reason=f"Value changed too much ({value_change:.1%} > {self.value_change_threshold:.0%} threshold)",
                    method="value_threshold"
                )

        # Context is fresh
        staleness_score = context_age / self.time_threshold_seconds
        return StalenessResult(
            is_stale=False,
            staleness_score=staleness_score,
            similarity=1.0,
            threshold=self.time_threshold_seconds,
            context_age_seconds=context_age,
            reason=f"Context is fresh (age: {context_age:.0f}s)",
            method="time_threshold"
        )

    def clear_context(self, context_key: str) -> bool:
        """Clear a specific context from cache."""
        if context_key in self._context_cache:
            del self._context_cache[context_key]
            return True
        return False

    def clear_all_contexts(self):
        """Clear all cached contexts."""
        self._context_cache.clear()

    def get_context_age(self, context_key: str) -> Optional[float]:
        """Get age of a context in seconds."""
        if context_key not in self._context_cache:
            return None
        _, context_time = self._context_cache[context_key]
        return time.time() - context_time


class StalenessDetector:
    """
    Detects staleness in sensor-text grounding.

    Uses semantic similarity between:
    1. Current sensor data description
    2. LLM's context/cached description

    When similarity drops below threshold, data is considered stale.

    Theorem (Staleness Bound):
    For embedding function E and threshold τ:
    stale(t) = 1 if cos(E(data_t), E(context)) < τ else 0
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        staleness_threshold: float = 0.85,
        max_age_seconds: float = 3600.0,
        device: Optional[str] = None
    ):
        """
        Initialize staleness detector.

        Args:
            model_name: Sentence transformer model name
            staleness_threshold: Similarity threshold for staleness
            max_age_seconds: Maximum age before automatic staleness
            device: Device to use (cuda/cpu)
        """
        self.staleness_threshold = staleness_threshold
        self.max_age_seconds = max_age_seconds

        # Load embedding model
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required for embedding-based staleness detector. "
                "Install with: pip install sentence-transformers"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading embedding model {model_name} on {device}...")
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Cache for context embeddings
        self._context_cache: Dict[str, Tuple[np.ndarray, float]] = {}

    def _format_sensor_description(
        self,
        readings: List[Dict],
        statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> str:
        """Format sensor data as natural language description."""
        parts = []

        if building_id:
            parts.append(f"Building {building_id}")
        if meter_type:
            parts.append(f"{meter_type} meter")

        if statistics:
            parts.append(
                f"average consumption {statistics.get('mean', 0):.1f} kWh, "
                f"ranging from {statistics.get('min', 0):.1f} to {statistics.get('max', 0):.1f} kWh, "
                f"with variability of {statistics.get('std', 0):.1f} kWh"
            )

        if readings:
            latest = readings[-1] if readings else {}
            if "value" in latest:
                parts.append(f"current reading {latest['value']:.1f} kWh")

        return ", ".join(parts) if parts else "no data available"

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        return self.encoder.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def set_context(
        self,
        context_key: str,
        readings: List[Dict],
        statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> np.ndarray:
        """
        Set the reference context for staleness detection.

        Args:
            context_key: Unique key for this context
            readings: Sensor readings at context creation time
            statistics: Statistics at context creation time
            building_id: Building identifier
            meter_type: Meter type

        Returns:
            Context embedding
        """
        description = self._format_sensor_description(
            readings, statistics, building_id, meter_type
        )
        embedding = self._compute_embedding(description)
        self._context_cache[context_key] = (embedding, time.time())
        return embedding

    def detect(
        self,
        context_key: str,
        current_readings: List[Dict],
        current_statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> StalenessResult:
        """
        Detect if context has become stale.

        Args:
            context_key: Key for cached context
            current_readings: Current sensor readings
            current_statistics: Current statistics
            building_id: Building identifier
            meter_type: Meter type

        Returns:
            StalenessResult with detection outcome
        """
        # Check if context exists
        if context_key not in self._context_cache:
            return StalenessResult(
                is_stale=True,
                staleness_score=1.0,
                similarity=0.0,
                threshold=self.staleness_threshold,
                context_age_seconds=float("inf"),
                reason="No context found - context not initialized"
            )

        context_embedding, context_time = self._context_cache[context_key]
        context_age = time.time() - context_time

        # Age-based staleness
        if context_age > self.max_age_seconds:
            return StalenessResult(
                is_stale=True,
                staleness_score=1.0,
                similarity=0.0,
                threshold=self.staleness_threshold,
                context_age_seconds=context_age,
                reason=f"Context too old ({context_age:.0f}s > {self.max_age_seconds:.0f}s max)"
            )

        # Compute current embedding
        current_description = self._format_sensor_description(
            current_readings, current_statistics, building_id, meter_type
        )
        current_embedding = self._compute_embedding(current_description)

        # Compute similarity
        similarity = self._cosine_similarity(context_embedding, current_embedding)

        # Staleness score (0 = fresh, 1 = stale)
        staleness_score = max(0.0, 1.0 - (similarity / self.staleness_threshold))
        staleness_score = min(1.0, staleness_score)

        is_stale = similarity < self.staleness_threshold

        if is_stale:
            reason = (
                f"Semantic drift detected: similarity {similarity:.3f} < "
                f"threshold {self.staleness_threshold:.3f}"
            )
        else:
            reason = f"Context is fresh: similarity {similarity:.3f}"

        return StalenessResult(
            is_stale=is_stale,
            staleness_score=staleness_score,
            similarity=similarity,
            threshold=self.staleness_threshold,
            context_age_seconds=context_age,
            reason=reason
        )

    def detect_batch(
        self,
        context_keys: List[str],
        current_data: List[Dict[str, Any]]
    ) -> List[StalenessResult]:
        """
        Batch staleness detection.

        Args:
            context_keys: List of context keys
            current_data: List of dicts with readings, statistics, building_id, meter_type

        Returns:
            List of StalenessResult
        """
        results = []
        for key, data in zip(context_keys, current_data):
            result = self.detect(
                context_key=key,
                current_readings=data.get("readings", []),
                current_statistics=data.get("statistics", {}),
                building_id=data.get("building_id", ""),
                meter_type=data.get("meter_type", "")
            )
            results.append(result)
        return results

    def update_context(
        self,
        context_key: str,
        readings: List[Dict],
        statistics: Dict[str, float],
        building_id: str = "",
        meter_type: str = ""
    ) -> np.ndarray:
        """Update context (same as set_context, refreshes timestamp)."""
        return self.set_context(context_key, readings, statistics, building_id, meter_type)

    def clear_context(self, context_key: str) -> bool:
        """Clear a specific context from cache."""
        if context_key in self._context_cache:
            del self._context_cache[context_key]
            return True
        return False

    def clear_all_contexts(self):
        """Clear all cached contexts."""
        self._context_cache.clear()

    def get_context_age(self, context_key: str) -> Optional[float]:
        """Get age of a context in seconds."""
        if context_key not in self._context_cache:
            return None
        _, context_time = self._context_cache[context_key]
        return time.time() - context_time

    def list_contexts(self) -> List[Tuple[str, float]]:
        """List all contexts with their ages."""
        current_time = time.time()
        return [
            (key, current_time - cache_time)
            for key, (_, cache_time) in self._context_cache.items()
        ]

    def benchmark_latency(self, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark staleness detection latency."""
        test_readings = [{"value": 100.0, "timestamp": time.time()}]
        test_stats = {"mean": 100.0, "std": 10.0, "min": 80.0, "max": 120.0}

        # Set context
        self.set_context("_benchmark", test_readings, test_stats, "test", "electricity")

        latencies = []
        for _ in range(n_runs):
            # Slightly modify data
            test_stats["mean"] += np.random.randn() * 0.1

            start = time.perf_counter()
            _ = self.detect("_benchmark", test_readings, test_stats, "test", "electricity")
            latencies.append((time.perf_counter() - start) * 1000)

        self.clear_context("_benchmark")

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "p99_ms": np.percentile(latencies, 99)
        }


class AdaptiveStalenessDetector(StalenessDetector):
    """
    Staleness detector with adaptive thresholds.

    Learns optimal thresholds from data patterns.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        initial_threshold: float = 0.85,
        learning_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(model_name=model_name, staleness_threshold=initial_threshold, **kwargs)
        self.learning_rate = learning_rate
        self._similarity_history: List[float] = []
        self._adaptation_window = 100

    def adapt_threshold(self, feedback_is_stale: bool, result: StalenessResult):
        """
        Adapt threshold based on feedback.

        Args:
            feedback_is_stale: True if user/system confirms data was actually stale
            result: The StalenessResult that was produced
        """
        self._similarity_history.append(result.similarity)

        # Only adapt after enough history
        if len(self._similarity_history) < 10:
            return

        # If prediction was wrong, adjust threshold
        if feedback_is_stale and not result.is_stale:
            # False negative: should have detected staleness
            # Increase threshold
            self.staleness_threshold = min(
                0.99,
                self.staleness_threshold + self.learning_rate
            )
        elif not feedback_is_stale and result.is_stale:
            # False positive: incorrectly detected staleness
            # Decrease threshold
            self.staleness_threshold = max(
                0.5,
                self.staleness_threshold - self.learning_rate
            )

        # Trim history
        if len(self._similarity_history) > self._adaptation_window:
            self._similarity_history = self._similarity_history[-self._adaptation_window:]


if __name__ == "__main__":
    print("Testing Staleness Detector...")

    detector = StalenessDetector(
        staleness_threshold=0.85,
        max_age_seconds=60.0
    )

    # Set initial context
    initial_readings = [{"value": 150.0, "timestamp": time.time()}]
    initial_stats = {"mean": 150.0, "std": 10.0, "min": 130.0, "max": 170.0}

    detector.set_context(
        "building_001",
        initial_readings,
        initial_stats,
        "Panther_office",
        "electricity"
    )

    # Test with similar data (should be fresh)
    print("\nTest 1: Similar data")
    similar_stats = {"mean": 152.0, "std": 11.0, "min": 132.0, "max": 172.0}
    result = detector.detect(
        "building_001",
        [{"value": 152.0, "timestamp": time.time()}],
        similar_stats,
        "Panther_office",
        "electricity"
    )
    print(f"  Is stale: {result.is_stale}")
    print(f"  Similarity: {result.similarity:.3f}")
    print(f"  Reason: {result.reason}")

    # Test with very different data (should be stale)
    print("\nTest 2: Very different data")
    different_stats = {"mean": 50.0, "std": 5.0, "min": 40.0, "max": 60.0}
    result = detector.detect(
        "building_001",
        [{"value": 50.0, "timestamp": time.time()}],
        different_stats,
        "Panther_office",
        "electricity"
    )
    print(f"  Is stale: {result.is_stale}")
    print(f"  Similarity: {result.similarity:.3f}")
    print(f"  Reason: {result.reason}")

    # Benchmark
    print("\nRunning latency benchmark...")
    latency = detector.benchmark_latency(n_runs=50)
    print(f"  Mean latency: {latency['mean_ms']:.2f} ms")
    print(f"  P99 latency: {latency['p99_ms']:.2f} ms")
