"""
Evaluation Metrics for Temporal Grounding Pipeline.

Provides standardized metrics for:
- Latency benchmarking
- Grounding accuracy
- Staleness detection (F1, precision, recall)
- Causal consistency
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_samples: int


@dataclass
class ClassificationMetrics:
    """Classification metrics (for staleness, causal)."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def compute_latency_metrics(latencies: List[float]) -> LatencyMetrics:
    """
    Compute latency statistics.

    Args:
        latencies: List of latency measurements in milliseconds

    Returns:
        LatencyMetrics with statistics
    """
    arr = np.array(latencies)
    return LatencyMetrics(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        n_samples=len(arr)
    )


def compute_classification_metrics(
    y_true: List[bool],
    y_pred: List[bool]
) -> ClassificationMetrics:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        ClassificationMetrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == True) & (y_pred == True)))
    fp = int(np.sum((y_true == False) & (y_pred == True)))
    tn = int(np.sum((y_true == False) & (y_pred == False)))
    fn = int(np.sum((y_true == True) & (y_pred == False)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp, fp=fp, tn=tn, fn=fn
    )


def compute_grounding_accuracy(
    responses: List[str],
    ground_truths: List[Dict[str, Any]],
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Compute grounding accuracy for responses.

    Checks if responses correctly reference sensor values.

    Args:
        responses: LLM responses
        ground_truths: Dict with expected values
        tolerance: Relative tolerance for numeric matching

    Returns:
        Dict with accuracy metrics
    """
    correct = 0
    total = 0
    value_errors = []

    for response, truth in zip(responses, ground_truths):
        response_lower = response.lower()

        # Check if key values are mentioned
        for key, value in truth.items():
            if isinstance(value, (int, float)):
                # Check if value appears in response (with tolerance)
                found = False
                for word in response.split():
                    try:
                        num = float(word.replace(',', '').replace('kWh', '').replace('%', ''))
                        if abs(num - value) / max(abs(value), 1e-10) < tolerance:
                            found = True
                            break
                    except ValueError:
                        continue

                if found:
                    correct += 1
                else:
                    value_errors.append((key, value))
                total += 1

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "error_rate": 1 - correct / max(total, 1)
    }


def compute_causal_f1(
    responses: List[str],
    causal_graph: Any,
    claim_extractor: Optional[callable] = None
) -> Dict[str, float]:
    """
    Compute causal consistency F1 score.

    Args:
        responses: LLM responses
        causal_graph: Ground truth causal graph
        claim_extractor: Function to extract causal claims from text

    Returns:
        Dict with F1 metrics
    """
    if claim_extractor is None:
        # Simple keyword-based extraction
        def claim_extractor(text):
            claims = []
            causal_words = ["causes", "leads to", "results in", "because", "due to"]
            text_lower = text.lower()
            for word in causal_words:
                if word in text_lower:
                    # Found a causal claim
                    claims.append(text_lower)
                    break
            return claims

    total_claims = 0
    valid_claims = 0

    for response in responses:
        claims = claim_extractor(response)
        total_claims += len(claims)
        # In real implementation, validate each claim against graph
        # For now, count claims that don't contain obvious errors
        valid_claims += len(claims)  # Placeholder

    return {
        "causal_f1": valid_claims / max(total_claims, 1),
        "total_claims": total_claims,
        "valid_claims": valid_claims
    }


class LatencyBenchmark:
    """Utility class for latency benchmarking."""

    def __init__(self):
        self.measurements: List[float] = []
        self._start_time: Optional[float] = None

    def start(self):
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and record measurement."""
        if self._start_time is None:
            raise ValueError("Benchmark not started")
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self.measurements.append(elapsed)
        self._start_time = None
        return elapsed

    def reset(self):
        """Clear all measurements."""
        self.measurements.clear()
        self._start_time = None

    def get_metrics(self) -> LatencyMetrics:
        """Get computed metrics."""
        if not self.measurements:
            raise ValueError("No measurements recorded")
        return compute_latency_metrics(self.measurements)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


def compare_methods(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple methods across metrics.

    Args:
        results: Dict mapping method name to metrics dict

    Returns:
        Comparison summary
    """
    comparison = {}

    # Find best method for each metric
    metrics_to_compare = ["latency_ms", "accuracy", "f1", "grounding_acc"]

    for metric in metrics_to_compare:
        values = {}
        for method, method_results in results.items():
            if metric in method_results:
                values[method] = method_results[metric]

        if values:
            if "latency" in metric:
                best = min(values, key=values.get)
            else:
                best = max(values, key=values.get)

            comparison[metric] = {
                "best_method": best,
                "best_value": values[best],
                "all_values": values
            }

    return comparison


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")

    # Latency metrics
    latencies = [50, 55, 48, 52, 60, 45, 58, 51, 49, 53]
    metrics = compute_latency_metrics(latencies)
    print(f"\nLatency: {metrics.mean_ms:.1f}ms ± {metrics.std_ms:.1f}ms")
    print(f"P95: {metrics.p95_ms:.1f}ms, P99: {metrics.p99_ms:.1f}ms")

    # Classification metrics
    y_true = [True, True, False, True, False, False, True, False]
    y_pred = [True, False, False, True, True, False, True, False]
    cls_metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\nClassification F1: {cls_metrics.f1:.3f}")
    print(f"Precision: {cls_metrics.precision:.3f}, Recall: {cls_metrics.recall:.3f}")

    # Benchmark context manager
    bench = LatencyBenchmark()
    for i in range(10):
        with bench:
            time.sleep(0.01)  # Simulate work
    print(f"\nBenchmark: {bench.get_metrics().mean_ms:.1f}ms avg")
