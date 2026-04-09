"""
Staleness Detection Heuristic Baselines.

Simple rule-based staleness detection methods for comparison:
- Time-only threshold
- Value-change threshold
- Combined heuristic
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class HeuristicResult:
    """Result from heuristic staleness detection."""
    is_stale: bool
    confidence: float  # 0-1
    reason: str
    method: str


class StalenessHeuristic(ABC):
    """Abstract base class for staleness heuristics."""

    @abstractmethod
    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness using heuristic."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Heuristic name."""
        pass


class TimeThresholdHeuristic(StalenessHeuristic):
    """
    Simple time-based staleness detection.

    Data is stale if context is older than threshold.
    """

    def __init__(self, threshold_seconds: float = 300):
        """
        Initialize time threshold heuristic.

        Args:
            threshold_seconds: Age threshold in seconds
        """
        self.threshold = threshold_seconds

    @property
    def name(self) -> str:
        return f"time_threshold_{self.threshold}s"

    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness based on time only."""
        age = current_time - context_time
        is_stale = age > self.threshold

        confidence = min(1.0, age / self.threshold) if is_stale else 0.0

        return HeuristicResult(
            is_stale=is_stale,
            confidence=confidence,
            reason=f"Context age {age:.0f}s {'>' if is_stale else '<='} threshold {self.threshold}s",
            method=self.name
        )


class ValueChangeHeuristic(StalenessHeuristic):
    """
    Value-change based staleness detection.

    Data is stale if mean value changed by more than threshold percent.
    """

    def __init__(self, threshold_percent: float = 0.2):
        """
        Initialize value change heuristic.

        Args:
            threshold_percent: Change threshold (0.2 = 20%)
        """
        self.threshold = threshold_percent

    @property
    def name(self) -> str:
        return f"value_change_{self.threshold:.0%}"

    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness based on value change."""
        ctx_mean = context_stats.get("mean", 0)
        cur_mean = current_stats.get("mean", 0)

        if ctx_mean == 0:
            change = 1.0 if cur_mean != 0 else 0.0
        else:
            change = abs(cur_mean - ctx_mean) / abs(ctx_mean)

        is_stale = change > self.threshold
        confidence = min(1.0, change / self.threshold) if is_stale else change / self.threshold

        return HeuristicResult(
            is_stale=is_stale,
            confidence=confidence,
            reason=f"Value change {change:.1%} {'>' if is_stale else '<='} threshold {self.threshold:.0%}",
            method=self.name
        )


class VarianceChangeHeuristic(StalenessHeuristic):
    """
    Variance-change based staleness detection.

    Data is stale if standard deviation changed significantly.
    """

    def __init__(self, threshold_ratio: float = 2.0):
        """
        Initialize variance change heuristic.

        Args:
            threshold_ratio: Ratio threshold (2.0 = double or half)
        """
        self.threshold = threshold_ratio

    @property
    def name(self) -> str:
        return f"variance_change_{self.threshold}x"

    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness based on variance change."""
        ctx_std = context_stats.get("std", 1)
        cur_std = current_stats.get("std", 1)

        if ctx_std == 0:
            ratio = float('inf') if cur_std > 0 else 1.0
        else:
            ratio = max(cur_std / ctx_std, ctx_std / cur_std)

        is_stale = ratio > self.threshold
        confidence = min(1.0, ratio / self.threshold) if is_stale else 0.0

        return HeuristicResult(
            is_stale=is_stale,
            confidence=confidence,
            reason=f"Variance ratio {ratio:.2f}x {'>' if is_stale else '<='} threshold {self.threshold}x",
            method=self.name
        )


class CombinedHeuristic(StalenessHeuristic):
    """
    Combined heuristic using multiple rules.

    Data is stale if ANY rule triggers.
    """

    def __init__(
        self,
        time_threshold: float = 300,
        value_threshold: float = 0.2,
        variance_threshold: float = 2.0
    ):
        """Initialize combined heuristic."""
        self.rules = [
            TimeThresholdHeuristic(time_threshold),
            ValueChangeHeuristic(value_threshold),
            VarianceChangeHeuristic(variance_threshold)
        ]

    @property
    def name(self) -> str:
        return "combined_heuristic"

    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness using all rules."""
        results = [
            rule.detect(context_time, context_stats, current_time, current_stats)
            for rule in self.rules
        ]

        # Stale if any rule triggers
        is_stale = any(r.is_stale for r in results)

        # Max confidence from triggered rules
        confidence = max((r.confidence for r in results if r.is_stale), default=0.0)

        # Combine reasons
        triggered = [r.reason for r in results if r.is_stale]
        reason = "; ".join(triggered) if triggered else "No rules triggered"

        return HeuristicResult(
            is_stale=is_stale,
            confidence=confidence,
            reason=reason,
            method=self.name
        )


class AdaptiveThresholdHeuristic(StalenessHeuristic):
    """
    Adaptive threshold based on historical patterns.

    Adjusts thresholds based on observed data variability.
    """

    def __init__(self, base_threshold: float = 0.2):
        """Initialize adaptive heuristic."""
        self.base_threshold = base_threshold
        self.history: List[float] = []
        self.max_history = 100

    @property
    def name(self) -> str:
        return "adaptive_threshold"

    def update_history(self, change: float):
        """Update change history."""
        self.history.append(change)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_adaptive_threshold(self) -> float:
        """Compute adaptive threshold from history."""
        if len(self.history) < 10:
            return self.base_threshold

        import numpy as np
        mean_change = np.mean(self.history)
        std_change = np.std(self.history)

        # Threshold at mean + 2 std
        return max(self.base_threshold, mean_change + 2 * std_change)

    def detect(
        self,
        context_time: float,
        context_stats: Dict[str, float],
        current_time: float,
        current_stats: Dict[str, float]
    ) -> HeuristicResult:
        """Detect staleness with adaptive threshold."""
        ctx_mean = context_stats.get("mean", 0)
        cur_mean = current_stats.get("mean", 0)

        if ctx_mean == 0:
            change = 1.0 if cur_mean != 0 else 0.0
        else:
            change = abs(cur_mean - ctx_mean) / abs(ctx_mean)

        # Update history
        self.update_history(change)

        # Get adaptive threshold
        threshold = self.get_adaptive_threshold()

        is_stale = change > threshold
        confidence = min(1.0, change / threshold) if is_stale else 0.0

        return HeuristicResult(
            is_stale=is_stale,
            confidence=confidence,
            reason=f"Change {change:.1%} vs adaptive threshold {threshold:.1%}",
            method=self.name
        )


def compare_heuristics(
    test_cases: List[Dict],
    heuristics: Optional[List[StalenessHeuristic]] = None
) -> Dict[str, Dict]:
    """
    Compare heuristics on test cases.

    Args:
        test_cases: List of dicts with context_time, context_stats, current_time, current_stats, is_stale
        heuristics: List of heuristics to compare

    Returns:
        Dict mapping heuristic name to metrics
    """
    if heuristics is None:
        heuristics = [
            TimeThresholdHeuristic(300),
            ValueChangeHeuristic(0.2),
            VarianceChangeHeuristic(2.0),
            CombinedHeuristic()
        ]

    results = {}

    for heuristic in heuristics:
        tp = fp = tn = fn = 0

        for case in test_cases:
            result = heuristic.detect(
                case["context_time"],
                case["context_stats"],
                case["current_time"],
                case["current_stats"]
            )

            actual = case["is_stale"]
            predicted = result.is_stale

            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif not actual and not predicted:
                tn += 1
            else:
                fn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        results[heuristic.name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    return results


if __name__ == "__main__":
    import numpy as np

    print("Testing staleness heuristics...")

    # Create test cases
    test_cases = []

    # Fresh cases
    for _ in range(50):
        base = np.random.uniform(100, 200)
        test_cases.append({
            "context_time": time.time() - np.random.uniform(10, 100),
            "context_stats": {"mean": base, "std": 10},
            "current_time": time.time(),
            "current_stats": {"mean": base + np.random.randn() * 5, "std": 10},
            "is_stale": False
        })

    # Stale cases
    for _ in range(50):
        base = np.random.uniform(100, 200)
        test_cases.append({
            "context_time": time.time() - np.random.uniform(600, 3600),
            "context_stats": {"mean": base, "std": 10},
            "current_time": time.time(),
            "current_stats": {"mean": base * np.random.uniform(0.5, 1.5), "std": 20},
            "is_stale": True
        })

    np.random.shuffle(test_cases)

    # Compare
    results = compare_heuristics(test_cases)

    print("\nResults:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
