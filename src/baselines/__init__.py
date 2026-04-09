"""Baseline implementations for comparison."""

from .api_baselines import (
    BaselineModel,
    BaselineResult,
    ClaudeBaseline,
    GPT4Baseline,
    StaticPromptBaseline,
    RAGBaseline,
    compare_baselines
)

from .staleness_heuristics import (
    StalenessHeuristic,
    HeuristicResult,
    TimeThresholdHeuristic,
    ValueChangeHeuristic,
    VarianceChangeHeuristic,
    CombinedHeuristic,
    AdaptiveThresholdHeuristic,
    compare_heuristics
)

__all__ = [
    # API baselines
    "BaselineModel",
    "BaselineResult",
    "ClaudeBaseline",
    "GPT4Baseline",
    "StaticPromptBaseline",
    "RAGBaseline",
    "compare_baselines",
    # Staleness heuristics
    "StalenessHeuristic",
    "HeuristicResult",
    "TimeThresholdHeuristic",
    "ValueChangeHeuristic",
    "VarianceChangeHeuristic",
    "CombinedHeuristic",
    "AdaptiveThresholdHeuristic",
    "compare_heuristics"
]
