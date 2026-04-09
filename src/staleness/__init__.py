"""Staleness detection module for temporal grounding."""

from .detector import (
    StalenessDetector,
    StalenessResult,
    AdaptiveStalenessDetector,
    TimeThresholdStalenessDetector
)

# TimeThresholdStalenessDetector is the primary detector (F1=1.0)
# StalenessDetector (embedding-based) is deprecated (poor F1)

__all__ = [
    "StalenessDetector",
    "StalenessResult",
    "AdaptiveStalenessDetector",
    "TimeThresholdStalenessDetector"
]
