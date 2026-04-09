"""
Temporal Grounding Pipeline (TGP) for Real-Time Sensor-Text Synchronization.

This package provides the core implementation for IEEE VTC2026-Spring paper:
"Real-Time Sensor-Text Grounding for SLM-Based Building Energy Systems"

Key Components:
- buffer: Redis-based circular buffer for <10ms sensor retrieval
- llm: TinyLLaMA/Phi-2 backbone with LoRA fine-tuning
- staleness: Embedding-based staleness detection
- causal: Causal validation using learned causal graphs
- pipeline: Main orchestrator integrating all components
- baselines: API baselines (Claude, GPT-4) for comparison
- simulation: Real-time sensor stream simulation for testing
"""

__version__ = "0.1.0"
__author__ = "Research Team"

# Main pipeline
from .pipeline import (
    TemporalGroundingPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline
)

# Core components
from .buffer import CircularBuffer, SensorReading
from .llm import LLMBackbone, ModelConfig
from .staleness import StalenessDetector, StalenessResult
from .causal import CausalValidator, CausalGraph, ValidationResult

__all__ = [
    # Pipeline
    "TemporalGroundingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "create_pipeline",
    # Buffer
    "CircularBuffer",
    "SensorReading",
    # LLM
    "LLMBackbone",
    "ModelConfig",
    # Staleness
    "StalenessDetector",
    "StalenessResult",
    # Causal
    "CausalValidator",
    "CausalGraph",
    "ValidationResult",
]
