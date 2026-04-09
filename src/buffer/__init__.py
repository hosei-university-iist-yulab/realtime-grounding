"""Buffer module for real-time sensor data storage.

Two buffer implementations available:
1. CircularBuffer: Redis-backed, for distributed systems
2. TemporalGroundingBuffer: In-process, for edge deployment (Novel)

Use create_buffer() factory to switch between implementations.
"""

from .circular_buffer import CircularBuffer, SensorReading
from .temporal_buffer import TemporalGroundingBuffer, create_buffer

__all__ = [
    "CircularBuffer",          # Redis-backed buffer
    "TemporalGroundingBuffer", # In-process buffer (Novel)
    "SensorReading",           # Shared data class
    "create_buffer",           # Factory function
]
