"""
Performance Tracker for GPU, Memory, and CO2 Monitoring.

Tracks computational costs for reproducibility and sustainability reporting.
"""

import os
import time
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path

import torch


@dataclass
class PerformanceMetrics:
    """Single performance measurement."""
    timestamp: str
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    gpu_utilization_percent: float
    power_watts: float
    inference_time_ms: float
    device_id: int = 0


@dataclass
class ExperimentMetrics:
    """Aggregate metrics for an experiment."""
    experiment_name: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    n_samples: int
    # GPU metrics
    peak_memory_gb: float
    avg_memory_gb: float
    avg_gpu_util: float
    avg_power_watts: float
    # Latency metrics
    mean_latency_ms: float
    p95_latency_ms: float
    # Emissions
    estimated_co2_kg: Optional[float] = None
    measurements: List[PerformanceMetrics] = field(default_factory=list)


class PerformanceTracker:
    """
    Track GPU memory, utilization, power, and inference time.

    Usage:
        tracker = PerformanceTracker(device=4)
        tracker.start_experiment("latency_benchmark")

        for query in queries:
            tracker.start_inference()
            result = model.generate(query)
            tracker.end_inference()

        metrics = tracker.end_experiment()
        tracker.save_results("output/results/exp01_perf.json")
    """

    # CO2 factor: kg CO2 per kWh (US average)
    CO2_PER_KWH = 0.42

    def __init__(self, device: int = 0, track_power: bool = True):
        """
        Initialize tracker.

        Args:
            device: GPU device ID (0-7)
            track_power: Whether to track power consumption
        """
        self.device = device
        self.track_power = track_power

        self._experiment_name: Optional[str] = None
        self._start_time: Optional[float] = None
        self._inference_start: Optional[float] = None
        self._measurements: List[PerformanceMetrics] = []
        self._latencies: List[float] = []

        # Verify GPU is available
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. GPU metrics will be limited.")

    def start_experiment(self, name: str):
        """Start tracking an experiment."""
        self._experiment_name = name
        self._start_time = time.time()
        self._measurements = []
        self._latencies = []

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_inference(self):
        """Start timing a single inference."""
        self._inference_start = time.perf_counter()

    def end_inference(self) -> float:
        """
        End timing and record metrics.

        Returns:
            Inference latency in milliseconds
        """
        if self._inference_start is None:
            raise ValueError("Inference not started")

        latency_ms = (time.perf_counter() - self._inference_start) * 1000
        self._latencies.append(latency_ms)

        # Record full metrics
        metrics = self._capture_metrics(latency_ms)
        self._measurements.append(metrics)

        self._inference_start = None
        return latency_ms

    def _capture_metrics(self, latency_ms: float) -> PerformanceMetrics:
        """Capture current GPU metrics."""
        gpu_mem_alloc = 0.0
        gpu_mem_reserved = 0.0
        gpu_util = 0.0
        power = 0.0

        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9

        if self.track_power:
            gpu_util, power = self._get_nvidia_smi_metrics()

        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            gpu_memory_allocated_gb=gpu_mem_alloc,
            gpu_memory_reserved_gb=gpu_mem_reserved,
            gpu_utilization_percent=gpu_util,
            power_watts=power,
            inference_time_ms=latency_ms,
            device_id=self.device
        )

    def _get_nvidia_smi_metrics(self) -> tuple:
        """Get GPU utilization and power from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    f"--id={self.device}"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                util = float(parts[0].strip())
                power = float(parts[1].strip())
                return util, power
        except Exception:
            pass
        return 0.0, 0.0

    def end_experiment(self) -> ExperimentMetrics:
        """
        End experiment and compute aggregate metrics.

        Returns:
            ExperimentMetrics with all statistics
        """
        if self._start_time is None:
            raise ValueError("Experiment not started")

        end_time = time.time()
        duration = end_time - self._start_time

        # Compute aggregates
        import numpy as np

        if self._measurements:
            peak_mem = max(m.gpu_memory_allocated_gb for m in self._measurements)
            avg_mem = np.mean([m.gpu_memory_allocated_gb for m in self._measurements])
            avg_util = np.mean([m.gpu_utilization_percent for m in self._measurements])
            avg_power = np.mean([m.power_watts for m in self._measurements])
        else:
            peak_mem = avg_mem = avg_util = avg_power = 0.0

        if self._latencies:
            mean_lat = np.mean(self._latencies)
            p95_lat = np.percentile(self._latencies, 95)
        else:
            mean_lat = p95_lat = 0.0

        # Estimate CO2 (simplified)
        energy_kwh = (avg_power * duration) / (1000 * 3600)  # W * s -> kWh
        co2_kg = energy_kwh * self.CO2_PER_KWH

        metrics = ExperimentMetrics(
            experiment_name=self._experiment_name,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_duration_seconds=duration,
            n_samples=len(self._latencies),
            peak_memory_gb=peak_mem,
            avg_memory_gb=avg_mem,
            avg_gpu_util=avg_util,
            avg_power_watts=avg_power,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            estimated_co2_kg=co2_kg,
            measurements=self._measurements
        )

        return metrics

    def save_results(self, path: str, include_measurements: bool = False):
        """
        Save experiment results to JSON.

        Args:
            path: Output file path
            include_measurements: Include per-inference measurements
        """
        if not hasattr(self, '_last_metrics'):
            raise ValueError("No experiment results to save")

        metrics = self._last_metrics
        data = asdict(metrics)

        if not include_measurements:
            data.pop("measurements", None)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "device_id": self.device,
            "name": torch.cuda.get_device_name(self.device),
            "total_memory_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__
        }

    def print_summary(self, metrics: ExperimentMetrics):
        """Print human-readable summary."""
        print(f"\n{'='*50}")
        print(f"Experiment: {metrics.experiment_name}")
        print(f"{'='*50}")
        print(f"Duration: {metrics.total_duration_seconds:.1f}s")
        print(f"Samples: {metrics.n_samples}")
        print(f"\nGPU Metrics:")
        print(f"  Peak Memory: {metrics.peak_memory_gb:.2f} GB")
        print(f"  Avg Memory: {metrics.avg_memory_gb:.2f} GB")
        print(f"  Avg Utilization: {metrics.avg_gpu_util:.1f}%")
        print(f"  Avg Power: {metrics.avg_power_watts:.1f} W")
        print(f"\nLatency:")
        print(f"  Mean: {metrics.mean_latency_ms:.1f} ms")
        print(f"  P95: {metrics.p95_latency_ms:.1f} ms")
        print(f"\nEnvironmental Impact:")
        print(f"  Estimated CO2: {metrics.estimated_co2_kg:.4f} kg")
        print(f"{'='*50}")


class CodeCarbonTracker:
    """
    Optional wrapper for CodeCarbon CO2 tracking.

    Install: pip install codecarbon
    """

    def __init__(self, project_name: str = "TGP"):
        """Initialize CodeCarbon tracker."""
        self.project_name = project_name
        self._tracker = None

        try:
            from codecarbon import EmissionsTracker
            self._tracker = EmissionsTracker(project_name=project_name)
        except ImportError:
            print("Warning: codecarbon not installed. Using estimation only.")

    def start(self):
        """Start CO2 tracking."""
        if self._tracker:
            self._tracker.start()

    def stop(self) -> Optional[float]:
        """
        Stop tracking and return emissions.

        Returns:
            CO2 emissions in kg, or None if not available
        """
        if self._tracker:
            return self._tracker.stop()
        return None


if __name__ == "__main__":
    print("Testing Performance Tracker...")

    # Create tracker
    tracker = PerformanceTracker(device=0, track_power=True)

    # Print GPU info
    print("\nGPU Info:")
    for k, v in tracker.get_gpu_info().items():
        print(f"  {k}: {v}")

    # Simulate experiment
    print("\nRunning simulated experiment...")
    tracker.start_experiment("test_benchmark")

    for i in range(10):
        tracker.start_inference()
        time.sleep(0.05)  # Simulate 50ms inference
        latency = tracker.end_inference()
        print(f"  Inference {i+1}: {latency:.1f}ms")

    metrics = tracker.end_experiment()
    tracker._last_metrics = metrics
    tracker.print_summary(metrics)
