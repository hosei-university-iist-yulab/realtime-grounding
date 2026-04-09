#!/usr/bin/env python3
"""
Quick Test - Fast end-to-end verification of TGP pipeline.

Runs a minimal test of all components in ~2 minutes.
Use this to verify the setup is working correctly.

Usage:
    python scripts/quick_test.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use GPU 4 by default
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")


def test_imports():
    """Test all required imports."""
    print("\n[1/6] Testing imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False

    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False

    try:
        import peft
        print(f"  ✓ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"  ✗ PEFT: {e}")
        return False

    try:
        import redis
        print("  ✓ Redis client")
    except ImportError as e:
        print(f"  ✗ Redis: {e}")
        return False

    try:
        import sentence_transformers
        print("  ✓ Sentence-Transformers")
    except ImportError as e:
        print(f"  ✗ Sentence-Transformers: {e}")
        return False

    return True


def test_buffer():
    """Test Redis circular buffer."""
    print("\n[2/6] Testing buffer module...")

    try:
        from src.buffer import CircularBuffer, SensorReading

        buffer = CircularBuffer()
        print("  ✓ Connected to Redis")

        # Add test data
        for i in range(10):
            reading = SensorReading(
                timestamp=time.time() + i,
                building_id="_test_quick",
                meter_type="electricity",
                value=100.0 + i
            )
            buffer.push(reading)

        # Query
        latest = buffer.get_latest("_test_quick", "electricity", n=5)
        print(f"  ✓ Push/get working ({len(latest)} readings)")

        # Benchmark
        latency = buffer.benchmark_latency(n_operations=100)
        print(f"  ✓ Latency: {latency['push_ms']:.2f}ms push, {latency['get_latest_ms']:.2f}ms get")

        # Cleanup
        buffer.clear("_test_quick", "electricity")
        return True

    except Exception as e:
        print(f"  ✗ Buffer test failed: {e}")
        return False


def test_staleness():
    """Test staleness detector."""
    print("\n[3/6] Testing staleness detector...")

    try:
        from src.staleness import StalenessDetector

        detector = StalenessDetector(staleness_threshold=0.85)
        print("  ✓ Detector initialized")

        # Set context
        readings = [{"value": 150.0, "timestamp": time.time()}]
        stats = {"mean": 150.0, "std": 10.0, "min": 130.0, "max": 170.0}
        detector.set_context("_test", readings, stats, "test", "electricity")
        print("  ✓ Context set")

        # Test fresh detection
        result = detector.detect("_test", readings, stats, "test", "electricity")
        print(f"  ✓ Detection: stale={result.is_stale}, similarity={result.similarity:.3f}")

        # Benchmark
        latency = detector.benchmark_latency(n_runs=50)
        print(f"  ✓ Latency: {latency['mean_ms']:.2f}ms")

        detector.clear_context("_test")
        return True

    except Exception as e:
        print(f"  ✗ Staleness test failed: {e}")
        return False


def test_causal():
    """Test causal validator."""
    print("\n[4/6] Testing causal validator...")

    try:
        from src.causal import CausalValidator, CausalGraph

        # Create graph
        graph = CausalGraph.create_energy_graph()
        print(f"  ✓ Created graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")

        # Validate
        validator = CausalValidator(graph)
        response = "High temperature causes increased HVAC load."
        result = validator.validate(response)
        print(f"  ✓ Validation: valid={result.is_valid}, score={result.score:.2f}")

        return True

    except Exception as e:
        print(f"  ✗ Causal test failed: {e}")
        return False


def test_baselines():
    """Test baseline implementations."""
    print("\n[5/6] Testing baselines...")

    try:
        from src.baselines import ClaudeBaseline, BaselineResult

        # Check API key
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            print("  ✓ ANTHROPIC_API_KEY found")

            # Quick API test (optional, costs money)
            # baseline = ClaudeBaseline()
            # result = baseline.generate("Test", {"building_id": "test"}, max_tokens=10)
            # print(f"  ✓ Claude API working")
        else:
            print("  ⚠ ANTHROPIC_API_KEY not set (baselines will be limited)")

        return True

    except Exception as e:
        print(f"  ✗ Baselines test failed: {e}")
        return False


def test_training_data():
    """Test training data generation."""
    print("\n[6/6] Testing training data generation...")

    try:
        from scripts.generate_training_data import TrainingDataGenerator

        generator = TrainingDataGenerator()

        # Generate small test set
        examples = generator._generate_synthetic()
        print(f"  ✓ Generated {len(examples)} synthetic examples")

        # Save test
        test_path = PROJECT_ROOT / "data" / "training" / "_quick_test.jsonl"
        generator.examples = examples[:10]
        generator.save(str(test_path), format="jsonl")
        print(f"  ✓ Saved to {test_path}")

        # Cleanup
        test_path.unlink()

        return True

    except Exception as e:
        print(f"  ✗ Training data test failed: {e}")
        return False


def main():
    """Run all quick tests."""
    print("=" * 50)
    print("TGP Quick Test")
    print("=" * 50)

    start_time = time.time()
    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Buffer", test_buffer()))
    results.append(("Staleness", test_staleness()))
    results.append(("Causal", test_causal()))
    results.append(("Baselines", test_baselines()))
    results.append(("Training Data", test_training_data()))

    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} passed in {elapsed:.1f}s")
    print("=" * 50)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    if passed == total:
        print("\n✓ All tests passed! TGP is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
