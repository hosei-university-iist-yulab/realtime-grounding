#!/usr/bin/env python3
"""Verify all dependencies and infrastructure are correctly installed."""

import sys
from pathlib import Path


def verify() -> bool:
    """Run all verification checks."""
    checks = []
    all_passed = True

    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        checks.append(f"✓ Python {py_version}")
    else:
        checks.append(f"✗ Python {py_version} (need >= 3.10)")
        all_passed = False

    # 2. PyTorch & CUDA
    try:
        import torch
        checks.append(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            checks.append(f"  CUDA available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                checks.append(f"  GPU {i}: {name}")
        else:
            checks.append("  ⚠ CUDA not available (CPU only)")
    except ImportError:
        checks.append("✗ PyTorch not installed")
        all_passed = False

    # 3. Transformers
    try:
        import transformers
        checks.append(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        checks.append("✗ Transformers not installed")
        all_passed = False

    # 4. PEFT (LoRA)
    try:
        import peft
        checks.append(f"✓ PEFT {peft.__version__}")
    except ImportError:
        checks.append("✗ PEFT not installed (pip install peft)")
        all_passed = False

    # 5. Sentence Transformers
    try:
        import sentence_transformers
        checks.append("✓ Sentence-Transformers")
    except ImportError:
        checks.append("✗ Sentence-Transformers not installed")
        all_passed = False

    # 6. Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        r.ping()
        checks.append("✓ Redis server running")
    except ImportError:
        checks.append("✗ redis-py not installed (pip install redis)")
        all_passed = False
    except Exception:
        checks.append("⚠ Redis not running (redis-server --daemonize yes)")

    # 7. Anthropic API
    try:
        import anthropic
        checks.append("✓ Anthropic SDK installed")
    except ImportError:
        checks.append("⚠ Anthropic SDK not installed (pip install anthropic)")

    # 8. Environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if os.getenv("ANTHROPIC_API_KEY"):
        checks.append("✓ ANTHROPIC_API_KEY set")
    else:
        checks.append("⚠ ANTHROPIC_API_KEY not set")

    if os.getenv("OPENAI_API_KEY"):
        checks.append("✓ OPENAI_API_KEY set")
    else:
        checks.append("⚠ OPENAI_API_KEY not set (optional)")

    # 9. Data directories
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "data/raw", "data/processed", "data/training",
        "output/models", "output/results", "output/figures"
    ]
    for d in required_dirs:
        dir_path = project_root / d
        dir_path.mkdir(parents=True, exist_ok=True)
    checks.append("✓ Data directories created")

    # 10. BDG2 dataset
    bdg2_path = project_root / "data/raw/bdg2"
    if bdg2_path.exists():
        checks.append("✓ BDG2 dataset present")
    else:
        checks.append("⚠ BDG2 not downloaded (git clone https://github.com/buds-lab/building-data-genome-project-2.git data/raw/bdg2)")

    # Print results
    print("\n" + "=" * 50)
    print("TGP Setup Verification")
    print("=" * 50)
    print("\n".join(checks))
    print("=" * 50)

    if all_passed:
        print("✓ All critical checks passed!")
    else:
        print("✗ Some checks failed - see above")

    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
