# Setup Guide: Real-Time Sensor-Text Grounding

## Quick Start

```bash
# 1. Activate existing environment
conda activate llms

# 2. Install additional dependencies
pip install redis sentence-transformers

# 3. Install Redis via conda
conda install -c conda-forge redis -y

# 4. Start Redis server
redis-server --daemonize yes --port 6379

# 5. Verify setup
python scripts/verify_setup.py
```

---

## Directory Structure

```
04-realtime-grounding/
│
├── src/                          # Core library
│   ├── __init__.py
│   ├── buffer/                   # Innovation 1: Sensor Snapshot Buffer
│   │   ├── __init__.py
│   │   ├── redis_buffer.py       # Redis circular buffer implementation
│   │   └── sensor_stream.py      # Real-time sensor stream handler
│   │
│   ├── llm/                      # Innovation 2: Grounding-Aware LLM
│   │   ├── __init__.py
│   │   ├── grounding_model.py    # TinyLLaMA/Phi-2 + LoRA model
│   │   └── training.py           # Fine-tuning with sensor-text pairs
│   │
│   ├── staleness/                # Innovation 3: Staleness Detection
│   │   ├── __init__.py
│   │   ├── detector.py           # Embedding-based staleness detector
│   │   └── embeddings.py         # Sentence embedding utilities
│   │
│   ├── causal/                   # Innovation 4: Causal Validation
│   │   ├── __init__.py
│   │   └── validator.py          # Causal grounding validator (uses Topic 1)
│   │
│   ├── pipeline/                 # Main TGP Pipeline
│   │   ├── __init__.py
│   │   └── tgp.py                # Temporal Grounding Pipeline
│   │
│   ├── baselines/                # Baseline implementations
│   │   ├── __init__.py
│   │   ├── cloud_api.py          # GPT-4/Claude API baseline
│   │   ├── postgresql_llm.py     # Traditional DB + LLM
│   │   ├── prompt_only.py        # No fine-tuning baseline
│   │   └── staleness_heuristics.py # Time/threshold baselines
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── metrics.py            # Evaluation metrics
│       ├── performance_tracker.py # GPU/memory/CO2 tracking
│       └── visualization.py      # Result plotting
│
├── scripts/                      # Execution scripts
│   ├── download_datasets.py      # Download BDG2, ASHRAE
│   ├── generate_training_data.py # Create sensor-query-response pairs
│   ├── train_grounding_model.py  # Fine-tune GPT-2 with LoRA
│   ├── verify_setup.py           # Verify all dependencies
│   └── quick_test.py             # Fast end-to-end test (~2 min)
│
├── experiments/                  # Experiment scripts
│   ├── 01_latency_benchmark.py   # Exp 1: Latency comparison
│   ├── 02_grounding_accuracy.py  # Exp 2: Grounding accuracy
│   ├── 03_staleness_detection.py # Exp 3: Staleness F1
│   ├── 04_causal_validation.py   # Exp 4: Causal consistency
│   ├── 05_ablation_study.py      # Exp 5: Component ablation
│   ├── 06_scalability.py         # Exp 6: Sensor count scaling
│   ├── 07_sota_comparison.py     # Exp 7: SOTA baselines
│   └── 08_computational_cost.py  # Exp 8: Resource usage
│
├── run/                          # Shell scripts (GPU handling)
│   ├── setup_environment.sh      # Environment setup
│   ├── run_all.sh                # Master orchestrator
│   ├── 01_download_data.sh       # Download datasets
│   ├── 02_generate_training.sh   # Generate training pairs
│   ├── 03_train_model.sh         # Train GPT-2 + LoRA
│   ├── 04_run_experiments.sh     # Run all 6 experiments
│   └── 05_generate_figures.sh    # Create paper figures
│
├── data/
│   ├── raw/                      # Downloaded datasets
│   │   ├── bdg2/                 # Building Data Genome 2
│   │   └── ashrae/               # ASHRAE Energy Predictor
│   ├── processed/                # Cleaned sensor streams
│   │   ├── building_*.csv        # Per-building sensor data
│   │   └── sensor_metadata.json  # Sensor descriptions
│   └── training/                 # Training data
│       ├── train.json            # Sensor-query-response pairs
│       ├── val.json
│       └── test.json
│
├── output/
│   ├── models/                   # Trained checkpoints
│   │   └── grounding_model_*.pt
│   ├── results/                  # Experiment outputs (JSON)
│   │   └── exp_*.json
│   ├── figures/                  # Paper figures (PDF)
│   └── tables/                   # LaTeX tables
│
├── paper/
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
├── tests/
│   ├── test_buffer.py
│   ├── test_llm.py
│   ├── test_staleness.py
│   └── test_pipeline.py
│
├── README.md
├── SETUP.md                      # This file
└── requirements.txt
```

---

## Environment Setup

### Use Existing `llms` Environment

```bash
# Activate
conda activate llms

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check existing packages
pip list | grep -E "torch|transformers|peft"
```

### Install Additional Dependencies

```bash
# Redis (server + Python client)
conda install -c conda-forge redis -y
pip install redis

# Sentence embeddings (for staleness detection)
pip install sentence-transformers

# Verify
python -c "import redis; import sentence_transformers; print('OK')"
```

### Redis Setup

```bash
# Start Redis server (background, no sudo needed)
redis-server --daemonize yes --port 6379

# Verify connection
redis-cli ping
# Expected output: PONG

# Check status
redis-cli info | head -20

# Stop Redis (when done)
redis-cli shutdown
```

---

## GPU Configuration

### Available Devices
```
GPU 4: RTX 3090 (24GB) - Primary inference
GPU 5: RTX 3090 (24GB) - Training
GPU 6: RTX 3090 (24GB) - Baselines
GPU 7: RTX 3090 (24GB) - Backup
```

### Device Assignment Pattern

```python
# In code: Explicit device assignment
import torch
torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, this is GPU 4

# Device map for multi-model
DEVICE_MAP = {
    'tinyllama': 'cuda:0',        # GPU 4 - TinyLLaMA 1.1B (primary)
    'phi2': 'cuda:1',             # GPU 5 - Phi-2 2.7B (backup/ensemble)
    'sentence_encoder': 'cuda:0', # GPU 4 (shared with TinyLLaMA)
    'baseline_gpt4': 'api',       # Cloud API
}
```

### Model Options

| Model | Params | VRAM | Inference | Use Case |
|-------|--------|------|-----------|----------|
| **TinyLLaMA** | 1.1B | ~4GB | ~50ms | Primary (fast) |
| **Phi-2** | 2.7B | ~8GB | ~80ms | Higher quality |
| **Qwen2-0.5B** | 0.5B | ~2GB | ~30ms | Ultra-fast fallback |

### Shell Script Pattern

```bash
#!/bin/bash
# run/03_train_model.sh

# Set visible GPUs (only 4-7)
export CUDA_VISIBLE_DEVICES=4,5

# Run training with TinyLLaMA (primary choice)
python scripts/train_grounding_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --epochs 5 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --device cuda:0

# Alternative: Phi-2 (higher quality, slower)
# python scripts/train_grounding_model.py \
#     --model microsoft/phi-2 \
#     --lora_rank 16 \
#     --epochs 5 \
#     --batch_size 4 \
#     --device cuda:0
```

---

## Datasets

### Building Data Genome 2 (Primary)

| Property | Value |
|----------|-------|
| **Source** | [GitHub](https://github.com/buds-lab/building-data-genome-project-2) |
| **License** | MIT |
| **Size** | 3,053 meters, 2 years |
| **Variables** | Energy, temperature, weather |
| **Format** | CSV |

```bash
# Download
git clone https://github.com/buds-lab/building-data-genome-project-2.git data/raw/bdg2
```

### ASHRAE Great Energy Predictor III (Secondary)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction) |
| **License** | Competition (free for research) |
| **Size** | 1,449 buildings |
| **Variables** | Energy, weather, building metadata |
| **Format** | CSV |

```bash
# Requires Kaggle API
pip install kaggle
kaggle competitions download -c ashrae-energy-prediction -p data/raw/ashrae
unzip data/raw/ashrae/*.zip -d data/raw/ashrae/
```

### Data Processing Pipeline

```
raw/bdg2/*.csv
    ↓
[scripts/download_datasets.py]
    ↓
processed/building_*.csv  (cleaned, resampled to 5-sec intervals)
    ↓
[scripts/generate_training_data.py]
    ↓
training/train.json  (sensor-query-response pairs)
```

---

## Training Data Generation

### Format

```json
{
  "sensor_snapshot": {
    "building_id": "office_1",
    "timestamp": "2024-01-15T14:32:15Z",
    "temperature": 22.5,
    "humidity": 45.0,
    "energy": 125.3,
    "occupancy": 42
  },
  "query": "Is the building energy-efficient right now?",
  "response": "Building office_1 is moderately efficient at 14:32. Energy consumption is 125.3 kWh with 42 occupants, giving 2.98 kWh/person. Temperature is 22.5°C (optimal for HVAC efficiency)."
}
```

### Generation Strategy (Rule-Based, Not API)

```python
# Template-based generation (reproducible, no API cost)
TEMPLATES = {
    "comfort": [
        "Is the {zone} comfortable?",
        "How is the temperature in {zone}?",
        "What's the air quality like?"
    ],
    "energy": [
        "Is energy consumption normal?",
        "Why is energy usage high?",
        "How efficient is the building?"
    ],
    "status": [
        "What's the current status of {zone}?",
        "Give me a summary of sensor readings.",
        "Any anomalies detected?"
    ]
}

def generate_response(snapshot, query_type):
    """Rule-based response generation from sensor values."""
    if query_type == "comfort":
        temp = snapshot["temperature"]
        comfort = "comfortable" if 20 <= temp <= 24 else "uncomfortable"
        return f"The {snapshot['zone']} is {comfort} at {temp}°C."
    # ... more rules
```

### Target Dataset Size

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 8,000 | Model fine-tuning |
| Val | 1,000 | Hyperparameter tuning |
| Test | 1,000 | Final evaluation |

---

## Execution Pipeline

### Master Orchestrator

```bash
#!/bin/bash
# run/run_all.sh

set -e  # Exit on error

echo "=== TGP COMPLETE PIPELINE ==="

# Step 1: Data
echo "[1/5] Downloading datasets..."
bash run/01_download_data.sh

# Step 2: Training data
echo "[2/5] Generating training pairs..."
bash run/02_generate_training.sh

# Step 3: Model training
echo "[3/5] Training grounding model..."
bash run/03_train_model.sh

# Step 4: Experiments
echo "[4/5] Running experiments..."
bash run/04_run_experiments.sh

# Step 5: Figures
echo "[5/5] Generating figures..."
bash run/05_generate_figures.sh

echo "=== COMPLETE ==="
```

### Step-by-Step Execution

```bash
# Option 1: Full pipeline (~4-6 hours)
./run/run_all.sh

# Option 2: Step-by-step
bash run/01_download_data.sh        # 10 min (download)
bash run/02_generate_training.sh    # 30 min (processing)
bash run/03_train_model.sh          # 2-3 hours (GPU training)
bash run/04_run_experiments.sh      # 1 hour (6 experiments)
bash run/05_generate_figures.sh     # 5 min (visualization)

# Option 3: Quick test (~2 min)
python scripts/quick_test.py
```

---

## Experiment Scripts

### Pattern (from causal-slm)

```python
#!/usr/bin/env python3
"""Experiment 1: Latency Benchmarking"""

import json
import time
from datetime import datetime
from pathlib import Path

def run_experiment():
    results = {
        "experiment": "latency_benchmark",
        "timestamp": datetime.now().isoformat(),
        "methods": {}
    }

    # Run each method
    for method in ["cloud_api", "postgresql", "tgp"]:
        latencies = benchmark_method(method, n_trials=1000)
        results["methods"][method] = {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p95_ms": np.percentile(latencies, 95)
        }

    # Save results
    output_path = Path("output/results") / f"exp01_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_experiment()
```

---

## Integration with Topic 1 (Causal SLM)

### Causal Graph Access

```python
# Load causal graph from Topic 1
import sys
sys.path.insert(0, "/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/01-causal-slm/src")

from innovations.causal_score_matching import CausalScoreMatching

# Or load pre-computed graph
import numpy as np
causal_graph = np.load("path/to/causal_graph.npy")
```

### Causal Validator Integration

```python
# src/causal/validator.py
class CausalGroundingValidator:
    def __init__(self, causal_graph: np.ndarray, variable_names: list):
        self.graph = causal_graph
        self.names = variable_names

    def validate(self, response: str) -> dict:
        """Check if response respects causal structure."""
        claims = self.extract_causal_claims(response)
        valid_claims = []
        for cause, effect in claims:
            if self.graph[self.names.index(cause), self.names.index(effect)] == 1:
                valid_claims.append((cause, effect))

        return {
            "total_claims": len(claims),
            "valid_claims": len(valid_claims),
            "causal_f1": len(valid_claims) / max(len(claims), 1)
        }
```

---

## Verification Script

```python
#!/usr/bin/env python3
"""scripts/verify_setup.py - Verify all dependencies are installed."""

def verify():
    checks = []

    # 1. Python packages
    try:
        import torch
        checks.append(f"✓ PyTorch {torch.__version__}")
        checks.append(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            checks.append(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError:
        checks.append("✗ PyTorch not installed")

    try:
        import transformers
        checks.append(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        checks.append("✗ Transformers not installed")

    try:
        import peft
        checks.append(f"✓ PEFT {peft.__version__}")
    except ImportError:
        checks.append("✗ PEFT not installed (pip install peft)")

    try:
        import sentence_transformers
        checks.append(f"✓ Sentence-Transformers")
    except ImportError:
        checks.append("✗ Sentence-Transformers not installed")

    # 2. Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        checks.append("✓ Redis server running")
    except:
        checks.append("✗ Redis not running (redis-server --daemonize yes)")

    # 3. Data directories
    from pathlib import Path
    for d in ["data/raw", "data/processed", "data/training", "output/results"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    checks.append("✓ Data directories created")

    print("\n".join(checks))
    return all("✓" in c for c in checks)

if __name__ == "__main__":
    success = verify()
    exit(0 if success else 1)
```

---

## API Configuration (Baselines)

### Secure API Key Setup

**Never commit API keys to git.** Use environment variables:

```bash
# Add to ~/.bashrc or ~/.zshrc (once)
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # Add later for GPT-4 baseline

# Or create a local .env file (gitignored)
echo 'ANTHROPIC_API_KEY=your-key-here' >> .env
echo 'OPENAI_API_KEY=your-key-here' >> .env
```

### Loading in Python

```python
# src/baselines/cloud_api.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify keys are set
assert ANTHROPIC_API_KEY, "Set ANTHROPIC_API_KEY environment variable"
```

### .gitignore Entry

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
```

---

## Baselines & SOTA Comparison

### Baseline Methods

| Category | Method | Description | Source |
|----------|--------|-------------|--------|
| **Cloud LLM** | GPT-4 API | Cloud-based, high latency baseline | OpenAI API |
| **Cloud LLM** | Claude API | Alternative cloud baseline | Anthropic API |
| **Traditional DB** | PostgreSQL + LLM | SQL query + local inference | Standard |
| **Cached** | No grounding | Pre-generated responses, no sync | Naive baseline |
| **Edge LLM** | Raw TinyLLaMA | No fine-tuning, prompt-only | HuggingFace |
| **Edge LLM** | Raw Phi-2 | No fine-tuning, prompt-only | HuggingFace |
| **Staleness** | Time-only | Flag stale if >5s old | Heuristic |
| **Staleness** | Threshold-based | Flag if sensor change >threshold | Heuristic |

### SOTA Comparison Matrix

```python
# experiments/baselines/sota_comparison.py

BASELINES = {
    # Latency baselines
    "cloud_gpt4": {
        "type": "cloud",
        "model": "gpt-4",
        "expected_latency": "500-1000ms"
    },
    "postgresql_llm": {
        "type": "traditional",
        "db": "postgresql",
        "model": "tinyllama",
        "expected_latency": "130ms"
    },

    # Grounding baselines
    "prompt_only": {
        "type": "no_finetune",
        "model": "tinyllama",
        "lora": False
    },
    "generic_llm": {
        "type": "no_context",
        "model": "tinyllama",
        "sensor_context": False
    },

    # Staleness baselines
    "time_threshold": {
        "type": "staleness",
        "method": "time_only",
        "threshold_sec": 5
    },
    "value_threshold": {
        "type": "staleness",
        "method": "value_change",
        "threshold_temp": 2.0
    }
}
```

### Expected Results Table

| Method | Latency | Grounding Acc | Staleness F1 | Causal F1 |
|--------|---------|---------------|--------------|-----------|
| Cloud GPT-4 | 500-1000ms | 98% | N/A | 85% |
| PostgreSQL + LLM | 130ms | 92% | N/A | 80% |
| Prompt-only | 60ms | 72% | N/A | 65% |
| Time-threshold | N/A | N/A | 0.76 | N/A |
| Value-threshold | N/A | N/A | 0.78 | N/A |
| **TGP (Ours)** | **<80ms** | **≥95%** | **≥0.90** | **≥0.90** |

---

## Ablation Studies

### Ablation Matrix

Systematically remove each component to measure contribution:

| Config | Buffer | LoRA | Staleness | Causal | Description |
|--------|--------|------|-----------|--------|-------------|
| `full` | ✓ | ✓ | ✓ | ✓ | Complete TGP system |
| `no_buffer` | ✗ | ✓ | ✓ | ✓ | Use PostgreSQL instead |
| `no_lora` | ✓ | ✗ | ✓ | ✓ | Prompt-only, no fine-tuning |
| `no_staleness` | ✓ | ✓ | ✗ | ✓ | No staleness detection |
| `no_causal` | ✓ | ✓ | ✓ | ✗ | No causal validation |
| `buffer_only` | ✓ | ✗ | ✗ | ✗ | Only Innovation 1 |
| `lora_only` | ✗ | ✓ | ✗ | ✗ | Only Innovation 2 |

### Ablation Script Pattern

```python
# experiments/05_ablation_study.py

ABLATION_CONFIGS = {
    "full_system": {
        "buffer": "redis",
        "model": "tinyllama",
        "lora": True,
        "staleness": True,
        "causal": True
    },
    "no_buffer": {
        "buffer": "postgresql",
        "model": "tinyllama",
        "lora": True,
        "staleness": True,
        "causal": True
    },
    "no_lora": {
        "buffer": "redis",
        "model": "tinyllama",
        "lora": False,
        "staleness": True,
        "causal": True
    },
    # ... more configs
}

def run_ablations():
    results = {}
    for name, config in ABLATION_CONFIGS.items():
        pipeline = build_pipeline(config)
        metrics = evaluate(pipeline, test_data)
        results[name] = metrics
    return results
```

### Expected Ablation Results

| Configuration | Latency | Grounding Acc | Staleness F1 | Notes |
|---------------|---------|---------------|--------------|-------|
| Full TGP | 80ms | 95% | 0.93 | Complete system |
| - No buffer | 140ms (+75%) | 95% | 0.93 | Redis critical for latency |
| - No LoRA | 80ms | 72% (-24%) | 0.93 | Fine-tuning critical for accuracy |
| - No staleness | 80ms | 95% | N/A | 18% stale responses |
| - No causal | 75ms | 95% | 0.93 | 30% causal errors |
| Buffer only | 40ms | 45% | N/A | Baseline |
| LoRA only | 130ms | 92% | N/A | No real-time capability |

---

## Computational Cost Tracking

### Metrics to Track

| Metric | Tool | Unit | Target |
|--------|------|------|--------|
| **GPU Memory** | `torch.cuda.max_memory_allocated()` | GB | <8GB |
| **GPU Utilization** | `nvidia-smi` | % | Report actual |
| **Training Time** | `time.time()` | hours | <4h |
| **Inference Latency** | `time.perf_counter()` | ms | <80ms |
| **Power Consumption** | `nvidia-smi --query-gpu=power.draw` | W | Report actual |
| **CO2 Emissions** | `codecarbon` | kg CO2 | Report actual |

### Performance Tracker Module

```python
# src/utils/performance_tracker.py

import torch
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class PerformanceMetrics:
    gpu_memory_gb: float
    gpu_utilization: float
    inference_time_ms: float
    power_watts: Optional[float]
    co2_kg: Optional[float]

class PerformanceTracker:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.start_time = None
        self.start_memory = None

    def start(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        self.start_memory = torch.cuda.memory_allocated(self.device)

    def stop(self) -> PerformanceMetrics:
        end_time = time.perf_counter()
        peak_memory = torch.cuda.max_memory_allocated(self.device)

        return PerformanceMetrics(
            gpu_memory_gb=peak_memory / 1e9,
            gpu_utilization=self._get_gpu_util(),
            inference_time_ms=(end_time - self.start_time) * 1000,
            power_watts=self._get_power(),
            co2_kg=None  # Computed post-hoc
        )

    def _get_gpu_util(self) -> float:
        # Parse nvidia-smi output
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split("\n")[0])

    def _get_power(self) -> float:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip().split("\n")[0])
```

### CO2 Tracking (Optional)

```bash
# Install codecarbon
pip install codecarbon

# In training script
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
# ... training code ...
emissions = tracker.stop()
print(f"CO2 emissions: {emissions:.4f} kg")
```

### Results Table Template

| Stage | GPU Memory | Time | Power | CO2 |
|-------|-----------|------|-------|-----|
| Dataset processing | 2GB | 30min | 150W | 0.05kg |
| LoRA fine-tuning | 8GB | 3h | 280W | 0.8kg |
| Inference (per query) | 4GB | 80ms | 200W | - |
| Full pipeline | 8GB | 4h | 250W | 1.0kg |

### Experiment Script Integration

```python
# experiments/01_latency_benchmark.py

from src.utils.performance_tracker import PerformanceTracker

def run_experiment():
    tracker = PerformanceTracker(device="cuda:0")
    results = {"methods": {}, "compute_costs": {}}

    for method_name, method in METHODS.items():
        # Track performance
        tracker.start()

        # Run benchmark
        latencies = []
        for query in test_queries:
            start = time.perf_counter()
            response = method.generate(query)
            latencies.append((time.perf_counter() - start) * 1000)

        # Stop tracking
        metrics = tracker.stop()

        # Store results
        results["methods"][method_name] = {
            "mean_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95)
        }
        results["compute_costs"][method_name] = {
            "gpu_memory_gb": metrics.gpu_memory_gb,
            "power_watts": metrics.power_watts
        }

    return results
```

---

## Common Commands

```bash
# Environment
conda activate llms

# Redis
redis-server --daemonize yes --port 6379  # Start
redis-cli ping                             # Test
redis-cli shutdown                         # Stop

# Training (TinyLLaMA with LoRA)
CUDA_VISIBLE_DEVICES=4 python scripts/train_grounding_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Training (Phi-2 with LoRA - alternative)
CUDA_VISIBLE_DEVICES=4 python scripts/train_grounding_model.py \
    --model microsoft/phi-2

# Experiments
CUDA_VISIBLE_DEVICES=4 python experiments/01_latency_benchmark.py

# Quick test
python scripts/quick_test.py

# Full pipeline
./run/run_all.sh
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection refused | Run `redis-server --daemonize yes` |
| CUDA out of memory | Reduce batch size or use GPU 5-7 |
| Import errors | Verify `conda activate llms` |
| Dataset download fails | Check internet / Kaggle API key |

---

**Last Updated**: 2025-12-22
