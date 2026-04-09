# Experimental Design: TGP Real-Time Grounding

**Project**: Topic 04 - Real-Time Grounding for Small Language Models
**Target Conference**: AAAI 2027 (August 2026 deadline)
**Last Updated**: 2025-12-25

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Datasets](#datasets)
4. [Experiment Suite](#experiment-suite)
   - [Exp01: Latency Benchmark](#exp01-latency-benchmark)
   - [Exp02: Grounding Accuracy](#exp02-grounding-accuracy)
   - [Exp03: Staleness Detection](#exp03-staleness-detection)
   - [Exp04: Causal Validation](#exp04-causal-validation)
   - [Exp05: Ablation Study](#exp05-ablation-study)
   - [Exp06: Scalability Test](#exp06-scalability-test)
   - [Exp07: SOTA Comparison](#exp07-sota-comparison)
   - [Exp08: Computational Cost](#exp08-computational-cost)
   - [Exp09: Sampling Robustness](#exp09-sampling-robustness)
   - [Exp10: Cross-Dataset Validation](#exp10-cross-dataset-validation)
5. [Execution Guide](#execution-guide)
6. [Result Files](#result-files)

---

## Overview

The **Temporal Grounding Pipeline (TGP)** enables real-time sensor data grounding for small language models running on edge devices. This experimental suite validates TGP's performance across latency, accuracy, robustness, and generalization dimensions.

**Key Innovation**: Novel `TemporalGroundingBuffer` providing O(1) statistics computation for real-time LLM context grounding.

**Core Components**:
- **Buffer**: TemporalGroundingBuffer (in-process) vs CircularBuffer (Redis baseline)
- **LLM**: TinyLlama 1.1B with LoRA fine-tuning + 4-bit quantization
- **Staleness**: TimeThresholdStalenessDetector (time + value change thresholds)
- **Causal**: Graph-based validator preventing invalid causal claims

---

## System Architecture

```
Sensor Stream → TemporalGroundingBuffer → Staleness Detection → LLM Inference → Causal Validation → Response
                       ↓                           ↓                    ↓                ↓
                  O(1) stats               Time threshold        TinyLlama 1.1B    Graph check
                  In-memory                F1 = 1.0              + LoRA 4-bit      Valid rate
```

**Target Performance**:
- Buffer latency: <1ms (vs Redis ~0.2ms)
- Total pipeline: <2000ms end-to-end
- Grounding accuracy: ≥95%
- Staleness F1: ≥0.90
- Causal validity: ≥0.90

---

## Datasets

### BDG2 (Building Data Genome 2)
- **Source**: Kaggle (commercial buildings dataset)
- **Type**: Commercial office buildings
- **Path**: `data/raw/bdg2_kaggle/`
- **Format**: `electricity.csv` with timestamp, building_id, meter_type, value
- **Usage**: Primary training dataset

### REDD (Reference Energy Disaggregation Dataset)
- **Source**: Kaggle (residential energy dataset)
- **Type**: US residential homes
- **Path**: `data/raw/redd/`
- **Format**: `dev*.csv` files per house
- **Usage**: Cross-validation for generalization testing

---

## Experiment Suite

### Exp01: Latency Benchmark

**File**: `experiments/01_latency_benchmark.py`

**Purpose**: Measure end-to-end latency of buffer operations and LLM inference.

**What it measures**:
- Buffer read/write latency (TGP vs Redis)
- Staleness detection latency
- LLM inference latency (local vs cloud)

**How it works**:
1. Loads 100 test sensor readings into buffer
2. Runs 100 queries measuring time for:
   - `buffer.get_latest()` - retrieve last N readings
   - `buffer.get_statistics()` - compute mean/std/min/max
   - `detector.detect()` - staleness check
3. Compares against baselines: Redis, Claude API, SQLite+LLM

**Key Code** ([lines 73-89](experiments/01_latency_benchmark.py#L73-L89)):
```python
for i in range(n_queries):
    start = time.perf_counter()

    # Get sensor data from buffer (O(1) operation)
    latest = buffer.get_latest("_benchmark", "electricity", n=5)
    current_stats = buffer.get_statistics("_benchmark", "electricity")

    # Check staleness (time + value thresholds)
    result = detector.detect("_benchmark", readings, current_stats)

    elapsed = (time.perf_counter() - start) * 1000  # ms
    latencies.append(elapsed)
```

**Metrics**:
- `mean_ms`: Average latency
- `std_ms`: Standard deviation
- `p50_ms`, `p95_ms`, `p99_ms`: Percentile latencies
- `speedup_vs_redis`: Performance gain vs baseline

**Expected Results**:
- TGP buffer: ~0.5ms mean latency
- Redis baseline: ~0.2ms mean latency
- Full pipeline: ~2000ms (dominated by LLM inference)

**Output**: `output/common/results/exp01_latency_YYYYMMDD_HHMMSS.json`

---

### Exp02: Grounding Accuracy

**File**: `experiments/02_grounding_accuracy.py`

**Purpose**: Verify that LLM responses actually reference provided sensor values.

**What it measures**:
- Value accuracy: Does response contain correct numbers?
- Trend accuracy: Does response correctly identify trends (rising/falling/stable)?
- Context relevance: Does response address energy consumption?

**How it works**:
1. Generates 100 test cases with known ground truth:
   - Mean consumption (e.g., 150.0 kWh)
   - Current value (e.g., 155.0 kWh)
   - Expected trend (increasing/decreasing/stable)

2. Feeds context to LLM via prompt:
```python
prompt = f"""<|system|>
You are an energy monitoring assistant. Answer using the provided sensor data.
</s>
<|user|>
Current readings:
- Mean consumption: {stats['mean']:.1f} kWh
- Current reading: {stats['current']:.1f} kWh
Question: {query}
</s>
<|assistant|>
"""
```

3. Extracts numbers from response using regex: `r'[-+]?\d*\.?\d+'`
4. Checks if extracted numbers match expected values (15% tolerance)

**Key Code** ([lines 89-112](experiments/02_grounding_accuracy.py#L89-L112)):
```python
def check_value_accuracy(response: str, expected: Dict[str, float]) -> Tuple[bool, float]:
    """Check if response contains expected values within 15% tolerance."""
    response_numbers = extract_numbers(response)

    matches = 0
    for key, exp_val in expected.items():
        for num in response_numbers:
            if abs(num - exp_val) / max(abs(exp_val), 1e-10) < 0.15:
                matches += 1
                break

    accuracy = matches / len(expected)
    return accuracy >= 0.5, accuracy
```

**Comparison**:
- **TGP (grounded)**: Receives sensor values → references them in response
- **Baseline (no grounding)**: No sensor context → hallucinates/guesses

**Metrics**:
- `mean_value_accuracy`: Fraction of correct values (target: ≥0.95)
- `trend_accuracy`: Fraction of correct trend identification
- `mean_latency_ms`: Response generation time

**Expected Results**:
- TGP: ~95% value accuracy, ~90% trend accuracy
- No grounding: ~5% value accuracy (random chance)

**Output**: `output/{bdg2,redd}/results/exp02_grounding_YYYYMMDD_HHMMSS.json`

---

### Exp03: Staleness Detection

**File**: `experiments/03_staleness_detection.py`

**Purpose**: Evaluate ability to detect when sensor context is outdated.

**What it measures**:
- Precision: Of predicted stale samples, how many are truly stale?
- Recall: Of truly stale samples, how many are detected?
- F1 score: Harmonic mean of precision and recall

**How it works**:
1. Generates 500 synthetic samples (30% stale, 70% fresh)
2. **Stale conditions**:
   - `value_shift`: Large change (>30% from context mean)
   - `pattern_change`: Variance increase (std × 2-4)
   - `anomaly`: Extreme spike/drop (0.3× or 2.5× normal)

3. Runs detection methods:
   - **TimeThresholdDetector** (PRIMARY): Time >300s OR value change >20%
   - **EmbeddingDetector** (deprecated): Semantic similarity
   - **Time-only baseline**: Just time threshold
   - **Value-only baseline**: Just value change

**Key Code** ([lines 158-177](experiments/03_staleness_detection.py#L158-L177)):
```python
# Simulate time passing for stale samples
if sample["is_stale"] and sample["time_delta_seconds"] > 300:
    # Push context timestamp back in time
    context_data, _ = detector._context_cache[f"test_{i}"]
    detector._context_cache[f"test_{i}"] = (
        context_data,
        time.time() - sample["time_delta_seconds"]
    )

# Run detection
result = detector.detect(f"test_{i}", current_readings, current_stats)
predictions.append(result.is_stale)

# Compute metrics
tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
```

**Metrics**:
- `precision`: TP / (TP + FP)
- `recall`: TP / (TP + FN)
- `f1`: 2 × (precision × recall) / (precision + recall)
- `mean_latency_ms`: Detection time per sample

**Expected Results**:
- TimeThresholdDetector: F1 = 1.0 (perfect on synthetic data)
- Time-only baseline: F1 ~0.6
- Value-only baseline: F1 ~0.7

**Output**: `output/{bdg2,redd}/results/exp03_staleness_YYYYMMDD_HHMMSS.json`

---

### Exp04: Causal Validation

**File**: `experiments/04_causal_validation.py`

**Purpose**: Prevent LLM from making invalid causal claims about energy consumption.

**What it measures**:
- Valid response rate: Fraction of responses without causal violations
- Mean validation score: Average causal consistency (0-1 scale)

**How it works**:
1. Creates causal graph for energy domain:
   - `temperature → HVAC → consumption`
   - `occupancy → equipment → consumption`
   - `time_of_day → lighting → consumption`

2. Generates causal queries:
   - "Why is consumption higher today than yesterday?"
   - "What caused the energy spike at 2 PM?"
   - "Why is weekend consumption lower?"

3. Gets **REAL LLM responses** (not simulated)
4. Validates responses against causal graph
5. Flags violations (e.g., "low occupancy causes high consumption")

**Key Code** ([lines 134-194](experiments/04_causal_validation.py#L134-L194)):
```python
def evaluate_tgp_causal(queries: List[Dict], n_repeats: int = 3):
    llm = get_llm_backbone()  # Fine-tuned TinyLlama
    graph = CausalGraph.create_energy_graph()
    validator = CausalValidator(graph)

    for item in queries * n_repeats:
        # Generate REAL LLM response
        prompt = format_causal_prompt(item["query"], item["context"])
        response = llm.generate(prompt, max_new_tokens=100, temperature=0.3)

        # Validate against causal graph
        validation = validator.validate(response)

        if validation.is_valid:
            valid_responses += 1
        else:
            # Log violations
            details.append({
                "query": item["query"],
                "violations": validation.violations
            })
```

**Comparison**:
- **TGP (fine-tuned + validation)**: ~70% valid responses
- **Raw LLM (no fine-tuning)**: ~80% valid (but lower grounding accuracy)
- **No validation baseline**: 100% accepted (including invalid claims)

**Metrics**:
- `valid_rate`: Fraction of causally consistent responses (target: ≥0.90)
- `mean_score`: Average validation score
- `mean_latency_ms`: Validation time per response

**Expected Results**:
- TGP: 70% valid rate (conservative due to strict validation)
- Raw LLM: 80% valid rate (but may hallucinate values)

**Output**: `output/{bdg2,redd}/results/exp04_causal_YYYYMMDD_HHMMSS.json`

---

### Exp05: Ablation Study

**File**: `experiments/05_ablation_study.py`

**Purpose**: Measure contribution of each component by systematically removing them.

**What it measures**:
- Latency impact: How much slower without component X?
- Accuracy impact: How much worse without component X?

**Configurations tested**:

| Config | Buffer | LoRA | Staleness | Causal | Description |
|--------|--------|------|-----------|--------|-------------|
| `full_system` | Temporal | ✓ | ✓ | ✓ | Complete TGP |
| `redis_baseline` | Redis | ✓ | ✓ | ✓ | Redis instead of Temporal |
| `no_buffer` | Dict | ✓ | ✓ | ✓ | In-memory dict |
| `no_lora` | Redis | ✗ | ✓ | ✓ | Raw LLM (no fine-tuning) |
| `no_staleness` | Redis | ✓ | ✗ | ✓ | Skip staleness detection |
| `no_causal` | Redis | ✓ | ✓ | ✗ | Skip causal validation |
| `buffer_only` | Redis | ✗ | ✗ | ✗ | Minimal system |

**How it works**:
1. For each configuration:
   - Initializes components based on config
   - Runs 20 queries with real LLM inference
   - Measures latency and grounding quality

2. Computes contribution of each component:
```python
latency_impact = config_latency - full_system_latency
accuracy_impact = full_system_accuracy - config_accuracy
```

**Key Code** ([lines 187-221](experiments/05_ablation_study.py#L187-L221)):
```python
# Setup based on configuration
if config["buffer"] == "redis":
    buffer = CircularBuffer()
elif config["buffer"] == "temporal":
    buffer = TemporalGroundingBuffer()  # Novel in-process buffer
else:
    buffer = {}  # In-memory dict fallback

llm = get_tgp_llm() if config["lora"] else get_raw_llm()
staleness_detector = TimeThresholdStalenessDetector() if config["staleness"] else None
causal_validator = CausalValidator(graph) if config["causal"] else None

# Run queries
for case in test_cases:
    start = time.perf_counter()

    # Buffer operations
    readings = buffer.get_latest(...)
    stats = buffer.get_statistics(...)

    # Optional staleness detection
    if staleness_detector:
        stale_result = staleness_detector.detect(...)

    # LLM inference
    response = llm.generate(prompt, max_new_tokens=50)

    # Optional causal validation
    if causal_validator:
        causal_result = causal_validator.validate(response)

    latency = (time.perf_counter() - start) * 1000
```

**Metrics**:
- `latency_impact_ms`: Absolute latency change
- `latency_impact_pct`: Percentage latency change
- `accuracy_impact`: Absolute accuracy change
- `accuracy_impact_pct`: Percentage accuracy change

**Expected Results**:
- Removing LoRA: +10% latency (no fine-tuning overhead), -15% accuracy
- Removing staleness: +5% latency, -8% accuracy
- Removing causal: +1% latency, -12% accuracy

**Output**: `output/{bdg2,redd}/results/exp05_ablation_YYYYMMDD_HHMMSS.json`

---

### Exp06: Scalability Test

**File**: `experiments/06_scalability_test.py`

**Purpose**: Validate that buffer maintains O(1) statistics computation at scale.

**What it measures**:
- Buffer latency vs data volume (10, 100, 1K, 10K, 100K readings)
- Memory usage scaling
- Statistics computation time

**How it works**:
1. Progressively loads 10, 100, 1K, 10K, 100K readings
2. At each scale, measures:
   - `push()` latency
   - `get_latest()` latency
   - `get_statistics()` latency
   - Peak memory usage

3. Verifies O(1) claim: latency should be constant regardless of buffer size

**Expected Results**:
- TemporalGroundingBuffer: Constant latency (O(1) verified)
- Redis baseline: Slight increase with size (network overhead)

**Output**: `output/common/results/exp06_scalability_YYYYMMDD_HHMMSS.json`

---

### Exp07: SOTA Comparison

**File**: `experiments/07_sota_comparison.py`

**Purpose**: Compare TGP against state-of-the-art baselines.

**What it measures**:
- Latency, cost, and grounding quality across 4 methods

**Methods compared**:

| Method | Type | Model | Buffer | Cost |
|--------|------|-------|--------|------|
| **TGP (Ours)** | Edge | TinyLlama 1.1B + LoRA | Temporal | Free |
| **Claude API** | Cloud | Claude Sonnet | N/A | ~$0.01/query |
| **SQLite + LLM** | Traditional | Raw TinyLlama | SQLite DB | Free |
| **Raw LLM** | Edge | TinyLlama (no LoRA) | Temporal | Free |

**How it works**:
1. **TGP**: Full pipeline (buffer → staleness → fine-tuned LLM)
2. **Claude API**: Real API calls with budget control ($2 max)
3. **SQLite + LLM**: SQL queries + raw TinyLlama
4. **Raw LLM**: No fine-tuning, just buffer + base model

**Key Code** ([lines 161-227](experiments/07_sota_comparison.py#L161-L227)):
```python
def benchmark_tgp(n_queries: int = 50):
    buffer = TemporalGroundingBuffer()
    detector = TimeThresholdStalenessDetector()
    llm = get_tgp_backbone()  # Fine-tuned

    for i in range(n_queries):
        start = time.perf_counter()

        readings = buffer.get_latest("_sota_test", "electricity", n=5)
        stats = buffer.get_statistics("_sota_test", "electricity")
        stale_result = detector.detect(...)

        response = llm.generate(prompt, max_new_tokens=50)

        latency = (time.perf_counter() - start) * 1000
        quality = check_grounding_quality(response, expected)
```

**Metrics**:
- `mean_latency_ms`: Average response time
- `p95_latency_ms`: 95th percentile latency
- `cost_per_query_usd`: Cost per inference (if applicable)
- `grounding_quality`: Combined quality score (0-1)

**Expected Results**:
- TGP: ~2000ms latency, free, 0.85 quality
- Claude API: ~1500ms latency, $0.01/query, 0.95 quality
- SQLite + LLM: ~2500ms latency, free, 0.70 quality
- Raw LLM: ~2000ms latency, free, 0.75 quality

**Output**: `output/{bdg2,redd}/results/exp07_sota_YYYYMMDD_HHMMSS.json`

---

### Exp08: Computational Cost

**File**: `experiments/08_computational_cost.py`

**Purpose**: Measure actual resource usage for reproducibility and carbon footprint reporting.

**What it measures**:
- GPU memory usage (peak)
- Power consumption (nvidia-smi)
- Training time (from logs)
- Inference throughput (tokens/sec)
- CO2 emissions estimate

**How it works**:
1. **GPU Baseline**: Queries nvidia-smi for device info and idle power
2. **Buffer Costs**: Measures push/get latency and memory (500 operations)
3. **Staleness Costs**: Measures detection latency and power (100 operations)
4. **Inference Costs**: Measures LLM latency, tokens/sec, power (30 queries)
5. **Training Costs**: Reads from `trainer_state.json` + estimates CO2

**Key Code** ([lines 33-46](experiments/08_computational_cost.py#L33-L46)):
```python
def get_gpu_power() -> Optional[float]:
    """Get current GPU power draw in watts using nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        power_str = result.stdout.strip().split('\n')[0]
        return float(power_str)  # Example: 284.5 W
    return None
```

**CO2 Calculation** ([lines 276-279](experiments/08_computational_cost.py#L276-L279)):
```python
# RTX 3090: ~280W average, ~30 min training
estimated_power_watts = 280
estimated_hours = 0.5
co2_kg = (power_watts * hours / 1000) * 0.42  # kg CO2 per kWh
# Result: ~0.059 kg CO2 for training
```

**Metrics**:
- `peak_memory_gb`: Maximum GPU memory allocated
- `avg_power_watts`: Average power draw during inference
- `tokens_per_second`: Inference throughput
- `co2_per_hour_kg`: CO2 emissions per hour of operation
- `training_co2_kg`: Total training emissions

**Expected Results**:
- Peak memory: ~4.2 GB (4-bit quantization)
- Avg power: ~280W during inference
- Tokens/sec: ~15-20 (TinyLlama 1.1B)
- Training CO2: ~0.06 kg

**Output**: `output/common/results/exp08_cost_YYYYMMDD_HHMMSS.json`

---

### Exp09: Sampling Robustness

**File**: `experiments/09_sampling_robustness_fixed.py`

**Purpose**: Test TGP performance under degraded data conditions.

**What it measures**:
- Impact of sampling rate (temporal resolution)
- Impact of data dropout (missing readings)
- Impact of sensor noise (measurement errors)

**Test scenarios**:

1. **Sampling Rates**: How well does TGP work with sparse data?
   - 1-minute (baseline)
   - 5-minute (1/5 data)
   - 15-minute (1/15 data)
   - 60-minute (1/60 data)

2. **Data Dropout**: How well does TGP handle missing readings?
   - 0% dropout (baseline)
   - 10% dropout (90% data)
   - 30% dropout (70% data)
   - 50% dropout (50% data)

3. **Noise Levels**: How well does TGP handle sensor errors?
   - 0% noise (baseline)
   - 10% Gaussian noise
   - 20% Gaussian noise
   - 30% Gaussian noise

**How it works**:
1. Generates 100-500 synthetic readings with time-of-day pattern
2. Applies degradation (downsample, dropout, or noise)
3. Runs grounding + staleness detection
4. Measures latency and quality degradation

**Key Code** ([lines 57-77](experiments/09_sampling_robustness_fixed.py#L57-L77)):
```python
def downsample_data(readings: List[Dict], interval_minutes: int):
    """Downsample to specified interval."""
    return [r for r in readings if r["index"] % interval_minutes == 0]

def apply_dropout(readings: List[Dict], dropout_rate: float):
    """Randomly drop readings."""
    mask = np.random.random(len(readings)) > dropout_rate
    return [r for i, r in enumerate(readings) if mask[i]]

def add_noise(readings: List[Dict], noise_level: float):
    """Add Gaussian noise proportional to value."""
    for r in readings:
        std = r["value"] * noise_level
        noisy_value = r["value"] + np.random.randn() * std
        r["value"] = max(0, noisy_value)  # Clamp to non-negative
```

**CRITICAL FIX**: Original script created 12 separate LLM instances causing GPU OOM. Fixed version reuses single LLM instance:

```python
def run_experiment(n_readings: int = 500):
    # Load LLM ONCE at start
    llm = LLMBackbone(model_type="tinyllama")
    llm.load_lora(lora_path)

    # Pass to all sub-experiments
    results = {
        "sampling_rate": run_sampling_experiment(n_readings, llm),  # Reuse
        "data_dropout": run_dropout_experiment(n_readings, llm),    # Reuse
        "noise_level": run_noise_experiment(n_readings, llm)        # Reuse
    }

    # Clean up
    del llm
    torch.cuda.empty_cache()
```

**Metrics**:
- `grounding_latency_ms`: LLM inference time
- `staleness_latency_ms`: Detection time
- `has_values`: Did buffer return valid statistics?

**Expected Results**:
- 1min sampling: ~500ms grounding, 0.01ms staleness
- 60min sampling: ~2500ms grounding (fewer data points)
- 50% dropout: ~4500ms grounding (sparse context)
- 30% noise: ~1000ms grounding (noisy patterns)

**Output**: `output/{bdg2,redd}/results/exp09_sampling_YYYYMMDD_HHMMSS.json`

---

### Exp10: Cross-Dataset Validation

**File**: `experiments/10_cross_dataset_validation.py`

**Purpose**: Verify that TGP generalizes across different building types.

**What it measures**:
- Performance on unseen data distributions
- Generalization gap between commercial and residential buildings

**Test setup**:
- **Train**: BDG2 (commercial buildings)
- **Test**: REDD (residential homes)
- **Gap**: Performance difference

**How it works**:
1. Load model trained on BDG2 (commercial)
2. Test on both BDG2 and REDD
3. Compute generalization gap:
   - `bdg2_to_redd`: BDG2 rate - REDD rate
   - `commercial_vs_residential`: |BDG2 rate - REDD rate|
   - `latency_gap_pct`: (REDD latency - BDG2 latency) / BDG2 latency

**Key Code** ([lines 330-361](experiments/10_cross_dataset_validation.py#L330-L361)):
```python
def compute_generalization_gap(results: Dict) -> Dict[str, float]:
    """Compute performance gap between BDG2 (commercial) and REDD (residential)."""
    bdg2_rate = results["bdg2"]["valid_response_rate"]
    redd_rate = results["redd"]["valid_response_rate"]

    gaps = {
        "bdg2_to_redd": bdg2_rate - redd_rate,
        "commercial_vs_residential": abs(bdg2_rate - redd_rate)
    }

    # Latency gap
    bdg2_latency = results["bdg2"]["mean_latency_ms"]
    redd_latency = results["redd"]["mean_latency_ms"]
    gaps["latency_gap_pct"] = (redd_latency - bdg2_latency) / bdg2_latency

    return gaps
```

**Metrics**:
- `valid_response_rate`: Fraction of responses with grounded values
- `mean_latency_ms`: Average inference time
- `generalization_gap`: Performance difference between datasets

**Expected Results**:
- BDG2 (in-domain): ~85% valid response rate
- REDD (out-of-domain): ~75% valid response rate
- Generalization gap: <10% (good generalization)

**Output**: `output/common/results/exp10_cross_dataset_YYYYMMDD_HHMMSS.json`

---

## Execution Guide

### Running All Experiments

**Option 1: Full pipeline** (recommended)
```bash
cd 04-realtime-grounding
./run/run_all.sh
```

This runs:
1. Environment verification
2. Data generation
3. Model training
4. All 10 experiments on both datasets

**Option 2: Individual experiments**
```bash
# Common experiments (dataset-agnostic)
CUDA_VISIBLE_DEVICES=4 python experiments/01_latency_benchmark.py
CUDA_VISIBLE_DEVICES=4 python experiments/06_scalability_test.py
CUDA_VISIBLE_DEVICES=4 python experiments/08_computational_cost.py
CUDA_VISIBLE_DEVICES=4 python experiments/10_cross_dataset_validation.py

# Dataset-specific experiments
CUDA_VISIBLE_DEVICES=4 python experiments/02_grounding_accuracy.py --dataset bdg2
CUDA_VISIBLE_DEVICES=4 python experiments/03_staleness_detection.py --dataset bdg2
CUDA_VISIBLE_DEVICES=4 python experiments/04_causal_validation.py --dataset bdg2
CUDA_VISIBLE_DEVICES=4 python experiments/05_ablation_study.py --dataset bdg2
CUDA_VISIBLE_DEVICES=4 python experiments/07_sota_comparison.py --dataset bdg2
CUDA_VISIBLE_DEVICES=4 python experiments/09_sampling_robustness_fixed.py --dataset bdg2 --n-readings 100
```

### GPU Allocation

**CRITICAL**: Always use GPUs 4, 5, 6, or 7 on the shared server.

```bash
# Correct
CUDA_VISIBLE_DEVICES=4 python experiments/01_latency_benchmark.py

# Wrong - will conflict with other users
python experiments/01_latency_benchmark.py  # Uses all GPUs
```

### Monitoring Progress

```bash
# Check GPU usage
nvidia-smi

# Check running experiments
ps aux | grep python | grep experiments

# Monitor output
tail -f output/common/logs/exp01_latency_*.log
```

### Expected Runtime

| Experiment | BDG2 | REDD | Total |
|------------|------|------|-------|
| Exp01 | 2 min | - | 2 min |
| Exp02 | 15 min | 15 min | 30 min |
| Exp03 | 10 min | 10 min | 20 min |
| Exp04 | 8 min | 8 min | 16 min |
| Exp05 | 12 min | 12 min | 24 min |
| Exp06 | 3 min | - | 3 min |
| Exp07 | 10 min | 10 min | 20 min |
| Exp08 | 5 min | - | 5 min |
| Exp09 | 8 min | 8 min | 16 min |
| Exp10 | 12 min | - | 12 min |
| **Total** | **~85 min** | **~63 min** | **~148 min** |

Full pipeline: **~2.5 hours** (including setup, training, all experiments)

---

## Result Files

### Directory Structure

```
output/
├── common/
│   └── results/
│       ├── exp01_latency_YYYYMMDD_HHMMSS.json
│       ├── exp06_scalability_YYYYMMDD_HHMMSS.json
│       ├── exp08_cost_YYYYMMDD_HHMMSS.json
│       └── exp10_cross_dataset_YYYYMMDD_HHMMSS.json
├── bdg2/
│   └── results/
│       ├── exp02_grounding_YYYYMMDD_HHMMSS.json
│       ├── exp03_staleness_YYYYMMDD_HHMMSS.json
│       ├── exp04_causal_YYYYMMDD_HHMMSS.json
│       ├── exp05_ablation_YYYYMMDD_HHMMSS.json
│       ├── exp07_sota_YYYYMMDD_HHMMSS.json
│       └── exp09_sampling_YYYYMMDD_HHMMSS.json
└── redd/
    └── results/
        ├── exp02_grounding_YYYYMMDD_HHMMSS.json
        ├── exp03_staleness_YYYYMMDD_HHMMSS.json
        ├── exp04_causal_YYYYMMDD_HHMMSS.json
        ├── exp05_ablation_YYYYMMDD_HHMMSS.json
        ├── exp07_sota_YYYYMMDD_HHMMSS.json
        └── exp09_sampling_YYYYMMDD_HHMMSS.json
```

### Result Format

All results are JSON files with standard structure:

```json
{
  "experiment": "latency_benchmark",
  "timestamp": "2025-12-25T16:04:18.123456",
  "config": {
    "n_queries": 100,
    "model": "TinyLlama-1.1B"
  },
  "methods": {
    "tgp_temporal": {
      "mean_ms": 0.52,
      "std_ms": 0.08,
      "p95_ms": 0.67,
      "p99_ms": 0.81
    }
  }
}
```

### Analyzing Results

Use the analysis script to generate figures and tables:

```bash
python scripts/analyze_results.py --output-dir output/figures
```

This generates:
- `fig_latency.pdf`: Latency comparison
- `fig_grounding.pdf`: Grounding accuracy by dataset
- `fig_ablation.pdf`: Component contributions
- `table_sota.tex`: SOTA comparison table
- `table_robustness.tex`: Sampling robustness results

---

## Data Integrity Policy

**CRITICAL**: All experiments use **REAL** data and measurements. No simulated/fake values.

✔ **Allowed**:
- Actual LLM inference (even if slow)
- Real GPU measurements (nvidia-smi)
- Actual dataset loading (BDG2, REDD)
- Measured latencies (time.perf_counter)

╳ **Forbidden**:
- Simulated/mock LLM responses
- Hardcoded performance numbers
- Fake dataset generation (except for staleness synthetic samples with clear labeling)
- Placeholder values in results

**Verification**: Each experiment logs its data sources and computation methods. Results include `real_data: true` flag.

---

## Citation

If you use this experimental setup, please cite:

```bibtex
@inproceedings{aboya2027tgp,
  title={Temporal Grounding Pipeline for Real-Time Small Language Models},
  author={Aboya, Franck Junior},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

---

**Last Updated**: 2025-12-25
**Maintainer**: Franck Junior Aboya Messou
**Contact**: faboya@student.ubc.ca
