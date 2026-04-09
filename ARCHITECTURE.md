# TGP Architecture: System Design & Component Interconnections

**Project**: Real-Time Grounding for Small Language Models (Topic 04)
**Target**: AAAI 2027 Conference
**Last Updated**: 2025-12-25

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Component Interconnection Map](#component-interconnection-map)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Experimental Coverage Map](#experimental-coverage-map)
5. [Baseline Comparison Architecture](#baseline-comparison-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Module Dependencies](#module-dependencies)

---

## System Architecture Overview

### High-Level Component View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TGP SYSTEM ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │ Sensor Stream│
                                    │  (IoT Data)  │
                                    └───────┬──────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────┐
                    │   TEMPORAL GROUNDING BUFFER (TGB)     │
                    │  ┌─────────────────────────────────┐  │
                    │  │ In-Memory Hash Table            │  │
                    │  │ Key: (building_id, meter_type)  │  │
                    │  │ Value: Deque[SensorReading]     │  │
                    │  └─────────────────────────────────┘  │
                    │  ┌─────────────────────────────────┐  │
                    │  │ Running Statistics (O(1))       │  │
                    │  │ • Mean, Std, Min, Max           │  │
                    │  │ • Trend detection               │  │
                    │  │ • Count, Sum, Sum²              │  │
                    │  └─────────────────────────────────┘  │
                    └───────┬───────────────────┬───────────┘
                            │                   │
                ┌───────────▼──────┐    ┌──────▼────────────┐
                │  get_latest()    │    │  get_statistics() │
                │  Returns: List[] │    │  Returns: Dict{}  │
                └───────────┬──────┘    └──────┬────────────┘
                            │                   │
                            └──────────┬────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │   STALENESS DETECTION MODULE         │
                    │  ┌────────────────────────────────┐  │
                    │  │ TimeThresholdStalenessDetector │  │
                    │  │ • Context cache: LRU           │  │
                    │  │ • Time threshold: 300s         │  │
                    │  │ • Value change: 20%            │  │
                    │  └────────────────────────────────┘  │
                    │  ┌────────────────────────────────┐  │
                    │  │ set_context()                  │  │
                    │  │ detect() → StalenessResult     │  │
                    │  │ clear_context()                │  │
                    │  └────────────────────────────────┘  │
                    └───────────┬──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────────┐
                    │        LLM BACKBONE MODULE           │
                    │  ┌────────────────────────────────┐  │
                    │  │ TinyLlama-1.1B-Chat            │  │
                    │  │ • 4-bit quantization (BNB)     │  │
                    │  │ • LoRA rank=16, alpha=32       │  │
                    │  │ • Target: q_proj, v_proj       │  │
                    │  │ • Trainable params: 4.5M       │  │
                    │  └────────────────────────────────┘  │
                    │  ┌────────────────────────────────┐  │
                    │  │ format_grounding_prompt()      │  │
                    │  │ generate() → Response          │  │
                    │  │ load_lora() / save_lora()      │  │
                    │  └────────────────────────────────┘  │
                    └───────────┬──────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────────┐
                    │    CAUSAL VALIDATION MODULE          │
                    │  ┌────────────────────────────────┐  │
                    │  │ CausalGraph                    │  │
                    │  │ Nodes: [temperature, HVAC,     │  │
                    │  │         occupancy, equipment,  │  │
                    │  │         consumption, ...]      │  │
                    │  │ Edges: temperature → HVAC      │  │
                    │  │        HVAC → consumption       │  │
                    │  │        occupancy → equipment    │  │
                    │  └────────────────────────────────┘  │
                    │  ┌────────────────────────────────┐  │
                    │  │ CausalValidator                │  │
                    │  │ validate() → ValidationResult  │  │
                    │  │ • Extracts claims from text    │  │
                    │  │ • Checks against graph         │  │
                    │  │ • Returns violations list      │  │
                    │  └────────────────────────────────┘  │
                    └───────────┬──────────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Final Response│
                        │  (Grounded,   │
                        │   Fresh,      │
                        │   Validated)  │
                        └───────────────┘
```

---

## Component Interconnection Map

### Internal Component Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     COMPONENT INTERCONNECTION GRAPH                      │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  SensorReading   │◄─────── Dataclass
                         │  (timestamp,     │         (immutable)
                         │   building_id,   │
                         │   meter_type,    │
                         │   value)         │
                         └────────┬─────────┘
                                  │ consumed by
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌────────────────┐     ┌──────────────────┐    ┌─────────────────┐
│ CircularBuffer │     │TemporalGrounding │    │  Statistics     │
│   (Redis)      │     │     Buffer       │    │  Computer       │
│                │     │  (In-process)    │    │                 │
│ • push()       │     │  • push()        │    │ • compute()     │
│ • get_latest() │     │  • get_latest()  │    │ • update()      │
│ • get_stats()  │     │  • get_stats()   │    │ • get_trend()   │
│ • clear()      │     │  • clear()       │    └────────┬────────┘
└────────┬───────┘     └────────┬─────────┘             │
         │                      │                        │
         └──────────┬───────────┘                        │
                    │ implements                         │
                    ▼                                    │
         ┌──────────────────────┐                       │
         │   BufferInterface    │                       │
         │   (Abstract Base)    │                       │
         └──────────────────────┘                       │
                    │                                    │
                    │ used by                            │
                    ▼                                    │
         ┌──────────────────────┐                       │
         │ StalenessDetector    │◄──────────────────────┘
         │ Base Class           │
         └──────────┬───────────┘
                    │
         ┌──────────┴──────────────┐
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ TimeThreshold    │    │ EmbeddingBased       │
│ StalenessDetector│    │ StalenessDetector    │
│ (PRIMARY)        │    │ (DEPRECATED)         │
│                  │    │                      │
│ • F1 = 1.0       │    │ • F1 = 0.35          │
│ • Latency: ~1ms  │    │ • Latency: ~50ms     │
└──────────┬───────┘    └──────────────────────┘
           │
           │ provides context to
           │
           ▼
┌──────────────────────────────────┐
│      LLMBackbone                 │
│                                  │
│  ┌────────────────────────────┐ │
│  │ ModelConfig                │ │
│  │ • model_name               │ │
│  │ • use_4bit                 │ │
│  │ • use_lora                 │ │
│  │ • lora_config              │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │ Tokenizer                  │ │
│  │ (AutoTokenizer)            │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │ Model                      │ │
│  │ (BitsAndBytes 4-bit)       │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │ LoRA Adapter               │ │
│  │ (PeftModel)                │ │
│  └────────────────────────────┘ │
└───────────┬──────────────────────┘
            │
            │ generates
            │
            ▼
┌──────────────────────────────────┐
│     CausalValidator              │
│                                  │
│  ┌────────────────────────────┐ │
│  │ CausalGraph                │ │
│  │ • NetworkX DiGraph         │ │
│  │ • Nodes: Variables         │ │
│  │ • Edges: Causal relations  │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │ ClaimExtractor             │ │
│  │ • Regex patterns           │ │
│  │ • Causal keywords          │ │
│  └────────────────────────────┘ │
│                                  │
│  ┌────────────────────────────┐ │
│  │ GraphChecker               │ │
│  │ • has_path()               │ │
│  │ • check_violation()        │ │
│  └────────────────────────────┘ │
└──────────────────────────────────┘
```

### Cross-Component Data Flow

```
INPUT: Sensor Reading
    │
    ├──► TemporalGroundingBuffer.push()
    │         │
    │         ├──► Update deque (append, popleft if full)
    │         └──► Update running statistics
    │                  │
    │                  ├──► sum += value
    │                  ├──► sum_sq += value²
    │                  ├──► count += 1
    │                  ├──► mean = sum / count
    │                  ├──► std = sqrt(sum_sq/count - mean²)
    │                  └──► trend = detect_trend(recent_values)
    │
QUERY: User question
    │
    ├──► TemporalGroundingBuffer.get_latest(n=5)
    │         │
    │         └──► Returns: List[SensorReading]
    │
    ├──► TemporalGroundingBuffer.get_statistics()
    │         │
    │         └──► Returns: {mean, std, min, max, trend}
    │
    ├──► StalenessDetector.detect(readings, stats)
    │         │
    │         ├──► Check time delta: time.time() - context_timestamp
    │         ├──► Check value change: |current - context_mean| / context_mean
    │         └──► Returns: StalenessResult(is_stale, reason, score)
    │
    ├──► LLMBackbone.format_grounding_prompt(stats, query)
    │         │
    │         └──► Creates ChatML format with sensor context
    │
    ├──► LLMBackbone.generate(prompt, max_new_tokens=100)
    │         │
    │         ├──► Tokenize input
    │         ├──► Forward pass (4-bit quantized)
    │         ├──► Apply LoRA adapter
    │         ├──► Sample tokens (temperature, top_p)
    │         └──► Decode to text
    │
    ├──► CausalValidator.validate(response)
    │         │
    │         ├──► Extract causal claims (regex + NLP)
    │         ├──► Parse cause-effect pairs
    │         ├──► Check each pair against graph
    │         ├──► Identify violations
    │         └──► Returns: ValidationResult(is_valid, score, violations)
    │
OUTPUT: Final Response
    └──► {
           "response": "Current consumption is 155.0 kWh, slightly above average...",
           "is_stale": false,
           "is_valid": true,
           "latency_ms": 2145.3
         }
```

---

## Data Flow Architecture

### End-to-End Request Processing

```
┌────────────────────────────────────────────────────────────────────┐
│                    REQUEST PROCESSING PIPELINE                      │
└────────────────────────────────────────────────────────────────────┘

TIME: t=0ms
┌─────────────────┐
│ User Query      │
│ "What is the    │
│  current energy │
│  consumption?"  │
└────────┬────────┘
         │
         ▼
TIME: t=0.5ms ──────────────────────────────────────────────────────┐
┌────────────────────────────────────┐                              │
│ TemporalGroundingBuffer            │                              │
│                                    │                              │
│ OPERATION: get_latest(n=5)         │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Hash lookup: O(1)              │ │ LATENCY: 0.05ms              │
│ │ Key: ("bldg_001", "electricity")│ │                              │
│ │ Result: Deque[5 readings]      │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ OPERATION: get_statistics()        │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Return cached stats: O(1)      │ │ LATENCY: 0.01ms              │
│ │ {mean: 150.3, std: 12.5, ...}  │ │                              │
│ └────────────────────────────────┘ │                              │
└────────────────────────────────────┘                              │
         │                                                           │
         ▼                                                           │
TIME: t=1.2ms ──────────────────────────────────────────────────────┤
┌────────────────────────────────────┐                              │
│ TimeThresholdStalenessDetector     │                              │
│                                    │                              │
│ INPUT: readings[], stats{}         │                              │
│                                    │                              │
│ CHECK 1: Time threshold            │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ time_delta = now - context_ts  │ │ LATENCY: 0.3ms               │
│ │ Result: 45s < 300s → FRESH     │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ CHECK 2: Value change              │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ change = |155-150.3| / 150.3   │ │                              │
│ │ Result: 3.1% < 20% → FRESH     │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ OUTPUT: StalenessResult            │                              │
│ {is_stale: false, score: 0.97}    │                              │
└────────────────────────────────────┘                              │
         │                                                           │
         ▼                                                           │
TIME: t=2ms ────────────────────────────────────────────────────────┤
┌────────────────────────────────────┐                              │
│ LLMBackbone.format_prompt()        │                              │
│                                    │                              │
│ TEMPLATE:                          │                              │
│ ┌────────────────────────────────┐ │                              │
│ │<|system|>                      │ │ LATENCY: 0.5ms               │
│ │You are an energy assistant.    │ │                              │
│ │</s>                            │ │                              │
│ │<|user|>                        │ │                              │
│ │Current: 155.0 kWh              │ │                              │
│ │Mean: 150.3 kWh                 │ │                              │
│ │Question: [query]               │ │                              │
│ │</s>                            │ │                              │
│ │<|assistant|>                   │ │                              │
│ └────────────────────────────────┘ │                              │
└────────────────────────────────────┘                              │
         │                                                           │
         ▼                                                           │
TIME: t=2.5ms ──────────────────────────────────────────────────────┤
┌────────────────────────────────────┐                              │
│ LLMBackbone.generate()             │                              │
│                                    │                              │
│ STEP 1: Tokenization               │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Input: 156 tokens              │ │ LATENCY: 15ms                │
│ │ [1, 2403, 338, 385, ...]       │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ STEP 2: Model Forward Pass         │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ 4-bit quantized inference      │ │ LATENCY: 1850ms              │
│ │ 22 transformer layers          │ │ (GPU: RTX 3090)              │
│ │ Memory: 4.2 GB                 │ │                              │
│ │ Power: 280W                    │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ STEP 3: LoRA Adapter               │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Low-rank adaptation            │ │ LATENCY: 120ms               │
│ │ Rank: 16, Alpha: 32            │ │                              │
│ │ Trainable: 4.5M params         │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ STEP 4: Token Sampling             │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Temperature: 0.3               │ │ LATENCY: 45ms                │
│ │ Top-p: 0.9                     │ │                              │
│ │ Generated: 42 tokens           │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ STEP 5: Decoding                   │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Output: "The current energy    │ │ LATENCY: 8ms                 │
│ │ consumption is 155.0 kWh..."   │ │                              │
│ └────────────────────────────────┘ │                              │
└────────────────────────────────────┘                              │
         │                                                           │
         ▼                                                           │
TIME: t=2040ms ─────────────────────────────────────────────────────┤
┌────────────────────────────────────┐                              │
│ CausalValidator.validate()         │                              │
│                                    │                              │
│ STEP 1: Claim Extraction           │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Regex patterns for causality   │ │ LATENCY: 15ms                │
│ │ Found: "consumption is higher" │ │                              │
│ │ Cause: temperature             │ │                              │
│ │ Effect: consumption            │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ STEP 2: Graph Validation           │                              │
│ ┌────────────────────────────────┐ │                              │
│ │ Check path in causal graph:    │ │ LATENCY: 2ms                 │
│ │ temperature → HVAC →           │ │                              │
│ │ consumption                    │ │                              │
│ │ Result: VALID ✓                │ │                              │
│ └────────────────────────────────┘ │                              │
│                                    │                              │
│ OUTPUT: ValidationResult           │                              │
│ {is_valid: true, score: 1.0}      │                              │
└────────────────────────────────────┘                              │
         │                                                           │
         ▼                                                           │
TIME: t=2057ms ─────────────────────────────────────────────────────┤
┌─────────────────┐                                                 │
│ Final Response  │                                                 │
│ ├─ response     │                                                 │
│ ├─ is_stale=F   │                                                 │
│ ├─ is_valid=T   │                                                 │
│ └─ latency=2057 │                                                 │
└─────────────────┘                                                 │
                                                                    │
TOTAL LATENCY: 2057ms ──────────────────────────────────────────────┘
  Buffer:     0.06ms  (0.003%)
  Staleness:  0.30ms  (0.015%)
  LLM:     2038.00ms (99.076%)  ← BOTTLENECK
  Causal:    17.00ms  (0.826%)
  Overhead:   1.64ms  (0.080%)
```

---

## Experimental Coverage Map

### Component Testing Matrix

```
┌──────────────────────────────────────────────────────────────────────┐
│              EXPERIMENTAL COVERAGE BY COMPONENT                       │
└──────────────────────────────────────────────────────────────────────┘

COMPONENT            │ Exp01 │ Exp02 │ Exp03 │ Exp04 │ Exp05 │ Exp06 │ Exp07 │ Exp08 │ Exp09 │ Exp10 │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
TemporalGrounding    │   ●   │   ●   │       │       │   ●   │   ●   │   ●   │   ●   │   ●   │   ●   │
Buffer               │       │       │       │       │       │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
CircularBuffer       │   ●   │       │       │       │   ●   │       │   ○   │       │       │       │
(Redis)              │       │       │       │       │       │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
Staleness            │   ●   │       │   ●   │       │   ●   │       │   ●   │   ●   │   ●   │       │
Detector             │       │       │       │       │       │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
LLM Backbone         │   ●   │   ●   │       │   ●   │   ●   │       │   ●   │   ●   │   ●   │   ●   │
(TinyLlama)          │       │       │       │       │       │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
LoRA Adapter         │       │   ●   │       │   ●   │   ●   │       │   ●   │       │       │   ●   │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
Causal Validator     │       │       │       │   ●   │   ●   │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
Data Loaders         │       │       │       │       │       │       │       │       │       │   ●   │
(BDG2, REDD)         │       │       │       │       │       │       │       │       │       │       │
─────────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
Statistics           │   ●   │   ●   │   ●   │       │   ●   │   ●   │   ●   │   ●   │   ●   │   ●   │
Computer             │       │       │       │       │       │       │       │       │       │       │
─────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

Legend:
  ● = Primary component being tested
  ○ = Comparison/baseline component
  (blank) = Not tested in this experiment

TESTING APPROACH:
┌────────────────────────────────────────────────────────────────┐
│ Exp01: Latency        → Measures speed of all operations      │
│ Exp02: Grounding      → Tests LLM value referencing           │
│ Exp03: Staleness      → Validates time/value threshold logic  │
│ Exp04: Causal         → Checks graph validation correctness   │
│ Exp05: Ablation       → Removes each component systematically │
│ Exp06: Scalability    → Tests buffer O(1) claim at scale      │
│ Exp07: SOTA           → Compares full system vs baselines     │
│ Exp08: Cost           → Measures GPU/power/memory usage       │
│ Exp09: Robustness     → Tests degraded data handling          │
│ Exp10: Cross-dataset  → Tests generalization BDG2 → REDD      │
└────────────────────────────────────────────────────────────────┘
```

### Metric Flow Diagram

```
EXPERIMENTS                    METRICS COLLECTED                    PAPER CLAIMS
─────────────────────────────────────────────────────────────────────────────────

Exp01: Latency         →   mean_ms, p95_ms, p99_ms      →   "TGP achieves <1ms
Benchmark                  speedup_vs_redis                   buffer latency"
                                                              (Table 2)

Exp02: Grounding       →   value_accuracy (0.95)         →   "95% grounding
Accuracy                   trend_accuracy (0.90)              accuracy on sensor
                                                              values" (Table 3)

Exp03: Staleness       →   precision, recall, F1         →   "F1=1.0 staleness
Detection                  (time_threshold: 1.0)              detection with time
                           (embedding: 0.35)                  thresholds" (Table 4)

Exp04: Causal          →   valid_rate (0.70)             →   "70% causal validity
Validation                 mean_score (0.82)                  vs 100% accepted
                                                              (no validation)"
                                                              (Figure 3)

Exp05: Ablation        →   latency_impact_ms             →   "LoRA contributes
Study                      accuracy_impact                    15% accuracy gain"
                           (per component)                    (Table 5, Figure 4)

Exp06: Scalability     →   latency vs size               →   "Constant O(1)
Test                       memory vs size                     latency up to 100K
                           (shows O(1) curve)                 readings" (Figure 5)

Exp07: SOTA            →   latency, cost, quality        →   "Comparable to
Comparison                 vs Cloud/SQL/Raw                   Claude API at zero
                                                              cost" (Table 6)

Exp08: Computational   →   peak_memory_gb (4.2)          →   "4.2GB memory,
Cost                       avg_power_watts (280)              0.06kg CO2 training"
                           co2_kg (0.059)                     (Table 7)

Exp09: Sampling        →   latency degradation           →   "Robust to 50%
Robustness                 quality degradation                dropout, 30% noise"
                           (vs sampling/dropout/noise)        (Figure 6)

Exp10: Cross-Dataset   →   generalization_gap            →   "10% gap between
Validation                 (BDG2 vs REDD)                     commercial/residential"
                           valid_rate_bdg2 (0.85)             (Table 8)
                           valid_rate_redd (0.75)
```

---

## Baseline Comparison Architecture

### System Variants Tested

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM CONFIGURATION VARIANTS                          │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 1: TGP (Ours) - FULL SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → TemporalGroundingBuffer → TimeThresholdDetector →             │
│           (In-process, O(1))        (F1=1.0)                             │
│                                                                          │
│           → TinyLlama + LoRA → CausalValidator → Response               │
│              (4-bit, 4.5M)      (Graph-based)                            │
│                                                                          │
│  LATENCY:  ~2000ms total                                                │
│  COST:     Free (on-device)                                             │
│  ACCURACY: 85% grounding, 70% causal                                    │
│  MEMORY:   4.2 GB                                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 2: Redis Baseline                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → CircularBuffer (Redis) → TimeThresholdDetector →              │
│           (Network I/O)            (Same as TGP)                         │
│                                                                          │
│           → TinyLlama + LoRA → CausalValidator → Response               │
│              (Same as TGP)         (Same as TGP)                         │
│                                                                          │
│  LATENCY:  ~2044ms (+2% vs TGP)                                         │
│  COST:     Free + Redis server                                          │
│  ACCURACY: Same as TGP                                                  │
│  MEMORY:   4.2 GB + Redis overhead                                     │
│  PURPOSE:  Compare buffer implementations                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 3: Cloud API (Claude)                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → (No buffer) → Anthropic API →                                 │
│                         (Claude Sonnet)                                  │
│                                                                          │
│                                  → Response                              │
│                                                                          │
│  LATENCY:  ~1500ms (-25% vs TGP)                                        │
│  COST:     $0.01 per query                                              │
│  ACCURACY: 95% grounding (superior)                                     │
│  MEMORY:   N/A (cloud)                                                  │
│  PURPOSE:  SOTA cloud baseline                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 4: SQLite + Raw LLM                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → SQLite Database → SQL Query → TinyLlama (Raw) →               │
│           (On-disk)         (SELECT AVG...) (No LoRA)                    │
│                                                                          │
│                                              → Response                  │
│                                                                          │
│  LATENCY:  ~2500ms (+25% vs TGP)                                        │
│  COST:     Free                                                         │
│  ACCURACY: 70% grounding (inferior)                                     │
│  MEMORY:   4.2 GB + SQLite                                              │
│  PURPOSE:  Traditional database baseline                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 5: No Fine-tuning (Raw LLM)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → TemporalGroundingBuffer → TinyLlama (Raw) → Response          │
│           (Same as TGP)              (NO LoRA)                           │
│                                                                          │
│  LATENCY:  ~1880ms (-6% vs TGP, no LoRA overhead)                       │
│  COST:     Free                                                         │
│  ACCURACY: 75% grounding (-10% vs TGP)                                  │
│  MEMORY:   4.0 GB (slightly less)                                       │
│  PURPOSE:  Measure LoRA contribution                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 6: No Staleness Detection                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → TemporalGroundingBuffer → (SKIP) →                            │
│                                                                          │
│           → TinyLlama + LoRA → CausalValidator → Response               │
│                                                                          │
│  LATENCY:  ~1999ms (-0.05% vs TGP, negligible)                          │
│  COST:     Free                                                         │
│  ACCURACY: 77% grounding (-8% vs TGP, stale context)                    │
│  PURPOSE:  Measure staleness contribution                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ VARIANT 7: No Causal Validation                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sensor → TemporalGroundingBuffer → TimeThresholdDetector →             │
│                                                                          │
│           → TinyLlama + LoRA → (SKIP) → Response                        │
│                                                                          │
│  LATENCY:  ~1983ms (-0.8% vs TGP, faster)                               │
│  COST:     Free                                                         │
│  ACCURACY: 85% grounding (same), 100% accepted (no filtering)           │
│  PURPOSE:  Measure causal validation necessity                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Edge Device Configuration

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     EDGE DEPLOYMENT ARCHITECTURE                          │
└──────────────────────────────────────────────────────────────────────────┘

HARDWARE: NVIDIA Jetson Xavier NX (or similar edge device)
──────────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│  EDGE DEVICE                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ GPU: NVIDIA Volta (384 CUDA cores)                                 │ │
│  │ RAM: 8 GB                                                           │ │
│  │ Storage: 32 GB eMMC                                                 │ │
│  │ Power: 10-20W                                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ RUNTIME ENVIRONMENT                                                 │ │
│  │                                                                      │ │
│  │  Python 3.11                                                        │ │
│  │  PyTorch 2.1 + CUDA 11.8                                            │ │
│  │  Transformers 4.36                                                  │ │
│  │  BitsAndBytes 0.41 (4-bit quantization)                             │ │
│  │  PEFT 0.7 (LoRA)                                                    │ │
│  │                                                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ TGP APPLICATION STACK                                               │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │ API Server (FastAPI)                          PORT: 8000     │  │ │
│  │  │ • /infer                                                      │  │ │
│  │  │ • /health                                                     │  │ │
│  │  │ • /metrics                                                    │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                        ▲                                             │ │
│  │  ┌─────────────────────┴─────────────────────────────────────────┐ │ │
│  │  │ TGP Core                                                       │ │ │
│  │  │ • TemporalGroundingBuffer (4.2 GB model + 500 MB buffer)     │ │ │
│  │  │ • TimeThresholdStalenessDetector                              │ │ │
│  │  │ • LLMBackbone (TinyLlama 1.1B 4-bit)                          │ │ │
│  │  │ • CausalValidator                                             │ │ │
│  │  └───────────────────────────────────────────────────────────────┘ │ │
│  │                        ▲                                             │ │
│  │  ┌─────────────────────┴─────────────────────────────────────────┐ │ │
│  │  │ Data Ingestion Service                                         │ │ │
│  │  │ • MQTT Subscriber (sensor streams)                            │ │ │
│  │  │ • Buffer writer (push to TGP)                                 │ │ │
│  │  └───────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ MQTT (1883)
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
             ┌──────▼──────┐              ┌─────────▼────────┐
             │ IoT Sensor  │              │  IoT Sensor      │
             │ (Building 1)│              │  (Building 2)    │
             │             │              │                  │
             │ • Power     │              │  • Power         │
             │ • HVAC      │              │  • HVAC          │
             │ • Occupancy │              │  • Occupancy     │
             └─────────────┘              └──────────────────┘

NETWORK TOPOLOGY:
────────────────────────────────────────────────────────────────────────

  Internet ◄─────┐
                 │ (Optional: Model updates)
                 │
  ┌──────────────▼────────────────┐
  │  Edge Gateway / Router        │
  └──────────────┬────────────────┘
                 │
  ┌──────────────▼────────────────┐
  │  Local Network (Building)     │
  │  • MQTT Broker                │
  │  • Edge Device (TGP)          │
  │  • IoT Sensors (50-200)       │
  └───────────────────────────────┘

DEPLOYMENT MODEL: Fully local, offline-capable
  ✓ All inference on-device
  ✓ No cloud dependency
  ✓ Privacy-preserving (data stays local)
  ✓ Low latency (no network round-trip)
  ✓ Cost-effective (no API fees)
```

### Training vs Inference Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  TRAINING ARCHITECTURE (Development Server)               │
└──────────────────────────────────────────────────────────────────────────┘

HARDWARE: NVIDIA RTX 3090 (24 GB)
GPU 4 on shared server (CUDA_VISIBLE_DEVICES=4)

┌─────────────────────────────────────────────────────────────────────────┐
│  Training Pipeline                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Data Preparation                                                    │ │
│  │ • Load BDG2 dataset (Kaggle)                                        │ │
│  │ • Generate synthetic queries + answers                             │ │
│  │ • Create train/val/test split (80/10/10)                           │ │
│  │ • Format as ChatML (system/user/assistant)                         │ │
│  │ • Save to data/training/train.jsonl                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                        ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Model Initialization                                                │ │
│  │ • Load TinyLlama-1.1B-Chat from HuggingFace                         │ │
│  │ • Apply 4-bit quantization (BitsAndBytes)                           │ │
│  │ • Add LoRA adapters (rank=16, alpha=32)                             │ │
│  │ • Trainable params: 4.5M (0.4% of total)                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                        ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Training Loop (HuggingFace Trainer)                                 │ │
│  │ • Epochs: 3                                                         │ │
│  │ • Batch size: 4 (effective: 16 with grad_accum=4)                   │ │
│  │ • Learning rate: 2e-4                                               │ │
│  │ • Optimizer: AdamW (8-bit)                                          │ │
│  │ • Scheduler: Linear warmup (100 steps)                              │ │
│  │ • Gradient clipping: 1.0                                            │ │
│  │ • Mixed precision: FP16                                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                        ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Checkpointing                                                       │ │
│  │ • Save every 100 steps                                              │ │
│  │ • Keep best 2 checkpoints (by val loss)                             │ │
│  │ • Final save: output/models/grounding_YYYYMMDD_HHMMSS/final/       │ │
│  │   ├── adapter_config.json                                           │ │
│  │   ├── adapter_model.bin (9 MB LoRA weights)                         │ │
│  │   └── trainer_state.json                                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  RESOURCES:                                                              │
│  • Training time: ~30 minutes                                           │
│  • GPU memory: 18 GB peak                                               │
│  • Power: 280W average                                                  │
│  • CO2: ~0.06 kg                                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                  INFERENCE ARCHITECTURE (Edge Device)                     │
└──────────────────────────────────────────────────────────────────────────┘

HARDWARE: NVIDIA Jetson / Edge GPU (4-8 GB)

┌─────────────────────────────────────────────────────────────────────────┐
│  Inference Pipeline                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Model Loading (Startup)                                             │ │
│  │ • Load TinyLlama-1.1B base model (4-bit)                            │ │
│  │ • Load LoRA weights from output/models/.../final/                   │ │
│  │ • Merge adapters into model (optional, for speed)                   │ │
│  │ • Move to GPU                                                       │ │
│  │ • Compile for inference (torch.compile)                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                        ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Serving Loop                                                        │ │
│  │ • Listen on FastAPI endpoint                                        │ │
│  │ • Receive query + sensor context                                    │ │
│  │ • Process through TGP pipeline                                      │ │
│  │ • Return response                                                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  RESOURCES:                                                              │
│  • Inference time: ~2000ms per query                                    │
│  • GPU memory: 4.2 GB                                                   │
│  • Power: 15W average                                                   │
│  • Throughput: ~0.5 queries/sec (single batch)                          │
└─────────────────────────────────────────────────────────────────────────┘

TRANSFER: Training → Inference
────────────────────────────────────────────────────────────────────────
Development Server                      Edge Device
(RTX 3090, 24GB)                        (Jetson, 8GB)
        │                                       │
        │ 1. Train LoRA adapters                │
        │    (30 min, 18GB memory)              │
        │                                       │
        │ 2. Save checkpoint                    │
        │    output/models/.../final/           │
        │    ├── adapter_model.bin (9 MB)       │
        │    └── adapter_config.json            │
        │                                       │
        ├───────── scp / rsync ─────────────────►
        │                                       │
        │                                       │ 3. Load base model + adapters
        │                                       │    (4.2GB memory)
        │                                       │
        │                                       │ 4. Serve inference
        │                                       │    (~2s latency)
        │                                       │
```

---

## Module Dependencies

### Python Package Hierarchy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCY GRAPH                           │
└──────────────────────────────────────────────────────────────────────────┘

src/
├── __init__.py
│
├── buffer/
│   ├── __init__.py
│   ├── base.py                  ← BufferInterface (ABC)
│   ├── circular_buffer.py       ← Redis-backed (baseline)
│   │       └── requires: redis-py
│   ├── temporal_buffer.py       ← In-process (novel)
│   │       └── requires: collections.deque
│   └── sensor_reading.py        ← @dataclass
│           └── requires: dataclasses
│
├── staleness/
│   ├── __init__.py
│   ├── base.py                  ← StalenessDetector (ABC)
│   ├── detector.py              ← TimeThresholdDetector
│   │       └── depends: buffer.base
│   └── embedding_detector.py    ← DEPRECATED
│           └── requires: sentence-transformers
│
├── llm/
│   ├── __init__.py
│   ├── backbone.py              ← LLMBackbone
│   │       └── requires: transformers, peft, bitsandbytes
│   ├── config.py                ← ModelConfig
│   └── prompts.py               ← Prompt templates
│
├── causal/
│   ├── __init__.py
│   ├── graph.py                 ← CausalGraph
│   │       └── requires: networkx
│   └── validator.py             ← CausalValidator
│           └── depends: causal.graph
│
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py          ← TemporalGroundingOrchestrator
│           └── depends: buffer, staleness, llm, causal
│
├── data/
│   ├── __init__.py
│   ├── loaders.py               ← BDG2Loader, REDDLoader
│   │       └── requires: pandas, pyarrow
│   └── generators.py            ← Synthetic data generation
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py               ← Accuracy, F1, latency
│   ├── visualization.py         ← Plotting utilities
│   │       └── requires: matplotlib, seaborn
│   └── performance_tracker.py   ← GPU/power monitoring
│           └── requires: pynvml
│
└── baselines/
    ├── __init__.py
    ├── claude.py                ← ClaudeBaseline
    │       └── requires: anthropic
    └── sqlite.py                ← SQLiteBaseline
            └── requires: sqlite3

experiments/
├── 01_latency_benchmark.py      → depends: buffer, staleness, llm
├── 02_grounding_accuracy.py     → depends: buffer, llm
├── 03_staleness_detection.py    → depends: staleness
├── 04_causal_validation.py      → depends: llm, causal
├── 05_ablation_study.py         → depends: ALL
├── 06_scalability_test.py       → depends: buffer
├── 07_sota_comparison.py        → depends: ALL + baselines
├── 08_computational_cost.py     → depends: ALL + utils.performance
├── 09_sampling_robustness_fixed.py → depends: buffer, llm
└── 10_cross_dataset_validation.py → depends: ALL + data.loaders

KEY DEPENDENCIES:
────────────────────────────────────────────────────────────────────────
torch >= 2.0.0
transformers >= 4.36.0
peft >= 0.7.0
bitsandbytes >= 0.41.0
accelerate >= 0.25.0
sentence-transformers >= 2.2.0  (for deprecated embedding detector)
networkx >= 3.0
redis >= 5.0
pandas >= 2.0
numpy >= 1.24
anthropic >= 0.8.0  (for Claude baseline)
```

### Import Dependency Matrix

```
MODULE                  │ torch │ trans │ peft │ bnb │ redis │ nx │ pandas │
────────────────────────┼───────┼───────┼──────┼─────┼───────┼────┼────────┤
buffer.circular_buffer  │       │       │      │     │   ●   │    │        │
buffer.temporal_buffer  │       │       │      │     │       │    │        │
staleness.detector      │       │       │      │     │       │    │        │
staleness.embedding     │   ●   │   ●   │      │     │       │    │        │
llm.backbone            │   ●   │   ●   │  ●   │  ●  │       │    │        │
causal.graph            │       │       │      │     │       │ ●  │        │
causal.validator        │       │       │      │     │       │ ●  │        │
data.loaders            │       │       │      │     │       │    │    ●   │
baselines.claude        │       │       │      │     │       │    │        │
────────────────────────┴───────┴───────┴──────┴─────┴───────┴────┴────────┘

Legend:
  ● = Required dependency
  trans = transformers
  bnb = bitsandbytes
  nx = networkx
```

---

## File Structure

```
04-realtime-grounding/
├── src/                          # Core implementation
│   ├── buffer/                   # Data buffering (novel TGB + Redis)
│   ├── staleness/                # Staleness detection
│   ├── llm/                      # LLM backbone + LoRA
│   ├── causal/                   # Causal validation
│   ├── pipeline/                 # End-to-end orchestrator
│   ├── data/                     # Dataset loaders
│   ├── utils/                    # Metrics, visualization, tracking
│   └── baselines/                # Comparison systems
│
├── experiments/                  # 10 experimental scripts
│   ├── 01_latency_benchmark.py
│   ├── 02_grounding_accuracy.py
│   ├── ...
│   └── 10_cross_dataset_validation.py
│
├── scripts/                      # Utility scripts
│   ├── download_datasets.py      # Download BDG2, REDD
│   ├── verify_setup.py           # Environment check
│   ├── train_model.py            # LoRA training
│   └── analyze_results.py        # Generate figures/tables
│
├── run/                          # Execution scripts
│   ├── run_all.sh                # Full pipeline
│   ├── 01_main_experiment.sh     # Run all experiments
│   └── 02_generate_figures.sh    # Create paper figures
│
├── data/                         # Datasets
│   ├── raw/
│   │   ├── bdg2_kaggle/          # Commercial buildings
│   │   └── redd/                 # Residential homes
│   └── training/
│       ├── train.jsonl           # Training data
│       ├── val.jsonl             # Validation data
│       └── test.jsonl            # Test data
│
├── output/                       # Results
│   ├── models/                   # Trained LoRA weights
│   ├── common/results/           # Dataset-agnostic experiments
│   ├── bdg2/results/             # BDG2-specific experiments
│   ├── redd/results/             # REDD-specific experiments
│   └── figures/                  # Generated plots (PDFs)
│
├── paper/                        # LaTeX paper
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
├── tests/                        # Unit tests
│   ├── test_buffer.py
│   ├── test_staleness.py
│   └── ...
│
├── ARCHITECTURE.md               # This file
├── EXPERIMENTS.md                # Experimental design doc
├── README.md                     # Project overview
├── SETUP.md                      # Installation guide
└── requirements.txt              # Python dependencies
```

---

**Last Updated**: 2026-04-09
**Maintainer**: Franck Junior Aboya Messou
**Contact**: franckjunioraboya.messou@ieee.org

