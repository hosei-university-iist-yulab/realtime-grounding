# Real-Time Sensor-Text Grounding for Edge-Deployed Small Language Models

**Authors**: Franck Junior Aboya Messou, Jinhua Chen, Tong Liu, Keping Yu  
**Affiliation**: Hosei University, Tokyo, Japan  
**Target**: IEEE Consumer Electronics Magazine, 2026

---

## Overview

The Temporal Grounding Pipeline (TGP) bridges the temporal gap between real-time sensor streams (1--5 Hz) and small language model inference, enabling grounded natural language interaction with IoT devices on consumer edge hardware.

### Key Results (V2, Multi-Dataset)

| Metric | Value | Comparison |
|--------|-------|------------|
| Value Accuracy | 80% avg (68--99%) | vs 1--9% without grounding |
| Buffer Latency | 0.085 ms mean | 10.4x faster than Redis |
| Staleness F1 | 0.98 | Perfect precision |
| Memory | 1.09 GB | Edge-deployable |
| Uptime | 100% | Zero memory leaks |
| Dependencies | 0 | No Redis/SQLite required |

---

## Repository Structure

```
04-realtime-grounding/
├── src/                          # Source implementation
│   ├── buffer/                   # Temporal grounding buffer (O(1) ops)
│   ├── staleness/                # Time+value staleness detector
│   ├── causal/                   # Causal validation graph
│   ├── llm/                      # TinyLLaMA + LoRA backbone
│   ├── pipeline/                 # End-to-end orchestrator
│   ├── baselines/                # Redis, API, prompt baselines
│   ├── simulation/               # Sensor stream simulation
│   ├── data/loaders/             # BDG2, UK-DALE, UCI dataset loaders
│   └── utils/                    # Metrics, visualization, tracking
├── experiments/                  # 21 experiment scripts (01--21)
├── scripts/                      # Training, data generation, plot generation
├── run/                          # Shell scripts for GPU-isolated execution
├── data/                         # Datasets (raw + processed + training)
├── output/                       # Experimental results (V2, multi-seed)
├── analysis/                     # Generated figures and LaTeX tables
├── IEEE_Consumer_Electronics_Magazine_2026/  # Paper source
└── tests/                        # Unit tests
```

---

## Datasets

Five publicly available energy datasets are used:

| Dataset | Type | Resolution | Source |
|---------|------|------------|--------|
| BDG2 | Commercial buildings | Hourly | Miller et al., 2020 |
| UK-DALE | Residential (UK) | 6-second | Kelly & Knottenbelt, 2015 |
| UCI-Household | Household power | 1-minute | UCI ML Repository |
| UCI-Steel | Industrial | 15-minute | UCI ML Repository |
| UCI-Tetouan | City load | 10-minute | UCI ML Repository |

---

## Quick Start

```bash
# Install dependencies
conda activate llms
pip install -r requirements.txt

# Verify setup
python scripts/verify_setup.py

# Run quick test (~2 min)
python scripts/quick_test.py

# Run full experiments (GPU required)
./run/run_all.sh
```

---

## Experiments

| Exp | Description | Key Metric |
|-----|-------------|------------|
| 01 | Latency benchmark | 0.085 ms buffer (10.4x Redis) |
| 02 | Grounding accuracy | 80% avg value accuracy |
| 03 | Staleness detection | F1 = 0.98 |
| 04 | Causal validation | 70% valid rate |
| 05 | Ablation study | Component contributions |
| 06 | Scalability | Sub-linear to 1,000 sensors |
| 07 | SOTA comparison | TGP vs Redis/SQLite/Cloud |
| 08 | Computational cost | 1.09 GB, 16 tok/s |
| 09 | Sampling robustness | Tolerates 50% dropout |
| 10 | Cross-dataset | Generalization validation |
| 11--14 | Extended ablations | Trend, multitask, causal weights |
| 18 | Backbone comparison | TinyLLaMA vs Phi-2 vs Qwen |
| 20 | Deployment simulation | 100% uptime, 0 leaks |
| 21 | Multi-dataset evaluation | 5 datasets, 2 seeds |

All results in `output/v2/` with two random seeds (2025, 2026).

---

## Citation

```bibtex
@article{messou2026tgp,
  title={Real-Time Sensor-Text Grounding for Edge-Deployed Small Language Models},
  author={Messou, Franck Junior Aboya and Chen, Jinhua and Liu, Tong and Yu, Keping},
  journal={IEEE Consumer Electronics Magazine},
  year={2026}
}
```

---

## Contact

Franck Junior Aboya Messou (franckjunioraboya.messou@ieee.org)  
Hosei University, Tokyo, Japan
