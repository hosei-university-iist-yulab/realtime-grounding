# Plan: Post-TGP Research Extensions

## Context

TGP (Topic 04) was submitted to IEEE CEM on 2026-04-15. The core contribution (80% grounding accuracy, 0.085ms buffer, 1.09GB footprint) is solid, but reviewer feedback and the paper's own Limitations section reveal four clearly journal-worthy gaps: (1) trend accuracy collapses on some datasets (2–100% range), (2) 2.5s latency limits closed-loop use, (3) energy-only evaluation, and (4) untested robustness against sensor drift and adversarial inputs. Meanwhile, Topic 05 (Federated-SLM) is a greenfield directory with only a `.gitkeep` file, targeting ICLR 2027 (Sep 2026 deadline).

This plan proposes **5 extension ideas for TGP** plus a **natural bridge to Topic 05**, so work on the journal extension and the ICLR paper can reuse infrastructure rather than duplicate it.

---

## Five TGP Extension Ideas

### E1. Trend Detection with a Dedicated Sub-Model (highest priority)
**Gap**: Trend accuracy is 2% on UCI-Tetouan (gradual monotonic loads) despite 99% value accuracy. Prompt engineering alone cannot fix this.
**Idea**: Add a lightweight trend classifier (1–2M params, CNN or small Transformer) that consumes the buffer's Welford statistics and outputs a trend label injected into the SLM prompt. SLM handles language, sub-model handles numeric pattern recognition.
**Novelty**: Hybrid numeric-plus-language grounding. Each component does what it's best at.
**Target**: IEEE IoT Journal or IEEE Trans. Smart Grid (journal extension).
**Effort**: Medium (~2 months). Trend classifier is small; main work is new evaluation protocol.

### E2. Speculative Decoding for Sub-500ms Grounding
**Gap**: 2.5s latency dominated by LLM inference, unsuitable for closed-loop control (<100ms).
**Idea**: Use a tiny draft model (125M, e.g., SmolLM-125M) to propose tokens verified by TinyLLaMA-1.1B via speculative decoding. Pair with 8-bit KV cache compression. Target: 500ms → 100ms.
**Novelty**: First speculative decoding evaluation on a grounding task with sensor context (vs. generic text).
**Target**: ICLR 2027 workshop or NeurIPS ML for Systems workshop.
**Effort**: Low-medium (~1 month) since `vllm` and `transformers` have speculative decoding built in.

### E3. Multi-Modal Grounding (Sensor + Vision)
**Gap**: Current TGP is numeric-only. Consumer AIoT increasingly includes cameras (security, thermal, occupancy).
**Idea**: Extend the buffer to store image tensors alongside sensor readings. Fuse a small vision encoder (CLIP-tiny or MobileCLIP) with TinyLLaMA. Query example: "Is anyone in the living room?" → grounded in live camera + motion sensor.
**Novelty**: First multi-modal real-time grounding on edge with zero cloud dependency.
**Target**: IEEE CEM follow-up or AAAI 2027.
**Effort**: High (~4 months). Need new dataset (consumer AIoT multi-modal scenes) and careful memory budgeting (vision + LLM > 1.09GB).

### E4. Cross-Domain Generalization Study
**Gap**: Evaluation limited to energy datasets. Generalization to healthcare wearables, agriculture, industrial monitoring untested.
**Idea**: Systematic evaluation across 4 domains (energy, health, agriculture, industrial). Analyze which domain characteristics (sensor update rate, value range, trend complexity) predict grounding accuracy. Propose a domain-adaptive prompt template.
**Novelty**: First generalization study of sensor-LLM grounding across domains. Valuable empirical contribution.
**Target**: IEEE Trans. IoT or ACM TOSN (journal extension).
**Effort**: Medium (~3 months). Data collection is the main cost.

### E5. Adversarial Robustness and Calibration Drift
**Gap**: Reviewer explicitly flagged that robustness evaluation misses long-term sensor calibration drift and adversarial inputs. Critical for the "Secure" keyword.
**Idea**: Two studies: (a) long-term drift using 6–12 month sensor traces, quantifying how staleness thresholds degrade; (b) adversarial sensor spoofing (gradually inject fabricated values) to see if TGP detects or hallucinates.
**Novelty**: First adversarial robustness study of sensor-LLM grounding. Directly addresses the "secure" keyword beyond just "on-device processing".
**Target**: IEEE Trans. Information Forensics and Security, or a security venue (CCS workshop, NDSS).
**Effort**: Medium (~3 months). Requires adversarial dataset generation.

---

## Bridge Idea: Federated TGP → Launches Topic 05

The single most strategic idea is to make **TGP federated**, which:
- Satisfies TGP's own Future Work bullet ("federated fine-tuning across multiple buildings")
- Kickstarts Topic 05 (Federated-SLM) with a concrete, publishable problem
- Reuses all TGP infrastructure (buffer, staleness detector, LoRA adapter, datasets)
- Natural ICLR 2027 story: federated grounding is a novel FL problem (not just federated training)

### Concrete Topic 05 Directions (pick one to start)

**T5.A. Federated LoRA Aggregation for Sensor-Grounded SLMs** *(recommended start)*
Each building trains a LoRA adapter on its own data. Central server aggregates adapters (FedAvg, FedProx) without touching sensor streams. Evaluate whether aggregated adapter generalizes to unseen buildings.
Extends: `chen2025fedlora` (our own VTC paper, already cited in TGP).

**T5.B. Heterogeneous Federated SLMs**
Different buildings have different sensor types and sampling rates. Propose a federated protocol that accommodates client heterogeneity (missing modalities, different LLM backbones).

**T5.C. Communication-Efficient Federated Fine-Tuning**
LoRA already reduces trainable parameters, but federated rounds still transmit full adapters. Further compress via quantization (INT4 adapters) or sparsification (top-k updates). Target: 100x communication reduction.

**T5.D. Federated Causal Graphs**
Each building learns local causal relationships (temperature → HVAC → consumption). Aggregate into a global causal graph while preserving privacy. Novel combination of FL + causal inference.

**Recommendation**: Start with **T5.A** because it directly extends Franck's existing co-author paper (`chen2025fedlora`) and reuses the TGP codebase (LoRA fine-tuning pipeline is already implemented).

---

## Recommended Execution Order

1. **Months 1–3**: E1 (trend sub-model) + T5.A (federated LoRA aggregation). Share infrastructure. Two submissions: one journal (E1 → IEEE IoT Journal), one conference (T5.A → ICLR 2027).
2. **Months 4–6**: E2 (speculative decoding) as a short paper — low effort, high perceived novelty.
3. **Months 7–12**: E3 (multi-modal) or E4 (cross-domain) as the long-term journal track. Pick based on which dataset becomes available first.
4. **Opportunistic**: E5 (adversarial robustness) whenever a security workshop deadline appears.

---

## Files That Would Be Modified/Created

### For E1 (trend sub-model):
- New: `projects/03-SLM-CoreMethods/04-realtime-grounding/src/trend/classifier.py`
- Modified: `src/pipeline/orchestrator.py` (insert trend classifier before LLM)
- Modified: `experiments/11_trend_features.py` (add new evaluation)

### For T5.A (Topic 05 kickoff):
- New: `projects/03-SLM-CoreMethods/05-federated-slm/` directory tree
  - `README.md`, `src/fl/`, `src/aggregator/`, `experiments/`, `run/`, `tests/`
- Reuse via `sys.path`: TGP's buffer, staleness detector, LoRA fine-tuning scripts
- New `paper/` for ICLR 2027 draft

### For E2 (speculative decoding):
- Modified: `src/llm/backbone.py` (add speculative decoder wrapper around TinyLLaMA)
- New: `experiments/22_speculative_latency.py`
- Dependencies: `vllm` or `transformers` with speculative decoding support

---

## Verification

- **E1**: Run `experiments/11_trend_features.py` on all 5 datasets; expect trend accuracy on UCI-Tetouan to rise from 2% → ≥70%. Measure latency overhead (should be <10ms for a 1–2M trend model).
- **E2**: Benchmark `experiments/22_speculative_latency.py` with and without speculative decoding. Target mean latency <500ms at P95.
- **E3**: New evaluation script across 4 domains; report per-domain accuracy and identify failure modes.
- **E4**: Per-domain ablation; publish dataset-characteristic-to-accuracy regression.
- **E5**: Inject adversarial traces; measure detection rate and hallucination rate.
- **T5.A**: Train on 3 building subsets, test on held-out building; measure accuracy gap between centralized and federated configurations.
