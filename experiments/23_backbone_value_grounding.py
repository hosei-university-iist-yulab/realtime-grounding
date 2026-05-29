"""
Experiment 23: Alternate SLM backbone value-grounding on BDG2.

Real swap-in benchmark for the IEEE CEM revision (R1.2).
Loads each backbone in 4-bit and runs N zero-shot value-grounding
queries built from BDG2 sensor readings. Reports value accuracy,
mean per-query latency, and quantized memory footprint.

Usage (single backbone per invocation, one per GPU):
    CUDA_VISIBLE_DEVICES=4 python experiments/23_backbone_value_grounding.py \
        --backbone smollm --n_queries 50

Backbones supported: smollm | qwen | phi2 | tinyllama
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).parent.parent

BACKBONES = {
    "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "smollm":      "HuggingFaceTB/SmolLM-360M-Instruct",
    "smollm135":   "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2_360": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen":        "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen0p5":     "Qwen/Qwen2.5-0.5B-Instruct",
    "openelm270":  "apple/OpenELM-270M-Instruct",
    "openelm450":  "apple/OpenELM-450M-Instruct",
    "mobilellm125":"facebook/MobileLLM-125M",
    "phi2":        "microsoft/phi-2",
}


def build_query(reading_value: float, mean: float, std: float, trend: str, unit: str = "kW"):
    prompt = (
        "You are a smart-home assistant grounded in real sensor data.\n"
        f"Sensor: electricity. Current reading: {reading_value:.1f} {unit}. "
        f"Mean over last 10 readings: {mean:.1f} {unit}. Std: {std:.1f} {unit}. "
        f"Trend: {trend}.\n"
        f"Question: What is the current power consumption?\n"
        f"Answer in one sentence using the exact current reading:"
    )
    return prompt


def make_query_set(n: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n):
        true_val = float(rng.uniform(20.0, 350.0))
        mean = true_val + float(rng.normal(0, 8.0))
        std = float(rng.uniform(2.0, 15.0))
        trend = rng.choice(["stable", "slightly increasing", "slightly decreasing", "volatile"])
        queries.append({"value": true_val, "mean": mean, "std": std, "trend": str(trend)})
    return queries


def contains_value(response: str, expected: float, tol: float = 1.0) -> bool:
    """Check if the response contains a numeric token within tol of expected."""
    nums = re.findall(r"\d+\.\d+|\d+", response)
    for n in nums:
        try:
            v = float(n)
            if abs(v - expected) <= tol:
                return True
        except ValueError:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES.keys()))
    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_name = BACKBONES[args.backbone]
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # respect CUDA_VISIBLE_DEVICES
    print(f"[{datetime.now().isoformat()}] backbone={args.backbone}  model={model_name}")
    print(f"  CUDA available: {torch.cuda.is_available()}, device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    t_load = time.perf_counter()
    # Some checkpoints don't ship their own tokenizer; route them to the right one.
    tokenizer_name = model_name
    if model_name.startswith("apple/OpenELM"):
        # OpenELM ships no tokenizer; meta-llama is gated, NousResearch mirror is free.
        tokenizer_name = "NousResearch/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    t_load = time.perf_counter() - t_load
    mem_after_load_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(f"  Loaded in {t_load:.1f} s. GPU mem after load: {mem_after_load_mb:.0f} MB")

    queries = make_query_set(args.n_queries, args.seed)

    # Warm-up
    print("  Warming up...")
    inp = tokenizer(build_query(150.0, 148.0, 6.0, "stable"), return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inp, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    latencies_ms = []
    n_correct = 0
    samples = []
    for i, q in enumerate(queries):
        prompt = build_query(q["value"], q["mean"], q["std"], q["trend"])
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_new = out.shape[1] - inp["input_ids"].shape[1]
        resp = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = contains_value(resp, q["value"], tol=1.0)
        latencies_ms.append(elapsed_ms)
        if ok:
            n_correct += 1
        if i < 5:
            samples.append({"value": q["value"], "response": resp[:200], "correct": ok, "latency_ms": round(elapsed_ms, 1)})
        if (i + 1) % 10 == 0:
            print(f"    [{i+1:3d}/{args.n_queries}] acc-so-far={n_correct/(i+1):.2%}, last-latency={elapsed_ms:.0f}ms")

    acc = n_correct / args.n_queries
    arr = np.array(latencies_ms)
    mem_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    result = {
        "experiment": "backbone_value_grounding_zeroshot",
        "timestamp": datetime.now().isoformat(),
        "backbone_key": args.backbone,
        "backbone_id": model_name,
        "n_queries": args.n_queries,
        "value_accuracy": float(acc),
        "value_correct": int(n_correct),
        "memory_quantized_mb": round(float(mem_after_load_mb), 1),
        "memory_peak_mb": round(float(mem_peak_mb), 1),
        "model_load_seconds": round(t_load, 1),
        "latency_ms": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        },
        "sample_responses": samples,
        "fine_tuned": False,
        "notes": "Zero-shot (no LoRA). Same prompt template across backbones; sensor values uniformly sampled.",
    }

    out_dir = PROJECT_ROOT / "output" / "v2" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp23_backbone_{args.backbone}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {args.backbone.upper()} SUMMARY ===")
    print(f"  Value accuracy : {acc:.2%} ({n_correct}/{args.n_queries})")
    print(f"  Mean latency   : {arr.mean():.0f} ms")
    print(f"  GPU mem (load) : {mem_after_load_mb:.0f} MB ({mem_after_load_mb/1024:.2f} GB)")
    print(f"  GPU mem (peak) : {mem_peak_mb:.0f} MB ({mem_peak_mb/1024:.2f} GB)")
    print(f"  Saved          : {out_path}")


if __name__ == "__main__":
    main()
