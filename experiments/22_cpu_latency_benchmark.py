"""
Experiment 22: CPU-only end-to-end latency benchmark.

Measures real TinyLLaMA-1.1B inference latency on CPU only (no GPU)
to provide a constrained-device data point for the IEEE CEM
revision (addressing reviewer R1.3).

Usage:
    python experiments/22_cpu_latency_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Force CPU-only: hide all GPUs before importing torch.cuda
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).parent.parent

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_QUERIES = 10
PROMPT = (
    "You are a smart home assistant grounded in real sensor data.\n"
    "Sensor: electricity. Current reading: 152.4 kW. "
    "Mean over last 10 readings: 148.7 kW. Std: 6.3 kW. "
    "Trend: slightly increasing.\n"
    "Question: What is the current power consumption?\nAnswer:"
)


def find_latest_lora():
    """Find latest LoRA fine-tuned multi-task model."""
    models_dir = PROJECT_ROOT / "output" / "models"
    if not models_dir.exists():
        return None
    candidates = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tinyllama_multitask_")]
    )
    return candidates[-1] / "final" if candidates else None


def measure_process_memory_mb():
    """RSS memory of this process in MB (Linux)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return -1.0
    return -1.0


def main():
    print(f"[{datetime.now().isoformat()}] CPU-only TGP latency benchmark")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")

    mem_start = measure_process_memory_mb()
    print(f"  Process RSS before model load: {mem_start:.1f} MB")

    t_load = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    lora_path = find_latest_lora()
    if lora_path is not None:
        print(f"  Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(base, str(lora_path))
    else:
        print("  WARN: no LoRA checkpoint found, using base TinyLLaMA")
        model = base
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t_load = time.perf_counter() - t_load
    mem_after_load = measure_process_memory_mb()
    print(f"  Model loaded in {t_load:.1f} s")
    print(f"  Process RSS after load: {mem_after_load:.1f} MB (delta {mem_after_load - mem_start:.1f} MB)")

    # Warm-up (first inference is slower)
    print("  Warming up...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    # Timed queries
    latencies_ms = []
    for i in range(N_QUERIES):
        inputs = tokenizer(PROMPT, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"    Query {i+1:2d}: {elapsed_ms:8.1f} ms  ({n_new} new tokens)")

    mem_final = measure_process_memory_mb()
    arr = np.array(latencies_ms)

    result = {
        "experiment": "cpu_latency_benchmark",
        "timestamp": datetime.now().isoformat(),
        "device": "cpu",
        "platform": "server x86 CPU (CUDA hidden)",
        "backbone": BASE_MODEL,
        "lora_loaded": lora_path is not None,
        "n_queries": N_QUERIES,
        "max_new_tokens": 50,
        "model_load_seconds": round(t_load, 1),
        "mem_rss_mb_start": round(mem_start, 1),
        "mem_rss_mb_after_load": round(mem_after_load, 1),
        "mem_rss_mb_final": round(mem_final, 1),
        "torch_num_threads": torch.get_num_threads(),
        "latency_ms": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "raw_ms": [round(x, 1) for x in latencies_ms],
        },
    }

    out_dir = PROJECT_ROOT / "output" / "v2" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp22_cpu_latency_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"  Device         : CPU only ({torch.get_num_threads()} threads, FP32)")
    print(f"  Backbone       : TinyLLaMA-1.1B + LoRA" if lora_path else "  Backbone       : TinyLLaMA-1.1B (base)")
    print(f"  Mean latency   : {arr.mean()/1000:.2f} s")
    print(f"  P95 latency    : {np.percentile(arr, 95)/1000:.2f} s")
    print(f"  P99 latency    : {np.percentile(arr, 99)/1000:.2f} s")
    print(f"  Peak RSS       : {mem_final:.0f} MB")
    print(f"  Result saved   : {out_path}")


if __name__ == "__main__":
    main()
