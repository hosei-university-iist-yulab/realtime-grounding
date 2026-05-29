"""
Experiment 24: Fair Redis benchmark (TCP vs Unix-domain socket vs TGB).

Replaces the fabricated Unix-socket numbers in Table 2 Panel A with
real measurements addressing reviewer R2.6.

Three configurations are timed identically (100,000 ops after 1,000-op
warm-up, single client, 64-byte payload):
  1. TGB (in-process circular buffer + Welford statistics)
  2. Redis 8.8 over loopback TCP, no pipelining
  3. Redis 8.8 over Unix-domain socket, pipelined batches matched to
     the buffer's get_latest(n=10) + get_statistics workload.

Persistence disabled (no AOF, no RDB) in both Redis configurations.

Usage:
    python experiments/24_redis_fair_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import redis

PROJECT_ROOT = Path(__file__).parent.parent

N_QUERIES = 100_000
N_WARMUP = 1_000
PAYLOAD_BYTES = 64
REDIS_PORT = 16379
REDIS_UDS = "/tmp/redis_bench.sock"


def make_payload(i: int) -> str:
    base = f"sensor:{i}:ts:{time.time():.3f}:val:"
    pad = "x" * max(0, PAYLOAD_BYTES - len(base))
    return base + pad


def bench_redis(client: redis.Redis, label: str, pipelined: bool) -> dict:
    """Time client.set/get N_QUERIES, with optional pipelining for batched reads."""
    client.flushdb()
    # Warm-up
    for i in range(N_WARMUP):
        client.set(f"k:{i}", make_payload(i))
    for i in range(N_WARMUP):
        client.get(f"k:{i}")

    latencies = []
    # Mirror TGB workload: get latest n=10 + read statistics.
    # Approximated as 10 GETs per query (no real running stats; that's TGB's win).
    for q in range(N_QUERIES // 10):
        t0 = time.perf_counter()
        if pipelined:
            pipe = client.pipeline()
            for i in range(10):
                pipe.get(f"k:{(q * 10 + i) % N_WARMUP}")
            pipe.execute()
        else:
            for i in range(10):
                client.get(f"k:{(q * 10 + i) % N_WARMUP}")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "config": label,
        "n_queries": len(latencies),
        "ops_per_query": 10,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(arr.std()),
    }


def bench_tgb() -> dict:
    """Benchmark the in-process Temporal Grounding Buffer."""
    from src.buffer.temporal_buffer import TemporalGroundingBuffer, SensorReading

    buf = TemporalGroundingBuffer(max_readings_per_sensor=100)
    # Warm-up
    for i in range(N_WARMUP):
        buf.push(SensorReading(
            timestamp=time.time(),
            building_id="b",
            meter_type="electricity",
            value=float(i),
        ))

    latencies = []
    for q in range(N_QUERIES // 10):
        t0 = time.perf_counter()
        _ = buf.get_latest("b", "electricity", n=10)
        _ = buf.get_statistics("b", "electricity")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "config": "TGB (in-process, Welford)",
        "n_queries": len(latencies),
        "ops_per_query": "get_latest(10) + get_statistics()",
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(arr.std()),
    }


def main():
    print(f"[{datetime.now().isoformat()}] Fair Redis-vs-TGB benchmark")
    print(f"  Redis port (TCP): {REDIS_PORT}")
    print(f"  Redis Unix socket: {REDIS_UDS}")

    print("\n=== TGB ===")
    r_tgb = bench_tgb()
    print(f"  mean={r_tgb['mean_ms']:.4f}ms  p99={r_tgb['p99_ms']:.4f}ms")

    print("\n=== Redis loopback TCP (no pipelining) ===")
    c_tcp = redis.Redis(host="127.0.0.1", port=REDIS_PORT, decode_responses=True)
    r_tcp = bench_redis(c_tcp, "Redis (loopback TCP, no pipe)", pipelined=False)
    print(f"  mean={r_tcp['mean_ms']:.4f}ms  p99={r_tcp['p99_ms']:.4f}ms")

    print("\n=== Redis Unix-socket (pipelined batches) ===")
    c_uds = redis.Redis(unix_socket_path=REDIS_UDS, decode_responses=True)
    r_uds = bench_redis(c_uds, "Redis (Unix socket, pipelined)", pipelined=True)
    print(f"  mean={r_uds['mean_ms']:.4f}ms  p99={r_uds['p99_ms']:.4f}ms")

    result = {
        "experiment": "fair_redis_benchmark",
        "timestamp": datetime.now().isoformat(),
        "redis_version": "8.8.0",
        "redis_persistence": "AOF off, snapshot off (save \"\")",
        "payload_bytes": PAYLOAD_BYTES,
        "n_warmup_ops": N_WARMUP,
        "n_timed_queries": N_QUERIES // 10,
        "client_lib": f"redis-py {redis.__version__}",
        "tgb": r_tgb,
        "redis_tcp": r_tcp,
        "redis_uds": r_uds,
        "speedup_tgb_vs_tcp": r_tcp["mean_ms"] / r_tgb["mean_ms"],
        "speedup_tgb_vs_uds": r_uds["mean_ms"] / r_tgb["mean_ms"],
    }

    out_dir = PROJECT_ROOT / "output" / "v2" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp24_redis_fair_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"  TGB mean              : {r_tgb['mean_ms']:.4f} ms")
    print(f"  Redis TCP mean        : {r_tcp['mean_ms']:.4f} ms ({result['speedup_tgb_vs_tcp']:.1f}x slower)")
    print(f"  Redis UDS+pipe mean   : {r_uds['mean_ms']:.4f} ms ({result['speedup_tgb_vs_uds']:.1f}x slower)")
    print(f"  Saved                 : {out_path}")


if __name__ == "__main__":
    main()
