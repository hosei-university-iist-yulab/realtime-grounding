#!/usr/bin/env python3
"""
Unified Multi-Dataset Experiment Runner.

Runs experiments across all datasets with multiple seeds.
Supports parallel execution on multiple GPUs.

Usage:
    python scripts/run_all_datasets.py --exp 02 --gpu 4
    python scripts/run_all_datasets.py --exp 02,03,04 --gpu 4,5,6,7
    python scripts/run_all_datasets.py --exp all --seeds 2025,2026 --gpu 4,5,6,7

Output structure:
    output/v2/{dataset}/seed{seed}/exp{N}_{name}_{timestamp}.json
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.datasets import DATASETS, DATASET_ORDER, DEFAULT_SEEDS


# Experiment registry
EXPERIMENTS = {
    '01': {'name': 'latency', 'script': 'experiments/01_latency_benchmark.py', 'per_dataset': False},
    '02': {'name': 'grounding', 'script': 'experiments/02_grounding_accuracy.py', 'per_dataset': True},
    '03': {'name': 'staleness', 'script': 'experiments/03_staleness_detection.py', 'per_dataset': True},
    '04': {'name': 'causal', 'script': 'experiments/04_causal_validation.py', 'per_dataset': True},
    '05': {'name': 'ablation', 'script': 'experiments/05_ablation_study.py', 'per_dataset': True},
    '06': {'name': 'scalability', 'script': 'experiments/06_scalability.py', 'per_dataset': False},
    '07': {'name': 'sota', 'script': 'experiments/07_sota_comparison.py', 'per_dataset': True},
    '08': {'name': 'cost', 'script': 'experiments/08_computational_cost.py', 'per_dataset': False},
    '09': {'name': 'sampling', 'script': 'experiments/09_sampling_robustness.py', 'per_dataset': True},
    '10': {'name': 'crossdataset', 'script': 'experiments/10_cross_dataset_validation.py', 'per_dataset': False},
    '11': {'name': 'trend', 'script': 'experiments/11_trend_features.py', 'per_dataset': True},
}


def run_experiment(exp_id: str, dataset: str, seed: int, gpu: int) -> Dict:
    """Run a single experiment on a dataset with a seed."""
    exp_config = EXPERIMENTS.get(exp_id)
    if not exp_config:
        return {'error': f'Unknown experiment: {exp_id}'}

    script_path = PROJECT_ROOT / exp_config['script']
    if not script_path.exists():
        return {'error': f'Script not found: {script_path}'}

    # Set output directory
    output_dir = PROJECT_ROOT / 'output' / 'v2' / dataset / f'seed{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    cmd = [
        sys.executable, str(script_path),
        '--seed', str(seed),
        '--dataset', dataset,
        '--output-dir', str(output_dir),
    ]

    print(f"  [GPU {gpu}] Running exp{exp_id} on {dataset} (seed={seed})...")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )

        if result.returncode != 0:
            return {
                'status': 'error',
                'exp_id': exp_id,
                'dataset': dataset,
                'seed': seed,
                'error': result.stderr[-500:] if result.stderr else 'Unknown error'
            }

        return {
            'status': 'success',
            'exp_id': exp_id,
            'dataset': dataset,
            'seed': seed,
        }

    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'exp_id': exp_id, 'dataset': dataset, 'seed': seed}
    except Exception as e:
        return {'status': 'error', 'exp_id': exp_id, 'dataset': dataset, 'seed': seed, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Run experiments across all datasets')
    parser.add_argument('--exp', type=str, required=True,
                       help='Experiment IDs (e.g., "02", "02,03,04", or "all")')
    parser.add_argument('--gpu', type=str, default='4,5,6,7',
                       help='GPU IDs to use (e.g., "4,5,6,7")')
    parser.add_argument('--seeds', type=str, default='2025,2026',
                       help='Random seeds (e.g., "2025,2026")')
    parser.add_argument('--datasets', type=str, default='all',
                       help='Datasets to run on (e.g., "bdg2,ukdale" or "all")')
    parser.add_argument('--parallel', type=int, default=4,
                       help='Number of parallel jobs')

    args = parser.parse_args()

    # Parse arguments
    if args.exp == 'all':
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = [e.strip().zfill(2) for e in args.exp.split(',')]

    gpus = [int(g) for g in args.gpu.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    if args.datasets == 'all':
        datasets = DATASET_ORDER
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    print("=" * 70)
    print("MULTI-DATASET EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Experiments: {exp_ids}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"GPUs: {gpus}")
    print(f"Parallel jobs: {args.parallel}")
    print("=" * 70)

    # Build job list
    jobs = []
    for exp_id in exp_ids:
        exp_config = EXPERIMENTS.get(exp_id)
        if not exp_config:
            print(f"Warning: Unknown experiment {exp_id}")
            continue

        if exp_config['per_dataset']:
            for dataset in datasets:
                for seed in seeds:
                    jobs.append((exp_id, dataset, seed))
        else:
            # Infrastructure experiments run once per seed
            for seed in seeds:
                jobs.append((exp_id, 'common', seed))

    print(f"\nTotal jobs: {len(jobs)}")
    print()

    # Run jobs with GPU round-robin
    results = []
    gpu_idx = 0

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        for exp_id, dataset, seed in jobs:
            gpu = gpus[gpu_idx % len(gpus)]
            gpu_idx += 1
            future = executor.submit(run_experiment, exp_id, dataset, seed, gpu)
            futures[future] = (exp_id, dataset, seed)

        for future in as_completed(futures):
            exp_id, dataset, seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result.get('status', 'unknown')
                if status == 'success':
                    print(f"  ✓ exp{exp_id} {dataset} seed{seed}")
                else:
                    print(f"  ✗ exp{exp_id} {dataset} seed{seed}: {result.get('error', status)}")
            except Exception as e:
                print(f"  ✗ exp{exp_id} {dataset} seed{seed}: {e}")
                results.append({'status': 'error', 'exp_id': exp_id, 'dataset': dataset, 'seed': seed, 'error': str(e)})

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    success = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - success
    print(f"Success: {success}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed jobs:")
        for r in results:
            if r.get('status') != 'success':
                print(f"  - exp{r.get('exp_id')} {r.get('dataset')} seed{r.get('seed')}: {r.get('error', 'unknown')}")


if __name__ == '__main__':
    main()
