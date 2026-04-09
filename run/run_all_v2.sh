#!/bin/bash
# ============================================================================
# Run All V2 Experiments on All Datasets (Parallel on GPUs 4-7)
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
GPUS="4 5 6 7"
SEEDS="2025 2026"
DATASETS="bdg2 ukdale uci_household uci_steel uci_tetouan"

# Experiments that run PER DATASET (have --dataset argument)
PER_DATASET_EXPS="02 03 04 05 07 09 11 11b"

# Infrastructure experiments (run ONCE per seed, no --dataset)
INFRA_EXPS="01 06 08 10"

# V2 experiments (run ONCE per seed, no --dataset)
V2_EXPS="11c 11d 12 13 14 20 21"

# Parse arguments
DRY_RUN=false
SPECIFIC_EXPS=""
RUN_INFRA=false
RUN_V2=false
RUN_PER_DATASET=true  # default: run per-dataset experiments

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --exp) SPECIFIC_EXPS="$2"; shift 2 ;;
        --infra) RUN_INFRA=true; RUN_PER_DATASET=false; shift ;;
        --v2) RUN_V2=true; RUN_PER_DATASET=false; shift ;;
        --all) RUN_INFRA=true; RUN_V2=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================================"
echo "V2 MULTI-DATASET EXPERIMENT RUNNER"
echo "============================================================================"
echo "GPUs: $GPUS | Seeds: $SEEDS"
echo "Datasets: $DATASETS"
echo "============================================================================"

LOG_DIR="$PROJECT_DIR/output/v2/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Build and run jobs
gpu_arr=($GPUS)
gpu_idx=0
job_count=0

# ============================================================================
# Per-dataset experiments (02, 03, 04, 05, 07, 09, 11, 11b)
# ============================================================================
if $RUN_PER_DATASET || [[ -n "$SPECIFIC_EXPS" ]]; then
for exp_id in $PER_DATASET_EXPS; do
    # Skip if not in specific list
    if [[ -n "$SPECIFIC_EXPS" ]]; then
        if [[ ! ",$SPECIFIC_EXPS," == *",$exp_id,"* ]]; then
            continue
        fi
    fi

    script_path=$(ls experiments/${exp_id}_*.py 2>/dev/null | head -1)
    if [[ -z "$script_path" ]]; then
        echo "  [SKIP] exp$exp_id: script not found"
        continue
    fi

    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            gpu=${gpu_arr[$((gpu_idx % ${#gpu_arr[@]}))]}
            output_dir="$PROJECT_DIR/output/v2/$dataset/seed$seed"
            log_file="$LOG_DIR/exp${exp_id}_${dataset}_seed${seed}.log"

            mkdir -p "$output_dir"

            if $DRY_RUN; then
                echo "  [DRY] GPU $gpu: exp$exp_id on $dataset (seed=$seed)"
            else
                echo "  [RUN] GPU $gpu: exp$exp_id on $dataset (seed=$seed)"
                CUDA_VISIBLE_DEVICES=$gpu python3 "$script_path" \
                    --seed "$seed" \
                    --dataset "$dataset" \
                    --output-dir "$output_dir" \
                    > "$log_file" 2>&1 &
            fi

            ((gpu_idx++))
            ((job_count++))

            # Wait every 4 jobs (one per GPU)
            if (( gpu_idx % ${#gpu_arr[@]} == 0 )) && ! $DRY_RUN; then
                wait
                echo "  Batch complete."
            fi
        done
    done
done
fi

# ============================================================================
# Infrastructure experiments (01, 06, 08, 10) - run once per seed
# ============================================================================
if $RUN_INFRA || [[ -n "$SPECIFIC_EXPS" ]]; then
    for exp_id in $INFRA_EXPS; do
        # Skip if not in specific list
        if [[ -n "$SPECIFIC_EXPS" ]]; then
            if [[ ! ",$SPECIFIC_EXPS," == *",$exp_id,"* ]]; then
                continue
            fi
        fi

        script_path=$(ls experiments/${exp_id}_*.py 2>/dev/null | head -1)
        if [[ -z "$script_path" ]]; then
            echo "  [SKIP] exp$exp_id: script not found"
            continue
        fi

        for seed in $SEEDS; do
            gpu=${gpu_arr[$((gpu_idx % ${#gpu_arr[@]}))]}
            output_dir="$PROJECT_DIR/output/v2/common/seed$seed"
            log_file="$LOG_DIR/exp${exp_id}_common_seed${seed}.log"

            mkdir -p "$output_dir"

            if $DRY_RUN; then
                echo "  [DRY] GPU $gpu: exp$exp_id (infrastructure, seed=$seed)"
            else
                echo "  [RUN] GPU $gpu: exp$exp_id (infrastructure, seed=$seed)"
                CUDA_VISIBLE_DEVICES=$gpu python3 "$script_path" \
                    --seed "$seed" \
                    --output-dir "$output_dir" \
                    > "$log_file" 2>&1 &
            fi

            ((gpu_idx++))
            ((job_count++))

            # Wait every 4 jobs (one per GPU)
            if (( gpu_idx % ${#gpu_arr[@]} == 0 )) && ! $DRY_RUN; then
                wait
                echo "  Batch complete."
            fi
        done
    done
fi

# ============================================================================
# V2 experiments (11c, 11d, 12, 13, 14, 20, 21) - run once per seed
# ============================================================================
if $RUN_V2 || [[ -n "$SPECIFIC_EXPS" ]]; then
    for exp_id in $V2_EXPS; do
        # Skip if not in specific list
        if [[ -n "$SPECIFIC_EXPS" ]]; then
            if [[ ! ",$SPECIFIC_EXPS," == *",$exp_id,"* ]]; then
                continue
            fi
        fi

        script_path=$(ls experiments/${exp_id}_*.py 2>/dev/null | head -1)
        if [[ -z "$script_path" ]]; then
            echo "  [SKIP] exp$exp_id: script not found"
            continue
        fi

        for seed in $SEEDS; do
            gpu=${gpu_arr[$((gpu_idx % ${#gpu_arr[@]}))]}
            output_dir="$PROJECT_DIR/output/v2/common/seed$seed"
            log_file="$LOG_DIR/exp${exp_id}_v2_seed${seed}.log"

            mkdir -p "$output_dir"

            if $DRY_RUN; then
                echo "  [DRY] GPU $gpu: exp$exp_id (V2, seed=$seed)"
            else
                echo "  [RUN] GPU $gpu: exp$exp_id (V2, seed=$seed)"
                CUDA_VISIBLE_DEVICES=$gpu python3 "$script_path" \
                    --seed "$seed" \
                    --output-dir "$output_dir" \
                    > "$log_file" 2>&1 &
            fi

            ((gpu_idx++))
            ((job_count++))

            # Wait every 4 jobs (one per GPU)
            if (( gpu_idx % ${#gpu_arr[@]} == 0 )) && ! $DRY_RUN; then
                wait
                echo "  Batch complete."
            fi
        done
    done
fi

# Wait for remaining jobs
if ! $DRY_RUN; then
    wait
fi

echo ""
echo "============================================================================"
echo "Total: $job_count jobs"
echo "Logs: $LOG_DIR"
echo "============================================================================"
