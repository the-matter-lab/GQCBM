#!/bin/bash
#SBATCH --job-name=qcbm_cvg
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:8
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_cvg_%j.out
#SBATCH --error=slurm_logs/qcbm_cvg_%j.err

# Re-run QCBM-SEL to convergence on all benchmarks.
# n_steps=5000, early_stop_patience=400 (stop when no improvement for 400 steps).
# History stored in JSON for plateau verification.
# Use $BATCH env var to select target group: small / medium_a / medium_b / large

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# Each invocation: filter to one target, both losses on 2 GPUs.
run_target() {  # $1=filter $2=gpu_kl $3=gpu_mmd
  CUDA_VISIBLE_DEVICES=$2 python -u qcbm_pennylane.py \
    --loss kl --n_starts 1 --n_steps 5000 --lr 0.05 \
    --early_stop_patience 400 --filter $1 \
    --out_json results/qcbmpl_v2_kl_$1.json \
    > slurm_logs/qcbm_v2_kl_$1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$3 python -u qcbm_pennylane.py \
    --loss mmd --n_starts 1 --n_steps 5000 --lr 0.1 \
    --early_stop_patience 400 --filter $1 \
    --out_json results/qcbmpl_v2_mmd_$1.json \
    > slurm_logs/qcbm_v2_mmd_$1.log 2>&1 &
}

case "${BATCH:-small}" in
  small)
    run_target bell_2q       0 1
    run_target ghz5_5q       2 3
    run_target bas4_4q       4 5
    run_target exp_5q        6 7
    ;;
  small_b)
    run_target gauss_5q_s42      0 1
    run_target bimodal_6q_s42    2 3
    run_target lognormal_5q      4 5
    run_target cardinality_5q    6 7
    ;;
  medium)
    run_target gauss_10q         0 1
    run_target lognormal_10q     2 3
    run_target cardinality_10q   4 5
    run_target ghz10_10q         6 7
    ;;
  large)
    run_target bas9_9q           0 1
    run_target gauss_15q_s42     2 3
    # GPU 4-7 idle for these heavy runs
    ;;
  *) echo "unknown BATCH"; exit 1 ;;
esac

wait
echo "batch ${BATCH:-small} done"
