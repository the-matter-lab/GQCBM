#!/bin/bash
#SBATCH --job-name=qcbm_ext
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:8
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_ext_%j.out
#SBATCH --error=slurm_logs/qcbm_ext_%j.err

# Extended-budget natural convergence for all PARTIAL targets from v2 batch.
# n_steps=10000, early_stop_patience=1000 (much longer patience).
# WARM-START from existing v2 ckpts so we don't restart from scratch.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# Reuse existing JSON as warm-start (run_one resumes if file exists).
# Output goes to v3 JSON so we keep v2 history; copy v2 → v3 first.

extend() {  # $1=filter $2=gpu_kl $3=gpu_mmd
  cp -f results/qcbmpl_v2_kl_$1.json  results/qcbmpl_v3_kl_$1.json  2>/dev/null
  cp -f results/qcbmpl_v2_mmd_$1.json results/qcbmpl_v3_mmd_$1.json 2>/dev/null
  CUDA_VISIBLE_DEVICES=$2 python -u qcbm_pennylane.py \
    --loss kl --n_starts 1 --n_steps 10000 --lr 0.05 \
    --early_stop_patience 1000 --filter $1 \
    --out_json results/qcbmpl_v3_kl_$1.json \
    > slurm_logs/qcbm_v3_kl_$1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$3 python -u qcbm_pennylane.py \
    --loss mmd --n_starts 1 --n_steps 10000 --lr 0.1 \
    --early_stop_patience 1000 --filter $1 \
    --out_json results/qcbmpl_v3_mmd_$1.json \
    > slurm_logs/qcbm_v3_mmd_$1.log 2>&1 &
}

case "${BATCH:-medium}" in
  medium)
    extend gauss_10q       0 1
    extend lognormal_10q   2 3
    extend cardinality_10q 4 5
    extend ghz10_10q       6 7
    ;;
  large)
    extend gauss_15q_s42   0 1
    extend bas9_9q         2 3
    ;;
esac

wait
echo "extend ${BATCH:-medium} done"
