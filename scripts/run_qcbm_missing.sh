#!/bin/bash
#SBATCH --job-name=qcbm_missing
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:2
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_missing_%j.out
#SBATCH --error=slurm_logs/qcbm_missing_%j.err

# Fill QCBM-SEL baselines for new targets added to BENCHMARKS:
#   lognormal_5q, lognormal_10q, cardinality_5q, cardinality_10q
# Run both KL and MMD loss variants for fair comparison.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# Loss=KL on lognormal + cardinality
CUDA_VISIBLE_DEVICES=0 python -u qcbm_pennylane.py \
  --loss kl --n_starts 1 --n_steps 200 --lr 0.05 \
  --filter lognormal \
  --out_json results/qcbmpl_kl_lognormal.json \
  > slurm_logs/qcbm_missing_kl_lognormal.log 2>&1 &

# Loss=MMD on lognormal + cardinality
CUDA_VISIBLE_DEVICES=1 python -u qcbm_pennylane.py \
  --loss mmd --n_starts 1 --n_steps 200 --lr 0.1 \
  --filter lognormal \
  --out_json results/qcbmpl_mmd_lognormal.json \
  > slurm_logs/qcbm_missing_mmd_lognormal.log 2>&1 &

wait
echo "QCBM baselines (lognormal) done"
