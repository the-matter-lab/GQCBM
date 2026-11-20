#!/bin/bash
#SBATCH --job-name=finalize
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:3
#SBATCH --time=00:40:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/finalize_%j.out
#SBATCH --error=slurm_logs/finalize_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# (a) cardinality QCBM baselines (KL + MMD)
CUDA_VISIBLE_DEVICES=0 python -u qcbm_pennylane.py \
  --loss kl --n_starts 1 --n_steps 200 --lr 0.05 \
  --filter cardinality \
  --out_json results/qcbmpl_kl_cardinality.json \
  > slurm_logs/qcbm_card_kl.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u qcbm_pennylane.py \
  --loss mmd --n_starts 1 --n_steps 200 --lr 0.1 \
  --filter cardinality \
  --out_json results/qcbmpl_mmd_cardinality.json \
  > slurm_logs/qcbm_card_mmd.log 2>&1 &

# (b) Re-run circuit_stats with fix on GPU 2
CUDA_VISIBLE_DEVICES=2 python -u compute_circuit_stats.py > slurm_logs/ckt_stats_v2.log 2>&1 &

wait
echo 'finalize batch done'
