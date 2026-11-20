#!/bin/bash
#SBATCH --job-name=qcbmcvg
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:5
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbmcvg_%j.out
#SBATCH --error=slurm_logs/qcbmcvg_%j.err
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# Converged QCBM-SEL+KL for Table 1 (replace under-trained 100-step defaults)
# 5 small targets, 2000 steps each, in parallel on 5 GPUs.

CUDA_VISIBLE_DEVICES=0 python -u qcbm_pennylane.py --filter bell_2q \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_cvg_bell_2q.json > slurm_logs/qcbmcvg_bell_2q.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u qcbm_pennylane.py --filter ghz5_5q \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_cvg_ghz5_5q.json > slurm_logs/qcbmcvg_ghz5_5q.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u qcbm_pennylane.py --filter bas4_4q \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_cvg_bas4_4q.json > slurm_logs/qcbmcvg_bas4_4q.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u qcbm_pennylane.py --filter bimodal_6q_s42 \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_cvg_bimodal_6q.json > slurm_logs/qcbmcvg_bimodal_6q.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u qcbm_pennylane.py --filter exp_5q \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_cvg_exp_5q.json > slurm_logs/qcbmcvg_exp_5q.log 2>&1 &
wait
echo "5 converged QCBM runs done"
