#!/bin/bash
#SBATCH --job-name=qcbm_nc
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:6
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_nc_%j.out
#SBATCH --error=slurm_logs/qcbm_nc_%j.err
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# Natural convergence: lower lr to avoid Adam blowup, fresh random init
# (new filenames → no carry-forward), step-ckpt saves partial progress.

# 5q: 3000 steps × 2 lr settings
CUDA_VISIBLE_DEVICES=0 python -u qcbm_pennylane.py --filter gauss_5q_s42 \
  --loss kl --lr 0.01 --n_steps 3000 --n_starts 1 \
  --out_json results/qcbm_nc_5q_lr0.01.json > slurm_logs/qcbm_nc_5q_lr0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u qcbm_pennylane.py --filter gauss_5q_s42 \
  --loss kl --lr 0.03 --n_steps 3000 --n_starts 1 \
  --out_json results/qcbm_nc_5q_lr0.03.json > slurm_logs/qcbm_nc_5q_lr0.03.log 2>&1 &

# 10q: 1000 steps × 2 lr (step-ckpt every 10, will hit wall ~step 800)
CUDA_VISIBLE_DEVICES=2 python -u qcbm_pennylane.py --filter gauss_10q \
  --loss kl --lr 0.01 --n_steps 1000 --n_starts 1 \
  --out_json results/qcbm_nc_10q_lr0.01.json > slurm_logs/qcbm_nc_10q_lr0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u qcbm_pennylane.py --filter gauss_10q \
  --loss kl --lr 0.03 --n_steps 1000 --n_starts 1 \
  --out_json results/qcbm_nc_10q_lr0.03.json > slurm_logs/qcbm_nc_10q_lr0.03.log 2>&1 &

# 15q: 500 steps × 2 lr (step-ckpt every 10)
CUDA_VISIBLE_DEVICES=4 python -u qcbm_pennylane.py --filter gauss_15q_s42 \
  --loss kl --lr 0.01 --n_steps 500 --n_starts 1 \
  --out_json results/qcbm_nc_15q_lr0.01.json > slurm_logs/qcbm_nc_15q_lr0.01.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u qcbm_pennylane.py --filter gauss_15q_s42 \
  --loss kl --lr 0.03 --n_steps 500 --n_starts 1 \
  --out_json results/qcbm_nc_15q_lr0.03.json > slurm_logs/qcbm_nc_15q_lr0.03.log 2>&1 &

wait
echo "Natural-convergence run done"
