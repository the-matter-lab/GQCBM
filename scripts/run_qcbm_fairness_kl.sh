#!/bin/bash
#SBATCH --job-name=qcbmfair_kl
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:6
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbmfair_kl_%j.out
#SBATCH --error=slurm_logs/qcbmfair_kl_%j.err
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# Real KL-loss fairness: vary training budget, measure if QCBM-SEL+KL catches GQCBM
# 5q Gaussian (test: does extra training close 0.1431 gap to GQCBM 0.0016?)
CUDA_VISIBLE_DEVICES=0 python -u qcbm_pennylane.py --filter gauss_5q_s42 \
  --n_steps 500  --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_5q_s500.json   > slurm_logs/qcbmfairkl_5q_s500.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u qcbm_pennylane.py --filter gauss_5q_s42 \
  --n_steps 2000 --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_5q_s2000.json  > slurm_logs/qcbmfairkl_5q_s2000.log 2>&1 &

# 10q Gaussian (test: does 10q QCBM 0.5735 close to GQCBM 0.0064 with more steps?)
CUDA_VISIBLE_DEVICES=2 python -u qcbm_pennylane.py --filter gauss_10q \
  --n_steps 500  --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_10q_s500.json  > slurm_logs/qcbmfairkl_10q_s500.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u qcbm_pennylane.py --filter gauss_10q \
  --n_steps 1000 --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_10q_s1000.json > slurm_logs/qcbmfairkl_10q_s1000.log 2>&1 &

# 15q Gaussian (test: does 15q QCBM 0.65 close to GQCBM 0.0055 with more steps?)
CUDA_VISIBLE_DEVICES=4 python -u qcbm_pennylane.py --filter gauss_15q_s42 \
  --n_steps 300  --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_15q_s300.json  > slurm_logs/qcbmfairkl_15q_s300.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -u qcbm_pennylane.py --filter gauss_15q_s42 \
  --n_steps 500  --n_starts 1 --loss kl \
  --out_json results/qcbm_fairkl_15q_s500.json  > slurm_logs/qcbmfairkl_15q_s500.log 2>&1 &
wait
echo "KL-loss fairness done"
