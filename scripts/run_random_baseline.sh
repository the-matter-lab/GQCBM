#!/bin/bash
#SBATCH --job-name=rnd_bl
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/rnd_bl_%j.out
#SBATCH --error=slurm_logs/rnd_bl_%j.err
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs
python -u random_baseline.py
