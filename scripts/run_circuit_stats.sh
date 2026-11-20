#!/bin/bash
#SBATCH --job-name=ckt_stats
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:15:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/ckt_stats_%j.out
#SBATCH --error=slurm_logs/ckt_stats_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
python -u compute_circuit_stats.py
python -u make_cx_vs_kl.py
