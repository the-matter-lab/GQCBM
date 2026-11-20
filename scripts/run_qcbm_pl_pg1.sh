#!/bin/bash
#SBATCH --job-name=qcbmpl_pg1
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:5
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbmpl_pg1_%j.out
#SBATCH --error=slurm_logs/qcbmpl_pg1_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

TARGETS=(bell_2q ghz5_5q bas4_4q gauss_5q_s42 gauss_5q_s123)
for i in "${!TARGETS[@]}"; do
  T="${TARGETS[$i]}"
  echo "[GPU${i}] ${T} start"
  CUDA_VISIBLE_DEVICES=${i} python -u qcbm_pennylane.py \
    --out_json results/qcbmpl_${T}.json \
    --filter "${T}" > slurm_logs/qcbmpl_${T}.log 2>&1 &
done
wait
echo "pg1 done"
