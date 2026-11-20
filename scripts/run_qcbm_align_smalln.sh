#!/bin/bash
#SBATCH --job-name=qcbm_smn_a
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=08:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_smn_align_%j.out
#SBATCH --error=slurm_logs/qcbm_smn_align_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

LOSS=${LOSS:?need LOSS}
LR=$([ "$LOSS" = "mmd" ] && echo 0.05 || echo 0.005)

# 12 cells: n=5/6/10/15 × 4 targets, split across 4 GPUs sequentially.
# Each GPU runs 3 cells in series. 8hr wall plenty.
declare -A GRP
GRP[0]="gauss_5q_s42 bimodal_6q_s42 exp_5q"
GRP[1]="lognormal_5q gauss_10q bimodal_10q"
GRP[2]="exp_10q lognormal_10q gauss_15q_s42"
GRP[3]="bimodal_15q exp_15q lognormal_15q"

for i in 0 1 2 3; do
  (
    for T in ${GRP[$i]}; do
      echo "[GPU$i] $T $LOSS start"
      CUDA_VISIBLE_DEVICES=${i} python -u qcbm_pennylane.py \
        --filter "${T}" --loss $LOSS --lr $LR \
        --n_steps 4000 --early_stop_patience 300 \
        --out_json results/qcbm_aligned_${LOSS}_${T}.json \
        >> slurm_logs/qcbm_aligned_${LOSS}_${T}.log 2>&1
    done
  ) &
done
wait
echo done
