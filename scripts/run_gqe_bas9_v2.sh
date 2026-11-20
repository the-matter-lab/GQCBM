#!/bin/bash
#SBATCH --job-name=bas9_v2
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:3
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/bas9_v2_%j.out
#SBATCH --error=slurm_logs/bas9_v2_%j.err

# Test row_col_sym_pool_v2: adds 6 DIAGONAL off-axis 2-body sum ops to
# the original row_col_sym (72 → 78 ops). Hypothesis: missing diagonals
# is why row_col_sym hit 0.288 plateau — stripes & bars subspaces are not
# coherently mixed by within-row + within-col ops alone.
# Variants combine v2 pool with the best entropy/leak findings from 65046.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# (A) v2 pool, baseline config — isolate the effect of adding diagonals
CUDA_VISIBLE_DEVICES=0 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 50 --lr 3e-4 \
  --target_kind bas --target_kw '{}' \
  --row_col_sym_pool_v2 '3,3' \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_bas9_v2_base.pt \
  --log_dir gqe_experiments/bas9_v2_base \
  > slurm_logs/gqe_bas9_v2_base.log 2>&1 &

# (B) v2 pool + ent_coef=0.10 (best entropy variant from 65046)
CUDA_VISIBLE_DEVICES=1 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 50 --lr 3e-4 \
  --target_kind bas --target_kw '{}' \
  --row_col_sym_pool_v2 '3,3' \
  --loss_metric kl --kl_temp_fixed 0.05 --ent_coef 0.10 \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_bas9_v2_ent.pt \
  --log_dir gqe_experiments/bas9_v2_ent \
  > slurm_logs/gqe_bas9_v2_ent.log 2>&1 &

# (C) v2 pool + kl_leak λ=3 (push leak down with new pool)
CUDA_VISIBLE_DEVICES=2 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 50 --lr 3e-4 \
  --target_kind bas --target_kw '{}' \
  --row_col_sym_pool_v2 '3,3' \
  --loss_metric kl_leak --leak_lambda 3.0 --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_bas9_v2_leak.pt \
  --log_dir gqe_experiments/bas9_v2_leak \
  > slurm_logs/gqe_bas9_v2_leak.log 2>&1 &

wait
echo "BAS-9 row_col_sym_pool_v2 variants done"
