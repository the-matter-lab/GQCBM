#!/bin/bash
#SBATCH --job-name=sym2
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:2
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/sym2_%j.out
#SBATCH --error=slurm_logs/sym2_%j.err

# Follow-up to the symmetric_pool breakthrough:
# (A) Confirm card_10q + symmetric_pool with FULL 4000 steps (previous
#     ckpt was step 1000 mid-run at 0.0430 — may go lower).
# (B) BAS-9 with NEW row_col_sym_pool: row-S_3 x col-S_3 mixed symmetry
#     pool (72 ops, vocab 1945) matches BAS structure better than pure
#     permutation-symmetric (which over-symmetrises) or all 3-body (525 ops
#     redundancy). Test if it breaks the 1.10 plateau.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# (A) card_10q + symmetric_pool + full 4000 steps + save_every=500
CUDA_VISIBLE_DEVICES=0 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 30 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":5}' \
  --symmetric_pool \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_sym_card_10q_full.pt \
  --log_dir gqe_experiments/sym_card_10q_full \
  > slurm_logs/gqe_sym_card_10q_full.log 2>&1 &

# (B) BAS-9 + row_col_sym_pool 3x3 (row × col symmetric, 72 ops)
CUDA_VISIBLE_DEVICES=1 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 50 --lr 3e-4 \
  --target_kind bas --target_kw '{}' \
  --row_col_sym_pool '3,3' \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_rcsym_bas9.pt \
  --log_dir gqe_experiments/rcsym_bas9 \
  > slurm_logs/gqe_rcsym_bas9.log 2>&1 &

wait
echo "symmetric_pool extended + row_col_sym BAS-9 done"
