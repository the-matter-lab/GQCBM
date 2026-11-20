#!/bin/bash
#SBATCH --job-name=gqcbm_symMMD
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:3
#SBATCH --time=01:30:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqcbm_symMMD_%j.out
#SBATCH --error=slurm_logs/gqcbm_symMMD_%j.err

# Target-symmetry-aware GQCBM+MMD retrains for D^5_2, D^10_5, BAS-3x3 --
# for a fair 4th (GQCBM+MMD) column in main-text Table 2. The existing
# gqe_hamMMD_cardinality_{5,10}q.pt / gqe_gridMMD_bas9.pt checkpoints used
# the DEFAULT (non-symmetric) operator pool, unlike the GQCBM+KL rows
# currently in Table 2 which use symmetric_pool / row_col_sym_pool_v2 --
# an apples-to-oranges comparison (2026-08-14 investigation). This rerun
# copies each target's exact GQCBM+KL architecture/budget/pool config
# (from the saved checkpoint's own config dict) and only swaps
# loss_metric kl->mmd, using the same mmd_kernel/sigmas already
# established for that target's (non-symmetric) MMD checkpoint.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

echo "===== D^5_2 cardinality (n=5), symmetric_pool + MMD/hamming ====="
CUDA_VISIBLE_DEVICES=0 python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 --angle_bins_K 12 \
  --n_embd 384 --n_layer 6 --n_head 8 \
  --num_samples 50 --lr 3e-4 --ent_coef 0.01 \
  --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
  --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 --temp_start 1.0 \
  --max_iters 2000 --symmetric_pool \
  --loss_metric mmd --mmd_kernel hamming --mmd_sigmas '0.5,1.0,2.0,4.0' \
  --target_kind cardinality --target_kw '{"k":2}' \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_symMMD_card_5q.pt --log_dir gqe_experiments/gqe_symMMD_card_5q \
  >> slurm_logs/gqe_symMMD_card_5q.log 2>&1 &

echo "===== D^10_5 cardinality (n=10), symmetric_pool + MMD/hamming ====="
CUDA_VISIBLE_DEVICES=1 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --n_embd 384 --n_layer 6 --n_head 8 \
  --num_samples 30 --lr 3e-4 --ent_coef 0.01 \
  --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
  --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 --temp_start 1.0 \
  --max_iters 4000 --symmetric_pool \
  --loss_metric mmd --mmd_kernel hamming --mmd_sigmas '0.5,1.0,2.0,4.0' \
  --target_kind cardinality --target_kw '{"k":5}' \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_symMMD_card_10q.pt --log_dir gqe_experiments/gqe_symMMD_card_10q \
  >> slurm_logs/gqe_symMMD_card_10q.log 2>&1 &

echo "===== BAS 3x3 (n=9), row_col_sym_pool_v2=3,3 + MMD/grid ====="
CUDA_VISIBLE_DEVICES=2 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 9 \
  --n_embd 128 --n_layer 4 --n_head 4 \
  --num_samples 20 --lr 3e-4 --ent_coef 0.02 \
  --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
  --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 --temp_start 1.0 \
  --max_iters 1500 --row_col_sym_pool_v2 '3,3' \
  --loss_metric mmd --mmd_kernel grid --mmd_sigmas '0.5,1.0,2.0,3.0' --mmd_grid_shape '3,3' \
  --target_kind bas --target_kw '{}' \
  --seed 42 --log_every 200 --save_every 500 \
  --save_ckpt ckpts/gqe_symMMD_bas9_9q.pt --log_dir gqe_experiments/gqe_symMMD_bas9_9q \
  >> slurm_logs/gqe_symMMD_bas9_9q.log 2>&1 &

wait
echo "===== all 3 done ====="
