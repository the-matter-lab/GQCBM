#!/bin/bash
#SBATCH --job-name=sym_2stage
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/sym_2stage_%j.out
#SBATCH --error=slurm_logs/sym_2stage_%j.err

# 2 new ideas tested in parallel:
#  (A) --symmetric_pool: global-sum operators (S_X, S_{ZZ} etc.) on Dicke
#      targets where Global VQ paper (2312.05060) proves arbitrary
#      symmetric states can be prepared with only these tokens.
#  (B) --leak_first_steps: 2-stage training, score=leak for first N steps
#      to push policy onto support, then switch to KL refinement.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

# (A1) cardinality 5q + symmetric_pool — theoretical fit for Dicke
CUDA_VISIBLE_DEVICES=0 python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 --angle_bins_K 12 \
  --max_iters 2000 --num_samples 50 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":2}' \
  --symmetric_pool \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_sym_card_5q.pt \
  --log_dir gqe_experiments/sym_card_5q \
  > slurm_logs/gqe_sym_card_5q.log 2>&1 &

# (A2) cardinality 10q + symmetric_pool
CUDA_VISIBLE_DEVICES=1 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 30 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":5}' \
  --symmetric_pool \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_sym_card_10q.pt \
  --log_dir gqe_experiments/sym_card_10q \
  > slurm_logs/gqe_sym_card_10q.log 2>&1 &

# (B1) BAS-9 + 3-body + 2-stage (leak first 500 steps, then KL+T_fix)
CUDA_VISIBLE_DEVICES=2 python -u gqe_single_target.py \
  --n_qubits 9 --n_gates 40 --angle_bins_K 12 \
  --max_iters 3000 --num_samples 50 --lr 3e-4 \
  --target_kind bas --target_kw '{}' \
  --add_all_3body \
  --loss_metric kl --kl_temp_fixed 0.05 \
  --leak_first_steps 500 \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_2stage_bas9.pt \
  --log_dir gqe_experiments/2stage_bas9 \
  > slurm_logs/gqe_2stage_bas9.log 2>&1 &

# (B2) cardinality 10q + 3-body + 2-stage (leak first 1000 steps)
CUDA_VISIBLE_DEVICES=3 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 30 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":5}' \
  --add_all_3body \
  --loss_metric kl_leak --leak_lambda 1.0 --kl_temp_fixed 0.05 \
  --leak_first_steps 1000 \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_2stage_card_10q.pt \
  --log_dir gqe_experiments/2stage_card_10q \
  > slurm_logs/gqe_2stage_card_10q.log 2>&1 &

wait
echo "symmetric_pool + 2-stage runs done"
