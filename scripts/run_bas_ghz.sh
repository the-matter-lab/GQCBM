#!/bin/bash
#SBATCH --job-name=bas_ghz
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/bas_ghz_%j.out
#SBATCH --error=slurm_logs/bas_ghz_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# Use proven settings: small model + ent_coef=0.05 + K=12 + adaptive kl_temp + temp_final=0.3
# Both KL and MMD recorded. MMD sigmas from PennyLane QCBM tutorial defaults.

echo "========== GHZ 5-qubit =========="
# GHZ: 50% |00000>, 50% |11111>
# Sparse target — should be easy in principle (2 non-zero entries)
python -u gqe_single_target.py \
    --n_qubits 5 \
    --n_gates 20 \
    --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 3000 --num_samples 50 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind ghz --target_kw '{}' \
    --mmd_sigmas '0.25,1.0,4.0' \
    --seed 42 --log_every 150 --save_every 1000 \
    --save_ckpt ckpts/gqe_ghz5.pt --log_dir gqe_experiments/ghz5

echo "========== GHZ 10-qubit =========="
python -u gqe_single_target.py \
    --n_qubits 10 \
    --n_gates 30 \
    --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 4000 --num_samples 30 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind ghz --target_kw '{}' \
    --mmd_sigmas '0.25,4.0,32.0' \
    --seed 42 --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/gqe_ghz10.pt --log_dir gqe_experiments/ghz10

echo "========== BAS 9-qubit (3x3 grid, matches PennyLane QCBM) =========="
# PennyLane QCBM tutorial result: KL=0.0755, qBAS not reported here directly
# 14 valid patterns out of 512 states
python -u gqe_single_target.py \
    --n_qubits 9 \
    --n_gates 30 \
    --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 4000 --num_samples 50 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind bas --target_kw '{}' \
    --mmd_sigmas '0.25,0.5,1.0' \
    --seed 42 --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/gqe_bas9.pt --log_dir gqe_experiments/bas9
