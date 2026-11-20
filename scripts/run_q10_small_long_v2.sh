#!/bin/bash
#SBATCH --job-name=q10_small_v2
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/q10_small_v2_%j.out
#SBATCH --error=slurm_logs/q10_small_v2_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# Fix: kl_temp_min 0.05→0.01, same as 5-qubit best setting
# Previous run (39175) stalled at KL=0.0147 because kl_temp floor was too high
echo "========== q10 small model + long training v2: kl_temp_min=0.01 =========="
python -u gqe_single_target.py \
    --n_qubits 10 \
    --n_gates 30 \
    --angle_bins_K 9 \
    --n_embd 128 \
    --n_layer 4 \
    --n_head 4 \
    --max_iters 4000 \
    --num_samples 30 \
    --lr 3e-4 \
    --temp_start 1.0 \
    --ent_coef 0.02 \
    --buf_capacity 1000 \
    --buf_batch_size 64 \
    --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 \
    --temp_final 0.3 \
    --target_kind gaussian \
    --target_kw '{"center":0.5,"sigma":0.10}' \
    --seed 42 \
    --log_every 200 \
    --save_every 500 \
    --save_ckpt ckpts/gqe_q10_small_v2.pt \
    --log_dir gqe_experiments/q10_small_v2
