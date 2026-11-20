#!/bin/bash
#SBATCH --job-name=q15_v4_finer
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=03:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/q15_v4_finer_%j.out
#SBATCH --error=slurm_logs/q15_v4_finer_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# v4: build on v3-A winner (ent_coef=0.05 → KL=0.0122)
#   Add: angle_bins_K 9→12 (21→27 angle values, smallest angle 0.35°→0.044°)
#   Add: max_iters 5000→8000 (v3-A was still improving at step 4999)
# vocab_size: 360*27+1 = 9721 (vs 7561 with K=9)
echo "========== q15 v4: K=12, longer training =========="
python -u gqe_single_target.py \
    --n_qubits 15 \
    --n_gates 40 \
    --angle_bins_K 12 \
    --n_embd 128 \
    --n_layer 4 \
    --n_head 4 \
    --max_iters 8000 \
    --num_samples 20 \
    --lr 3e-4 \
    --temp_start 1.0 \
    --ent_coef 0.05 \
    --buf_capacity 1000 \
    --buf_batch_size 64 \
    --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 \
    --temp_final 0.3 \
    --target_kind gaussian \
    --target_kw '{"center":0.5,"sigma":0.10}' \
    --seed 42 \
    --log_every 400 \
    --save_every 1000 \
    --save_ckpt ckpts/gqe_q15_v4.pt \
    --log_dir gqe_experiments/q15_v4
