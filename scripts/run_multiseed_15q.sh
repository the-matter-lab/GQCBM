#!/bin/bash
#SBATCH --job-name=multiseed_15q
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=04:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/multiseed_15q_%j.out
#SBATCH --error=slurm_logs/multiseed_15q_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# 15q Gaussian best config (v4: K=12, ent=0.05, 8000 iters) — 3 seeds
for SEED in 42 123 456; do
  echo "========== 15q Gaussian seed=${SEED} =========="
  python -u gqe_single_target.py \
    --n_qubits 15 --n_gates 40 --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 8000 --num_samples 20 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind gaussian --target_kw '{"center":0.5,"sigma":0.10}' \
    --mmd_max_n_states 4096 \
    --seed ${SEED} --log_every 400 --save_every 1000 \
    --save_ckpt ckpts/seed_15q_${SEED}.pt \
    --log_dir gqe_experiments/seed_15q_${SEED}
done
