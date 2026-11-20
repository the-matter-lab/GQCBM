#!/bin/bash
#SBATCH --job-name=multiseed_5q
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/multiseed_5q_%j.out
#SBATCH --error=slurm_logs/multiseed_5q_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# 5q Gaussian baseline — 5 seeds for variance reporting
for SEED in 42 123 456 789 1000; do
  echo "========== 5q Gaussian seed=${SEED} =========="
  python -u gqe_single_target.py \
    --n_qubits 5 --n_gates 20 --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 2000 --num_samples 50 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind gaussian --target_kw '{"center":0.5,"sigma":0.20}' \
    --mmd_sigmas '0.25,1.0,4.0' \
    --seed ${SEED} --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/seed_5q_${SEED}.pt \
    --log_dir gqe_experiments/seed_5q_${SEED}
done
