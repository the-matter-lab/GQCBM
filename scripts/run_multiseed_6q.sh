#!/bin/bash
#SBATCH --job-name=multiseed_6q
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=03:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/multiseed_6q_%j.out
#SBATCH --error=slurm_logs/multiseed_6q_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

# 6q QCBM head-to-head (PennyLane reports KL=0.0260) — 5 seeds
for SEED in 42 123 456 789 1000; do
  echo "========== 6q QCBM seed=${SEED} =========="
  python -u gqe_single_target.py \
    --n_qubits 6 --n_gates 20 --angle_bins_K 12 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 2000 --num_samples 50 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.05 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 \
    --n_final_samples 512 --temp_final 0.3 \
    --target_kind bimodal --target_kw '{"c1":0.2857,"c2":0.7143,"sigma":0.125}' \
    --mmd_sigmas '0.25,60.0' \
    --seed ${SEED} --log_every 200 --save_every 1000 \
    --save_ckpt ckpts/seed_6q_${SEED}.pt \
    --log_dir gqe_experiments/seed_6q_${SEED}
done
