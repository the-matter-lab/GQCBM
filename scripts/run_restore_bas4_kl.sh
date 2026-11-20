#!/bin/bash
#SBATCH --job-name=restore_bas4
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:30:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/restore_bas4_%j.out
#SBATCH --error=slurm_logs/restore_bas4_%j.err

# Restores ckpts/extras_bas4.pt (GQCBM+KL, BAS 2x2), lost during a
# repository reorganisation. Exact recipe copied from
# scripts/run_extras.sh's BAS-4 block (same seed=42, same all args) so the
# result should reproduce the paper's existing Table 2 BAS 2x2 GQCBM+KL
# number (KL=0.0003, CX=26) for verification, not a fresh/different result.
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

COMMON_ARGS="--angle_bins_K 12 --n_embd 128 --n_layer 4 --n_head 4 \
  --num_samples 50 --lr 3e-4 --temp_start 1.0 --ent_coef 0.05 \
  --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
  --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
  --seed 42 --save_every 1000"

echo "========== BAS 4-qubit (2x2 grid) restore =========="
python -u gqe_single_target.py \
  --n_qubits 4 --n_gates 16 ${COMMON_ARGS} \
  --max_iters 2500 \
  --target_kind bas --target_kw '{}' \
  --mmd_sigmas '0.25,0.5,1.0' \
  --log_every 150 \
  --save_ckpt ckpts/extras_bas4.pt --log_dir gqe_experiments/extras_bas4_restore
