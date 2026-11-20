#!/bin/bash
#SBATCH --job-name=extras
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/extras_%j.out
#SBATCH --error=slurm_logs/extras_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"

COMMON_ARGS="--angle_bins_K 12 --n_embd 128 --n_layer 4 --n_head 4 \
  --num_samples 50 --lr 3e-4 --temp_start 1.0 --ent_coef 0.05 \
  --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
  --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
  --seed 42 --save_every 1000"

echo "========== Bell state (2-qubit GHZ): (|00>+|11>)/sqrt(2) =========="
python -u gqe_single_target.py \
  --n_qubits 2 --n_gates 8 ${COMMON_ARGS} \
  --max_iters 1500 \
  --target_kind ghz --target_kw '{}' \
  --mmd_sigmas '0.25,1.0' \
  --log_every 100 \
  --save_ckpt ckpts/extras_bell.pt --log_dir gqe_experiments/extras_bell

echo "========== W state 4-qubit: (|0001>+|0010>+|0100>+|1000>)/2 =========="
python -u gqe_single_target.py \
  --n_qubits 4 --n_gates 16 ${COMMON_ARGS} \
  --max_iters 2500 \
  --target_kind w_state --target_kw '{}' \
  --mmd_sigmas '0.25,1.0,4.0' \
  --log_every 150 \
  --save_ckpt ckpts/extras_w4.pt --log_dir gqe_experiments/extras_w4

echo "========== W state 5-qubit =========="
python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 ${COMMON_ARGS} \
  --max_iters 3000 \
  --target_kind w_state --target_kw '{}' \
  --mmd_sigmas '0.25,1.0,4.0' \
  --log_every 200 \
  --save_ckpt ckpts/extras_w5.pt --log_dir gqe_experiments/extras_w5

echo "========== Exponential 5-qubit =========="
python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 ${COMMON_ARGS} \
  --max_iters 2000 \
  --target_kind exponential --target_kw '{"scale":0.25}' \
  --mmd_sigmas '0.25,1.0,4.0' \
  --log_every 200 \
  --save_ckpt ckpts/extras_exp5.pt --log_dir gqe_experiments/extras_exp5

echo "========== BAS 4-qubit (2x2 grid, 6 patterns; matches Benedetti 2018 BAS(2,2)) =========="
python -u gqe_single_target.py \
  --n_qubits 4 --n_gates 16 ${COMMON_ARGS} \
  --max_iters 2500 \
  --target_kind bas --target_kw '{}' \
  --mmd_sigmas '0.25,0.5,1.0' \
  --log_every 150 \
  --save_ckpt ckpts/extras_bas4.pt --log_dir gqe_experiments/extras_bas4
