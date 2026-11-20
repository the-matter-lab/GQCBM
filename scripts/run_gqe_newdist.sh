#!/bin/bash
#SBATCH --job-name=gqe_newdist
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_newdist_%j.out
#SBATCH --error=slurm_logs/gqe_newdist_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# log-normal + cardinality-constrained, 5q and 10q. Same hyperparameters
# as the smooth-target multi-seed runs (n_gates/max_iters/num_samples).

CUDA_VISIBLE_DEVICES=0 python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 --angle_bins_K 12 \
  --max_iters 2000 --num_samples 50 --lr 3e-4 \
  --target_kind lognormal --target_kw '{"mu":-0.7,"sigma":0.5}' \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_lognormal_5q.pt \
  --log_dir gqe_experiments/lognormal_5q \
  > slurm_logs/gqe_lognormal_5q.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 30 --lr 3e-4 \
  --target_kind lognormal --target_kw '{"mu":-0.7,"sigma":0.5}' \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_lognormal_10q.pt \
  --log_dir gqe_experiments/lognormal_10q \
  > slurm_logs/gqe_lognormal_10q.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u gqe_single_target.py \
  --n_qubits 5 --n_gates 20 --angle_bins_K 12 \
  --max_iters 2000 --num_samples 50 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":2}' \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_cardinality_5q.pt \
  --log_dir gqe_experiments/cardinality_5q \
  > slurm_logs/gqe_cardinality_5q.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u gqe_single_target.py \
  --n_qubits 10 --n_gates 30 --angle_bins_K 12 \
  --max_iters 4000 --num_samples 30 --lr 3e-4 \
  --target_kind cardinality --target_kw '{"k":5}' \
  --seed 42 --log_every 200 --save_every 1000 \
  --save_ckpt ckpts/gqe_cardinality_10q.pt \
  --log_dir gqe_experiments/cardinality_10q \
  > slurm_logs/gqe_cardinality_10q.log 2>&1 &

wait
echo "log-normal + cardinality (5q, 10q) done"
