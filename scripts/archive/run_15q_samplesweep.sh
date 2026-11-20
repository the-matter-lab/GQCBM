#!/bin/bash
#SBATCH --job-name=15q_Nsweep
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:6
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/15q_Nsweep_%j.out
#SBATCH --error=slurm_logs/15q_Nsweep_%j.err

# 15-qubit Gaussian × num_samples sweep.
# Single variable: N ∈ {10, 20, 40, 60, 80, 100}.
# Fixed: n_gates=40, max_iters=5000, lr=3e-4, ent_coef=0.05, seed=42.
# Use n_qubits=15, sigma=0.10*n_s — same target as the headline result
# (where GQCBM beat QCBM-MMD 143×).
# Watch convergence speed AND final best_kl: small N = noisy gradient but
# more steps per wall-time; large N = stable gradient but fewer steps.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs
nvidia-smi -L > /dev/null || { echo "ERROR: no CUDA device"; exit 1; }

run() {  # $1=num_samples $2=gpu
  CUDA_VISIBLE_DEVICES=$2 python -u gqe_single_target.py \
    --n_qubits 15 --n_gates 40 --angle_bins_K 12 \
    --max_iters 5000 --num_samples $1 --lr 3e-4 \
    --ent_coef 0.05 \
    --target_kind gaussian --target_kw '{"center":0.5,"sigma":0.10}' \
    --seed 42 --log_every 100 --save_every 500 \
    --save_ckpt ckpts/gqe_15q_N$1.pt \
    --log_dir gqe_experiments/15q_N$1 \
    > slurm_logs/gqe_15q_N$1.log 2>&1 &
}

run 10  0
run 20  1
run 40  2
run 60  3
run 80  4
run 100 5

wait
echo "15q num_samples sweep done"
