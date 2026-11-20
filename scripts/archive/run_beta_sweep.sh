#!/bin/bash
#SBATCH --job-name=beta_sw
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:6
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/beta_sw_%j.out
#SBATCH --error=slurm_logs/beta_sw_%j.err
REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# Single variable: --ent_coef. All else IDENTICAL to seed_5q_42 (default 0.05).
BASE="--n_qubits 5 --n_gates 20 --max_iters 2000 --num_samples 50 --lr 3e-4 \
  --target_kind gaussian --target_kw {\"center\":0.5,\"sigma\":0.20} \
  --mmd_sigmas 0.25,1.0,4.0 --angle_bins_K 12 --seed 42"
for i in 0 1 2 3 4 5; do
  case $i in
    0) BETA=0.01 ;;
    1) BETA=0.02 ;;
    2) BETA=0.03 ;;
    3) BETA=0.05 ;;
    4) BETA=0.10 ;;
    5) BETA=0.15 ;;
  esac
  echo "[GPU${i}] beta=${BETA} start"
  CUDA_VISIBLE_DEVICES=${i} python -u gqe_single_target.py $BASE \
    --ent_coef ${BETA} \
    --save_ckpt ckpts/beta_${BETA}.pt \
    --log_dir gqe_experiments/beta_${BETA} \
    > slurm_logs/beta_${BETA}.log 2>&1 &
done
wait
echo "beta sweep done"
