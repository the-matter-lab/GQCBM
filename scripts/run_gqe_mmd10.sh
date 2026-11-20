#!/bin/bash
#SBATCH --job-name=gqe_mmd10
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=02:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_mmd10_%j.out
#SBATCH --error=slurm_logs/gqe_mmd10_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# GQCBM+MMD at n=10, matching each target's existing GQCBM+KL architecture/
# budget exactly (n_gates, angle_bins_K, n_embd/layer/head, num_samples, lr,
# max_iters, ent_coef, target_kw all copied from the KL ckpt configs), just
# --loss_metric kl -> mmd (mmd_sigmas=0.25,60.0, kernel=index -- matches
# QCBM-SEL+MMD and the existing small-n gqe_mmdloss2_*.pt convention). New
# checkpoint family gqe_mmd10_*.pt, does not touch the KL-trained ckpts.
# n=10 state space (1024) is 1024x smaller than n=20 -- expect this to
# complete in a single job, no chaining needed (unlike n=20).

run_one () {
  local GPU=$1; local T=$2; local KIND=$3; local KW=$4
  local NGATES=$5; local K=$6; local NEMBD=$7; local NLAYER=$8; local NHEAD=$9
  local NSAMP=${10}; local MAXIT=${11}; local ENT=${12}
  echo "[GPU${GPU}] ${T} MMD n=10 start"
  CUDA_VISIBLE_DEVICES=${GPU} python -u gqe_single_target.py \
    --n_qubits 10 --n_gates ${NGATES} --angle_bins_K ${K} \
    --n_embd ${NEMBD} --n_layer ${NLAYER} --n_head ${NHEAD} \
    --max_iters ${MAXIT} --num_samples ${NSAMP} --lr 3e-4 \
    --temp_start 1.0 --ent_coef ${ENT} \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
    --loss_metric mmd --mmd_sigmas 0.25,60.0 --mmd_kernel index \
    --target_kind ${KIND} --target_kw "${KW}" \
    --seed 42 --log_every 200 --save_every 200 \
    --save_ckpt   ckpts/gqe_mmd10_${T}.pt \
    --log_dir gqe_experiments/gqe_mmd10_${T} \
    >> slurm_logs/gqe_mmd10_${T}.log 2>&1 &
}

# T KIND KW NGATES K NEMBD NLAYER NHEAD NSAMP MAXIT ENT
run_one 0 gauss_10q     gaussian    '{"center":0.5,"sigma":0.10}' 30 9  128 4 4 30 4000 0.02
run_one 1 bimodal_10q   bimodal     '{"c1":0.2857,"c2":0.7143,"sigma":0.125}' 30 9 128 4 4 20 3000 0.02
run_one 2 exp_10q       exponential '{}' 20 9 128 4 4 20 3000 0.02
run_one 3 lognormal_10q lognormal   '{"mu":-0.7,"sigma":0.5}' 30 12 384 6 8 30 4000 0.01
wait
echo "n=10 GQCBM+MMD done"
