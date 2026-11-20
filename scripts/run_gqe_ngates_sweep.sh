#!/bin/bash
#SBATCH --job-name=gqe_ngates
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=08:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_ngates_%j.out
#SBATCH --error=slurm_logs/gqe_ngates_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# GQCBM n_gates sweep — symmetric to QCBM-SEL layer sweep.
# 4 targets × 2 n × 3 n_gates = 24 cells; distributed across 4 GPUs sequentially.
# Each cell trains 3000 iters; configuration matches the original paper runs
# except for n_gates (and angle_bins_K which we keep at 9 for fair comparison).

NQ=${NQ:?need NQ (10 or 15)}     # which n_qubits to sweep this job

declare -A TARGET_KW=(
  [gauss]='{"center":0.5,"sigma":0.10}'
  [bimodal]='{"c1":0.2857,"c2":0.7143,"sigma":0.125}'
  [exp]='{}'
  [lognormal]='{"mu":-0.7,"sigma":0.5}'
)
declare -A TARGET_KIND=(
  [gauss]=gaussian [bimodal]=bimodal [exp]=exponential [lognormal]=lognormal
)

run_cell () {
  local GPU=$1; local T=$2; local NG=$3
  local KIND="${TARGET_KIND[$T]}"
  local KW="${TARGET_KW[$T]}"
  local CKPT="ckpts/gqe_${T}_${NQ}q_g${NG}.pt"
  local LOG="slurm_logs/gqe_${T}_${NQ}q_g${NG}.log"
  echo "[GPU${GPU}] ${T} n=${NQ} n_gates=${NG} start"
  eval CUDA_VISIBLE_DEVICES=${GPU} python -u gqe_single_target.py \
    --n_qubits $NQ --target_kind $KIND --target_kw "'$KW'" \
    --n_gates $NG --angle_bins_K 9 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 3000 --num_samples 20 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.02 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
    --seed 42 --log_every 250 --save_every 500 \
    --save_ckpt $CKPT \
    --log_dir gqe_experiments/gqe_${T}_${NQ}q_g${NG} \
    \>\> $LOG 2\>\&1
}

# 4 targets × 3 n_gates = 12 cells per NQ. Distribute across 4 GPUs sequentially.
# GPU 0: gauss g20, g30, g50
# GPU 1: bimodal g20, g30, g50
# GPU 2: exp g20, g30, g50
# GPU 3: lognormal g20, g30, g50

(for NG in 20 30 50; do run_cell 0 gauss $NG; done) &
(for NG in 20 30 50; do run_cell 1 bimodal $NG; done) &
(for NG in 20 30 50; do run_cell 2 exp $NG; done) &
(for NG in 20 30 50; do run_cell 3 lognormal $NG; done) &
wait
echo "===== GQCBM n_gates sweep n=${NQ} done ====="
