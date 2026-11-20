#!/bin/bash
#SBATCH --job-name=gqe_n20mmd_lr
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=44:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_n20mmd_lr_%j.out
#SBATCH --error=slurm_logs/gqe_n20mmd_lr_%j.err

# Long-run GQCBM+MMD n=20 chain targeting real budget parity with GQCBM+KL's
# n=20 chain-plateau (~10,000-15,000 cumulative iterations; Table 1's Sec
# footnote previously said 150 vs 1500, which understated the true gap --
# GQCBM+KL's n=20 actually used a multi-cycle chain totalling 10-15k
# iterations per Fig 3's caption). Runs on the 8gpus partition (44h budget,
# 4h safety margin under the 48h/2-day limit) instead of dev (4h), with an
# INTERNAL retry-safe cycle loop instead of relying on external
# sbatch-per-cycle resubmission, because auditing the prior run
# (ckpts/gqe_mmd20_*.pt, cycles 258331/258332) found 2 of 5 cycles silently
# CRASHED at `import cudaq` (custatevec GPU-compatibility error) and were
# misread by the old plateau-check as "no improvement" -- a real robustness
# bug, not genuine convergence. This version retries a crashed cycle up to
# 3 times before giving up on it for this pass, and logs failures loudly.
#
# Self-chains by calling sbatch at the END of the script if the per-target
# cycle target (CYCLES_TARGET) isn't reached. Because that resubmit line is
# only reached on normal completion, each target loop stops launching new
# cycles after WALL_GUARD_S seconds (40h of the 44h budget) so the script
# always finishes before SLURM's walltime kill -- the 2026-08 audit found
# job 263059 died at 38/70 cycles exactly because this guard was missing.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

CYCLES_TARGET=${CYCLES_TARGET:-70}   # 70 * 150 = 10,500 cumulative iterations
MAX_RETRIES=3
WALL_GUARD_S=$((40*3600))            # stop starting new cycles after 40h (walltime is 44h)

run_target () {
  local GPU=$1; local T=$2; local KIND=$3; local KW=$4
  local COUNTFILE="slurm_logs/gqe_mmd20_${T}_cycles_done.count"
  local done_cycles=0
  [ -f "$COUNTFILE" ] && done_cycles=$(cat "$COUNTFILE")

  while [ "$done_cycles" -lt "$CYCLES_TARGET" ]; do
    if [ "$SECONDS" -gt "$WALL_GUARD_S" ]; then
      echo "[GPU${GPU}] ${T} walltime guard hit at ${SECONDS}s, leaving remaining cycles to the resubmit" \
        >> slurm_logs/gqe_mmd20_${T}_longrun.log
      break
    fi
    local RESUME=""
    [ -f "ckpts/gqe_mmd20_${T}.pt" ] && RESUME="--resume_from ckpts/gqe_mmd20_${T}.pt"
    local attempt=0
    local ok=0
    while [ $attempt -lt $MAX_RETRIES ]; do
      attempt=$((attempt+1))
      echo "[GPU${GPU}] ${T} cycle $((done_cycles+1))/${CYCLES_TARGET} attempt ${attempt} $(date)" \
        >> slurm_logs/gqe_mmd20_${T}_longrun.log
      CUDA_VISIBLE_DEVICES=${GPU} python -u gqe_single_target.py \
        --n_qubits 20 --n_gates 50 --angle_bins_K 9 \
        --n_embd 128 --n_layer 4 --n_head 4 \
        --max_iters 150 --num_samples 20 --lr 3e-4 \
        --temp_start 1.0 --ent_coef 0.02 \
        --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
        --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
        --loss_metric mmd --mmd_sigmas 0.25,60.0 --mmd_kernel index \
        --target_kind ${KIND} --target_kw "${KW}" \
        --seed 42 --log_every 20 --save_every 20 \
        ${RESUME} \
        --save_ckpt   ckpts/gqe_mmd20_${T}.pt \
        --log_dir gqe_experiments/gqe_mmd20_${T} \
        >> slurm_logs/gqe_mmd20_${T}_longrun.log 2>&1
      local exit_code=$?
      if [ $exit_code -eq 0 ]; then
        ok=1
        break
      else
        echo "[GPU${GPU}] ${T} cycle $((done_cycles+1)) attempt ${attempt} FAILED, retrying..." \
          >> slurm_logs/gqe_mmd20_${T}_longrun.log
        sleep 30
      fi
    done
    if [ $ok -eq 1 ]; then
      done_cycles=$((done_cycles+1))
      echo "$done_cycles" > "$COUNTFILE"
    else
      echo "[GPU${GPU}] ${T} cycle $((done_cycles+1)) FAILED after ${MAX_RETRIES} attempts, aborting this target for this job" \
        >> slurm_logs/gqe_mmd20_${T}_longrun.log
      break
    fi
  done
  echo "[GPU${GPU}] ${T} reached ${done_cycles}/${CYCLES_TARGET} cycles $(date)" \
    >> slurm_logs/gqe_mmd20_${T}_longrun.log
}

run_target 0 gauss_20q     gaussian    '{"center":0.5,"sigma":0.10}' &
run_target 1 bimodal_20q   bimodal     '{"c1":0.2857,"c2":0.7143,"sigma":0.125}' &
run_target 2 exp_20q       exponential '{}' &
run_target 3 lognormal_20q lognormal   '{"mu":-0.7,"sigma":0.5}' &
wait

echo "===== job done $(date), per-target cycle counts: ====="
for T in gauss_20q bimodal_20q exp_20q lognormal_20q; do
  C=0; [ -f "slurm_logs/gqe_mmd20_${T}_cycles_done.count" ] && C=$(cat "slurm_logs/gqe_mmd20_${T}_cycles_done.count")
  echo "  ${T}: ${C}/${CYCLES_TARGET}"
done

# Self-chain if any target hasn't reached the cycle target yet.
NEED_MORE=0
for T in gauss_20q bimodal_20q exp_20q lognormal_20q; do
  C=0; [ -f "slurm_logs/gqe_mmd20_${T}_cycles_done.count" ] && C=$(cat "slurm_logs/gqe_mmd20_${T}_cycles_done.count")
  [ "$C" -lt "$CYCLES_TARGET" ] && NEED_MORE=1
done
if [ $NEED_MORE -eq 1 ]; then
  echo "===== resubmitting for more cycles ====="
  sbatch --export=ALL,CYCLES_TARGET=${CYCLES_TARGET} scripts/run_gqe_n20_mmd_longrun.sh
else
  echo "===== all targets reached cycle target ====="
fi
