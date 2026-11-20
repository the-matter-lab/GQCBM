#!/bin/bash
#SBATCH --job-name=gqe_n20_rs
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=01:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_n20_rs_%j.out
#SBATCH --error=slurm_logs/gqe_n20_rs_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# Resume each n=20 GQCBM training from its existing partial ckpt
# (job 67558 stopped at OOM after ~1000-1500 iters). One per GPU, 1500 more
# iters per cycle. Same hyperparameters as run_gqe_n20_continuous.sh.
# This script self-chains: at the end it inspects best_kl improvement and,
# if relative gain > 1%, resubmits itself for another cycle (until plateau
# or CYCLE >= MAX_CYCLES).

CYCLE=${CYCLE:-1}
MAX_CYCLES=${MAX_CYCLES:-6}
TAG="cyc${CYCLE}"

echo "===== GQCBM n=20 resume cycle ${CYCLE}/${MAX_CYCLES} ====="

run_one () {
  local GPU=$1; local T=$2; local KIND=$3; local KW=$4
  echo "[GPU${GPU}] ${T} resume start"
  CUDA_VISIBLE_DEVICES=${GPU} python -u gqe_single_target.py \
    --n_qubits 20 --n_gates 50 --angle_bins_K 9 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 1500 --num_samples 20 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.02 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
    --target_kind ${KIND} --target_kw "${KW}" \
    --seed 42 --log_every 100 --save_every 500 \
    --resume_from ckpts/gqe_${T}.pt \
    --save_ckpt   ckpts/gqe_${T}.pt \
    --log_dir gqe_experiments/gqe_${T} \
    >> slurm_logs/gqe_${T}_resume.log 2>&1 &
}

PREV_FILE=/tmp/gqe_n20_prev_kl_$$.txt
:> $PREV_FILE
for T in gauss_20q bimodal_20q exp_20q lognormal_20q; do
  PREV=$(python3 -c "import torch; print(torch.load('ckpts/gqe_${T}.pt', map_location='cpu', weights_only=False).get('best_kl', float('inf')))" 2>/dev/null)
  echo "$T $PREV" >> $PREV_FILE
done
cat $PREV_FILE

run_one 0 gauss_20q     gaussian    '{"center":0.5,"sigma":0.10}'
run_one 1 bimodal_20q   bimodal     '{"c1":0.2857,"c2":0.7143,"sigma":0.125}'
run_one 2 exp_20q       exponential '{}'
run_one 3 lognormal_20q lognormal   '{"mu":-0.7,"sigma":0.5}'
wait

echo "===== cycle ${CYCLE} done, checking plateau ====="

PLATEAU=1
while read T PREV; do
  NEW=$(python3 -c "import torch; print(torch.load('ckpts/gqe_${T}.pt', map_location='cpu', weights_only=False).get('best_kl', float('inf')))" 2>/dev/null)
  REL=$(python3 -c "p=$PREV; n=$NEW; print('plateau' if (p-n)/abs(p) < 0.01 else 'improving', (p-n)/abs(p))")
  echo "  $T prev=$PREV new=$NEW $REL"
  echo "$REL" | grep -q "^improving" && PLATEAU=0
done < $PREV_FILE
rm -f $PREV_FILE

if [ $PLATEAU -eq 0 ] && [ $CYCLE -lt $MAX_CYCLES ]; then
  NEXT=$((CYCLE+1))
  echo "===== resubmitting cycle ${NEXT} ====="
  # Retry on QOSMaxSubmitJobPerUserLimit (5-job dev cap)
  T=0
  while [ $T -lt 10 ]; do
    OUT=$(sbatch --export=ALL,CYCLE=${NEXT},MAX_CYCLES=${MAX_CYCLES} \
                 scripts/run_gqe_n20_resume.sh 2>&1)
    echo "$OUT"
    echo "$OUT" | grep -q "Submitted batch job" && break
    echo "$OUT" | grep -q "QOSMaxSubmitJobPerUserLimit" || break
    echo "$(date +%T) QoS full, sleep 90s try=$T"
    sleep 15; T=$((T+1))
  done
else
  echo "===== plateau reached or max cycles ====="
fi
