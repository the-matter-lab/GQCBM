#!/bin/bash
#SBATCH --job-name=qcbm_n20_m2
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=01:55:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_n20_m2_%j.out
#SBATCH --error=slurm_logs/qcbm_n20_m2_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# QCBM-SEL n=20 matched-protocol rerun v2, one script for BOTH objectives
# (LOSS=kl or LOSS=mmd, passed via --export). Motivation (2026-08-13 audit):
#   H1  the previous "matched" pair was not matched -- KL ran 10 steps/cycle
#       patience 8 x 3 cycles, MMD ran 20/12 x up-to-7 cycles (~4-6x budget gap),
#       and the two columns also had asymmetric warm starts.
#   H3  lognormal-20 MMD cell was an interrupted mid-cycle value (partial=true).
#   H4  adam_optimise stored best_params one Adam step past the argmin
#       (fixed in qcbm_pennylane.py on 2026-08-13); all previous QCBM-SEL
#       numbers carry a 2.6-4.6% KL inflation at lr=0.1.
# This rerun: BOTH objectives start from the SAME fresh random init (same
# seed, no warm start), same lr=0.1, same 20 steps/cycle, same
# early_stop_patience 12, same <1% plateau rule on their OWN objective,
# same MAX_CYCLES=10 -- a genuinely matched protocol, with the fixed code.
# Fresh output files (qcbm_n20m2_{kl,mmd}_*.json); nothing old is overwritten.

LOSS=${LOSS:?set LOSS=kl or LOSS=mmd via --export}
CYCLE=${CYCLE:-1}
MAX_CYCLES=${MAX_CYCLES:-10}
LR=0.1
TARGETS=(gauss_20q bimodal_20q exp_20q lognormal_20q)

METRIC_FIELD=$([ "$LOSS" = "kl" ] && echo best_kl || echo best_mmd)
EXTRA=$([ "$LOSS" = "kl" ] && echo "--skip_mmd_eval")

echo "===== QCBM-SEL+${LOSS^^} n=20 matched-v2 cycle ${CYCLE}/${MAX_CYCLES} lr=${LR} ====="

PREV_FILE=/tmp/qcbm_n20m2_${LOSS}_prev_$$.txt; :>$PREV_FILE
for i in "${!TARGETS[@]}"; do
  T="${TARGETS[$i]}"; J="results/qcbm_n20m2_${LOSS}_${T}.json"
  PREV=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r['$METRIC_FIELD'])" 2>/dev/null)
  echo "$T $PREV" >> $PREV_FILE
  CUDA_VISIBLE_DEVICES=${i} python -u qcbm_pennylane.py \
    --filter "${T}" --loss ${LOSS} --lr ${LR} ${EXTRA} \
    --n_steps 20 --early_stop_patience 12 \
    --out_json ${J} \
    >> slurm_logs/qcbm_n20m2_${LOSS}_${T}.log 2>&1 &
done
wait

PLATEAU=1
while read T PREV; do
  J="results/qcbm_n20m2_${LOSS}_${T}.json"
  NEW=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r['$METRIC_FIELD'])" 2>/dev/null)
  if [ -z "$PREV" ] || [ -z "$NEW" ]; then
    echo "  $T prev=$PREV new=$NEW UNKNOWN (treating as improving)"
    PLATEAU=0
    continue
  fi
  REL=$(python3 -c "p=$PREV; n=$NEW; r=(p-n)/abs(p) if p != 0 else 1.0; print('IMPROVING' if r>0.01 else 'PLATEAU', f'rel={r:.4f}')")
  echo "  $T prev=$PREV new=$NEW $REL"
  echo "$REL" | grep -q "^IMPROVING" && PLATEAU=0
done < $PREV_FILE
rm -f $PREV_FILE

if [ $PLATEAU -eq 0 ] && [ $CYCLE -lt $MAX_CYCLES ]; then
  NEXT=$((CYCLE+1))
  echo "===== resubmitting ${LOSS} cycle ${NEXT} ====="
  TRY=0
  while [ $TRY -lt 10 ]; do
    OUT=$(sbatch --export=ALL,LOSS=${LOSS},CYCLE=${NEXT},MAX_CYCLES=${MAX_CYCLES} scripts/run_qcbm_n20_matched_v2.sh 2>&1)
    echo "$OUT"
    echo "$OUT" | grep -q "Submitted batch job" && break
    echo "$OUT" | grep -q "QOSMaxSubmitJobPerUserLimit" || break
    echo "$(date +%T) QoS full, sleep 60s try=$TRY"
    sleep 60; TRY=$((TRY+1))
  done
else
  echo "===== ${LOSS} plateau reached or max cycles ====="
fi
