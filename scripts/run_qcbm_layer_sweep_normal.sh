#!/bin/bash
#SBATCH --job-name=qcbm_lsweep
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:4
#SBATCH --time=08:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qcbm_lsweep_%j.out
#SBATCH --error=slurm_logs/qcbm_lsweep_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p results slurm_logs

# Layer-ablation sweep for QCBM-SEL: covers (n, L) ∈ {10,15,20} × {4,6}.
# Each invocation specifies SWEEP_N and SWEEP_L via --export; the script
# runs MMD then KL stages, each with 4 targets in parallel on 4 GPUs.
# Self-chains: at end of cycle, checks plateau (>1% improvement per cycle)
# and resubmits with annealed lr until plateau or MAX_CYCLES.

SWEEP_N=${SWEEP_N:?need SWEEP_N (10|15|20)}
SWEEP_L=${SWEEP_L:?need SWEEP_L (4|6)}
CYCLE=${CYCLE:-1}
MAX_CYCLES=${MAX_CYCLES:-6}
LR_MMD=${LR_MMD:-0.05}
LR_KL=${LR_KL:-0.005}

# Per-n step caps to fit dev 1hr (both stages combined).
case $SWEEP_N in
  10) NS_MMD=300; NS_KL=400; PAT=100; SKIP="";;
  15) NS_MMD=200; NS_KL=200; PAT=80;  SKIP="";;
  20) NS_MMD=999;  NS_KL=40;  PAT=15;  SKIP="--skip_mmd_eval";;
esac

TARGETS=(gauss_${SWEEP_N}q bimodal_${SWEEP_N}q exp_${SWEEP_N}q lognormal_${SWEEP_N}q)
# Existing TARGETS entry naming: gauss_15q uses gauss_15q_s42
[ "$SWEEP_N" = "15" ] && TARGETS[0]=gauss_15q_s42

echo "===== sweep n=$SWEEP_N L=$SWEEP_L cycle $CYCLE/$MAX_CYCLES ====="
echo "      LR_MMD=$LR_MMD LR_KL=$LR_KL n_steps_mmd=$NS_MMD n_steps_kl=$NS_KL"

# ---------- MMD stage ----------
if [ "$SWEEP_N" != "20" ]; then    # MMD@n=20 infeasible (8TB kernel)
  PREV_FILE=/tmp/lsweep_mmd_${SLURM_JOB_ID}.txt; :>$PREV_FILE
  for i in "${!TARGETS[@]}"; do
    T="${TARGETS[$i]}"
    J="results/sweep_L${SWEEP_L}_mmd_${T}.json"
    PREV=$(python3 -c "import json,os; r=json.load(open('$J'))[0] if os.path.exists('$J') else None; print(r['best_mmd'] if r else 'INF')" 2>/dev/null)
    echo "$T $J $PREV" >> $PREV_FILE
    CUDA_VISIBLE_DEVICES=${i} python -u qcbm_pennylane.py \
      --filter "${T}" --loss mmd --n_layers $SWEEP_L --lr $LR_MMD \
      --n_steps $NS_MMD --early_stop_patience $PAT \
      --out_json $J \
      >> slurm_logs/lsweep_L${SWEEP_L}_mmd_${T}.log 2>&1 &
  done
  wait
  echo "===== MMD stage done ====="
fi

# ---------- KL stage ----------
PREV_FILE_KL=/tmp/lsweep_kl_${SLURM_JOB_ID}.txt; :>$PREV_FILE_KL
for i in "${!TARGETS[@]}"; do
  T="${TARGETS[$i]}"
  J="results/sweep_L${SWEEP_L}_kl_${T}.json"
  PREV=$(python3 -c "import json,os; r=json.load(open('$J'))[0] if os.path.exists('$J') else None; print(r['best_kl'] if r else 'INF')" 2>/dev/null)
  echo "$T $J $PREV" >> $PREV_FILE_KL
  CUDA_VISIBLE_DEVICES=${i} python -u qcbm_pennylane.py \
    --filter "${T}" --loss kl --n_layers $SWEEP_L --lr $LR_KL $SKIP \
    --n_steps $NS_KL --early_stop_patience $PAT \
    --out_json $J \
    >> slurm_logs/lsweep_L${SWEEP_L}_kl_${T}.log 2>&1 &
done
wait
echo "===== KL stage done ====="

# ---------- plateau check ----------
PLATEAU=1
if [ "$SWEEP_N" != "20" ]; then
  while read T J PREV; do
    NEW=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r['best_mmd'])" 2>/dev/null)
    REL=$(python3 -c "p='$PREV'; n='$NEW'; p=float(p) if p!='INF' else float('inf'); n=float(n); r=(p-n)/abs(p) if p<float('inf') else 1.0; print('IMPROVING' if r>0.01 else 'PLATEAU', f'rel={r:.4f}')")
    echo "  MMD $T prev=$PREV new=$NEW $REL"
    echo "$REL" | grep -q "^IMPROVING" && PLATEAU=0
  done < $PREV_FILE
  rm -f $PREV_FILE
fi
while read T J PREV; do
  NEW=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r['best_kl'])" 2>/dev/null)
  REL=$(python3 -c "p='$PREV'; n='$NEW'; p=float(p) if p!='INF' else float('inf'); n=float(n); r=(p-n)/abs(p) if p<float('inf') else 1.0; print('IMPROVING' if r>0.01 else 'PLATEAU', f'rel={r:.4f}')")
  echo "  KL  $T prev=$PREV new=$NEW $REL"
  echo "$REL" | grep -q "^IMPROVING" && PLATEAU=0
done < $PREV_FILE_KL
rm -f $PREV_FILE_KL

if [ $PLATEAU -eq 0 ] && [ $CYCLE -lt $MAX_CYCLES ]; then
  NEXT=$((CYCLE+1))
  NEW_MMD=$(python3 -c "print(f'{$LR_MMD*0.6:.5f}')")
  NEW_KL=$(python3  -c "print(f'{$LR_KL*0.6:.5f}')")
  echo "===== resubmit cycle $NEXT lr_mmd=$NEW_MMD lr_kl=$NEW_KL ====="
  T=0
  while [ $T -lt 10 ]; do
    OUT=$(sbatch --export=ALL,SWEEP_N=$SWEEP_N,SWEEP_L=$SWEEP_L,CYCLE=$NEXT,MAX_CYCLES=$MAX_CYCLES,LR_MMD=$NEW_MMD,LR_KL=$NEW_KL scripts/run_qcbm_layer_sweep_normal.sh 2>&1)
    echo "$OUT"
    echo "$OUT" | grep -q "Submitted batch job" && break
    echo "$OUT" | grep -q "QOSMaxSubmitJobPerUserLimit" || break
    echo "$(date +%T) QoS full, sleep 90s try=$T"
    sleep 15; T=$((T+1))
  done
else
  echo "===== sweep n=$SWEEP_N L=$SWEEP_L plateau reached or max cycles ====="
fi
