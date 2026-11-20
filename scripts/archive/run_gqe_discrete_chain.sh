#!/bin/bash
#SBATCH --job-name=gqe_disc
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:6
#SBATCH --time=08:00:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/gqe_disc_%j.out
#SBATCH --error=slurm_logs/gqe_disc_%j.err

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
mkdir -p ckpts gqe_experiments slurm_logs

# Chain-to-plateau for all 6 discrete GQCBM targets in one normal-partition
# job (6 GPUs, 8hr wall). Each target resumes from its existing ckpt with
# the same pool/kernel config; trains MAX_CYCLES of 1500 iters; saves back.
# Total compute per target ≈ MAX_CYCLES × ~15-25 min. Plateau check at end
# extends chain if any target improved > 1% (saves to same ckpt path).

CYCLE=${CYCLE:-1}
MAX_CYCLES=${MAX_CYCLES:-5}

declare -A CFGS=(
  [ghz5_5q]="--n_qubits 5 --target_kind ghz --target_kw '{}'"
  [bas4_4q]="--n_qubits 4 --target_kind bas --target_kw '{}'"
  [ghz10_10q]="--n_qubits 10 --target_kind ghz --target_kw '{}' --loss_metric mmd --mmd_kernel hamming"
  [bas9_9q]="--n_qubits 9 --target_kind bas --target_kw '{}' --row_col_sym_pool_v2 3,3 --loss_metric kl"
  [card_5q]="--n_qubits 5 --target_kind cardinality --target_kw '{\"k\":2}' --symmetric_pool --loss_metric kl"
  [card_10q]="--n_qubits 10 --target_kind cardinality --target_kw '{\"k\":5}' --symmetric_pool --loss_metric kl"
)
declare -A CKPTS=(
  [ghz5_5q]=ckpts/gqe_ghz5.pt
  [bas4_4q]=ckpts/extras_bas4.pt
  [ghz10_10q]=ckpts/gqe_hamMMD_ghz10_10q.pt
  [bas9_9q]=ckpts/gqe_bas9_v2_ent.pt
  [card_5q]=ckpts/gqe_sym_card_5q.pt
  [card_10q]=ckpts/gqe_sym_card_10q_full.pt
)
declare -A NGATES=(
  [ghz5_5q]=20 [bas4_4q]=20 [ghz10_10q]=30 [bas9_9q]=40 [card_5q]=30 [card_10q]=50
)

TARGETS=(ghz5_5q bas4_4q ghz10_10q bas9_9q card_5q card_10q)

echo "===== GQCBM discrete chain cycle $CYCLE/$MAX_CYCLES ====="

PREV_FILE=/tmp/gqe_disc_${SLURM_JOB_ID}.txt; :>$PREV_FILE
for i in "${!TARGETS[@]}"; do
  T="${TARGETS[$i]}"
  CK="${CKPTS[$T]}"
  PREV=$(python3 -c "import torch; d=torch.load('$CK', map_location='cpu', weights_only=False); print(d.get('best_kl', float('inf')))" 2>/dev/null)
  echo "$T $CK $PREV" >> $PREV_FILE
  CFG="${CFGS[$T]}"
  NG="${NGATES[$T]}"
  echo "[GPU${i}] $T resume start (prev_kl=$PREV)"
  eval CUDA_VISIBLE_DEVICES=${i} python -u gqe_single_target.py \
    $CFG --n_gates $NG --angle_bins_K 9 \
    --n_embd 128 --n_layer 4 --n_head 4 \
    --max_iters 1500 --num_samples 20 --lr 3e-4 \
    --temp_start 1.0 --ent_coef 0.02 \
    --buf_capacity 1000 --buf_batch_size 64 --buf_min_size 50 \
    --kl_temp_min 0.01 --n_final_samples 512 --temp_final 0.3 \
    --seed 42 --log_every 200 --save_every 500 \
    --resume_from $CK --save_ckpt $CK \
    --log_dir gqe_experiments/gqe_disc_${T} \
    \>\> slurm_logs/gqe_disc_${T}_resume.log 2\>\&1 \&
done
wait
echo "===== cycle $CYCLE training done ====="

PLATEAU=1
while read T CK PREV; do
  NEW=$(python3 -c "import torch; d=torch.load('$CK', map_location='cpu', weights_only=False); print(d.get('best_kl', float('inf')))" 2>/dev/null)
  REL=$(python3 -c "p=$PREV; n=$NEW; r=(p-n)/abs(p) if abs(p)>1e-12 else 0; print('IMPROVING' if r>0.01 else 'PLATEAU', f'rel={r:.4f}')")
  echo "  $T prev=$PREV new=$NEW $REL"
  echo "$REL" | grep -q "^IMPROVING" && PLATEAU=0
done < $PREV_FILE
rm -f $PREV_FILE

if [ $PLATEAU -eq 0 ] && [ $CYCLE -lt $MAX_CYCLES ]; then
  NEXT=$((CYCLE+1))
  echo "===== resubmit cycle $NEXT ====="
  T=0
  while [ $T -lt 10 ]; do
    OUT=$(sbatch --export=ALL,CYCLE=$NEXT,MAX_CYCLES=$MAX_CYCLES scripts/archive/run_gqe_discrete_chain.sh 2>&1)
    echo "$OUT"
    echo "$OUT" | grep -q "Submitted batch job" && break
    echo "$OUT" | grep -q "QOSMaxSubmitJobPerUserLimit" || break
    sleep 15; T=$((T+1))
  done
else
  echo "===== GQCBM discrete plateau or max cycles ====="
fi
