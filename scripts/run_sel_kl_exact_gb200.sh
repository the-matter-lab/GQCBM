#!/bin/bash
#SBATCH -A <your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH -J sel-kl-exact
#SBATCH -o /path/to/storage/sel_kl_pilot/logs/%x_%j.out
#
# Re-run of every QCBM-SEL+KL number in the manuscript with the EXACT KL gradient
# (qcbm_pennylane.py --kl_grad exact), on a GB200 cluster.
# Each GROUP reproduces one manuscript protocol, read from the script that produced the original
# numbers; only the gradient (and the CUDA-Q version, see build_env_sel_gb200.sh) differ.
#
# Prerequisite, once:  sbatch scripts/build_env_sel_gb200.sh   (-> $L/env/selenv-aarch64.tar)
# Submit from the repository root (code in /home). Partition and time per group. Measured in the
# 2026-11-10 production jobs (4 runs per node): n=10 2.95 s/step, n=15 ~10.7 s/step, n=20 ~40 s/step
# (the 1-run-per-node pilot 8801667 was faster: 2.28 s/step at n=10).
#   sbatch -p batch      -t 03:30:00 --export=ALL,GROUP=t1_n10    scripts/run_sel_kl_exact_gb200.sh  # ~2.5 h
#   sbatch -p batch_long -t 24:00:00 --export=ALL,GROUP=t1_n15    scripts/run_sel_kl_exact_gb200.sh  # ~12 h
#   sbatch -p batch      -t 03:00:00 --export=ALL,GROUP=t1_n20    scripts/run_sel_kl_exact_gb200.sh  # <= 200 steps (plateaued after 60)
#   sbatch -p batch      -t 02:00:00 --export=ALL,GROUP=t2_small  scripts/run_sel_kl_exact_gb200.sh
#   sbatch -p batch_long -t 24:00:00 --export=ALL,GROUP=t2_medium scripts/run_sel_kl_exact_gb200.sh  # <= 15000 steps, 12-19 h
#   sbatch -p batch      -t 02:00:00 --export=ALL,GROUP=figS4     scripts/run_sel_kl_exact_gb200.sh
#   sbatch -p batch_long -t 24:00:00 --export=ALL,GROUP=t1_n20_aligned scripts/run_sel_kl_exact_gb200.sh  # option 1, ~44 h at 40 s/step
#
# Groups (protocol source):
#   t1_n10, t1_n15  Table 1 aligned   run_qcbm_align_smalln.sh   lr 0.005, 4000 steps, patience 300
#   t1_n20          Table 1 n=20      run_qcbm_n20_matched_v2.sh lr 0.1, 20-step cycles, patience 12,
#                                     --skip_mmd_eval, <=10 cycles, stop at <1% improvement
#   t2_small        Table 2 GHZ-5     run_qcbm_converged.sh      lr 0.1 (default), 2000 steps
#                   Table 2 BAS2x2,D5_2 run_qcbm_converge.sh (v2) lr 0.05, 5000 steps, patience 400
#   t2_medium       GHZ-10, D10_5, BAS3x3: v2 as above, then run_qcbm_extend.sh (v3) continues the
#                                     same file: lr 0.05, 10000 steps, patience 1000
#   figS4           run_qcbm_natconv.sh  Gaussian 5q/10q/15q x lr {0.01,0.03}, 3000/1000/500 steps.
#                   NOTE: make_supp_figures.parse_qcbm_log's regex does not match the trainer's log
#                   format; plot Fig. S4 from the JSON 'history' (step, kl, best_loss) instead.
# Output names are the manuscript's, so the figure scripts only need a different directory.
# Resume (preemption/requeue-safe): on start the job copies in its own earlier output ($OUT) and, if set,
# SEED_FROM=<an earlier job dir with results/ and logs/>. Outputs marked .done are skipped; a cut-short run
# warm-starts from its saved best params and runs only the remaining steps of its --n_steps (counted from
# its log; Adam state is not saved, so it restarts - as in the manuscript's own chained invocations).
# CPUs: request the node's 144 cores (--cpus-per-task); the 2026-11-10 jobs got NumCPUs=1 and their
# s/step grew with the number of runs per node.
set -uo pipefail
GROUP=${GROUP:?set GROUP via --export}
L=/path/to/storage/sel_kl_pilot
OUT=$L/runs/$GROUP/job_${SLURM_JOB_ID}
S=/path/to/scratch
W=$S/selkl_${SLURM_JOB_ID}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
export HF_HOME=$S/cache/huggingface TORCH_HOME=$S/cache/torch PIP_CACHE_DIR=$S/cache/pip TMPDIR=$S/tmp
mkdir -p "$W"/{code,results,logs} "$TMPDIR" "$OUT"
echo "== job $SLURM_JOB_ID group $GROUP on $(hostname) $(date -Is)"

# ---- environment: unpack the packed venv once per node (never reinstall) ----
if [ ! -f "$S/selenv/.unpacked" ]; then
  rm -rf "$S/selenv" && tar -C "$S" -xf "$L/env/selenv-aarch64.tar" && touch "$S/selenv/.unpacked" \
    || { echo "cannot unpack $L/env/selenv-aarch64.tar (run build_env_sel_gb200.sh first)"; exit 1; }
fi
PY=$S/selenv/venv/bin/python
rsync -a --exclude ckpts --exclude results --exclude slurm_logs --exclude paper \
      --exclude gqe_experiments --exclude legacy "$REPO_ROOT"/ "$W"/code/
cd "$W"/code
git -C "$REPO_ROOT" rev-parse HEAD > "$W"/logs/code_commit.txt 2>/dev/null || cp -f COMMIT.txt "$W"/logs/ 2>/dev/null
for SRC in "$OUT" "${SEED_FROM:-}"; do   # own earlier attempt first, then the seed; never clobber
  [ -n "$SRC" ] && [ -d "$SRC/results" ] && { cp -n "$SRC"/results/* "$W"/results/ 2>/dev/null; cp -n "$SRC"/logs/*.log "$W"/logs/ 2>/dev/null; echo "== resumed from $SRC: $(ls "$W"/results | tr '\n' ' ')"; }
done
echo "== cpus: nproc=$(nproc) affinity=$(taskset -cp $$ | cut -d: -f2)"
"$PY" - <<'EOF' || { echo "GPU smoke test failed"; exit 1; }
import cudaq, numpy as np
from qcbm_pennylane import circuit_probs
p = circuit_probs(4, 2, np.linspace(0.1, 2.0, 24)); assert abs(p.sum() - 1) < 1e-9
print('smoke ok:', cudaq.__version__.split(' (')[0])
EOF

# ---- helpers ----
remaining () {  # seconds of walltime left
  squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4;
    else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }'; }
sync_out () { cp -r "$W"/results "$W"/logs "$OUT"/ 2>/dev/null; }
best () { "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); r=r[0] if isinstance(r,list) else r; print(r['best_kl'])" "$1" 2>/dev/null; }
logged_steps () {  # cumulative steps: last attempt's steps_already_done + its last "step N/"
  # (a preempted attempt dies before writing .steps, so the log must carry the running total)
  [ -f "$1" ] && awk '/^== attempt/{b=$0; sub(/.*steps_already_done=/,"",b); s=0} match($0,/step +[0-9]+\//){s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s)} END{print b+s}' "$1" || echo 0; }
train () {  # gpu filter outname args...   (CHAIN=1: no .done marker, no step accounting)
  local g=$1 f=$2 o=$3; shift 3
  local log="$W/logs/${o%.json}.log" done="$W/results/$o.done" side="$W/results/$o.steps"
  [ -f "$done" ] && { echo "== $f -> $o already complete, skipped"; return 0; }
  local args=("$@") i n prev=0 rc now
  if [ "${CHAIN:-0}" = 0 ]; then
    # Steps done = the larger of the sidecar and the log. Both can only undercount: the sidecar is written
    # only on a normal exit (stale after a preemption), and a log seeded from an older job may hold its
    # pre-3c4fdc50 per-attempt counts. Sidecar first wins was the 9015307 bug (count reset on each requeue).
    [ -f "$side" ] && prev=$(cat "$side")
    if [ -f "$log" ]; then local l; l=$(logged_steps "$log"); [ "$l" -gt "$prev" ] && prev=$l; fi
    for i in "${!args[@]}"; do
      if [ "${args[$i]}" = --n_steps ]; then n=${args[$((i+1))]}; args[$((i+1))]=$(( n - prev > 1 ? n - prev : 1 )); fi
    done
    if [ -n "${n:-}" ] && [ "$prev" -ge "$n" ]; then
      echo "$prev" > "$side"; touch "$done"; echo "== $f -> $o reached $prev/$n steps, marked done"; return 0
    fi
  fi
  local budget=$(( $(remaining) - 600 ))
  [ "$budget" -gt 60 ] || { echo "== $f ($o): no time left, skipped"; return 0; }
  echo "== attempt $(date -Is) steps_already_done=$prev" >> "$log"
  CUDA_VISIBLE_DEVICES=$g timeout "$budget" "$PY" -u qcbm_pennylane.py --filter "$f" --loss kl \
    --kl_grad exact --out_json "$W/results/$o" "${args[@]}" >> "$log" 2>&1
  rc=$?
  if [ "${CHAIN:-0}" = 0 ]; then
    now=$(( $(logged_steps "$log") - prev )); echo $(( prev + now )) > "$side"
    [ $rc -eq 0 ] && touch "$done"
  fi
  echo "== $f -> $o exit $rc steps_total=$(( prev + ${now:-0} )) best_kl=$(best "$W/results/$o") $(date -Is)"
}
n20_chain () {  # gpu filter outname: run_qcbm_n20_matched_v2.sh, cycles inside one job
  local g=$1 f=$2 o=$3 c prev new
  [ -f "$W/results/$o.done" ] && { echo "== $f -> $o chain already complete, skipped"; return 0; }
  for c in $(seq 1 10); do
    prev=$(best "$W/results/$o")
    CHAIN=1 train "$g" "$f" "$o" --lr 0.1 --n_steps 20 --early_stop_patience 12 --skip_mmd_eval
    new=$(best "$W/results/$o")
    [ -n "$prev" ] && [ -n "$new" ] && "$PY" -c "import sys; p,n=float(sys.argv[1]),float(sys.argv[2]); sys.exit(0 if (p-n)/abs(p) > 0.01 else 1)" "$prev" "$new" \
      || { [ -z "$prev" ] && continue; echo "== $f: plateau after cycle $c"; break; }
  done
  touch "$W/results/$o.done"
}
v2_then_v3 () {  # gpu filter: run_qcbm_converge.sh (v2) then run_qcbm_extend.sh (v3)
  local g=$1 f=$2
  train "$g" "$f" "qcbmpl_v2_kl_$f.json" --lr 0.05 --n_steps 5000 --early_stop_patience 400 --n_starts 1
  [ -f "$W/results/qcbmpl_v2_kl_$f.json.done" ] || return 0      # v2 cut short: v3 must not start
  [ -f "$W/results/qcbmpl_v3_kl_$f.json" ] || cp "$W/results/qcbmpl_v2_kl_$f.json" "$W/results/qcbmpl_v3_kl_$f.json"
  train "$g" "$f" "qcbmpl_v3_kl_$f.json" --lr 0.05 --n_steps 10000 --early_stop_patience 1000 --n_starts 1
}

( while sleep 300; do sync_out; done ) & SYNC=$!
trap 'sync_out; echo "== TERM: synced"; exit 143' TERM
ALIGNED=(--lr 0.005 --n_steps 4000 --early_stop_patience 300)
pids=()
case "$GROUP" in
  t1_n10)
    i=0; for T in gauss_10q bimodal_10q exp_10q lognormal_10q; do
      train $i $T "qcbm_aligned_kl_$T.json" "${ALIGNED[@]}" & pids+=($!); i=$((i+1)); done ;;
  t1_n15)
    i=0; for T in gauss_15q_s42 bimodal_15q exp_15q lognormal_15q; do
      train $i $T "qcbm_aligned_kl_$T.json" "${ALIGNED[@]}" & pids+=($!); i=$((i+1)); done ;;
  t1_n20_aligned)   # backlog option 1: the n<=15 aligned protocol at n=20 (decided 2026-11-10)
    i=0; for T in gauss_20q bimodal_20q exp_20q lognormal_20q; do
      train $i $T "qcbm_aligned_kl_$T.json" "${ALIGNED[@]}" --skip_mmd_eval & pids+=($!); i=$((i+1)); done ;;
  t1_n20)
    i=0; for T in gauss_20q bimodal_20q exp_20q lognormal_20q; do
      n20_chain $i $T "qcbm_n20m2_kl_$T.json" & pids+=($!); i=$((i+1)); done ;;
  t2_small)
    train 0 ghz5_5q "qcbm_cvg_ghz5_5q.json" --n_steps 2000 --n_starts 1 & pids+=($!)
    train 1 bas4_4q "qcbmpl_v2_kl_bas4_4q.json" --lr 0.05 --n_steps 5000 --early_stop_patience 400 --n_starts 1 & pids+=($!)
    train 2 cardinality_5q "qcbmpl_v2_kl_cardinality_5q.json" --lr 0.05 --n_steps 5000 --early_stop_patience 400 --n_starts 1 & pids+=($!) ;;
  t2_medium)
    v2_then_v3 0 ghz10_10q & pids+=($!)
    v2_then_v3 1 cardinality_10q & pids+=($!)
    v2_then_v3 2 bas9_9q & pids+=($!) ;;
  figS4)
    train 0 gauss_5q_s42  "qcbm_nc_5q_lr0.01.json"  --lr 0.01 --n_steps 3000 --n_starts 1 & pids+=($!)
    train 1 gauss_5q_s42  "qcbm_nc_5q_lr0.03.json"  --lr 0.03 --n_steps 3000 --n_starts 1 & pids+=($!)
    train 2 gauss_10q     "qcbm_nc_10q_lr0.01.json" --lr 0.01 --n_steps 1000 --n_starts 1 & pids+=($!)
    train 3 gauss_10q     "qcbm_nc_10q_lr0.03.json" --lr 0.03 --n_steps 1000 --n_starts 1 & pids+=($!)
    train 0 gauss_15q_s42 "qcbm_nc_15q_lr0.01.json" --lr 0.01 --n_steps 500 --n_starts 1 & pids+=($!)
    train 1 gauss_15q_s42 "qcbm_nc_15q_lr0.03.json" --lr 0.03 --n_steps 500 --n_starts 1 & pids+=($!) ;;
  *) echo "unknown GROUP=$GROUP"; kill $SYNC; exit 1 ;;
esac
wait "${pids[@]}"        # only the training processes (the pilot hung on a bare `wait`)
kill $SYNC 2>/dev/null; sync_out
"$PY" - "$W"/results <<'EOF' | tee "$W"/logs/summary.txt
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    r = json.load(open(f)); r = r[0] if isinstance(r, list) else r
    print(f"{f.split('/')[-1]:40s} kl_grad={r.get('kl_grad')} best_kl={float(r['best_kl']):.5f} "
          f"partial={r.get('partial')} starts_done={r.get('starts_done')} elapsed={float(r.get('elapsed', 0)):.0f}s")
EOF
sync_out
echo "== done $(date -Is); results in $OUT"
