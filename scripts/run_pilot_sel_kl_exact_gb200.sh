#!/bin/bash
#SBATCH --job-name=sel_kl_exact_pilot
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Pilot for the QCBM-SEL+KL re-run on a GB200 cluster.
# 2 cells x {exact, legacy} KL gradient, same seed/init, one GPU each:
#   GPU0 exp_10q     exact   aligned protocol (run_qcbm_align_smalln.sh: lr 0.005, patience 300),
#   GPU1 exp_10q     legacy    capped at N10_STEPS steps (full protocol is 4000 = ~4.3 h at H200 speed)
#   GPU2 bimodal_20q exact   N20_STEPS steps of the n=20 matched protocol (one full cycle = 20)
#   GPU3 bimodal_20q legacy    (run_qcbm_n20_matched_v2.sh: lr 0.1, patience 12, --skip_mmd_eval)
# Defaults are the 1-hour timing pilot (N10_STEPS=300, N20_STEPS=3). For the 3.5-hour comparison
# pilot: sbatch -t 03:30:00 --export=ALL,...,N10_STEPS=1500,N20_STEPS=20
# Goals: (1) does the pinned CUDA-Q/CuPy stack run on GB200 (aarch64, sm_100)?  (2) s/step on GB200
# vs the H200 numbers (~3.9 s/step at n=10, ~260 s/step at n=20)  (3) exact vs legacy from one init.
# Every output JSON is fresh (no warm start); nothing in results/ is read or overwritten.
#
# Submit from the repository root (code lives in /home; never run from Lustre):
#   mkdir -p slurm_logs
#   sbatch -A <your_account> --export=ALL,RESULTS_DIR=/path/to/storage \
#          scripts/run_pilot_sel_kl_exact_gb200.sh
# Optional: N10_STEPS (300), N20_STEPS (3), COPY_MARGIN_S (300), PYTHON (python3), PIN=0 skips the pins.
# torch is only imported (gqe_single_target.py), never used for SEL, so its CPU build of the pinned
# version is installed; that saves a multi-GB CUDA download in a short job.
set -uo pipefail

RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR (a Lustre path) via --export}
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
N10_STEPS=${N10_STEPS:-300}
N20_STEPS=${N20_STEPS:-3}
COPY_MARGIN_S=${COPY_MARGIN_S:-300}
PYTHON=${PYTHON:-python3}
PIN=${PIN:-1}

# the cluster rules: node-local NVMe for code, env, caches and temp files (wiped between jobs).
WORK=/path/to/scratch/sel_kl_pilot_${SLURM_JOB_ID}
export HF_HOME=/path/to/scratch/cache/huggingface
export TORCH_HOME=/path/to/scratch/cache/torch
export PIP_CACHE_DIR=/path/to/scratch/cache/pip
export TMPDIR=/path/to/scratch/tmp
mkdir -p "$WORK"/{code,results,logs} "$TMPDIR" "$PIP_CACHE_DIR"

# Stage code only (no checkpoints, results, logs or paper files).
rsync -a --exclude ckpts --exclude results --exclude slurm_logs --exclude paper \
      --exclude gqe_experiments --exclude legacy "$REPO_ROOT"/ "$WORK"/code/
cd "$WORK"/code
LOG="$WORK"/logs/pilot.log
exec > >(tee -a "$LOG") 2>&1
echo "== job $SLURM_JOB_ID on $(hostname)  $(date -Is)  arch $(uname -m)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- environment: pinned versions first (requirements.txt), newest as a fallback ----
$PYTHON -m venv "$WORK"/venv && source "$WORK"/venv/bin/activate
pip install -q --upgrade pip
smoke () {
  python - <<'EOF'
import cudaq, cupy, numpy as np
cudaq.set_target('nvidia', option='fp64')
from qcbm_pennylane import circuit_probs
p = circuit_probs(4, 2, np.linspace(0.1, 2.0, 24))
print('cudaq', cudaq.__version__, '| cupy', cupy.__version__, '|', cupy.cuda.runtime.getDeviceProperties(0)['name'].decode(),
      '| sum p =', float(p.sum()))
assert abs(p.sum() - 1) < 1e-9
EOF
}
OK=0
if [ "$PIN" = 1 ]; then
  pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -q cuda-quantum==0.8.0 cupy-cuda12x==13.6.0 numpy==1.26.4 scipy==1.13.1 \
    && smoke && OK=1 && echo "ENV: pinned versions (requirements.txt)"
fi
if [ $OK = 0 ]; then
  echo "ENV: pinned stack failed or skipped; trying current releases"
  pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -q --upgrade cuda-quantum-cu12 cupy-cuda12x numpy scipy && smoke && OK=1 \
    && echo "ENV: current releases (NOT the paper's pinned versions; record this)"
fi
pip freeze > "$WORK"/logs/pip_freeze.txt
if [ $OK = 0 ]; then
  echo "ENV FAILED: cudaq/cupy cannot run on this node. See $LOG"
  mkdir -p "$RESULTS_DIR"; cp -r "$WORK"/logs "$RESULTS_DIR"/pilot_${SLURM_JOB_ID}_logs; exit 1
fi

# ---- gradient self-test on the real stack: exact vs finite difference (n=4, 6) ----
timeout 8m python -u qcbm_kl_grad_exact.py > "$WORK"/logs/selftest.log 2>&1; cat "$WORK"/logs/selftest.log

# ---- the four pilot runs, one GPU each ----
STOP=$(( $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - COPY_MARGIN_S ))
echo "== run budget ${STOP}s (remaining walltime minus ${COPY_MARGIN_S}s for copy-back)"
run () {  # gpu  cell  grad  extra-args...
  local g=$1 c=$2 k=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g timeout "$STOP" python -u qcbm_pennylane.py --filter "$c" --loss kl --kl_grad "$k" \
      --out_json "$WORK/results/pilot_${k}_${c}.json" "$@" > "$WORK/logs/pilot_${k}_${c}.log" 2>&1
  echo "== $c $k exit $?  $(date -Is)"
}
run 0 exp_10q     exact  --lr 0.005 --n_steps "$N10_STEPS" --early_stop_patience 300 &
run 1 exp_10q     legacy --lr 0.005 --n_steps "$N10_STEPS" --early_stop_patience 300 &
run 2 bimodal_20q exact  --lr 0.1 --n_steps "$N20_STEPS" --early_stop_patience 12 --skip_mmd_eval &
run 3 bimodal_20q legacy --lr 0.1 --n_steps "$N20_STEPS" --early_stop_patience 12 --skip_mmd_eval &
wait

# ---- summary: best KL, steps done, s/step (from the history wall_s field) ----
python - "$WORK"/results <<'EOF'
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + '/pilot_*.json')):
    r = json.load(open(f)); r = r[0] if isinstance(r, list) else r
    h = r.get('history') or []
    if h:   # wall time of the optimiser loop at the last logged step
        sps, how = h[-1]['wall_s'] / h[-1]['step'], f"history, step {h[-1]['step']}"
    else:   # <10 steps at n>=10 log nothing: whole run_one time / steps (slight overestimate)
        sps, how = float(r['elapsed']) / int(r['n_steps']), f"elapsed/{r['n_steps']} steps"
    print(f"{f.split('/')[-1]:34s} kl_grad={r.get('kl_grad')} best_kl={float(r['best_kl']):.5f} "
          f"partial={r.get('partial')} s/step={sps:.2f} ({how})")
print('manuscript reference (legacy, H200): exp_10q 0.4626, bimodal_20q 0.6368')
EOF

mkdir -p "$RESULTS_DIR"
cp -r "$WORK"/results "$RESULTS_DIR"/pilot_${SLURM_JOB_ID}
cp -r "$WORK"/logs "$RESULTS_DIR"/pilot_${SLURM_JOB_ID}/logs
echo "== copied to $RESULTS_DIR/pilot_${SLURM_JOB_ID}  $(date -Is)"
