#!/bin/bash
#SBATCH --job-name=qn20m2_logn_mmd
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=01:50:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qn20m2_logn_mmd_%j.out
#SBATCH --error=slurm_logs/qn20m2_logn_mmd_%j.err

# Continues results/qcbm_n20m2_mmd_lognormal_20q.json (currently
# starts_done=2, partial=True) to parity with its siblings
# (gauss/bimodal/exp reached 6-7 starts). This target is the slowest
# per-cycle (~40min/start, 3 MMD bandwidths vs 2 for the others) and was
# repeatedly OOM-killed mid-cycle by the shared 4-GPU parallel run
# (2026-08-14); running it alone on 1 GPU here to reduce host-memory
# pressure per attempt. CYCLE loop resumes from the existing JSON each
# time (qcbm_pennylane.py's own resume-by-out_json logic), one start per
# invocation, same lr=0.1/20-steps/patience=12 protocol as the rest of
# matched-v2.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
J=results/qcbm_n20m2_mmd_lognormal_20q.json
TARGET_STARTS=7

for i in $(seq 1 6); do
  NOW=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r.get('starts_done',0))" 2>/dev/null || echo 0)
  echo "===== attempt $i, starts_done currently $NOW (target $TARGET_STARTS) $(date) ====="
  if [ "$NOW" -ge "$TARGET_STARTS" ]; then
    echo "reached target, stopping"
    break
  fi
  python -u qcbm_pennylane.py \
    --filter "lognormal_20q" --loss mmd --lr 0.1 \
    --n_steps 20 --early_stop_patience 12 \
    --out_json ${J} \
    >> slurm_logs/qcbm_n20m2_mmd_lognormal_20q.log 2>&1
done

FINAL=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r.get('starts_done',0), r.get('partial'))" 2>/dev/null)
echo "===== done, final starts_done/partial: $FINAL ====="
