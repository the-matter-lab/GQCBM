#!/bin/bash
#SBATCH --job-name=qn20m2_logn_sc
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1
#SBATCH --time=01:50:00
#SBATCH --account=<your_account>
#SBATCH --output=slurm_logs/qn20m2_logn_sc_%j.out
#SBATCH --error=slurm_logs/qn20m2_logn_sc_%j.err

# lognormal_20q QCBM-SEL+MMD kept OOM-killing at the same point every retry
# with --n_steps 20 (matched-v2 protocol), even running alone on 1 GPU.
# Root cause: qcbm_pennylane.py's circuit_probs uses cudaq.get_state()
# (1201 calls/gradient step for parameter-shift at 600 params) -- the same
# CUDA-Q host-memory-grows-with-wall-clock issue already documented for
# GQCBM at n=20 (Methods). lognormal is slower per-step than its siblings
# (~42min/start vs ~28min for gauss), so it accumulates more memory before
# finishing a start and crosses the ceiling first. Mitigation: shorter
# --n_steps per invocation (5 instead of 20) so each process finishes
# before hitting the ceiling, checkpointing to disk between cycles (same
# resume-by-out_json pattern). More cycles needed for the same total step
# budget, but each individual process survives.

REPO_ROOT="${REPO_ROOT:-/path/to/GQCBM}"
cd "$REPO_ROOT"
J=results/qcbm_n20m2_mmd_lognormal_20q.json
TARGET_STARTS=7

for i in $(seq 1 20); do
  NOW=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r.get('starts_done',0))" 2>/dev/null || echo 0)
  echo "===== attempt $i, starts_done currently $NOW (target $TARGET_STARTS) $(date) ====="
  if [ "$NOW" -ge "$TARGET_STARTS" ]; then
    echo "reached target, stopping"
    break
  fi
  python -u qcbm_pennylane.py \
    --filter "lognormal_20q" --loss mmd --lr 0.1 \
    --n_steps 5 --early_stop_patience 12 \
    --out_json ${J} \
    >> slurm_logs/qcbm_n20m2_mmd_lognormal_20q.log 2>&1
done

FINAL=$(python3 -c "import json; r=json.load(open('$J'))[0]; print(r.get('starts_done',0), r.get('partial'))" 2>/dev/null)
echo "===== done, final starts_done/partial: $FINAL ====="
