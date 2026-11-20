#!/bin/bash
# Build the QCBM-SEL environment ONCE, pack it as ONE tarball on Lustre, smoke-test the imports.
# Every GPU job (run_sel_kl_exact_gb200.sh) unpacks it to the same node-local path (a venv is not
# relocatable) instead of reinstalling. Modelled on
# spectral-born-machine-for-conformer/slurm/build_env_mlp.sh.
#
#   sbatch scripts/build_env_sel_gb200.sh           (from the repository root, code in /home)
#
# Versions: exactly scripts/selenv-gb200.freeze.txt, the stack the GB200 pilot ran (job 8801667).
# The paper's pinned cuda-quantum==0.8.0 has no aarch64/Python 3.12 wheel (oldest is 0.9.0), so the
# re-run uses CUDA-Q 0.16.0; the pilot's gradient self-test reproduced the H200 numbers exactly.
# torch is only imported by gqe_single_target.py, never used by the SEL trainer: CPU build.
# A CPU node cannot test the GPU path; each GPU job runs a smoke test before training.
#SBATCH -A <your_account>
#SBATCH -p <your_partition>
#SBATCH --qos=<your_qos>
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J sel-build-env
#SBATCH -o /path/to/storage/sel_kl_pilot/logs/%x_%j.out
set -euo pipefail
L=/path/to/storage/sel_kl_pilot
S=/path/to/scratch
E=$S/selenv                       # the path every job unpacks to
FREEZE=${SLURM_SUBMIT_DIR:-$PWD}/scripts/selenv-gb200.freeze.txt
export TMPDIR=$S/tmp UV_CACHE_DIR=$S/cache/uv PIP_CACHE_DIR=$S/cache/pip
export UV_PYTHON_INSTALL_DIR=$E/python UV_NO_MODIFY_PATH=1
[ -f "$FREEZE" ] || { echo "missing $FREEZE"; exit 1; }
rm -rf "$E"; mkdir -p "$TMPDIR" "$E/bin" "$L/env"
echo "== $(date -Is) $(hostname) $(uname -m) $(nproc) cpus"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$E/bin sh
UV=$E/bin/uv
$UV venv "$E/venv" --python 3.12.13            # the system Python the pilot used
PY="$E/venv/bin/python"
$UV pip install --python "$PY" -r "$FREEZE" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$UV pip freeze --python "$PY" > "$L/env/selenv-aarch64.freeze.txt"
diff <(grep -v '^#' "$FREEZE" | sort) <(sort "$L/env/selenv-aarch64.freeze.txt") \
  && echo "freeze matches the pilot exactly" || echo "WARNING: freeze differs from the pilot (see diff)"
"$PY" -c "import cudaq, cupy, numpy, scipy, torch; print('imports ok:', cudaq.__version__.split(' (')[0], '| numpy', numpy.__version__)"
echo "== $(date -Is) packing"
tar -C "$S" -cf "$L/env/selenv-aarch64.tar.tmp" selenv && mv "$L/env/selenv-aarch64.tar.tmp" "$L/env/selenv-aarch64.tar"
ls -la "$L/env"
echo "== $(date -Is) done"
