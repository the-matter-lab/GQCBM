# scripts/archive/

Older job scripts that are no longer the recommended way to run anything. They are kept because
checkpoints or result files used in the paper were produced with them, so they record how those
results were made (training provenance).

- `run_beta_sweep.sh`: entropy-coefficient sweep (Fig. S10.2 / `fig:s10_beta`).
- `run_15q_samplesweep.sh`: 15-qubit per-step sample-budget sweep (`fig:nsweep`).
- `run_gqe_discrete_chain.sh`: resumed discrete-target runs in place with a mismatched architecture,
  which overwrote the stored `config` of three checkpoints that the paper still uses. Read those
  checkpoints' settings from the paper's tables, not from the checkpoint `config`.

Each script sets `REPO_ROOT` and the `#SBATCH` fields to placeholders. Set them for your own cluster
before submitting.
