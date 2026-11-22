# G-QCBM: learning quantum circuit structure for Born machines with a generative transformer

Code, results and figure data for

> Y. Nakamura, J.-Y. Chang, *et al.*, *Learning Quantum Circuit Structure for Born Machines with a
> Generative Transformer*, submitted to *Digital Discovery* (2026).

A quantum circuit Born machine (QCBM) is a quantum circuit whose measurement outcomes follow a target
probability distribution. **G-QCBM** adapts the generative quantum eigensolver (GQE) from energy
minimisation to distribution matching. A transformer writes a circuit as a sequence of
`(Pauli operator, rotation angle)` tokens, so the circuit *structure* is learned as well as its angles.
It is compared with the conventional variational QCBM (**V-QCBM**, a fixed layered ansatz trained by
gradient descent), each trained on either the Kullback–Leibler divergence (KL) or the maximum mean
discrepancy (MMD).

## Quick start: regenerate the figures and tables

```bash
git clone https://github.com/the-matter-lab/GQCBM && cd GQCBM
python -m venv .venv && source .venv/bin/activate
pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4      # enough for every CPU row
python make_tables.py                                          # Tables 1, 2, S1, S2
python make_fig2_n20.py                                        # Fig. 3
python make_fig_cx_vs_kl_pairs.py                              # Fig. 4 and Fig. S3
```

`REPRODUCE.md` is the full manifest: one row per figure and table, with the command, the files it
reads, and whether it needs checkpoints or a GPU. No row needs retraining. Rows marked *ckpt* re-simulate
circuits from the trained G-QCBM checkpoints on CPU; fetch those first:

```bash
pip install torch==2.8.0 huggingface_hub
python scripts/download_ckpts.py        # into ckpts/, from huggingface.co/the-matter-lab/GQCBM
```

## Repository layout

```
gqe_single_target.py        G-QCBM training (the paper's method); --loss_metric {kl,mmd}
seed_buffer.py              replay buffer used by gqe_single_target.py
qcbm_pennylane.py           V-QCBM baseline (layered ansatz); --loss {kl,mmd}, --kl_grad exact
qcbm_kl_grad_exact.py       exact KL gradient for V-QCBM, with a finite-difference self-test
fast_mmd.py                 exact O(N log N) MMD^2 evaluator (index / Hamming / grid kernels)
test_fast_mmd.py            checks fast_mmd.py against the dense O(N^2) reference (NumPy only)
circuit_ops.py, pool_select.py   operator pools and CUDA-Q state kernel shared by all scripts
random_baseline.py          no-learning control: random circuits from the G-QCBM vocabulary
compute_circuit_stats.py    KL and compiled CX count of every model -> results/circuit_stats.json
export_figure_data.py       Level-2 CSVs for Figs. S12 and S14
make_*.py                   one script per figure (see REPRODUCE.md); make_tables.py for the tables
figures/                    shared matplotlib style and colour palette
scripts/                    SLURM job scripts that trained each checkpoint and result file,
                            Level-2 exporters (export_level2_cpu.py, export_fig2_s8_gpu.py),
                            analysis scripts behind numbers quoted in the Discussion,
                            and download_ckpts.py
results/                    V-QCBM result files, circuit_stats.json, analysis outputs
results/sel_kl_exact_gb200/ raw outputs of the exact-gradient V-QCBM+KL runs (cell map in its README)
gqe_experiments/            training logs read by Fig. S13
paper/gqcbm_preprint.pdf    preprint of the paper (main text + Supplementary Information)
paper/figs/                 figure PDFs as submitted
paper/figure_data/          CSVs behind each figure
```

Some `make_*.py` modules that no longer draw a paper figure (`make_all_dist_compare.py`,
`make_bas_figures.py`, `make_fig1_bimodal20_3way.py`, `make_fig_exp10_bars.py`) are kept because the
Level-2 exporters import functions from them.

In the code, `GQCBM` / `gqe` denotes G-QCBM and `QCBM-SEL` / `qcbm` denotes V-QCBM.

## Environment

Every reported training run used Python 3.9.21 on a single NVIDIA H200 GPU, except the exact-gradient
V-QCBM+KL runs, which used CUDA-Q 0.16 on a GB200 cluster (`scripts/selenv-gb200.freeze.txt`).
`requirements.txt` pins the versions. `cuda-quantum` provides the `cudaq` module, and `cupy-cuda12x`
assumes a CUDA 12 driver. All circuit evaluation is exact state-vector simulation
(`cudaq.set_target('nvidia', option='fp64')`, falling back to `qpp-cpu` without a GPU).
`python verify_env.py` checks a CUDA-Q installation.

## Training

Training is not needed to reproduce any figure or table. Each job script in `scripts/` records how one
group of checkpoints or result files was made; the "Training scripts" section of `REPRODUCE.md` maps
every script to its outputs. Each script sets `REPO_ROOT` and its `#SBATCH` account and partition to
placeholders, so set them for your cluster before submitting:

```bash
export REPO_ROOT=$PWD
sbatch --account=<your_account> scripts/run_gqe_n20_resume.sh
```

The basic entry points are

```bash
python -u gqe_single_target.py --target_kind <kind> --n_qubits <n> --loss_metric kl \
    --save_ckpt ckpts/<name>.pt --log_dir gqe_experiments/<name>
python -u qcbm_pennylane.py --filter <target> --loss kl --kl_grad exact --out_json results/<name>.json
```

## Notes on the stored results

- **MMD-trained checkpoints:** the `best_kl` field of a checkpoint trained with `loss_metric == 'mmd'`
  is the lowest KL seen among all sampled circuits, not the KL of its saved best-MMD circuit.
  `compute_circuit_stats.py` rebuilds that circuit and recomputes its KL; do the same if you read these
  checkpoints directly.
- **Three checkpoints' stored `config` is incomplete:** `gqe_bas9_v2_ent.pt`, `gqe_ghz5.pt` and
  `gqe_hamMMD_ghz10_10q.pt` were resumed in several stages, and the stored `config` records only the last
  stage (see `scripts/archive/README.md`). Take their settings from the paper's tables.
- **V-QCBM+KL uses the exact KL gradient.** An earlier `kl_grad` in `qcbm_pennylane.py` (kept, but not
  the default for any paper run) applied the parameter-shift rule to the KL value, which is not the
  gradient of KL. Every V-QCBM+KL number in the paper comes from `--kl_grad exact`
  (`results/sel_kl_exact_gb200/`).
- **Multi-cycle training at n = 20:** some n = 20 runs were trained as chains of short jobs, each resuming
  from the previous best parameters (Methods, "multi-cycle training"). This is not claimed to be
  equivalent to one uninterrupted run.

## Citation

See `CITATION.cff`. Please cite the paper once it is published; the DOI will be added here.

## License

Code: MIT License (`LICENSE`). Checkpoints on Hugging Face: see the model card there.
