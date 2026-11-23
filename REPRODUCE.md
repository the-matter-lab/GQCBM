# Reproducing the figures and tables

One row per figure and table of *Learning Quantum Circuit Structure for Born Machines with a Generative
Transformer* (main text and Supplementary Information, as submitted). Run every command from the
repository root. Environment and training are described in `README.md`.

**Compute levels**

- **CPU**: plots from files in this repository (`results/`, `paper/figure_data/`, `gqe_experiments/`).
  Needs `numpy`, `scipy`, `matplotlib`.
- **CPU + ckpt**: re-simulates circuits stored in G-QCBM checkpoints with NumPy on CPU (needs `torch` to
  load them). Fetch the checkpoints first: `pip install huggingface_hub && python scripts/download_ckpts.py`
  (Hugging Face [`the-matter-lab/GQCBM`](https://huggingface.co/the-matter-lab/GQCBM), 52 checkpoints, 1.08 GB,
  SHA-256-checked). Each of these scripts also has a CPU path that plots the CSV it writes.
- **GPU**: needs CUDA-Q on an NVIDIA GPU. Only used to rebuild `results/circuit_stats.json` and a few
  Level-2 CSVs; no figure or table needs it, because the outputs are in the repository.
- **Train**: re-running training. Not needed for any row below; see "Training scripts".

Fonts depend on the machine, so a regenerated PDF differs from the submitted one in glyph shapes only.
Figure PDFs as submitted are in `paper/figs/`.

## Main text

| Item | Output | Command | Inputs | Compute |
|---|---|---|---|---|
| Fig. 1 `fig:pipeline` | (manuscript only) | none: drawn by hand | none | none |
| Fig. 2 `fig:circuits` | `paper/figs/fig_circuits_exp_n10.pdf` | `python make_fig_circuits.py --from-csv` (or `--ckpt-dir ckpts --case exp_n10`) | `paper/figure_data/fig_circuits_exp_n10.csv`; G-QCBM circuit from `ckpts/gqe_exp_10q_g20.pt`, V-QCBM layer count from the Table 1 n=10 V-QCBM+KL run | CPU (+ckpt) |
| Fig. 3 `fig:bimodal20_3way` | `paper/figs/fig2_n20_4dist.pdf` | `python make_fig_continuous_grid.py --from-csv`, then `python make_fig2_n20.py` | n=20 rows of `paper/figure_data/fig_continuous_grid.csv` (rebuilt by `make_fig_continuous_grid.py --ckpt-dir ckpts` from `ckpts/gqe_{gauss,exp,lognormal}_20q.pt`, `paper/figure_data/fig2_bimodal20_3way.csv`, `results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/`, `results/qcbm_n20m2_mmd_*_20q.json`) | CPU (+ckpt) |
| Fig. 4 `fig:cx_vs_kl` | `paper/figs/fig_cx_vs_kl_pairs_n20.pdf` | `python make_fig_cx_vs_kl_pairs.py` (also writes Fig. S3) | `results/circuit_stats.json`; n=20 V-QCBM+KL at step 4,000 from `results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/logs/`. Data: `paper/figure_data/fig_cx_vs_kl_pairs.csv` | CPU |
| Table 1 `tab:continuous` | in manuscript | `python make_tables.py continuous` | `results/circuit_stats.json` (G-QCBM checkpoints, V-QCBM result files); n=20 V-QCBM+KL from the run logs as in Fig. 4 | CPU |
| Table 2 `tab:discrete` | in manuscript | `python make_tables.py discrete` | `results/circuit_stats.json` | CPU |
| Table 3 `tab:kl_vs_mmd` | in manuscript | none: definitions | none | none |

`results/circuit_stats.json` is rebuilt by `python compute_circuit_stats.py` (GPU, CUDA-Q; reads every
checkpoint and result file named in its `ENTRIES`), or `--sel-only` to refresh only the V-QCBM rows on CPU.

## Supplementary Information

| Item | Output | Command | Inputs | Compute |
|---|---|---|---|---|
| Fig. S1 `fig:supp_exp10_bars` | `paper/figs/fig_continuous_grid_7in.pdf` | `python make_fig_continuous_grid.py --from-csv` (or `--ckpt-dir ckpts` to rebuild the CSV) | `paper/figure_data/fig_continuous_grid.csv`; G-QCBM+KL from the Table 1 checkpoints, V-QCBM+KL from `results/sel_kl_exact_gb200/runs/{t1_n10,t1_n15,t1_n20_aligned}/`, V-QCBM+MMD from `results/sweep_L4_mmd_*.json` and `results/qcbm_n20m2_mmd_*_20q.json` | CPU (+ckpt) |
| Fig. S2 `fig:supp_n10_circuits` | `paper/figs/fig_si_n10_circuits.pdf` | `python make_fig_si_n10_circuits.py` | n=10 rows of `paper/figure_data/fig_continuous_grid.csv` and `paper/figure_data/fig_circuits_*_n10.csv` (from `make_fig_circuits.py --ckpt-dir ckpts`: `ckpts/{gqe_q10_small_v2,gqe_bimodal_10q_g30,gqe_exp_10q_g20,gqe_lognormal_10q}.pt`) | CPU |
| Fig. S3 `fig:supp_cx_pairs_n10_n15` | `paper/figs/fig_cx_vs_kl_pairs_n10_n15.pdf` | `python make_fig_cx_vs_kl_pairs.py` | as Fig. 4 | CPU |
| Fig. S4 `fig:supp_qubit_shape` | `paper/figs/fig_si_qubit_shape.pdf` | `python make_fig_si_qubit_shape.py --from-csv` (or `--ckpt-dir ckpts`) | `paper/figure_data/fig_si_qubit_shape{,_top4,_circuit}.csv`, rebuilt from `ckpts/{gqe_q15_v4,gqe_bimodal_15q_g30,gqe_exp_15q_g30,gqe_lognormal_15q_g50}.pt` and `ckpts/gqe_{gauss,bimodal,exp,lognormal}_20q.pt`; each CPU replay is asserted to reproduce the stored `best_kl` | CPU (+ckpt) |
| Fig. S5 `fig:supp_dist_ghz10_10q` | `paper/figs/dist/ghz10_10q.pdf` | `python make_fig_ghz10.py` | `paper/figure_data/fig_s7_ghz10_dist.csv` (G-QCBM+MMD `ckpts/gqe_hamMMD_ghz10_10q.pt`, V-QCBM+MMD `results/qcbmpl_v3_mmd_ghz10_10q.json`); V-QCBM+KL re-simulated from `results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/`. Data: `paper/figure_data/fig_ghz10.csv` | CPU (torch) |
| Fig. S6 `fig:supp_dist_cardinality_10q` | `paper/figs/dist/cardinality_10q.pdf` | `python make_fig_dicke10.py` (also writes Fig. S7) | `paper/figure_data/fig_s8_cardinality10_dist.csv` (G-QCBM+KL `ckpts/gqe_sym_card_10q_full.pt`, V-QCBM+MMD); V-QCBM+KL re-simulated from `results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/` | CPU (torch) |
| Fig. S7 `fig:supp_dist_cardinality_10q_weights` | `paper/figs/dist/cardinality_10q_weights.pdf` | `python make_fig_dicke10.py` | as Fig. S6. Data: `paper/figure_data/fig_dicke10_weights.csv` | CPU (torch) |
| Fig. S8 `fig:supp_dist_bas4_4q` | `paper/figs/dist/bas2x2_merged.pdf` | `python make_fig_bas.py --ckpt-dir ckpts` (also writes Fig. S9) | G-QCBM+KL `ckpts/extras_bas4.pt`; V-QCBM+KL `results/sel_kl_exact_gb200/runs/t2_small/job_8803245/`; V-QCBM+MMD `results/archive/qcbmpl_v2_mmd_bas4_4q.json`; `paper/figure_data/fig_s6_bas4_patterns.csv`. Data: `paper/figure_data/fig_bas2x2{,_states}.csv` | CPU + ckpt |
| Fig. S9 `fig:supp_dist_bas9_9q` | `paper/figs/dist/bas3x3_merged.pdf` | `python make_fig_bas.py --ckpt-dir ckpts` | G-QCBM+KL `ckpts/gqe_bas9_v2_ent.pt`; V-QCBM+KL `results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/`; V-QCBM+MMD `results/qcbmpl_v3_mmd_bas9_9q.json`; `paper/figure_data/fig_s9_bas9_patterns.csv`. Data: `paper/figure_data/fig_bas3x3{,_states}.csv` | CPU + ckpt |
| Fig. S10 `fig:supp_cx_discrete` | `paper/figs/fig_cx_vs_kl_discrete.pdf` | `python make_cx_vs_kl.py` | `results/circuit_stats.json` | CPU |
| Fig. S11 `fig:s10_budget` | `paper/figs/fig_s10_5_budget.pdf` | `python make_supp_figures.py budget` | `results/sel_kl_exact_gb200/runs/figS4/job_8803247/results/qcbm_nc_*.json` → `paper/figure_data/fig_s10_5_qcbm_budget.csv` | CPU |
| Fig. S12 `fig:s10_random` | `paper/figs/fig_s10_3_random.pdf` | `python make_supp_figures.py random` | `paper/figure_data/fig_s10_3_random_baseline.csv` (from `results/random_baseline.json` via `export_figure_data.py`). The n=5 G-QCBM+KL reference 0.0016 is the best of five seeds of `scripts/run_multiseed_5q.sh` (job 39630), reproduced by rerun job 796452 (`ckpts/seed_5q_123_rerun.pt`, `best_kl` 0.001642; logs on Hugging Face under `logs/`) | CPU |
| Fig. S13 `fig:nsweep` | `paper/figs/fig_15q_Nsweep.pdf` | `python make_15q_Nsweep.py` | `gqe_experiments/15q_N{10,20,40,60,80,100}/log.json` | CPU |
| Fig. S14 `fig:s10_beta` | `paper/figs/fig_s10_2_beta.pdf` | `python make_supp_figures.py beta` | `paper/figure_data/fig_s10_2_beta_sweep.csv` (from the `beta_*.pt` checkpoints of `scripts/archive/run_beta_sweep.sh`) | CPU |
| Table S1 `tab:n20_protocol` | in manuscript | `python make_tables.py n20_protocol` | KL: `results/sel_kl_exact_gb200/runs/t1_n20/job_8803244/` (multi-cycle) and `t1_n20_aligned/job_9015307/logs/` (single-trajectory, step 4,000). Steps and wall-clock: `logs/summary.txt` and `logs/*.log` in the same folders | CPU |
| Table S2 `tab:abl_ngates` | in manuscript | `python make_tables.py ngates` | `paper/figs/fig_ngates_sweep.csv`, rebuilt by `python make_ngates_sweep.py` from `ckpts/gqe_gauss_{10,15,20}q_g{20,30,50}.pt` and `ckpts/gqe_gauss_20q.pt` | CPU (+ckpt) |
| Tables S3–S6 `tab:arch`, `tab:buffer`, `tab:hyperparams`, `tab:per_experiment` | in manuscript | none: configuration | defaults in `gqe_single_target.py`, `qcbm_pennylane.py`; per-run flags in `scripts/run_*.sh` | none |

## Training scripts (how each checkpoint and result file was made)

Each job script below wrote checkpoints (`ckpts/*.pt`, on Hugging Face) or result files (`results/*.json`)
that the figures and tables above read. The mapping comes from expanding every script's loops and variables
(including the `NQ`, `LOSS` and `SWEEP_*` settings each script takes) and matching their output paths. Scripts
that produced nothing the paper uses were removed.

| Script | Paper-used outputs |
|---|---|
| `scripts/archive/run_beta_sweep.sh` | `beta_0.01.pt`, `beta_0.02.pt`, `beta_0.03.pt`, `beta_0.05.pt`, `beta_0.10.pt`, `beta_0.15.pt` |
| `scripts/run_bas_ghz.sh` | `gqe_ghz5.pt` |
| `scripts/run_circuit_stats.sh` | `circuit_stats.json` |
| `scripts/run_extras.sh` | `extras_bas4.pt`, `extras_exp5.pt` |
| `scripts/run_finalize.sh` | `qcbmpl_kl_cardinality.json` |
| `scripts/run_gqcbm_mmd_symmetric_discrete.sh` | `gqe_symMMD_bas9_9q.pt`, `gqe_symMMD_card_10q.pt`, `gqe_symMMD_card_5q.pt` |
| `scripts/run_gqe_bas9_v2.sh` | `gqe_bas9_v2_ent.pt` |
| `scripts/run_gqe_hammingmmd.sh` | `gqe_hamMMD_ghz10_10q.pt` |
| `scripts/run_gqe_mmd10.sh` | `gqe_mmd10_bimodal_10q.pt`, `gqe_mmd10_exp_10q.pt`, `gqe_mmd10_gauss_10q.pt`, `gqe_mmd10_lognormal_10q.pt` |
| `scripts/run_gqe_mmd15.sh` | `gqe_mmd15_bimodal_15q.pt`, `gqe_mmd15_exp_15q.pt`, `gqe_mmd15_gauss_15q.pt`, `gqe_mmd15_lognormal_15q.pt` |
| `scripts/run_gqe_mmdloss_v2.sh` | `gqe_mmdloss2_bas4_4q.pt`, `gqe_mmdloss2_ghz5_5q.pt` |
| `scripts/run_gqe_n20_mmd_longrun.sh` | `gqe_mmd20_bimodal_20q.pt`, `gqe_mmd20_exp_20q.pt`, `gqe_mmd20_gauss_20q.pt`, `gqe_mmd20_lognormal_20q.pt` |
| `scripts/run_gqe_n20_resume.sh` | `gqe_bimodal_20q.pt`, `gqe_exp_20q.pt`, `gqe_gauss_20q.pt`, `gqe_lognormal_20q.pt` |
| `scripts/run_gqe_newdist.sh` | `gqe_lognormal_10q.pt`, `gqe_lognormal_5q.pt` |
| `scripts/run_gqe_ngates_sweep.sh` | `gqe_bimodal_10q_g30.pt`, `gqe_bimodal_15q_g30.pt`, `gqe_exp_10q_g20.pt`, `gqe_exp_15q_g30.pt`, `gqe_gauss_10q_g20.pt`, … (14 files) |
| `scripts/run_gqe_sym2.sh` | `gqe_sym_card_10q_full.pt` |
| `scripts/run_gqe_sym_2stage.sh` | `gqe_sym_card_5q.pt` |
| `scripts/run_multiseed_15q.sh` | `seed_15q_42.pt` |
| `scripts/run_multiseed_5q.sh` | `seed_5q_42.pt` (seeds 123/456/789/1000 lost; seeds 123 and 42 rerun as `seed_5q_{123,42}_rerun.pt`, HF `f764882d`) |
| `scripts/run_multiseed_6q.sh` | `seed_6q_789.pt` |
| `scripts/run_q10_small_long_v2.sh` | `gqe_q10_small_v2.pt` |
| `scripts/run_q15_v4_finer.sh` | `gqe_q15_v4.pt` |
| `scripts/run_qcbm_align_smalln.sh` | `qcbm_aligned_kl_bimodal_10q.json`, `qcbm_aligned_kl_bimodal_15q.json`, `qcbm_aligned_kl_exp_10q.json`, `qcbm_aligned_kl_exp_15q.json`, `qcbm_aligned_kl_gauss_10q.json`, … (8 files) |
| `scripts/run_qcbm_converge.sh` | `qcbmpl_v2_kl_bas4_4q.json`, `qcbmpl_v2_mmd_bas4_4q.json`, `qcbmpl_v2_mmd_bell_2q.json`, `qcbmpl_v2_mmd_exp_5q.json`, `qcbmpl_v2_mmd_ghz5_5q.json` |
| `scripts/run_qcbm_converged.sh` | `qcbm_cvg_bimodal_6q.json`, `qcbm_cvg_exp_5q.json`, `qcbm_cvg_ghz5_5q.json` |
| `scripts/run_qcbm_extend.sh` | `qcbmpl_v3_kl_cardinality_10q.json`, `qcbmpl_v3_kl_ghz10_10q.json`, `qcbmpl_v3_mmd_cardinality_10q.json`, `qcbmpl_v3_mmd_gauss_10q.json`, `qcbmpl_v3_mmd_ghz10_10q.json` |
| `scripts/run_qcbm_fairness_kl.sh` | `qcbm_fairkl_10q_s500.json`, `qcbm_fairkl_15q_s500.json`, `qcbm_fairkl_5q_s500.json` |
| `scripts/run_qcbm_layer_sweep.sh` | `sweep_L4_mmd_bimodal_10q.json`, `sweep_L4_mmd_bimodal_15q.json`, `sweep_L4_mmd_exp_10q.json`, `sweep_L4_mmd_exp_15q.json`, `sweep_L4_mmd_gauss_10q.json`, … (8 files) |
| `scripts/run_qcbm_layer_sweep_normal.sh` | `sweep_L4_mmd_bimodal_10q.json`, `sweep_L4_mmd_bimodal_15q.json`, `sweep_L4_mmd_exp_10q.json`, `sweep_L4_mmd_exp_15q.json`, `sweep_L4_mmd_gauss_10q.json`, … (8 files) |
| `scripts/run_qcbm_missing.sh` | `qcbmpl_kl_lognormal.json`, `qcbmpl_mmd_lognormal.json` |
| `scripts/run_qcbm_n20_matched_v2.sh` | `qcbm_n20m2_kl_bimodal_20q.json`, `qcbm_n20m2_kl_exp_20q.json`, `qcbm_n20m2_kl_gauss_20q.json`, `qcbm_n20m2_kl_lognormal_20q.json`, `qcbm_n20m2_mmd_bimodal_20q.json`, … (8 files) |
| `scripts/run_qcbm_n20m2_lognormal_mmd_continue.sh` | `qcbm_n20m2_mmd_lognormal_20q.json` |
| `scripts/run_qcbm_n20m2_lognormal_mmd_shortcycle.sh` | `qcbm_n20m2_mmd_lognormal_20q.json` |
| `scripts/run_qcbm_pl_pg1.sh` | `qcbmpl_gauss_5q_s42.json`, `qcbmpl_ghz5_5q.json` |
| `scripts/run_qcbm_pl_pg2.sh` | `qcbmpl_bimodal_6q_s42.json` |
| `scripts/run_qcbm_pl_pg3.sh` | `qcbmpl_exp_5q.json` |
| `scripts/run_qcbm_pl_pg4.sh` | `qcbmpl_gauss_15q_s42.json` |
| `scripts/run_random_baseline.sh` | `random_baseline.json` |
| `scripts/run_restore_bas4_kl.sh` | `extras_bas4.pt` |
| `scripts/run_qcbm_natconv.sh` | `qcbm_nc_{5,10,15}q_lr{0.01,0.03}.json` (training trajectories behind `fig:s10_budget`) |
| `scripts/archive/run_15q_samplesweep.sh` | `gqe_experiments/15q_N*/log.json` (`fig:nsweep`) |
| `scripts/archive/run_gqe_discrete_chain.sh` | none written, but it resumed three discrete checkpoints in place and overwrote their stored `config` (see `scripts/archive/README.md`) |
| `scripts/run_sel_kl_exact_gb200.sh` (+ `build_env_sel_gb200.sh`, `run_pilot_sel_kl_exact_gb200.sh`) | `results/sel_kl_exact_gb200/` (exact-gradient QCBM-SEL+KL re-run; the pilot holds the gradient self-test) |


**Producer not in this repository:** `qcbmpl_v2_kl_cardinality_5q.json`, `qcbmpl_v2_mmd_bimodal_6q_s42.json`, `qcbmpl_v2_mmd_cardinality_5q.json`, `qcbmpl_v2_mmd_gauss_5q_s42.json`, `qcbmpl_v3_kl_bas9_9q.json`, `qcbm_ghz10_10q.json`, `qcbmpl_v3_mmd_bas9_9q.json`, `qcbmpl_v3_mmd_cardinality_5q.json`. These files are
in `results/` and are read by the figures and tables, but no surviving job script writes them; they were
made by scripts removed from the development history earlier.


## Last verified

2026-11-20, from a clean export of this repository: every CPU row above, in a fresh environment
(Python 3.11, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.4, torch 2.8.0 CPU), with checkpoints from
`scripts/download_ckpts.py`. Each regenerated PDF was compared with the submitted figure, and
`make_tables.py` matches every value in Tables 1, 2, S1 and S2. GPU rows were not re-run.
