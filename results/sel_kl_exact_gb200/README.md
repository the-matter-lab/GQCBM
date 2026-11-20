# QCBM-SEL+KL re-run with the exact KL gradient (GB200, 2026-11-10 to 11-15)

Raw outputs of the re-run of every manuscript QCBM-SEL+KL number with `qcbm_pennylane.py --kl_grad exact`.
The legacy `kl_grad` in `qcbm_pennylane.py` applied the two-term parameter-shift rule to the KL value itself, which is not the gradient of KL; the exact gradient is in `qcbm_kl_grad_exact.py`.
The jobs ran `scripts/run_sel_kl_exact_gb200.sh` (commit hashes in `runs/*/logs/COMMIT.txt` refer to the development repository).

- **Cluster:** a GB200 cluster, one 4-GPU node per group.
- **Software:** CUDA-Q 0.16.0 (env `scripts/selenv-gb200.freeze.txt`).
- **Fidelity:** a copy, with cluster paths, accounts and host names replaced by placeholders, of the `sel_kl_pilot/runs/` (every attempt, including preempted ones) and `sel_kl_pilot/logs/` (here `slurm/`). There are 155 files, each verified with sha1 against the cluster on 2026-11-15.
- **Untouched:** the manuscript's legacy result files in `results/` are not changed. Nothing here overwrites them.

## Which file is which manuscript cell (use these; one seed each, as in the manuscript)

| Manuscript cell | File (under `runs/`) | best KL |
|---|---|---|
| Table 1 n=10 Gauss / Bimodal / Exp / Log-normal | `t1_n10/job_8803242/results/qcbm_aligned_kl_{gauss,bimodal,exp,lognormal}_10q.json` | 0.1277 / 0.1798 / 0.0244 / 0.1885 |
| Table 1 n=15 (same order) | `t1_n15/job_8828095/results/qcbm_aligned_kl_{gauss_15q_s42,bimodal_15q,exp_15q,lognormal_15q}.json` | 0.3884 / 0.5379 / 0.8100 (early stop at 3180) / 0.6425 ¹ |
| SI Table `tab:n20_protocol`, multi-cycle column | `t1_n20/job_8803244/results/qcbm_n20m2_kl_{gauss,bimodal,exp,lognormal}_20q.json` | 0.5861 / 0.5752 / 0.9592 / 0.6215 |
| Table 2 GHZ-5 | `t2_small/job_8803245/results/qcbm_cvg_ghz5_5q.json` | −2e-10 (≈0) |
| Table 2 BAS 2×2 | `t2_small/job_8803245/results/qcbmpl_v2_kl_bas4_4q.json` | −6e-10 (≈0) |
| Table 2 D⁵₂ | `t2_small/job_8803245/results/qcbmpl_v2_kl_cardinality_5q.json` | 2.4e-8 |
| Table 2 GHZ-10 / D¹⁰₅ / BAS 3×3 | `t2_medium/job_8806630/results/qcbmpl_v3_kl_{ghz10_10q,cardinality_10q,bas9_9q}.json` | −2e-10 (≈0) / 0.6920 / 0.0115 (v2: 0.0210) |
| SI `fig:s10_budget` (Gaussian 5/10/15q × lr 0.01/0.03) | `figS4/job_8803247/results/qcbm_nc_{5,10,15}q_lr{0.01,0.03}.json` | 5q ≈0; 10q 0.154 / 0.107; 15q 0.600 / 0.600 |
| Table 1 n=20 V-QCBM+KL, and the single-trajectory column of `tab:n20_protocol` | `t1_n20_aligned/job_9015307/` | step 4000: 0.5367 / 0.5727 / 0.9222 / 0.6016 ² |

Each JSON is a one-element list holding `best_kl`, `best_params`, `history`, `partial`, `n_steps`, `starts_done`, `kl_grad`, `elapsed` and related fields.
Small negative KL values are floating-point error around 0.

¹ **n=15 was preempted five times and resumed from saved best parameters.** The JSON `history` covers only the last attempt. The full trajectory is in `runs/t1_n15/job_8828095/logs/*.log` (one `== attempt` block per attempt). Because of a step-accounting bug (fixed afterwards), Gauss, bimodal and log-normal ran ~4,920 real steps. The values above are the best KL at real step 4000, read from the logs. They differ from the JSON `best_kl` only in the 5th digit. The JSON `best_kl` is the end-of-run value.

² **The single-trajectory n=20 run ran as job 8828096 (to ~3,065 steps, 40 h budget) and then top-up job 9015307.** Both were preempted repeatedly, so the JSONs there are `partial=True` and have no `history`. A stale `.steps` sidecar reset the count at each requeue (fixed afterwards), so the cells ran 4,400–4,450 real steps. The JSON `best_kl` is the end-of-run value (0.53659 / 0.57265 / 0.92193 / 0.60149). The step-4000 values come from the logs.

Folders named `*_preempt1` and `t1_n15/job_8806629` hold earlier, superseded attempts. They are kept for the record and are not used. `.steps` / `.done` are the job script's resume markers. `slurm/` holds each job's Slurm stdout; it also contains the pilot (8801667) and the environment build (8803241).

## Reading the step-4000 value from a log

```bash
awk '/^== attempt/{cum+=s; s=0} match($0,/step +[0-9]+\//){s=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); s+=0;
     k=$0; sub(/.*best=/,"",k); sub(/ .*/,"",k); if(!hit && cum+s>=4000){hit=1; print k}}' <file>.log
```

This works because each attempt's `step N/` counter restarts at 0 and every attempt warm-starts from the previous best parameters.
