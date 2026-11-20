"""New Fig. 1: target + GQCBM+KL + QCBM-SEL+KL + QCBM-SEL+MMD overlay on the
20-qubit bimodal Gaussian -- the headline n=20 result (GQCBM+KL wins by
1-2 orders of magnitude over both QCBM-SEL baselines), now that
QCBM-SEL+MMD is trainable at n=20 via the fast MMD evaluator.

Extends make_fig_bimodal20.py (kept as-is) with a third curve using
results/qcbm_n20m2_kl_bimodal_20q.json (QCBM-SEL+KL; since 2026-11-15 replaced by the
exact-gradient 4,000-step re-run, SEL_KL_JSON below) and
results/qcbm_n20m2_mmd_bimodal_20q.json (QCBM-SEL+MMD) -- the matched-v2
chained-protocol checkpoints (genuinely identical budget between KL/MMD,
Adam-argmin-bugfixed, 2026-08-14), superseding the earlier
qcbm_aligned_kl_matched_/qcbmpl_v4_mmd_ files used before that rerun.

Outputs paper/figs/fig1_bimodal20_3way.{pdf,png}

    python make_fig1_bimodal20_3way.py              # GPU (CUDA-Q) recompute
    python make_fig1_bimodal20_3way.py --from-csv   # CPU redraw from figure_data CSV
"""
import os, sys, json, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GQCBM_CKPT = 'ckpts/gqe_bimodal_20q.pt'
# QCBM-SEL+KL: exact-gradient re-run, 4,000-step protocol (results/sel_kl_exact_gb200/README.md).
# best_params / best_kl are the end-of-run values (~4,400 real steps after preemptions).
SEL_KL_JSON = ('results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/'
               'qcbm_aligned_kl_bimodal_20q.json')
SEL_MMD_JSON = 'results/qcbm_n20m2_mmd_bimodal_20q.json'
FIG_CSV = 'paper/figure_data/fig2_bimodal20_3way.csv'
N_BINS = 512
nq = 20
bin_size = (2 ** nq) // N_BINS


def load_rec(json_path):
    rec = json.load(open(json_path))
    return rec[-1] if isinstance(rec, list) else rec


if '--from-csv' in sys.argv:
    # CPU redraw: binned series from the figure-data CSV (GQCBM / SEL+MMD columns come
    # from scripts/export_fig2_s8_gpu.py; the SEL+KL column from
    # `python scripts/export_level2_cpu.py Fig2`, KL-checked). Legend values: the
    # GQCBM checkpoint's best_kl as recorded in results/circuit_stats.json, and the
    # stored best_kl of each QCBM-SEL JSON.
    rows = list(csv.DictReader(open(FIG_CSV)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    t_b, g_b, skl_b, smmd_b = (col('target'), col('gqcbm_kl'),
                               col('qcbm_sel_kl'), col('qcbm_sel_mmd'))
    kl_gqcbm = next(d['best_kl'] for d in json.load(open('results/circuit_stats.json'))
                    if d['target'] == 'bimodal_20q' and d['method'] == 'GQCBM')
    rk, rm = load_rec(SEL_KL_JSON), load_rec(SEL_MMD_JSON)
    kl_sel_kl, kl_sel_mmd = float(rk['best_kl']), float(rm['best_kl'])
    # The 4,000-step run ended because of the step budget, not a walltime cut:
    # its JSON is marked partial only because the job was preempted and resumed.
    partial_kl, partial_mmd = False, bool(rm.get('partial', False))
else:
    import cudaq
    cudaq.set_target('nvidia', option='fp64')
    from make_all_dist_compare import gqcbm_dist
    from qcbm_pennylane import circuit_probs as sel_probs

    target, probs_gqcbm, nq, kl_gqcbm = gqcbm_dist(GQCBM_CKPT)

    def load_sel(json_path):
        rec = load_rec(json_path)
        params = np.asarray(rec['best_params'])
        probs = sel_probs(rec['n_qubits'], rec['n_layers'], params)
        return probs, float(rec['best_kl']), bool(rec.get('partial', False))

    probs_sel_kl, kl_sel_kl, _ = load_sel(SEL_KL_JSON)
    partial_kl = False  # see the --from-csv branch
    probs_sel_mmd, kl_sel_mmd, partial_mmd = load_sel(SEL_MMD_JSON)

    # Coarse-grain into 512 equal-width bins to compare apples-to-apples
    # at 2^20 = ~1M basis states. Each bin sums (2^20)/512 = 2048 raw states.
    bin_size = (2 ** nq) // N_BINS
    def coarse_grain(p):
        return p.reshape(N_BINS, bin_size).sum(axis=1)

    t_b = coarse_grain(target)
    g_b = coarse_grain(probs_gqcbm)
    skl_b = coarse_grain(probs_sel_kl)
    smmd_b = coarse_grain(probs_sel_mmd)
xb  = np.arange(N_BINS) * bin_size + bin_size / 2

mpl.rcParams.update({'font.size': 10, 'axes.linewidth': 0.9})
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.fill_between(xb, t_b, alpha=0.30, color='#444', step='mid', label='Target')
ax.step(xb, g_b, where='mid', color='#C00000', lw=1.4,
        label=fr'GQCBM+KL ($D_{{\mathrm{{KL}}}}={kl_gqcbm:.4f}$)')
dag_kl = '^{\\dagger}' if partial_kl else ''
dag_mmd = '^{\\dagger}' if partial_mmd else ''
ax.step(xb, skl_b, where='mid', color='#1F4E79', lw=1.2, alpha=0.9,
        label=fr'QCBM-SEL+KL ($D_{{\mathrm{{KL}}}}={kl_sel_kl:.3f}{dag_kl}$)')
ax.step(xb, smmd_b, where='mid', color='#E07B00', lw=1.2, alpha=0.9,
        label=fr'QCBM-SEL+MMD ($D_{{\mathrm{{KL}}}}={kl_sel_mmd:.3f}{dag_mmd}$)')
ax.set_xlabel(fr'basis-state index $x \in \{{0, \ldots, 2^{{20}}-1\}}$ '
              fr'(binned, {bin_size} states/bin)')
ax.set_ylabel(r'binned mass $\sum_{x\in\mathrm{bin}}p(x)$')
ax.set_xlim(0, 2 ** nq)
ax.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#666')
ax.set_title('20-qubit bimodal Gaussian: target vs. all three methods '
             fr'($2^{{20}} = 1{{,}}048{{,}}576$ basis states, '
             fr'binned into ${N_BINS}$ bins)')
ax.grid(True, which='major', linestyle=':', alpha=0.35)

os.makedirs('paper/figs', exist_ok=True)
out = 'paper/figs/fig1_bimodal20_3way'
fig.tight_layout()
fig.savefig(f'{out}.pdf', bbox_inches='tight')
fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
print(f'saved {out}.{{pdf,png}}  '
      f'(GQCBM+KL={kl_gqcbm:.6f}, SEL+KL={kl_sel_kl:.4f}, SEL+MMD={kl_sel_mmd:.4f})')
