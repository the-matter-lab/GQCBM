"""SI figure for the GHZ-10 target (fig:supp_dist_ghz10_10q), 1x2:
(a) probability on |0>^10, on all other 1,022 states (summed), and on |1>^10, log axis;
(b) probability summed over each Hamming-weight class 0..10, log axis.

Series follow Table 2: GHZ-10 has no G-QCBM+KL entry, so the G-QCBM series is
ckpts/gqe_hamMMD_ghz10_10q.pt labelled G-QCBM+MMD (light blue, dashed). V-QCBM+KL is
the exact-gradient re-run; V-QCBM+MMD the Table 2 run.

Data:
  G-QCBM+MMD, V-QCBM+MMD  paper/figure_data/fig_s7_ghz10_dist.csv (per-state export,
                          scripts/export_level2_cpu.py, Fig. S7)
  V-QCBM+KL               re-simulated on CPU from best_params of
                          results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/
                          results/qcbmpl_v3_kl_ghz10_10q.json
Each series must reproduce its results/circuit_stats.json best_kl, or nothing is written.

    python make_fig_ghz10.py
Writes paper/figure_data/fig_ghz10.csv and paper/figs/dist/ghz10_10q.{pdf,png}.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, 'figures'), os.path.join(HERE, 'scripts')]

import numpy as np  # noqa: E402
import export_level2_cpu as E  # noqa: E402

N = 10
SRC_CSV = 'paper/figure_data/fig_s7_ghz10_dist.csv'
SEL_KL = ('results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/'
          'qcbmpl_v3_kl_ghz10_10q.json')
OUT_CSV = 'paper/figure_data/fig_ghz10.csv'
OUT = 'paper/figs/dist/ghz10_10q'
# (palette key, circuit_stats method, column)
SERIES = [('gqcbm_mmd', 'GQCBM+MMD', 'gqcbm_mmd'),
          ('sel_kl', 'QCBM-SEL+KL', 'sel_kl'),
          ('sel_mmd', 'QCBM-SEL+MMD', 'sel_mmd')]
LO = 1e-11  # lower edge of both log axes


def load():
    rows = list(csv.DictReader(open(SRC_CSV)))
    col = lambda c: np.array([float(r[c]) for r in rows])
    t = col('target')
    p_vk, _ = E.sel_probs(SEL_KL)
    p = {'gqcbm_mmd': col('gqcbm_kl'),  # column name from the export; checkpoint is Hamming-MMD
         'sel_kl': p_vk, 'sel_mmd': col('qcbm_sel_mmd')}
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    ok = True
    for key, method, _ in SERIES:
        s = stats[('ghz10_10q', method)]
        got, want = E.kl(t, p[key]), float(s['best_kl'])
        good = math.isclose(got, want, rel_tol=1e-4, abs_tol=1e-9)
        print(f'  {method:13s} KL {got:.6g}  circuit_stats {want:.6g}  ({s["source"]})  '
              f'{"ok" if good else "MISMATCH"}')
        ok &= good
    if not ok:
        sys.exit('KL check failed; nothing written')
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['basis_index', 'hamming_weight', 'target'] + [c for _, _, c in SERIES])
        for i in range(2 ** N):
            w.writerow([i, bin(i).count('1'), repr(float(t[i]))] + [repr(float(p[k][i])) for k, _, _ in SERIES])
    print(f'wrote {OUT_CSV}')
    return t, p


def plot(t, p):
    import nature_style
    nature_style.apply()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import gqcbm_palette as P

    hw = np.array([bin(i).count('1') for i in range(2 ** N)])
    mpl.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(width_ratios=[1, 1.35]))

    # (a) |0..0>, all other states, |1..1>
    x, w = np.arange(3), 0.2
    a.bar(x - 1.5 * w, [0.5, 0, 0.5], w, color=P.TARGET_FILL)
    for i, (k, _, _) in enumerate(SERIES):
        q = p[k]
        v = [q[0], max(1 - q[0] - q[-1], 0.0), q[-1]]
        xi = x + (i - 0.5) * w
        a.bar(xi, v, w, color=P.COLOR[k], edgecolor='none')
        txt = f'{v[1]:.1e}' if v[1] > 1e-9 else f'{P.LABEL[k]}: ' + r'$<10^{-9}$'
        a.text(xi[1], max(v[1], LO) * 2, txt, rotation=90, ha='center', va='bottom',
               fontsize=6.5, color=P.TEXT[k])
    a.text(1 - 1.5 * w, LO * 2, 'target: 0', rotation=90, ha='center', va='bottom',
           fontsize=6.5, color='#777777')
    a.set_yscale('log')
    a.set_ylim(LO, 3)
    a.set_xticks(x, [r'$|0\rangle^{\otimes 10}$', 'all other\n1,022 states', r'$|1\rangle^{\otimes 10}$'])
    a.set_ylabel('probability')
    a.set_title('(a) GHZ states and all other states', loc='left', fontsize=8.5)

    # (b) probability per Hamming-weight class
    wt = np.array([t[hw == j].sum() for j in range(N + 1)])
    b.bar(np.arange(N + 1), np.where(wt > 0, wt, np.nan), 0.75, color=P.TARGET_FILL)
    for k, _, _ in SERIES:
        y = np.array([p[k][hw == j].sum() for j in range(N + 1)])
        kw = P.line_kw(k)
        kw.update(marker='o', ms=3.2)
        b.plot(np.arange(N + 1), np.maximum(y, LO * 1e-3), **kw)  # below LO: drawn off-axis
    b.set_yscale('log')
    b.set_ylim(LO, 3)
    b.set_xlim(-0.6, N + 0.6)
    b.set_xticks(range(N + 1))
    b.set_xlabel('Hamming weight of $x$ (number of 1s)')
    b.set_ylabel('probability per weight class')
    b.set_title('(b) where the off-target probability goes', loc='left', fontsize=8.5)
    b.text(0.5, 0.02, r'points below $10^{-11}$ are off-axis', transform=b.transAxes,
           ha='center', va='bottom', fontsize=6.5, color='#777777')

    for ax in (a, b):
        ax.grid(False)  # figures/nature_style.py turns the grid on
    hs = [Patch(facecolor=P.TARGET_FILL, label='Target')] + [
        Patch(facecolor=P.COLOR[k], label=P.LABEL[k]) for k, _, _ in SERIES]
    fig.legend(handles=hs, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=220, bbox_inches='tight')
    print(f'saved {OUT}.{{pdf,png}}')


if __name__ == '__main__':
    plot(*load())
