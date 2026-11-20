"""SI figure for the Dicke target D^10_5 (cardinality_10q), 1x2, placed after the per-state
overlay paper/figs/dist/cardinality_10q (which is kept):
(a) probability summed over each Hamming-weight class 0..10, log axis (target: 1 at weight 5);
(b) the 252 weight-5 states sorted by model probability, as p(x) / (1/252) (target: 1).

Series follow Table 2: G-QCBM+KL (ckpts/gqe_sym_card_10q_full.pt), V-QCBM+KL (exact-gradient
re-run), V-QCBM+MMD. G-QCBM+MMD (ckpts/gqe_symMMD_card_10q.pt) is not drawn: the checkpoint is
not on Hugging Face.

Data:
  G-QCBM+KL, V-QCBM+MMD  paper/figure_data/fig_s8_cardinality10_dist.csv (per-state export,
                         scripts/export_fig2_s8_gpu.py)
  V-QCBM+KL              re-simulated on CPU from best_params of
                         results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/
                         qcbmpl_v3_kl_cardinality_10q.json (the CSV's qcbm_sel_kl column is the
                         older run, KL 0.7103, and is not used)
Each series must reproduce its results/circuit_stats.json best_kl, or nothing is written.

    python make_fig_dicke10.py
Writes paper/figure_data/fig_dicke10_weights.csv, paper/figs/dist/cardinality_10q_weights.{pdf,png}
and the per-state overlay paper/figs/dist/cardinality_10q.{pdf,png} (SI Fig. S7; no grid lines).
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

N, K = 10, 5
SRC_CSV = 'paper/figure_data/fig_s8_cardinality10_dist.csv'
SEL_KL = ('results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/'
          'qcbmpl_v3_kl_cardinality_10q.json')
OUT_CSV = 'paper/figure_data/fig_dicke10_weights.csv'
OUT = 'paper/figs/dist/cardinality_10q_weights'
SERIES = [('gqcbm_kl', 'GQCBM'), ('sel_kl', 'QCBM-SEL+KL'), ('sel_mmd', 'QCBM-SEL+MMD')]
LO = 1e-4  # lower edge of panel (a)


def load():
    rows = list(csv.DictReader(open(SRC_CSV)))
    col = lambda c: np.array([float(r[c]) for r in rows])
    t = col('target')
    p_vk, _ = E.sel_probs(SEL_KL)
    p = {'gqcbm_kl': col('gqcbm_kl'), 'sel_kl': p_vk, 'sel_mmd': col('qcbm_sel_mmd')}
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    ok = True
    for key, method in SERIES:
        s = stats[('cardinality_10q', method)]
        got, want = E.kl(t, p[key]), float(s['best_kl'])
        good = math.isclose(got, want, rel_tol=1e-4)
        print(f'  {method:13s} KL {got:.6g}  circuit_stats {want:.6g}  ({s["source"]})  '
              f'{"ok" if good else "MISMATCH"}')
        ok &= good
    if not ok:
        sys.exit('KL check failed; nothing written')
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['basis_index', 'hamming_weight', 'target'] + [k for k, _ in SERIES])
        for i in range(2 ** N):
            w.writerow([i, bin(i).count('1'), repr(float(t[i]))] + [repr(float(p[k][i])) for k, _ in SERIES])
    print(f'wrote {OUT_CSV}')
    return t, p


def plot(t, p):
    import nature_style
    nature_style.apply()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import gqcbm_palette as P

    hw = np.array([bin(i).count('1') for i in range(2 ** N)])
    valid = t > 0
    nv = int(valid.sum())  # 252
    mpl.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))

    # (a) probability per Hamming weight
    wt = np.array([t[hw == j].sum() for j in range(N + 1)])
    a.bar(np.arange(N + 1), np.where(wt > 0, wt, np.nan), 0.75, color=P.TARGET_FILL)
    for k, _ in SERIES:
        y = np.array([p[k][hw == j].sum() for j in range(N + 1)])
        kw = P.line_kw(k)
        kw.update(marker='o', ms=3.2)
        a.plot(np.arange(N + 1), np.maximum(y, LO * 1e-3), **kw)  # below LO: drawn off-axis
    a.set_yscale('log')
    a.set_ylim(LO, 2)
    a.set_xlim(-0.6, N + 0.6)
    a.set_xticks(range(N + 1))
    a.set_xlabel('Hamming weight of $x$ (number of 1s)')
    a.set_ylabel('probability per weight class')
    a.set_title('(a) probability per Hamming weight', loc='left', fontsize=8.5)

    # (b) the weight-5 states, sorted, relative to the target 1/252
    rk = np.arange(1, nv + 1)
    b.fill_between(rk, 1, step='mid', color=P.TARGET_FILL, lw=0, zorder=0)
    for k, _ in SERIES:
        b.step(rk, np.sort(p[k][valid])[::-1] * nv, where='mid', **P.line_kw(k))
    b.set_xlim(1, nv)
    b.set_ylim(0, None)
    b.set_xlabel(f'weight-{K} states, sorted by model probability')
    b.set_ylabel(f'$p(x)\\,/\\,(1/{nv})$')
    b.set_title(f'(b) uniformity over the {nv} valid states', loc='left', fontsize=8.5)
    b.text(nv - 2, 1.03, 'target = 1', ha='right', va='bottom', fontsize=6.5, color='#666666')

    for ax in (a, b):
        ax.grid(False)  # figures/nature_style.py turns the grid on
    hs = [Patch(facecolor=P.TARGET_FILL, label='Target')] + [
        Line2D([], [], label=P.LABEL[k], **P.line_kw(k, with_zorder=False)) for k, _ in SERIES]
    fig.legend(handles=hs, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + '.pdf', bbox_inches='tight')
    fig.savefig(OUT + '.png', dpi=220, bbox_inches='tight')
    print(f'saved {OUT}.{{pdf,png}}')
    for k, _ in SERIES:
        q = p[k][valid] * nv
        print(f'  {k:9s} mass on weight {K}: {p[k][valid].sum():.4f}   p/(1/{nv}) range {q.min():.3f}-{q.max():.3f}')


def plot_overlay(t, p):
    """SI Fig. S7: the per-state distribution over all 1,024 basis states (paper/figs/dist/
    cardinality_10q), redrawn without grid lines and with the paper names and palette."""
    import nature_style
    nature_style.apply()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import gqcbm_palette as P

    mpl.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    x = np.arange(2 ** N)
    w = 0.8 / 4
    ax.bar(x - 1.5 * w, t, w, color=P.TARGET_FILL, label='Target')
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    for i, (k, m) in enumerate(SERIES):
        kl = float(stats[('cardinality_10q', m)]['best_kl'])
        ax.bar(x + (i - 0.5) * w, p[k], w, color=P.COLOR[k], label=f'{P.LABEL[k]} ($D_{{KL}}$ = {P.sig2(kl)})')
    ax.set_xlim(-8, 2 ** N + 8)
    ax.set_xlabel('basis-state index $x$')
    ax.set_ylabel('$p(x)$')
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    fig.tight_layout()
    out = 'paper/figs/dist/cardinality_10q'
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}.{{pdf,png}}')


if __name__ == '__main__':
    data = load()
    plot(*data)
    plot_overlay(*data)
