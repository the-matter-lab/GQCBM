"""Fig. 2: the four continuous targets at n=20 (Gaussian, bimodal, exponential,
log-normal), target vs GQCBM+KL vs QCBM-SEL+KL vs QCBM-SEL+MMD, each coarse-
grained into 512 bins of 2,048 states. Replaces the single-panel bimodal figure
(make_fig1_bimodal20_3way.py).

Data: the n=20 rows of paper/figure_data/fig_continuous_grid.csv, written by
make_fig_continuous_grid.py (every series there is KL-checked against its stored
value; all four n=20 GQCBM+KL curves are the Table 1 checkpoints). Numbers in
each panel are the full, unbinned forward KL of each method, printed to four
decimals exactly as in Table 1; for QCBM-SEL+KL that is the best KL at step 4,000
(from the run log), while its curve uses the end-of-run parameters.

    python make_fig_continuous_grid.py --ckpt-dir ckpts   # (re)build the CSV
    python make_fig2_n20.py                               # plot

Writes paper/figs/fig2_n20_4dist.{pdf,png}. Option: --out PATH_WITHOUT_EXT.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, 'figures'))
try:
    import nature_style
    nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mpl  # noqa: E402

import gqcbm_palette as P  # noqa: E402  (figures/gqcbm_palette.py)
from make_fig_continuous_grid import (CSV_PATH, FAMILIES, N_BINS, sel_mmd_json,  # noqa: E402
                                      step4000_kl, stored_kl)

OUT = 'paper/figs/fig2_n20_4dist'
N = 20


def main():
    rows = {}
    for r in csv.DictReader(open(CSV_PATH)):
        if int(r['n_qubits']) == N:
            rows.setdefault(r['family'], []).append(r)
    gq = {d['target']: d['best_kl'] for d in json.load(open('results/circuit_stats.json'))
          if d['method'] == 'GQCBM'}
    mpl.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), sharex='col')
    bin_size = 2 ** N // N_BINS
    xb = (np.arange(N_BINS) * bin_size + bin_size / 2) / 2 ** N  # index as a fraction of 2^20
    for ax, (label, key, _, _), tag in zip(axes.flat, FAMILIES, 'abcd'):
        rs = rows[key]
        assert rs[0]['gqcbm_source'] == 'table1', key
        col = lambda k: np.array([float(r[k]) for r in rs])
        ax.fill_between(xb, col('target'), color=P.TARGET_FILL, lw=0, step='mid', zorder=1)
        ax.step(xb, col('qcbm_sel_kl'), where='mid', **P.line_kw('sel_kl'))
        ax.step(xb, col('qcbm_sel_mmd'), where='mid', **P.line_kw('sel_mmd'))
        ax.step(xb, col('gqcbm_kl'), where='mid', **P.line_kw('gqcbm_kl'))
        # D_KL block (2 significant figures). Values are the Table 1 sources:
        # GQCBM+KL results/circuit_stats.json; QCBM-SEL+KL the step-4,000 best KL from
        # the run log (its curve uses the end-of-run params, <=3e-4 lower);
        # QCBM-SEL+MMD the stored best_kl of results/qcbm_n20m2_mmd_*_20q.json.
        left, block_bottom = P.kl_block(ax, [
            ('gqcbm_kl', P.sig2(gq[f'{key}_{N}q'])),
            ('sel_kl', P.sig2(step4000_kl(key))),
            ('sel_mmd', P.sig2(stored_kl(sel_mmd_json(key, N))))])
        ax.set_title(f'({tag}) {label}', loc='left', fontsize=9.5)
        ax.set_xlim(0, 1)
        # y range: 5 % above the data, raised only as far as needed for the data
        # under the D_KL block (x >= its left edge) to stay below the block.
        series = [col(c) for c in ('target', 'gqcbm_kl', 'qcbm_sel_kl', 'qcbm_sel_mmd')]
        top = max(v.max() for v in series)
        under = max(v[xb >= left].max() for v in series)
        ax.set_ylim(0, max(1.05 * top, 1.05 * under / block_bottom))
        ax.grid(False)  # figures/nature_style.py turns the grid on by default
    for ax in axes[1]:
        ax.set_xlabel(r'basis-state index $x / 2^{20}$')
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 0]:
        ax.set_ylabel('probability per bin')
    handles = P.legend_handles(('gqcbm_kl', 'sel_kl', 'sel_mmd'), proposed=True)
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    print(f'saved {out}.{{pdf,png}}')


if __name__ == '__main__':
    main()
