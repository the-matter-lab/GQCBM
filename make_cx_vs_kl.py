"""CX-count vs KL scatter plots (Figures 2 & 4).

Reads results/circuit_stats.json (from compute_circuit_stats.py) and
generates two scatter plots: one for continuous targets (Fig 2), one
for discrete (Fig 4). Both use the same uniform decomposition convention
documented in compute_circuit_stats.py.
"""
import os, json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 10, 'axes.linewidth': 0.9,
                     'xtick.major.width': 0.9, 'ytick.major.width': 0.9})
os.makedirs('paper/figs', exist_ok=True)

CONTINUOUS = {
    'gauss_10q', 'gauss_15q', 'gauss_20q',
    'bimodal_10q', 'bimodal_15q', 'bimodal_20q',
    'exp_10q', 'exp_15q', 'exp_20q',
    'lognormal_10q', 'lognormal_15q', 'lognormal_20q',
}
# Table 2 targets, in its row order. Bell (2-qubit GHZ) is not plotted: it was used only in the
# MMD-loss comparison (old SI S10.3), which was dropped on 2026-11-17, so the paper no longer
# defines or discusses it (GHZ is defined for n = 5 and 10 only).
DISCRETE_ORDER = ('ghz5_5q', 'ghz10_10q', 'cardinality_5q', 'cardinality_10q', 'bas4_4q', 'bas9_9q')
DISCRETE = set(DISCRETE_ORDER)
PAPER_NAME = {'ghz5_5q': 'GHZ-5', 'ghz10_10q': 'GHZ-10', 'cardinality_5q': r'Dicke $D^{5}_{2}$',
              'cardinality_10q': r'Dicke $D^{10}_{5}$', 'bas4_4q': r'BAS 2$\times$2',
              'bas9_9q': r'BAS 3$\times$3'}

import gqcbm_palette as P  # figures/gqcbm_palette.py (paper-wide palette)
_KEY = {'GQCBM': 'gqcbm_kl', 'GQCBM+MMD': 'gqcbm_mmd',
        'QCBM-SEL+KL': 'sel_kl', 'QCBM-SEL+MMD': 'sel_mmd'}
COLOR = {m: P.COLOR[k] for m, k in _KEY.items()}
# GQCBM+KL on top; the other three share one layer below it.
ZORDER = {'GQCBM': 5, 'GQCBM+MMD': 4, 'QCBM-SEL+KL': 4, 'QCBM-SEL+MMD': 4}
# Legend order: GQCBM+KL, QCBM-SEL+KL, QCBM-SEL+MMD, then GQCBM+MMD.
METHOD_ORDER = ('GQCBM', 'QCBM-SEL+KL', 'QCBM-SEL+MMD', 'GQCBM+MMD')
# Legend text from the palette (G-QCBM+KL, V-QCBM+KL, ...); data keys stay as above.
LEGEND_NAME = {m: P.LABEL[k] for m, k in _KEY.items()}
# Marker shape: encodes target family; size will encode n (set in _scatter).
MARKER = {
    'gauss_10q': 'o', 'gauss_15q': 'o', 'gauss_20q': 'o',
    'bimodal_10q': 's', 'bimodal_15q': 's', 'bimodal_20q': 's',
    'exp_10q': '^', 'exp_15q': '^', 'exp_20q': '^',
    'lognormal_10q': 'D', 'lognormal_15q': 'D', 'lognormal_20q': 'D',
    'bell_2q': 'o', 'ghz5_5q': 's', 'bas4_4q': '^',
    'ghz10_10q': 'D', 'cardinality_5q': 'v',
    'cardinality_10q': 'P', 'bas9_9q': 'X',
}
# Size by qubit count (visual cue for scaling).
N_SIZE = {5: 60, 6: 70, 9: 95, 10: 95, 15: 140, 20: 200}


def load_stats(path='results/circuit_stats.json'):
    data = json.load(open(path))
    grouped = {'continuous': [], 'discrete': []}
    for d in data:
        if d.get('best_kl') is None or d['best_kl'] != d['best_kl']:
            continue
        if d['best_kl'] <= 0:
            d['best_kl'] = 1e-10  # plotting floor
        if d['target'] in CONTINUOUS:
            grouped['continuous'].append(d)
        elif d['target'] in DISCRETE:
            # GHZ-10 has no G-QCBM+KL result (Table 2: "--"; the forward-KL retrain never
            # trained). circuit_stats.json fills that row with the Hamming-MMD checkpoint,
            # which is plotted once, as G-QCBM+MMD.
            if (d['target'], d['method']) == ('ghz10_10q', 'GQCBM'):
                continue
            grouped['discrete'].append(d)
    return grouped


def _scatter(ax, points, x_key, y_key, log_x=True, log_y=True):
    for p in points:
        m = MARKER.get(p['target'], 'o')
        c = COLOR[p['method']]
        s = N_SIZE.get(int(p['n_qubits']), 110)
        ax.scatter(p[x_key], p[y_key], marker=m, color=c, s=s,
                   edgecolor='black', lw=0.7, alpha=0.9, zorder=ZORDER[p['method']])
    if log_x:
        ax.set_xscale('log')
        # Label a few plain-number ticks instead of every minor tick (they overlapped).
        lo, hi = ax.get_xlim()
        ticks = [v for v in (5, 10, 20, 50, 100, 200, 500, 1000) if lo <= v <= hi]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if log_y: ax.set_yscale('log')
    ax.grid(False)  # paper convention: no grid (figures/nature_style.py turns it on)


def _legend(fig, group, methods):
    handles, labels = [], []
    for m, c in methods.items():
        handles.append(plt.scatter([], [], color=c, marker='o', s=80,
                                   edgecolor='black', lw=0.7))
        labels.append(LEGEND_NAME.get(m, m))
    # Family-level markers (continuous: 4 families) + size legend for n.
    if group == 'continuous':
        family_marker = [('Gaussian', 'o'), ('Bimodal', 's'),
                         ('Exponential', '^'), ('Log-normal', 'D')]
        for name, mk in family_marker:
            handles.append(plt.scatter([], [], color='lightgray', marker=mk,
                                       s=120, edgecolor='black', lw=0.7))
            labels.append(name)
        for n in (10, 15, 20):
            handles.append(plt.scatter([], [], color='lightgray', marker='o',
                                       s=N_SIZE[n], edgecolor='black', lw=0.7))
            labels.append(f'n={n}')
    else:
        for t in DISCRETE_ORDER:
            handles.append(plt.scatter([], [], color='lightgray',
                                       marker=MARKER[t], s=110,
                                       edgecolor='black', lw=0.7))
            labels.append(PAPER_NAME[t])
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=True,
               labelspacing=0.6)


def fig_cx_vs_kl(group, title, out_path):
    grouped = load_stats()
    pts = grouped[group]
    if not pts:
        print(f'[skip] no points for {group}'); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter(ax1, pts, 'cx', 'best_kl')
    ax1.set_xlabel('CX gate count')
    ax1.set_ylabel(r'best $D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    ax1.set_title('(a) CX gate count vs. KL')
    _scatter(ax2, pts, 'depth', 'best_kl')
    # depth is an analytic sequential count (compute_circuit_stats.py), not a compiler output
    ax2.set_xlabel('Compiled depth' if group == 'continuous' else 'Circuit depth')
    ax2.set_ylabel(r'best $D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    ax2.set_title('(b) Compiled depth vs. KL' if group == 'continuous' else '(b) Circuit depth vs. KL')
    methods = {m: COLOR[m] for m in METHOD_ORDER}
    _legend(fig, group, methods)
    if title:
        fig.suptitle(title, fontsize=11, y=1.00)
    plt.tight_layout(rect=[0, 0, 0.86, 1.0])
    plt.savefig(out_path + '.pdf', bbox_inches='tight')
    plt.savefig(out_path + '.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_path}.{{pdf,png}}')


if __name__ == '__main__':
    fig_cx_vs_kl('continuous',
                 'Continuous (smooth) distributions: G-QCBM vs. V-QCBM',
                 'paper/figs/fig_cx_vs_kl_continuous')
    fig_cx_vs_kl('discrete',
                 None,   # no suptitle (SI Fig. S11; caption carries it)
                 'paper/figs/fig_cx_vs_kl_discrete')
    print('Done.')
