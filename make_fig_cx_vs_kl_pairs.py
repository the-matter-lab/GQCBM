"""Paired CX-vs-KL figure for the continuous targets (redesign of main-text Fig. 4,
fig_cx_vs_kl_continuous; that figure and make_cx_vs_kl.py are left as they are).

Message: G-QCBM+KL's lower KL is not bought with extra two-qubit gates.
Per (target, n) cell, one arrow runs from the strongest V-QCBM baseline (the lower
D_KL of V-QCBM+KL and V-QCBM+MMD) to G-QCBM+KL. One panel per n.

Data (nothing is recomputed; plotting only):
  results/circuit_stats.json  (compute_circuit_stats.py): KL, CX, compiled depth,
                              V-QCBM n_layers, for every Table 1 cell.
  V-QCBM+KL n=20: Table 1 reports the best KL at real step 4,000 read from the run
                  log (make_fig_continuous_grid.step4000_kl), not the JSON end-of-run
                  value stored in circuit_stats.json.
Every KL is asserted against TABLE1 (main-text tab:continuous, 4 decimals), and every
V-QCBM CX against n * L_q, before anything is drawn.

    python make_fig_cx_vs_kl_pairs.py
Writes
  paper/figure_data/fig_cx_vs_kl_pairs.csv
  paper/figs/fig_cx_vs_kl_pairs.{pdf,png}        main figure (CX)
  paper/figs/fig_depth_vs_kl_pairs.{pdf,png}     SI: same design, compiled depth
  paper/figs/fig_cx_kl_ratio.{pdf,png}           SI: CX ratio vs KL ratio per cell
  paper/figs/fig_cx_vs_kl_pairs_n20.{pdf,png}    main-text candidate: n=20 panel only
  paper/figs/fig_cx_vs_kl_n20_ratio.{pdf,png}    main-text candidate: n=20 panel + ratio panel
  paper/figs/fig_cx_vs_kl_pairs_n10_n15.{pdf,png} SI: n=10 and n=15 panels
At n=20 the four V-QCBM markers share CX=200 and are drawn offset horizontally
(dodge(); data x unchanged) so they do not overlap and their arrows do not cross.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, 'figures')]

import numpy as np  # noqa: E402

FAMS = [('gauss', 'Gaussian'), ('bimodal', 'Bimodal'), ('exp', 'Exponential'),
        ('lognormal', 'Log-normal')]
NS = (10, 15, 20)
# main-text Table 1 (tab:continuous), manuscript as submitted:
# (family, n): (G-QCBM+KL, G-QCBM+MMD, V-QCBM+KL, V-QCBM+MMD)
TABLE1 = {
    ('gauss', 10): (0.0064, 0.0335, 0.1277, 0.0126),
    ('gauss', 15): (0.0055, 0.0511, 0.3884, 0.0121),
    ('gauss', 20): (0.0093, 0.2193, 0.5367, 0.6095),
    ('bimodal', 10): (0.0045, 0.0385, 0.1798, 0.0023),
    ('bimodal', 15): (0.0110, 0.0515, 0.5379, 0.0430),
    ('bimodal', 20): (0.0028, 0.5430, 0.5727, 0.5798),
    ('exp', 10): (0.0011, 0.0076, 0.0244, 0.0080),
    ('exp', 15): (0.0021, 0.0129, 0.8100, 0.0025),
    ('exp', 20): (0.0031, 0.8077, 0.9222, 0.9527),
    ('lognormal', 10): (0.0271, 0.0503, 0.1885, 0.0227),
    ('lognormal', 15): (0.0409, 0.3708, 0.6425, 0.0629),
    ('lognormal', 20): (0.0441, 0.1865, 0.6016, 0.6475),
}
METHODS = ('GQCBM', 'GQCBM+MMD', 'QCBM-SEL+KL', 'QCBM-SEL+MMD')  # TABLE1 column order


def load():
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    rows, bad = [], []
    for fam, _ in FAMS:
        for n in NS:
            t = f'{fam}_{n}q'
            cell = {}
            for j, m in enumerate(METHODS):
                r = dict(stats[(t, m)])
                if m == 'QCBM-SEL+KL' and n == 20:
                    import make_fig_continuous_grid as G
                    r['best_kl'] = G.step4000_kl(fam)
                if round(r['best_kl'], 4) != TABLE1[(fam, n)][j]:
                    bad.append((t, m, r['best_kl'], TABLE1[(fam, n)][j]))
                if m.startswith('QCBM-SEL') and r['cx'] != n * r['n_layers']:
                    bad.append((t, m, 'cx', r['cx'], n * r['n_layers']))
                cell[m] = r
            g, vk, vm = cell['GQCBM'], cell['QCBM-SEL+KL'], cell['QCBM-SEL+MMD']
            v = vk if vk['best_kl'] <= vm['best_kl'] else vm
            rows.append(dict(family=fam, n=n,
                             g_kl=g['best_kl'], g_cx=g['cx'], g_depth=g['depth'],
                             v_obj='KL' if v is vk else 'MMD', v_kl=v['best_kl'], v_cx=v['cx'],
                             v_depth=v['depth'], v_lq=v['n_layers'],
                             vk_kl=vk['best_kl'], vk_lq=vk['n_layers'],
                             vm_kl=vm['best_kl'], vm_lq=vm['n_layers']))
    if bad:
        for b in bad:
            print('MISMATCH', b)
        sys.exit('Table 1 check failed; nothing written')
    print('all 48 KL values match Table 1 to 4 decimals; V-QCBM CX = n * L_q in all 24 cells')
    with open('paper/figure_data/fig_cx_vs_kl_pairs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def style():
    import nature_style
    nature_style.apply()
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    import gqcbm_palette as P
    # one hue per method; dark blue vs mid orange also differ in grey (lightness ~0.30 vs ~0.50)
    return P.COLOR['gqcbm_kl'], P.COLOR['sel_kl']


LABEL_OFF = {  # direct target labels next to the G-QCBM+KL point (points offset)
    ('gauss', 10): (5, 3), ('bimodal', 10): (5, -5), ('exp', 10): (5, -2), ('lognormal', 10): (5, 3),
    ('gauss', 15): (5, -3), ('bimodal', 15): (-5, 3), ('exp', 15): (-5, -7), ('lognormal', 15): (5, 3),
    ('gauss', 20): (5, 0), ('bimodal', 20): (5, -4), ('exp', 20): (-5, 3), ('lognormal', 20): (-5, 2),
}
LABEL_HA = {k: ('right' if dx < 0 else 'left') for k, (dx, _) in LABEL_OFF.items()}


def dodge(cells, metric, step=7.0, span=0.3):
    """Horizontal offsets (points) for V-QCBM markers when all of a panel's baselines sit
    at one x within `span` decades of KL (n=20: four markers at 200 CX, KL 0.54-0.92).
    The data x is unchanged; only the drawn marker and arrow tail move. Markers are
    ordered left to right by where their arrow crosses a common mid height, so the
    arrows do not cross."""
    xs = {r[f'v_{metric}'] for r in cells}
    ky = np.log10([r['v_kl'] for r in cells])
    if len(xs) > 1 or ky.max() - ky.min() >= span:
        return {r['family']: 0.0 for r in cells}
    y_ref = np.mean(np.log10([r['v_kl'] for r in cells] + [r['g_kl'] for r in cells]))

    def x_at_ref(r):  # log-x of the arrow at log-y = y_ref
        lx0, ly0 = np.log10(r[f'v_{metric}']), np.log10(r['v_kl'])
        lx1, ly1 = np.log10(r[f'g_{metric}']), np.log10(r['g_kl'])
        return lx0 + (lx1 - lx0) * (ly0 - y_ref) / (ly0 - ly1)

    order = sorted(cells, key=x_at_ref)
    return {r['family']: (k - (len(order) - 1) / 2) * step for k, r in enumerate(order)}


def draw_panel(ax, rows, n, metric, xlabel, xlim, xticks, CG, CV, label_fs=6.5, ylo=5e-4):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import offset_copy
    names = dict(FAMS)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(ylo, 5)
    cells = [r for r in rows if r['n'] == n]
    lq = {r['v_lq'] for r in cells}   # all four best baselines share one L_q per panel
    assert len(lq) == 1, (n, lq)
    lq = lq.pop()
    vx = cells[0][f'v_{metric}']
    ax.axvline(vx, color='#999999', ls=':', lw=0.8, zorder=0)
    unit = 'CX' if metric == 'cx' else 'depth'
    ax.text(vx / 1.06, 4.2, f'V-QCBM ansatz\n($L_q$={lq}): {vx} {unit}', fontsize=6.3,
            color='#666666', ha='right', va='top')
    off = dodge(cells, metric)
    for r in cells:
        x0, y0, x1, y1 = r[f'v_{metric}'], r['v_kl'], r[f'g_{metric}'], r['g_kl']
        tr = offset_copy(ax.transData, fig=ax.figure, x=off[r['family']], y=0, units='points')
        # a pair closer than ~0.15 decades (log-normal n=10) would lose its arrow under the
        # markers: draw it straight on top, starting inside the hollow V-QCBM circle
        short = np.hypot(np.log10(x1 / x0), np.log10(y1 / y0)) < 0.15
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), textcoords=tr, zorder=5 if short else 2,
                    arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8,
                                    shrinkA=0 if short else 3, shrinkB=3.2 if short else 3,
                                    mutation_scale=6 if short else 7))
        filled = r['v_obj'] == 'KL'
        ax.plot(x0, y0, 'o', ms=5.5, mec=CV, mew=1.3, mfc=CV if filled else 'white', zorder=3,
                transform=tr)
        ax.plot(x1, y1, 's', ms=5.5, mec=CG, mew=1.0, mfc=CG, zorder=4)
        dx, dy = LABEL_OFF[(r['family'], n)]
        ax.annotate(names[r['family']], (x1, y1), xytext=(dx, dy), textcoords='offset points',
                    fontsize=label_fs, ha=LABEL_HA[(r['family'], n)], va='center', color='#333333')
    ax.set_xticks(xticks, [str(v) for v in xticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(f'$n$ = {n}', fontsize=9)
    ax.set_xlabel(xlabel)
    ax.grid(False)


def legend_handles(CG, CV, objs=('KL', 'MMD')):
    """objs: the V-QCBM objectives that win a cell in the panels shown. With both, the
    entries read 'best V-QCBM: +KL won' / '+MMD won'; with one (n=20: all KL; n=10/15:
    all MMD), the single entry is just that method's name."""
    from matplotlib.lines import Line2D
    both = len(set(objs)) > 1
    hs = [Line2D([], [], ls='', marker='s', ms=5.5, mfc=CG, mec=CG, label='G-QCBM+KL')]
    if 'KL' in objs:
        hs.append(Line2D([], [], ls='', marker='o', ms=5.5, mfc=CV, mec=CV, mew=1.3,
                         label='best V-QCBM: +KL won' if both else 'V-QCBM+KL'))
    if 'MMD' in objs:
        hs.append(Line2D([], [], ls='', marker='o', ms=5.5, mfc='white', mec=CV, mew=1.3,
                         label='best V-QCBM: +MMD won' if both else 'V-QCBM+MMD'))
    hs.append(Line2D([], [], color='#555555', lw=0.8, marker='>', ms=4, markevery=[1],
                     label=r'same target: V-QCBM $\rightarrow$ G-QCBM+KL'))
    return hs


BETTER = r'$\swarrow$ better' '\n(fewer CX, lower KL)'


def sig2int(v):
    """2 significant figures, as an integer for v >= 10 (297.7 -> 300, 13.65 -> 14)."""
    import math
    r = round(v, -int(math.floor(math.log10(v))) + 1)
    return f'{int(r)}' if r >= 10 else f'{r:g}'


def save(fig, out):
    import matplotlib.pyplot as plt
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}.{{pdf,png}}')


def pairs_figure(rows, metric, out, xlabel, xlim, xticks, ns=NS):
    """One panel per n in `ns`, shared y."""
    import matplotlib.pyplot as plt
    CG, CV = style()
    w = {3: 7.1, 2: 4.9, 1: 3.4}[len(ns)]
    fig, axes = plt.subplots(1, len(ns), figsize=(w, 2.75), sharey=True, squeeze=False)
    axes = axes[0]
    ylo = 2.5e-4 if 20 not in ns else 5e-4   # SI n=10/15: room for the 'better' line
    for ax, n in zip(axes, ns):
        draw_panel(ax, rows, n, metric, xlabel, xlim, xticks, CG, CV, ylo=ylo)
    axes[0].set_ylabel(r'$D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    if ns[-1] == 20:   # empty lower-left corner of the n=20 panel
        axes[-1].text(0.04, 0.04, BETTER, transform=axes[-1].transAxes, fontsize=6.5,
                      color='#444444', ha='left', va='bottom')
    else:              # n=10/15 (SI): one line under the n=10 panel's lowest point
        axes[0].text(0.03, 0.03, r'$\swarrow$ better (fewer CX, lower KL)',
                     transform=axes[0].transAxes, fontsize=6.5, color='#444444',
                     ha='left', va='bottom')
    if ns == (20,):    # headline numbers, computed from the plotted cells
        c = [r for r in rows if r['n'] == 20]
        cx = [r['v_cx'] / r['g_cx'] for r in c]
        kl = [r['v_kl'] / r['g_kl'] for r in c]
        axes[0].text(0.04, 0.96, f'{min(cx):.1f}\u2013{max(cx):.1f}\u00d7 fewer CX,\n'
                     f'{sig2int(min(kl))}\u2013{sig2int(max(kl))}\u00d7 lower KL',
                     transform=axes[0].transAxes, fontsize=7.5, color='#222222',
                     ha='left', va='top')
    ncol = 4 if len(ns) == 3 else 2
    objs = sorted({r['v_obj'] for r in rows if r['n'] in ns})
    fig.legend(handles=legend_handles(CG, CV, objs), loc='upper center', ncol=ncol, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.07 if ncol == 4 else 1.13),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, out)


def n20_ratio_figure(rows, out):
    """Main-text candidate: (a) the n=20 pairs; (b) CX ratio vs KL ratio for all 12 cells."""
    import matplotlib.pyplot as plt
    CG, CV = style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    draw_panel(a, rows, 20, 'cx', 'CX gate count', (10, 320), [10, 20, 50, 100, 200], CG, CV,
               label_fs=7)
    a.set_title('')
    a.set_title('(a) $n$ = 20', loc='left', fontsize=9)
    a.set_ylabel(r'$D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    a.text(0.04, 0.04, BETTER, transform=a.transAxes, fontsize=6.5, color='#444444',
           ha='left', va='bottom')
    draw_ratio(b, rows)
    b.set_title('(b) all 12 cells, $n$ = 10, 15, 20', loc='left', fontsize=9)
    fig.legend(handles=legend_handles(CG, CV, ('KL',)), loc='upper center', ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.07), handletextpad=0.4,
               columnspacing=1.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, out)


def draw_ratio(ax, rows):
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    cn = {10: '#9ECAE1', 15: '#4292C6', 20: '#08306B'}  # sequential: lighter = smaller n
    mk = {'gauss': 'o', 'bimodal': 's', 'exp': '^', 'lognormal': 'D'}
    ax.axvline(1, color='#888888', ls='--', lw=0.8, zorder=0)
    ax.axhline(1, color='#888888', ls='--', lw=0.8, zorder=0)
    for r in rows:
        ax.plot(r['g_cx'] / r['v_cx'], r['g_kl'] / r['v_kl'], mk[r['family']], ms=6,
                mfc=cn[r['n']], mec='black', mew=0.5, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.2, 3)
    ax.set_ylim(1e-3, 5)
    ax.set_xticks([0.25, 0.5, 1, 2], ['0.25', '0.5', '1', '2'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.text(0.04, 0.04, 'cheaper and better', transform=ax.transAxes, fontsize=7.5,
            color='#333333', ha='left', va='bottom')
    ax.set_xlabel('CX(G-QCBM+KL) / CX(best V-QCBM)')
    ax.set_ylabel(r'$D_{\mathrm{KL}}$(G-QCBM+KL) / $D_{\mathrm{KL}}$(best V-QCBM)')
    ax.grid(False)
    hs = [Line2D([], [], ls='', marker='o', ms=6, mfc=cn[n], mec='black', mew=0.5,
                 label=f'$n$ = {n}') for n in NS]
    hs += [Line2D([], [], ls='', label='')]  # legend fills by column: n | family
    hs += [Line2D([], [], ls='', marker=mk[f], ms=5, mfc='#DDDDDD', mec='black', mew=0.5,
                  label=lab) for f, lab in FAMS]
    ax.legend(handles=hs, fontsize=6.5, frameon=False, loc='lower right', ncol=2,
              handletextpad=0.3, columnspacing=0.8)


def ratio_figure(rows, out):
    import matplotlib.pyplot as plt
    style()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    draw_ratio(ax, rows)
    fig.tight_layout()
    save(fig, out)


if __name__ == '__main__':
    rows = load()
    pairs_figure(rows, 'cx', 'paper/figs/fig_cx_vs_kl_pairs', 'CX gate count',
                 (10, 320), [10, 20, 50, 100, 200])
    pairs_figure(rows, 'depth', 'paper/figs/fig_depth_vs_kl_pairs', 'compiled depth',
                 (25, 380), [30, 50, 100, 200])
    ratio_figure(rows, 'paper/figs/fig_cx_kl_ratio')
    # main-text candidates: n=20 alone, or n=20 + the all-cell ratio panel; n=10/15 for the SI
    pairs_figure(rows, 'cx', 'paper/figs/fig_cx_vs_kl_pairs_n20', 'CX gate count',
                 (10, 320), [10, 20, 50, 100, 200], ns=(20,))
    n20_ratio_figure(rows, 'paper/figs/fig_cx_vs_kl_n20_ratio')
    pairs_figure(rows, 'cx', 'paper/figs/fig_cx_vs_kl_pairs_n10_n15', 'CX gate count',
                 (10, 320), [10, 20, 50, 100, 200], ns=(10, 15))
    for r in rows:
        up, right = r['g_kl'] > r['v_kl'], r['g_cx'] > r['v_cx']
        if up or right:
            print(f"  {r['family']:9s} n={r['n']}: G {r['g_kl']:.4f} @ {r['g_cx']} CX vs best V "
                  f"(+{r['v_obj']}) {r['v_kl']:.4f} @ {r['v_cx']} CX  ->"
                  f"{' KL worse' if up else ''}{' more CX' if right else ''}")
