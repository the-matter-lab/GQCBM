"""SI figures for Bars-and-Stripes: BAS 2x2 (Fig. S9) and BAS 3x3 (Fig. S10).

BAS 2x2 (paper/figs/dist/bas2x2_merged):
  (a) one valid pattern with each pixel's bit and qubit, and its basis index x = sum_k b_k 2^k;
  (b) one row per method, one column per valid pattern, probability printed above each pattern;
  (c) probability of every basis state (bars over x = 0..15), in the former S9 layout.
BAS 3x3 (paper/figs/dist/bas3x3_merged): panels (b) and (c) only, labelled (a) and (b); the
  encoding is shown once, in the 2x2 figure.

Pixel (r, c) is qubit k = r*C + c, as in gqe_single_target.build_target('bas'). Dark = 1.

Series follow Table 2: G-QCBM+KL, V-QCBM+KL (exact-gradient re-run), V-QCBM+MMD.
Data:
  per pattern            paper/figure_data/fig_s6_bas4_patterns.csv, fig_s9_bas9_patterns.csv
                         (G-QCBM+KL, V-QCBM+MMD; scripts/export_level2_cpu.py)
  every basis state      re-simulated on CPU: G-QCBM+KL from ckpts/extras_bas4.pt and
                         ckpts/gqe_bas9_v2_ent.pt (Hugging Face); V-QCBM from best_params of the
                         run JSONs in CASES (V-QCBM+KL: results/sel_kl_exact_gb200/runs/...;
                         the 2x2 CSV's QCBM-SEL+KL rows are the older run and are not used).
Every series' D_KL, from the per-pattern and from the full distribution, must match
results/circuit_stats.json, or nothing is written.

    python make_fig_bas.py --ckpt-dir ckpts
Writes paper/figure_data/fig_bas{2x2,3x3}{,_states}.csv and paper/figs/dist/bas{2x2,3x3}_merged.{pdf,png}.
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

RUNS = 'results/sel_kl_exact_gb200/runs/'
CASES = {
    '2x2': dict(R=2, target='bas4_4q', csv='paper/figure_data/fig_s6_bas4_patterns.csv',
                sel_kl=RUNS + 't2_small/job_8803245/results/qcbmpl_v2_kl_bas4_4q.json', example=5,
                ckpt='extras_bas4.pt', sel_mmd='results/archive/qcbmpl_v2_mmd_bas4_4q.json',
                # layout of the S9-style figure: figure size, axes (a), (b), (c), cell / pitch sizes
                fig=(7.1, 5.3), ax_a=[0.0, 0.45, 0.31, 0.53], ax_b=[0.33, 0.45, 0.67, 0.53],
                ax_c=[0.09, 0.07, 0.90, 0.28], cell=(0.42, 0.95, 1.25), ylim=0.19, xstep=2, bar_w=0.2, xlim=(-1, 16)),
    '3x3': dict(R=3, target='bas9_9q', csv='paper/figure_data/fig_s9_bas9_patterns.csv',
                sel_kl=RUNS + 't2_medium/job_8806630/results/qcbmpl_v3_kl_bas9_9q.json', example=146,
                ckpt='gqe_bas9_v2_ent.pt', sel_mmd='results/qcbmpl_v3_mmd_bas9_9q.json',
                # no encoding panel (shown once, in the BAS 2x2 figure); wider bars over 512 states
                fig=(7.1, 5.4), ax_a=None, ax_b=[0.0, 0.50, 1.0, 0.48], ax_c=[0.08, 0.07, 0.91, 0.33],
                cell=(0.2, 0.72, 1.0), ylim=None, xstep=64, bar_w=1.6, xlim=(-7, 518)),
}
SERIES = [('gqcbm_kl', 'GQCBM'), ('sel_kl', 'QCBM-SEL+KL'), ('sel_mmd', 'QCBM-SEL+MMD')]
ON, OFF = '#222222', 'white'


def valid_patterns(R, C):
    out = set()
    for m in range(2 ** R):  # stripes: rows uniform
        out.add(sum(1 << (r * C + c) for r in range(R) if (m >> r) & 1 for c in range(C)))
    for m in range(2 ** C):  # bars: columns uniform
        out.add(sum(1 << (r * C + c) for c in range(C) if (m >> c) & 1 for r in range(R)))
    return sorted(out)


def load(name, case):
    R = case['R']
    V = valid_patterns(R, R)
    rows = list(csv.DictReader(open(case['csv'])))
    # the export's target rows list the valid patterns of build_target('bas')
    assert sorted(int(r['pattern']) for r in rows if r['method'] == 'target' and r['pattern'] != 'leakage') == V, 'pattern rule mismatch'
    p = {}
    for key, method in SERIES:
        if key == 'sel_kl':
            continue
        d = {r['pattern']: float(r['prob']) for r in rows if r['method'] == method}
        p[key] = np.array([d[str(v)] for v in V])
    full, _ = E.sel_probs(case['sel_kl'])
    p['sel_kl'] = full[V]
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    nv = len(V)
    ok = True
    for key, method in SERIES:
        got = -np.mean(np.log(nv * p[key] + 1e-10 * nv))  # kl_of adds 1e-10 to p
        want = float(stats[(case['target'], method)]['best_kl'])
        good = math.isclose(got, want, rel_tol=1e-3, abs_tol=1e-8)
        print(f'  BAS {name} {method:13s} KL {got:.6g}  circuit_stats {want:.6g}  {"ok" if good else "MISMATCH"}')
        ok &= good
    if not ok:
        sys.exit('KL check failed; nothing written')
    out = f'paper/figure_data/fig_bas{name}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pattern_x', 'target'] + [k for k, _ in SERIES])
        for i, v in enumerate(V):
            w.writerow([v, repr(1 / nv)] + [repr(float(p[k][i])) for k, _ in SERIES])
        w.writerow(['invalid', 0] + [repr(float(1 - p[k].sum())) for k, _ in SERIES])
    print(f'wrote {out}')
    return V, p


def draw_pattern(ax, x0, y0, s, idx, R, labels=False, fs=11, qfs=6):
    from matplotlib.patches import Rectangle
    for r in range(R):
        for c in range(R):
            b = (idx >> (r * R + c)) & 1
            ax.add_patch(Rectangle((x0 + c * s, y0 - (r + 1) * s), s, s, fc=ON if b else OFF,
                                   ec='#777777', lw=0.4))
            if labels:
                col = 'white' if b else '#222222'
                ax.text(x0 + (c + .5) * s, y0 - (r + .45) * s, str(b), ha='center', va='center',
                        fontsize=fs, color=col)
                ax.text(x0 + (c + .5) * s, y0 - (r + .85) * s, f'$q_{{{r * R + c}}}$', ha='center',
                        va='center', fontsize=qfs, color=col)
    ax.add_patch(Rectangle((x0, y0 - R * s), R * s, R * s, fc='none', ec='#222222', lw=0.8))


def plot_grid(name, case, V, p, states, out=None):
    """Former Fig. S9 style: (a) one pattern with bits and qubits (2x2 only);
    (b) one row per method, one column per valid pattern, pixels in the method's colour, with the
    probability printed above (the former S9 also scaled opacity with p(x), but at 0.161-0.170
    the difference was invisible, so it is dropped)."""
    import nature_style
    nature_style.apply()
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    import gqcbm_palette as P

    R, nv = case['R'], len(V)
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    rows = [('Target', ON, np.full(nv, 1 / nv), None, None, False)]
    for k, m in SERIES:
        kl = float(stats[(case['target'], m)]['best_kl'])
        rows.append((P.LABEL[k], P.COLOR[k], p[k], kl, P.TEXT[k], k == 'gqcbm_kl'))
    mpl.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    fig = plt.figure(figsize=case['fig'])
    a = fig.add_axes(case['ax_a']) if case['ax_a'] else None
    b = fig.add_axes(case['ax_b'])
    cax = fig.add_axes(case['ax_c'])
    for ax in (a, b):
        if ax is not None:
            ax.set_aspect('equal')
            ax.axis('off')
    tags = ('(a)', '(b)', '(c)') if a is not None else (None, '(a)', '(b)')

    # (a) one pattern with bits and qubits
    ex = case['example']
    if a is not None:
        draw_pattern(a, 0.3, 3.0, 2.4 / R, ex, R, labels=True, fs=16 if R == 2 else 13, qfs=11 if R == 2 else 9)
        bits = ''.join(str((ex >> k) & 1) for k in range(R * R))
        a.text(1.5, 0.35, f'$b_0\\cdots b_{{{R * R - 1}}}$ = {bits}\n$x=\\sum_k b_k 2^k$ = {ex}',
               ha='center', va='top', fontsize=8.5)
        a.set_xlim(0, 3)
        a.set_ylim(-0.7, 3.3)
        a.set_title('(a) one valid pattern', loc='left', fontsize=8.5)

    # (b) methods x patterns
    (s, gx, gy), x_lab = case['cell'], 2.55   # cell size, column / row pitch, width of label column
    for ri, (lab, colr, q, kl, tcol, bold) in enumerate(rows):
        y0 = -ri * gy
        leak = 1 - q.sum()
        if kl is None:
            txt = lab
        else:
            kl_txt = f'= {P.sig2(kl)}' if kl > 1e-6 else '< 1e-6'
            inv_txt = f'= {leak:.1e}' if leak >= 1e-9 else '< 1e-9'
            txt = f'{lab}\n$D_{{KL}}$ {kl_txt}\nP(invalid) {inv_txt}'
        b.text(0, y0 - R * s / 2, txt, ha='left', va='center', fontsize=7.5,
               color=tcol or '#222222', fontweight='bold' if bold else 'normal')
        base = to_rgba(colr)
        for j, v in enumerate(V):
            x0 = x_lab + j * gx
            for r in range(R):
                for c in range(R):
                    on = (v >> (r * R + c)) & 1
                    b.add_patch(Rectangle((x0 + c * s, y0 - (r + 1) * s), s, s,
                                          fc=base if on else '#F5F5F5', ec='none'))
            b.add_patch(Rectangle((x0, y0 - R * s), R * s, R * s, fc='none', ec='#BBBBBB', lw=0.5))
            b.text(x0 + R * s / 2, y0 + 0.05, f'{q[j]:.3f}', ha='center', va='bottom', fontsize=6.5 if R == 2 else 6,
                   color=tcol or '#222222')
            if ri == len(rows) - 1:
                b.text(x0 + R * s / 2, y0 - R * s - 0.08, f'$x$={v}', ha='center', va='top', fontsize=6.5)
    b.set_xlim(-0.05, x_lab + nv * gx)
    b.set_ylim(-(len(rows) - 1) * gy - R * s - 0.45, 0.35)
    b.set_title(f'{tags[1]} probability on each of the {nv} valid patterns', loc='left', fontsize=8.5)
    draw_states(cax, *states, tags[2], case)
    out = out or f'paper/figs/dist/bas{name}_merged'
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}.{{pdf,png}}')


def load_full(name, ckpt_dir):
    """All 2^n basis-state probabilities (panel (c)), each series re-simulated on CPU and
    checked against results/circuit_stats.json."""
    case = CASES[name]
    t, p_g, _ = E.gqcbm_probs(os.path.join(ckpt_dir, case['ckpt']))
    full = {'gqcbm_kl': p_g,
            'sel_kl': E.sel_probs(case['sel_kl'])[0],
            'sel_mmd': E.sel_probs(case['sel_mmd'])[0]}
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    ok = True
    for key, method in SERIES:
        got, want = E.kl(t, full[key]), float(stats[(case['target'], method)]['best_kl'])
        good = math.isclose(got, want, rel_tol=1e-3, abs_tol=1e-8)
        print(f'  BAS {name} full {method:13s} KL {got:.6g}  circuit_stats {want:.6g}  {"ok" if good else "MISMATCH"}')
        ok &= good
    if not ok:
        sys.exit('KL check failed; nothing written')
    out = f'paper/figure_data/fig_bas{name}_states.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['basis_index', 'target'] + [k for k, _ in SERIES])
        for i in range(len(t)):
            w.writerow([i, repr(float(t[i]))] + [repr(float(full[k][i])) for k, _ in SERIES])
    print(f'wrote {out}')
    return t, full


def draw_states(ax, t, full, tag, case):
    """Former Fig. S9 layout: bars over every basis index x, linear y; paper colours and names."""
    import gqcbm_palette as P
    stats = {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}
    x = np.arange(len(t))
    w = case['bar_w']  # per series; 4 series side by side around each index
    ax.bar(x - 1.5 * w, t, w, color=P.TARGET_FILL, label='Target')
    for i, (k, m) in enumerate(SERIES):
        kl = float(stats[(case['target'], m)]['best_kl'])
        ax.bar(x + (i - 0.5) * w, full[k], w, color=P.COLOR[k],
               label=f'{P.LABEL[k]} ($D_{{KL}}$ ' + (f'= {P.sig2(kl)})' if kl > 1e-6 else '< 1e-6)'))
    ax.set_xlim(*case['xlim'])  # wide enough that the bar groups at both ends are not clipped
    ax.set_xticks(range(0, len(t) + 1, case['xstep']))
    ax.set_ylim(0, case['ylim'] or 1.6 * max(max(full[k].max() for k, _ in SERIES), t.max()))
    ax.set_xlabel('basis index $x$')
    ax.set_ylabel('$p(x)$')
    ax.set_title(f'{tag} probability of every basis state', loc='left', fontsize=8.5)
    # above the axes: inside it would cover the bars at x = 12 and 15 (as in the former S9)
    if case['ylim']:
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=7, frameon=False)
    else:  # narrow panel: legend inside, above the data (the y range leaves room for it)
        ax.legend(loc='upper center', ncol=2, fontsize=6.5, frameon=False)
    ax.grid(False)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='ckpts',
                    help='folder with extras_bas4.pt and gqe_bas9_v2_ent.pt (Hugging Face)')
    args = ap.parse_args()
    for name, case in CASES.items():
        V, p = load(name, case)
        plot_grid(name, case, V, p, load_full(name, args.ckpt_dir))
