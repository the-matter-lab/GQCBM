"""SI figure: the four continuous targets at n=10, one row each (Gaussian, bimodal,
exponential, log-normal). Left: target + G-QCBM+KL + V-QCBM+KL + V-QCBM+MMD,
the n=10 column of the 4x3 grid (make_fig_continuous_grid.draw_cell). Right:
the decoded G-QCBM+KL circuit behind each blue curve, as in make_fig_circuits.py
(G-QCBM panel only; the V-QCBM ansatz is in main-text Fig. 3).

Data: paper/figure_data/fig_continuous_grid.csv (n=10 rows) and
paper/figure_data/fig_circuits_{family}_n10.csv. Both are written, with KL
checks, by the two scripts above; this script only reads them. All four circuits
share one gate size, so wide and narrow circuits are drawn at the same scale.

    python make_fig_continuous_grid.py --ckpt-dir ckpts   # (re)build the grid CSV
    python make_fig_circuits.py --ckpt-dir ckpts          # (re)build circuit CSVs
    python make_fig_si_n10_circuits.py                     # plot

Writes paper/figs/fig_si_n10_circuits.{pdf,png}.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, 'figures'))

import make_fig_circuits as C  # noqa: E402  (applies nature_style, imports pyplot)
import make_fig_continuous_grid as G  # noqa: E402
import gqcbm_palette as P  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = 'paper/figs/fig_si_n10_circuits'
N = 10
CASE = {'gauss': 'gauss_n10', 'bimodal': 'bimodal_n10', 'exp': 'exp_n10',
        'lognormal': 'lognormal_n10'}


def main():
    mpl.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
    data = {}
    for r in csv.DictReader(open(G.CSV_PATH)):
        if int(r['n_qubits']) == N:
            data.setdefault(r['family'], []).append(r)
    stored = G.gqcbm_stored_kl()
    gates = {k: C.load_csv(c) for k, c in CASE.items()}
    # one gate size for all rows: the widest circuit sets the scale
    total = max(C.extent(CASE[k], g, show_v=False) for k, g in gates.items())
    wc = 6.0                                  # circuit panel width (in)
    unit = wc / total
    hr = C.height_in(N, unit, *C.vert(unit, notes_below=True, has_title=False, n_lines=2,
                                                show_v=False))
    wd, ml, gap, mt, mb, rg = 2.7, 0.75, 0.35, 0.65, 0.5, 0.25
    W = ml + wd + gap + wc + 0.05
    H = mt + 4 * hr + 3 * rg + mb
    fig = plt.figure(figsize=(W, H))
    for i, (label, key, _, _) in enumerate(G.FAMILIES):
        y_top = H - mt - i * (hr + rg)        # top of this row (in, from bottom)
        # distribution panel: below the panel title, down to the last wire
        ad = fig.add_axes([ml / W, (y_top - hr + 0.05) / H, wd / W, (hr - 0.3) / H])
        has_g, is_old = G.draw_cell(ad, key, N, data[key], stored, fontsize=6.6)
        assert has_g and not is_old, key       # every n=10 cell is its Table 1 checkpoint
        ad.set_title(f'({"abcd"[i]}) {label}', loc='left', fontsize=9.5)
        ad.set_ylabel('probability per bin')
        if i == 3:
            ad.set_xlabel(f'basis-state index $x / 2^{{{N}}}$')
        else:
            ad.tick_params(labelbottom=False)
        # circuit panel, same height as the row
        acx = fig.add_axes([(ml + wd + gap) / W, (y_top - hr) / H, total * unit / W, hr / H])
        C.draw(CASE[key], gates[key], C.v_layers(CASE[key]), None, ax=acx, unit=unit, total=total,
               titles=(None, None), show_v=False, notes_below=True)
    # two rows: "Target, G-QCBM+KL (proposed)" / "V-QCBM+KL, V-QCBM+MMD". Legend
    # columns fill top to bottom, so the handles go in column order.
    t, g, vk, vm = P.legend_handles(('gqcbm_kl', 'sel_kl', 'sel_mmd'), proposed=True)
    fig.legend(handles=[t, vk, g, vm], loc='upper left', ncol=2, frameon=False,
               bbox_to_anchor=(0.1 / W, 1.0), fontsize=9.5)
    fig.savefig(OUT + '.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(OUT + '.png', dpi=220, bbox_inches='tight', pad_inches=0.03)
    print(f'saved {OUT}.{{pdf,png}}  ({W:.2f} x {H:.2f} in, unit {unit:.3f} in/gate)')


if __name__ == '__main__':
    main()
