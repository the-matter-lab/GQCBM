"""SI figure (DRAFT): which qubits carry the shape of the learned G-QCBM+KL distributions.

Explains the step-like G-QCBM+KL curves of main Fig. 3 (fig2_n20_4dist): at n = 15 and
n = 20 the learned distribution is (to numerical precision) uniform over the least
significant qubits, so p(x) is constant over blocks of 2^m consecutive basis states and
only the top k = n - m qubits carry the shape.

Measure (exact, additive): per-qubit conditional entropy deficit
    delta_q = 1 - H(b_q | b_{q+1}, ..., b_{n-1})        [bits]
where b_q is bit q of the basis index x (CUDA-Q little-endian: q_0 = LSB, q_{n-1} = MSB)
and the lower bits are marginalised. By the chain rule
    sum_q delta_q = n - H(p) = D_KL(p || uniform) / ln 2,
so the bars split the total departure from uniform over the qubits. delta_q = 0 exactly
when bit q is a fair coin given every value of the more significant bits; p(x) is
constant over blocks of 2^m states iff delta_0 = ... = delta_{m-1} = 0. k is the number
of most significant qubits above the highest such m (threshold TOL below).

Panels: (a) the n = 20 exponential circuit (Fig. 2b style, make_fig_circuits.draw);
(b) delta_q vs q for the four continuous targets at n = 15 and n = 20;
(c) n = 20 exponential: target vs learned marginal of q16..q19 (16 blocks of 2^16 states).

Circuits: Table 1 G-QCBM+KL checkpoints (Hugging Face the-matter-lab/GQCBM,
fetched by scripts/download_ckpts.py). Each circuit is replayed on CPU
(make_fig_circuits.replay) and asserted to reproduce the stored best_kl.

    python make_fig_si_qubit_shape.py --ckpt-dir ckpts   # decode, verify, write CSVs, plot
    python make_fig_si_qubit_shape.py --from-csv         # plot from paper/figure_data
Writes paper/figure_data/fig_si_qubit_shape{,_top4,_circuit}.csv and
paper/figs/fig_si_qubit_shape.{pdf,png}.
"""
import csv
import json
import math
import os
import sys

import numpy as np

import make_fig_circuits as C          # chdir to repo root, nature_style, palette, draw()
import gqcbm_palette as P
import matplotlib.pyplot as plt

CELLS = [  # (n, family, checkpoint)
    (15, 'Gaussian', 'gqe_q15_v4'), (15, 'Bimodal', 'gqe_bimodal_15q_g30'),
    (15, 'Exponential', 'gqe_exp_15q_g30'), (15, 'Log-normal', 'gqe_lognormal_15q_g50'),
    (20, 'Gaussian', 'gqe_gauss_20q'), (20, 'Bimodal', 'gqe_bimodal_20q'),
    (20, 'Exponential', 'gqe_exp_20q'), (20, 'Log-normal', 'gqe_lognormal_20q'),
]
TOL = 1e-12         # bits; delta_q above this counts as non-uniform (replay noise is ~1e-15)
CSV_D = 'paper/figure_data/fig_si_qubit_shape.csv'
CSV_T = 'paper/figure_data/fig_si_qubit_shape_top4.csv'
CSV_C = 'paper/figure_data/fig_si_qubit_shape_circuit.csv'
OUT = 'paper/figs/fig_si_qubit_shape'
CASE = 'exp_n20'
C.CASES[CASE] = dict(n=20, family='Exponential', ckpt='gqe_exp_20q', vjson=None)


def deficits(p, n):
    """delta_q = 1 - H(b_q | b_{>q}) in bits, q = 0 (LSB) .. n-1 (MSB)."""
    out = []
    for q in range(n):
        P2 = p.reshape(2 ** (n - q - 1), 2, 2 ** q).sum(2)      # [higher bits, b_q]
        ph = P2.sum(1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(P2 > 0, P2 * np.log2(P2 / ph), 0.0)
        out.append(1.0 + float(t.sum()))
    return out


def target_of(d, cfg, n):
    if 'target_arr' in d:
        t = np.asarray(d['target_arr'], float)
    else:
        from make_fig_continuous_grid import build_target
        t = np.asarray(build_target(cfg['target_kind'], 2 ** n,
                                    json.loads(cfg.get('target_kw') or '{}')), float)
    return t / t.sum()


def build_csv(ckpt_dir):
    import torch
    rows, excluded, top4, circ = [], [], None, None
    for n, fam, ck in CELLS:
        d = torch.load(os.path.join(ckpt_dir, ck + '.pt'), map_location='cpu', weights_only=False)
        cfg = d['config'] if isinstance(d['config'], dict) else vars(d['config'])
        assert cfg['n_qubits'] == n and not any(cfg.get(k) for k in C.SPECIAL_POOL_FLAGS), ck
        pool = C.default_pool(n)
        gates = [(pool[int(i)][0], pool[int(i)][1], float(a))
                 for i, a in zip(d['best_ops'].tolist(), d['best_theta'].tolist()) if abs(a) > 1e-8]
        tgt = target_of(d, cfg, n)
        p = C.replay(n, gates)
        m = tgt > 0
        kl = float(np.sum(tgt[m] * np.log(tgt[m] / (p[m] + 1e-10))))
        stored = float(d['best_kl'])
        if not math.isclose(kl, stored, rel_tol=1e-6):
            excluded.append((n, fam, ck, kl, stored))
            print(f'EXCLUDED n={n} {fam} {ck}: replay KL {kl:.6g} != stored {stored:.6g}')
            continue
        dq = deficits(p, n)
        tot = sum(dq)
        assert math.isclose(tot, n + float(np.sum(np.where(p > 0, p * np.log2(p), 0))), abs_tol=1e-9)
        print(f'n={n} {fam:11s} {ck:22s} KL {kl:.6g} = stored; sum delta {tot:.4f} bits')
        dt = deficits(tgt, n)       # same measure on the ground-truth target
        for q, (v, vt) in enumerate(zip(dq, dt)):
            rows.append(dict(n=n, target=fam, ckpt=ck, stored_best_kl=repr(stored),
                             replay_kl=repr(kl), q=q, delta_bits=repr(v),
                             delta_target_bits=repr(vt)))
        if (n, fam) == (20, 'Exponential'):
            top4 = [(h, float(tgt.reshape(16, -1)[h].sum()), float(p.reshape(16, -1)[h].sum()))
                    for h in range(16)]
            circ = gates
    os.makedirs('paper/figure_data', exist_ok=True)
    with open(CSV_D, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(CSV_T, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['block_q19_to_q16', 'target', 'model'])
        w.writerows([(h, repr(a), repr(b)) for h, a, b in top4])
    with open(CSV_C, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['step', 'pauli', 'qubits', 'theta', 'source_ckpt'])
        for s, (qs, ps, th) in enumerate(circ):
            w.writerow([s, ps, ' '.join(map(str, qs)), repr(th), 'gqe_exp_20q'])
    return excluded


def load():
    D = {}
    for r in csv.DictReader(open(CSV_D)):
        D.setdefault((int(r['n']), r['target']), []).append(float(r['delta_bits']))
        D.setdefault((int(r['n']), r['target'], 'target'), []).append(float(r['delta_target_bits']))
    top4 = [(int(r['block_q19_to_q16']), float(r['target']), float(r['model']))
            for r in csv.DictReader(open(CSV_T))]
    gates = [(tuple(int(q) for q in r['qubits'].split()), r['pauli'], float(r['theta']))
             for r in csv.DictReader(open(CSV_C))]
    return D, top4, gates


def k_of(dq, lo=0):
    """Number of most significant qubits that carry shape: n minus the length of the run of
    qubits q = lo, lo+1, ... with delta_q <= TOL (lo = 0: full state resolution; lo = n - 9:
    the 512-bin resolution of Fig. 3, where bits below q_{n-9} are summed inside a bin)."""
    m = lo
    while m < len(dq) and dq[m] <= TOL:
        m += 1
    return len(dq) - m


FAMS = ('Gaussian', 'Bimodal', 'Exponential', 'Log-normal')
VMIN = 1e-12        # colour scale lower bound (bits) = TOL


CMAP_COLORS = {'target': ['#F2F2F2', '#9A9A9A', '#2B2B2B'],
               'model': ['#E8EEF7', P.COLOR['gqcbm_kl'], '#0B1F45']}


def cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    out = {}
    for key, cols in CMAP_COLORS.items():
        cm = LinearSegmentedColormap.from_list(key, cols)
        cm.set_bad('white')
        out[key] = cm
    return out


def heat(ax, D, n, tag):
    """Two rows per target: ground truth (grey) above the learned G-QCBM+KL output (blue)."""
    from matplotlib.colors import LogNorm
    cm = cmaps()
    norm = LogNorm(VMIN, 1)
    fams = [f for f in FAMS if (n, f) in D]
    rows = [(f, kind) for f in fams for kind in ('target', 'model')]
    ims = {}
    for kind in ('target', 'model'):
        Z = np.full((len(rows), n), np.nan)
        for i, (f, k) in enumerate(rows):
            if k == kind:
                dq = D[(n, f, 'target')] if kind == 'target' else D[(n, f)]
                Z[i] = [np.clip(v, VMIN, 1) if v > TOL else np.nan for v in dq]
        Zm = np.ma.masked_invalid(Z)
        if kind == 'model':           # white only where the target row is not drawn
            Zm = np.ma.masked_where([[k != 'model'] * n for _, k in rows], Z)
        else:
            Zm = np.ma.masked_where([[k != 'target'] * n for _, k in rows], Z)
        cmk = cm[kind].copy()
        cmk.set_bad((0, 0, 0, 0))
        ax.imshow(np.where(np.isnan(Z), np.nan, Z), cmap=cmk, norm=norm, aspect='auto',
                  interpolation='nearest', extent=(-0.5, n - 0.5, len(rows) - 0.5, -0.5),
                  alpha=np.array([[1.0 if k == kind else 0.0] * n for _, k in rows]))
        ims[kind] = plt.cm.ScalarMappable(norm=norm, cmap=cm[kind])
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color='white', lw=2.4 if i % 2 == 0 else 0.5)
    for q in range(n + 1):
        ax.axvline(q - 0.5, color='white', lw=0.4)
    xb = n - 9 - 0.5
    ax.axvline(xb, color='#555555', lw=0.8, ls=(0, (3, 2)))
    ax.text(xb + 0.15, -0.62, 'resolved at 512 bins', fontsize=6.8, color='#555555',
            ha='left', va='bottom')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{f}, target' if k == 'target' else P.LABEL['gqcbm_kl'] for f, k in rows],
                       fontsize=6.2)
    for lab, (f, k) in zip(ax.get_yticklabels(), rows):
        lab.set_color('#333333' if k == 'target' else P.TEXT['gqcbm_kl'])
    for i, (f, k) in enumerate(rows):
        dq = D[(n, f, 'target')] if k == 'target' else D[(n, f)]
        ax.text(n - 0.3, i, f'  $k={k_of(dq)}$ ({k_of(dq, n - 9)})', fontsize=6.0, va='center',
                ha='left', color='#333333' if k == 'target' else P.TEXT['gqcbm_kl'])
    ticks = sorted({0, n - 9, n - 1})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'$q_{{{t}}}$' + ('\n(LSB)' if t == 0 else '\n(MSB)' if t == n - 1 else '')
                        for t in ticks], fontsize=7.5)
    ax.tick_params(length=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ims


def plot(D, top4, gates):
    W = 7.0
    total = C.extent(CASE, gates, show_v=False)
    unit = W / total
    head, foot = C.vert(unit, show_v=False)
    h_a = C.height_in(20, unit, head, foot)
    h_b, gap = 5.4, 0.1
    H = h_a + h_b + gap
    fig = plt.figure(figsize=(W, H))
    ax_a = fig.add_axes([0, (h_b + gap) / H, 1, h_a / H])
    C.draw(CASE, gates, 10, None, ax=ax_a, unit=unit, total=total, show_v=False,
           titles=(None, '(a) G-QCBM+KL circuit, Exponential, $n = 20$'))
    for t in ax_a.texts:               # panel (a) title and notes in black, like the other panels
        if t.get_text().startswith(('(a)', 'generated circuit')):
            t.set_color('black')
    fy = lambda inch: inch / H            # inches from the bottom -> figure fraction
    L = 0.012                             # common left edge: (b)/(c) titles, colour bars, notes
    x0, wc = 0.15, 0.33                   # heatmap left edge and width of the n = 20 panel
    top = 5.27                            # top of the left column = top of (d) (inches)
    ax_c = fig.add_axes([x0, fy(1.65), wc, fy(1.45)])
    ax_b = fig.add_axes([x0, fy(3.7), wc * 15 / 20, fy(1.4)])
    ims = heat(ax_b, D, 15, '(b)')
    heat(ax_c, D, 20, '(c)')
    fig.text(L, fy(top), '(b) $n = 15$', fontsize=7.5, va='top', ha='left')
    fig.text(L, fy(1.65 + 1.45 + 0.24), '(c) $n = 20$', fontsize=7.5, va='top', ha='left')
    # colour scales, equation and k note at the bottom of the left column
    xc = 0.30                             # centre of the left column (row labels .. k labels)
    wb = 0.36
    xbar = xc - wb / 2                    # colour bars and the equation are centred on xc
    for j, (kind, lab) in enumerate((('model', P.LABEL['gqcbm_kl']), ('target', 'target'))):
        yb = 0.77 - j * 0.2
        cax = fig.add_axes([xbar, fy(yb), wb, fy(0.05)])
        cb = fig.colorbar(ims[kind], cax=cax, orientation='horizontal')
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.3, length=2, pad=1, labelbottom=(j == 1))
        fig.text(xbar + wb + 0.008, fy(yb + 0.025), lab, fontsize=6.5, va='center',
                 color=P.TEXT['gqcbm_kl'] if kind == 'model' else '#333333')
    fig.text(xc, fy(0.94), r'$\delta_q = 1 - H(b_q \mid b_{q+1}\ldots b_{n-1})$ [bits]'
             r'   (white: $\delta_q \leq 10^{-12}$)', fontsize=8.5, va='bottom', ha='center')
    fig.text(L, fy(0.0), r'basis index $x = {\sum}_q\, b_q\,2^q$, $b_q$ = measured bit of qubit $q_q$'
             r' ($q_0$ = LSB)' '\n'
             '$k$ = number of top qubits that carry the shape, over all $2^n$ states\n'
             '(in brackets: at the 512-bin resolution of Fig. 3)', fontsize=6.3, color='#333333',
             va='bottom', ha='left')
    # (d) spans the same height as the left column: x label at the bottom, title at `top`
    ax = fig.add_axes([0.665, fy(0.42), 0.32, fy(top - 0.2 - 0.42)])
    col = P.COLOR['gqcbm_kl']
    h = np.array([t[0] for t in top4])
    ax.bar(h, [t[1] for t in top4], width=0.9, color=P.TARGET_FILL, label='Target')
    ax.step(np.r_[h - 0.5, 15.5], np.r_[[t[2] for t in top4], top4[-1][2]], where='post',
            color=col, lw=1.6, label=P.LABEL['gqcbm_kl'])
    ax.set_xlim(-0.6, 15.6)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel(r'value of $(q_{19} q_{18} q_{17} q_{16})$, i.e. block $\lfloor x / 2^{16} \rfloor$',
                  fontsize=7.5)
    ax.set_ylabel(r'probability per block of $2^{16}$ states', fontsize=6.5)
    ax.set_title(r'(d) Exponential, $n = 20$: marginal of $q_{16}$–$q_{19}$', fontsize=7.5, loc='left', pad=5)
    ax.tick_params(labelsize=6.3)
    ax.legend(fontsize=6.2, frameon=False)
    ax.grid(False)
    fig.savefig(OUT + '.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'saved {OUT}.{{pdf,png}}')


def main():
    if '--from-csv' not in sys.argv:
        ckpt_dir = sys.argv[sys.argv.index('--ckpt-dir') + 1] if '--ckpt-dir' in sys.argv else 'ckpts'
        build_csv(ckpt_dir)
    D, top4, gates = load()
    for key, dq in sorted(D.items(), key=str):
        n, fam = key[0], ' '.join(key[1:])
        print(f'n={n} {fam:18s} k={k_of(dq):2d} k512={k_of(dq, n - 9):2d}  delta_q: ' + ' '.join(f'{v:.1e}' for v in dq))
    plot(D, top4, gates)


if __name__ == '__main__':
    main()
