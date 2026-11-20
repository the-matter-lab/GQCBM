"""Circuit figure: V-QCBM ansatz (one layer, repeated L_q times) next to an actual
G-QCBM generated circuit.

V-QCBM is drawn as the code runs it (qcbm_pennylane.sel_kernel): per layer,
RZ, RY, RZ on every qubit, then CNOT(i, i+1 mod n) for i = 0..n-1. Its layer
count L_q is read from the run JSON of that cell.

G-QCBM is the saved best circuit of the checkpoint: a Hadamard on every qubit,
then exp(i theta P) for each token (circuit_ops.circuit_kernel), P from the
default 1+2-body pool (circuit_ops.build_op_pool), theta in {+-pi/2^k}. Gates
with theta = +-pi equal -I (identity up to a global phase); they are drawn like
the others and counted, as compute_circuit_stats.py counts them. A second file,
fig_circuits_{case}_faded, draws them faded and dashed.

  case          G-QCBM checkpoint        V-QCBM run (L_q)
  gauss_n5      seed_5q_42               figS4 qcbm_nc_5q_lr0.01 (sigma 0.20; not a Table 1 cell)
  gauss_n10     gqe_q10_small_v2         Table 1
  bimodal_n10   gqe_bimodal_10q_g30      Table 1
  exp_n10       gqe_exp_10q_g20          Table 1
  lognormal_n10 gqe_lognormal_10q        Table 1
  gauss_n15     gqe_q15_v4               Table 1

The decode runs on CPU (no CUDA-Q): the script replays the circuit in numpy,
asserts it reproduces the checkpoint's stored best_kl, and writes the gate list to
paper/figure_data/fig_circuits_{case}.csv.

    python make_fig_circuits.py --ckpt-dir ckpts          # decode, verify, plot
    python make_fig_circuits.py --from-csv                # plot from the CSVs
Writes paper/figs/fig_circuits_{case}{,_faded}.{pdf,png} (and gauss_n10_notesbelow, notes under the circuits). Option: --case gauss_n10 (one only).
"""
import csv
import itertools
import json
import math
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
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

import gqcbm_palette as P  # noqa: E402

_T1 = 'results/sel_kl_exact_gb200/runs/t1_n10/job_8803242/results/qcbm_aligned_kl_{}_10q.json'
CASES = {
    'gauss_n5': dict(n=5, family='Gaussian', ckpt='seed_5q_42',
                     vjson='results/sel_kl_exact_gb200/runs/figS4/job_8803247/results/qcbm_nc_5q_lr0.01.json'),
    'gauss_n10': dict(n=10, family='Gaussian', ckpt='gqe_q10_small_v2', vjson=_T1.format('gauss')),
    'bimodal_n10': dict(n=10, family='Bimodal', ckpt='gqe_bimodal_10q_g30', vjson=_T1.format('bimodal')),
    'exp_n10': dict(n=10, family='Exponential', ckpt='gqe_exp_10q_g20', vjson=_T1.format('exp')),
    'lognormal_n10': dict(n=10, family='Log-normal', ckpt='gqe_lognormal_10q',
                          vjson=_T1.format('lognormal')),
    'gauss_n15': dict(n=15, family='Gaussian', ckpt='gqe_q15_v4',
                      vjson='results/sel_kl_exact_gb200/runs/t1_n15/job_8828095/results/'
                            'qcbm_aligned_kl_gauss_15q_s42.json'),
}
CSV = 'paper/figure_data/fig_circuits_{case}.csv'
OUT = 'paper/figs/fig_circuits_{case}'
SPECIAL_POOL_FLAGS = ('symmetric_pool', 'row_col_sym_pool_v2', 'row_col_sym_pool', 'bas_full_pool',
                      'bas_rich_pool', 'add_all_3body', 'add_nn_3body')


# ── decode + verify ─────────────────────────────────────────────────────────
def default_pool(n):
    """CPU replica of circuit_ops.build_op_pool: (qubits, Pauli letters)."""
    ops = [((q,), p) for q in range(n) for p in 'XYZ']
    for a, b in itertools.combinations(range(n), 2):
        ops += [((a, b), p * 2) for p in 'XYZ']
    return ops


_PM = {'X': np.array([[0, 1], [1, 0]], complex), 'Y': np.array([[0, -1j], [1j, 0]]),
       'Z': np.diag([1, -1]).astype(complex)}


def replay(n, gates):
    """Statevector of H^n then exp(i theta P) per gate; CUDA-Q qubit q is bit q
    (little-endian), i.e. axis n-1-q of the big-endian tensor."""
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for qs, ps, th in gates:
        t = psi.reshape([2] * n)
        u = t
        for q, p in zip(qs, ps):
            ax = n - 1 - q
            u = np.moveaxis(np.tensordot(_PM[p], u, axes=([1], [ax])), 0, ax)
        psi = (math.cos(th) * t + 1j * math.sin(th) * u).reshape(-1)
    return np.abs(psi) ** 2


def decode(case, ckpt_dir):
    import torch
    c = CASES[case]
    n = c['n']
    d = torch.load(os.path.join(ckpt_dir, c['ckpt'] + '.pt'), map_location='cpu', weights_only=False)
    cfg = d.get('config', {}) or {}
    assert not any(cfg.get(k) for k in SPECIAL_POOL_FLAGS), 'non-default pool'
    pool = default_pool(n)
    gates = [(pool[int(i)][0], pool[int(i)][1], float(a))
             for i, a in zip(d['best_ops'].tolist(), d['best_theta'].tolist()) if abs(a) > 1e-8]
    tgt = np.asarray(d['target_arr'], float)
    assert len(tgt) == 2 ** n
    p = replay(n, gates)
    m = tgt > 0
    kl = float(np.sum(tgt[m] * np.log(tgt[m] / (p[m] + 1e-10))))
    assert math.isclose(kl, float(d['best_kl']), rel_tol=1e-6), (kl, d['best_kl'])
    print(f'{case}: {len(gates)} gates, replay KL {kl:.6g} = stored {float(d["best_kl"]):.6g}')
    with open(CSV.format(case=case), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'pauli', 'qubits', 'theta', 'source_ckpt', 'stored_best_kl'])
        for s, (qs, ps, th) in enumerate(gates):
            w.writerow([s, ps, ' '.join(map(str, qs)), repr(th), c['ckpt'], repr(float(d['best_kl']))])
    return gates


def load_csv(case):
    gates = []
    for r in csv.DictReader(open(CSV.format(case=case))):
        assert r['source_ckpt'] == CASES[case]['ckpt'], (case, r['source_ckpt'])
        gates.append((tuple(int(q) for q in r['qubits'].split()), r['pauli'], float(r['theta'])))
    return gates


def v_layers(case):
    rec = json.load(open(CASES[case]['vjson']))
    rec = rec[-1] if isinstance(rec, list) else rec
    assert rec['n_qubits'] == CASES[case]['n']
    return int(rec['n_layers'])


def is_identity(th):
    return abs(abs(th) - math.pi) < 1e-6


def angle_label(th):
    k = round(-math.log2(abs(th) / math.pi))
    assert math.isclose(abs(th), math.pi / 2 ** k, rel_tol=1e-6), th
    s = '-' if th < 0 else ''
    return f'{s}\\pi' if k == 0 else f'{s}\\pi/{2 ** k}'


# ── layout ──────────────────────────────────────────────────────────────────
def pack(gates):
    """ASAP columns; a two-qubit gate occupies every wire it spans, so its
    connector never runs through another gate's box. Every gate also keeps one
    free wire above and below it within its column, so gates never touch end to
    end (two touching gates would read as one larger gate)."""
    free, cols = {}, []
    for qs, _, _ in gates:
        lo, hi = min(qs), max(qs)
        c = max(free.get(q, 0) for q in range(lo - 1, hi + 2))
        for q in range(lo, hi + 1):
            free[q] = c + 1
        cols.append(c)
    return cols


def underbrace(ax, x0, x1, y, h, **kw):
    """Horizontal curly brace opening upward (tips at y, point at y + h; y grows down)."""
    t = np.linspace(0, 1, 200)
    half = lambda s: h * (1 / (1 + np.exp(-30 * (s - 0.08))) + 1 / (1 + np.exp(-30 * (s - 0.92)))) / 2
    xm = (x0 + x1) / 2
    xs = np.concatenate([x0 + t * (xm - x0), xm + t * (x1 - xm)])
    ys = np.concatenate([y + half(t), y + half(1 - t)])
    ax.plot(xs, ys, **kw)


G_ONLY_X0 = 1.4  # x of the H layer when the V-QCBM panel is omitted (room for q labels)


def extent(case, gates, show_v=True):
    """Total width (gate units) of draw()'s layout."""
    n = CASES[case]['n']
    v_w = 2 * 1.85 + 0.8 + 0.5 + (n - 1) * 0.5 + 0.8
    gx0 = 1.6 + v_w + 3.2 if show_v else G_ONLY_X0
    return gx0 + 1 + max(pack(gates)) + 1 + 0.6


FS_N = 6.8  # note font size (pt)


def vert(unit, notes_below=False, has_title=True, n_lines=3, show_v=True):
    """(head in inches above the first wire, foot in gate units below the last wire).
    notes_below puts the note lines under the circuit (below the brace and x L_q)."""
    if not notes_below:
        return 0.62, 0.9
    pt = unit * 72
    head = 0.3 if has_title else 0.05
    return head, notes_y(unit, show_v) + n_lines * FS_N * 1.3 / pt + 0.1


def notes_y(unit, show_v=True):
    """y (gate units below the last wire) of the first note line when notes are below
    (under the brace and x L_q when the V-QCBM panel is drawn)."""
    return (0.8 + (FS_N + 4.5) / (unit * 72)) if show_v else 0.75


def height_in(n, unit, head=0.62, foot=0.9):
    """Figure height (inches) of draw()'s layout for n wires at `unit` inches/gate unit."""
    return (n - 1 + foot + 0.75) * unit + head


def draw(case, gates, L, out, faded=False, ax=None, unit=None, total=None,
         titles=('(a) V-QCBM', '(b) G-QCBM'), show_v=True, notes_below=False):
    """Draw one case. With ax/unit/total given, draw into that axes (whose physical
    size must be total*unit x height_in(n, unit)) and do not save."""
    n, family = CASES[case]['n'], CASES[case]['family']
    vcol = P.COLOR['sel_kl']
    gcol = P.COLOR['gqcbm_kl']
    gcols = pack(gates)
    n_g = max(gcols) + 1
    # x positions (gate units): V-QCBM R_Z R_Y R_Z rectangles at vx0 + j*rsp,
    # CNOT ring after; G-QCBM H at gx0, gates after.
    vx0 = 1.6
    rw, rsp = 1.6, 1.85                 # rotation rectangle width / pitch
    cxp = 0.5                           # CNOT pitch in the ring (squeezed)
    cx0 = vx0 + 2 * rsp + rw / 2 + 0.5  # first CNOT
    v_w = cx0 - vx0 + (n - 1) * cxp + 0.8
    gx0 = vx0 + v_w + 3.2 if show_v else G_ONLY_X0
    own = ax is None
    if own:
        total = gx0 + 1 + n_g + 0.6
        unit = 7.0 / total              # inches per gate unit
    # head: inches above the first wire (title, notes); foot: gate units below the
    # last wire (brace + x L_q, and the notes when notes_below)
    n_lines = 2 + bool(faded and any(is_identity(th) for _, _, th in gates))
    head, foot = vert(unit, notes_below, has_title=titles[1] is not None, n_lines=n_lines,
                      show_v=show_v)
    if own:
        fig = plt.figure(figsize=(7.0, height_in(n, unit, head, foot)))
        ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total)
    ax.set_ylim(n - 1 + foot, -0.75 - head / unit)
    ax.axis('off')
    pt = unit * 72                      # points per gate unit
    fs = min(7.0, 0.5 * pt)             # gate-letter font size (fits the box)
    fs_t, fs_n = 8.0, FS_N              # panel title / note, same in every figure
    fs_a = max(3.2, 0.3 * pt)           # parameter labels (small, for the sense only)
    y_t = -0.75 - head / unit + 0.05
    bw = 0.62                           # box width (gate units)

    def box(x, y, text, ec, size=None, w=bw, alpha=1.0, ls='-'):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - bw / 2), w, bw, boxstyle='square,pad=0',
                                    fc='white', ec=ec, lw=0.8, alpha=alpha, ls=ls, zorder=3))
        ax.text(x, y, text, ha='center', va='center', fontsize=size or fs, color=ec, alpha=alpha,
                zorder=4)

    def plabel(x, y, text, color, alpha=1.0):
        ax.text(x, y - bw / 2 - 0.04, text, ha='center', va='bottom', fontsize=fs_a,
                color=color, alpha=alpha, zorder=4)

    def header(x, title, lines, key):
        """Bold title on top; note lines under it, or under the circuit if notes_below."""
        if title is not None:
            ax.text(x, y_t, title, ha='left', va='top', fontsize=fs_t, fontweight='bold',
                    color=P.TEXT[key])
        y_n = (n - 1 + notes_y(unit, show_v) if notes_below
               else y_t + (13 / pt if title is not None else 0))
        ax.text(x, y_n, '\n'.join(lines), ha='left', va='top', fontsize=fs_n,
                color=P.TEXT[key], linespacing=1.25)

    # wires and labels
    for q in range(n):
        if show_v:
            ax.plot([vx0 - rw / 2 - 0.35, vx0 + v_w - 0.2], [q, q], color='#555', lw=0.6, zorder=1)
        ax.plot([gx0 - 0.6, gx0 + 1 + n_g - 0.2], [q, q], color='#555', lw=0.6, zorder=1)
        for x in ((vx0 - rw / 2 + 0.3, gx0) if show_v else (gx0,)):
            ax.text(x - 0.75, q, f'$q_{{{q}}}$', ha='right', va='center', fontsize=min(fs, 6.5))

    if show_v:
        # V-QCBM: one layer, RZ RY RZ shown explicitly
        fs_r = min(fs, 0.95 * rw * pt / 3.3)   # R_Z(phi_q) fits the rectangle
        for q in range(n):
            for j, (g, par) in enumerate((('Z', '\\phi'), ('Y', '\\theta'), ('Z', '\\omega'))):
                box(vx0 + j * rsp, q, f'$R_{g}({par}_{{{q}}})$', vcol, size=fs_r, w=rw)
        for i in range(n):
            x = cx0 + i * cxp
            c, t = i, (i + 1) % n
            ax.plot([x, x], [c, t], color=vcol, lw=0.9, zorder=2)
            ax.add_patch(Circle((x, c), 0.09, color=vcol, zorder=3))
            ax.add_patch(Circle((x, t), 0.2, fc='white', ec=vcol, lw=0.9, zorder=3))
            ax.plot([x - 0.2, x + 0.2], [t, t], color=vcol, lw=0.9, zorder=4)
            ax.plot([x, x], [t - 0.2, t + 0.2], color=vcol, lw=0.9, zorder=4)
        # layer edges (dashed), brace underneath, "x L_q" below the brace
        lx0, lx1 = vx0 - rw / 2 - 0.25, vx0 + v_w - 0.5
        for x in (lx0, lx1):
            ax.plot([x, x], [-0.55, n - 1 + 0.4], color=vcol, lw=0.7, ls=(0, (3, 2)), zorder=1)
        underbrace(ax, lx0, lx1, n - 1 + 0.45, 0.3, color=vcol, lw=0.8)
        ax.text((lx0 + lx1) / 2, n - 1 + 0.8, f'$\\times\\,{L}$', ha='center', va='top',
                fontsize=fs_n + 0.5, color=P.TEXT['sel_kl'])
        header(vx0 - rw / 2 - 1.0, titles[0], [f'fixed ansatz, one layer shown ($L_q={L}$)',
                                           f'{3 * n * L} trained angles, {n * L} CNOTs'], 'sel_kl')

    # G-QCBM: H layer, then gates
    for q in range(n):
        box(gx0, q, 'H', '#666666')
    n_id = sum(is_identity(th) for _, _, th in gates)
    for (qs, ps, th), c in zip(gates, gcols):
        x = gx0 + 1 + c
        a, ls = (0.35, '--') if faded and is_identity(th) else (1.0, '-')
        if len(qs) == 2:
            ax.plot([x, x], [qs[0], qs[1]], color=gcol, lw=0.9, alpha=a, ls=ls, zorder=2)
        for q, p in zip(qs, ps):
            box(x, q, p, gcol, alpha=a, ls=ls)
        plabel(x, min(qs), f'${angle_label(th)}$', P.TEXT['gqcbm_kl'], alpha=a)
    g_cx = sum(2 * (len(qs) - 1) for qs, _, _ in gates)
    lines = [f'generated circuit, {family.lower()} target',
             f'{len(gates)} gates $e^{{i\\theta P}}$, $\\theta$ from $\\{{\\pm\\pi/2^k\\}}$, '
             f'{g_cx} CNOTs after decomposition']
    if faded and n_id:
        lines.append(f'faded: $\\theta=\\pm\\pi$ (identity up to phase), {n_id} of {len(gates)}')
    header(gx0 - 1.3, None if titles[1] is None else
           titles[1] + (P.PROPOSED if CASES[case]['n'] == 10 else ''), lines, 'gqcbm_kl')
    if not own:
        return
    fig.savefig(out + '.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(out + '.png', dpi=250, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'saved {out}.{{pdf,png}}  (V: L={L}, {n * L} CX; G: {len(gates)} gates, {g_cx} CX, '
          f'{n_g} columns, {n_id} identity)')


def main():
    cases = [sys.argv[sys.argv.index('--case') + 1]] if '--case' in sys.argv else list(CASES)
    ckpt_dir = sys.argv[sys.argv.index('--ckpt-dir') + 1] if '--ckpt-dir' in sys.argv else 'ckpts'
    for case in cases:
        gates = load_csv(case) if '--from-csv' in sys.argv else decode(case, ckpt_dir)
        draw(case, gates, v_layers(case), OUT.format(case=case))
        draw(case, gates, v_layers(case), OUT.format(case=case) + '_faded', faded=True)
        if case == 'gauss_n10':  # variant with the notes under the circuits
            draw(case, gates, v_layers(case), OUT.format(case=case) + '_notesbelow',
                 notes_below=True)


if __name__ == '__main__':
    main()
