"""BAS-specific figures: render the 6 / 14 valid grid patterns as
$\sqrt{n}{\times}\sqrt{n}$ images annotated with each method's
probability mass on that pattern. Replaces the probability-bar overlay
for BAS-$2{\times}2$ and BAS-$3{\times}3$ in paper/figs/dist/.
"""
import os, sys, math, json
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import torch, cudaq, cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cudaq.set_target('nvidia', option='fp64')

from gqe_single_target import (
    build_target, build_op_pool_symmetric,
    build_op_pool_row_col_sym, build_op_pool_row_col_sym_v2,
    build_op_pool_with_all_3body, build_op_pool_with_nn_3body,
    build_op_pool_bas_full, build_op_pool_bas_rich,
)
from circuit_ops import build_op_pool, circuit_kernel
from qcbm_pennylane import circuit_probs as qcbm_probs


mpl.rcParams.update({'font.size': 9, 'axes.linewidth': 0.7})


def _select_pool(nq, cfg):
    if cfg.get('symmetric_pool'):
        return build_op_pool_symmetric(nq)
    if cfg.get('row_col_sym_pool_v2'):
        R, C = (int(s) for s in cfg['row_col_sym_pool_v2'].split(','))
        return build_op_pool_row_col_sym_v2(nq, (R, C))
    if cfg.get('row_col_sym_pool'):
        R, C = (int(s) for s in cfg['row_col_sym_pool'].split(','))
        return build_op_pool_row_col_sym(nq, (R, C))
    if cfg.get('bas_full_pool'):
        R, C = (int(s) for s in cfg['bas_full_pool'].split(','))
        return build_op_pool_bas_full(nq, (R, C))
    if cfg.get('bas_rich_pool'):
        R, C = (int(s) for s in cfg['bas_rich_pool'].split(','))
        return build_op_pool_bas_rich(nq, (R, C))
    if cfg.get('add_all_3body'):
        return build_op_pool_with_all_3body(nq)
    if cfg.get('add_nn_3body'):
        return build_op_pool_with_nn_3body(nq)
    return build_op_pool(nq)


def gqcbm_probs(ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = d.get('config', {}) or {}
    if 'target_arr' in d:
        target = (d['target_arr'].numpy() if hasattr(d['target_arr'], 'numpy')
                  else np.array(d['target_arr']))
        nq = int(round(math.log2(len(target))))
    else:
        nq = int(cfg['n_qubits'])
        tkw = json.loads(cfg.get('target_kw', '{}') or '{}')
        target = build_target(cfg['target_kind'], 2 ** nq, **tkw)
    pool = _select_pool(nq, cfg)
    fc, fw = [], []
    for idx, ang in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
        if abs(float(ang)) < 1e-8:
            continue
        for term in float(ang) * pool[int(idx)]:
            c = term.get_coefficient().real
            w = term.to_string(nq).split(']')[-1].strip().ljust(nq, 'I')
            fc.append(c); fw.append(cudaq.pauli_word(w))
    if not fc:
        return target, np.ones(2 ** nq) / (2 ** nq), nq, float(d.get('best_kl', float('nan')))
    st = cudaq.get_state(circuit_kernel, nq, fc, fw)
    probs = cp.asnumpy(cp.abs(cp.array(st, dtype=cp.complex128)) ** 2)
    return target, probs, nq, float(d.get('best_kl', float('nan')))


def qcbm_sel_probs(json_path, name=None):
    if not json_path or not os.path.exists(json_path):
        return None, None
    d = json.load(open(json_path))
    if not isinstance(d, list):
        d = [d]
    if name:
        d = [r for r in d if r.get('name') == name]
        if not d:
            return None, None
    rec = d[-1]
    if 'best_params' not in rec or rec['best_params'] is None:
        return None, None
    return (qcbm_probs(rec['n_qubits'], rec['n_layers'],
                      np.array(rec['best_params'])),
            float(rec.get('best_kl', float('nan'))))


def valid_bas_indices(R, C):
    out = set()
    for sp in range(2 ** R):                # stripes
        idx = 0
        for r in range(R):
            if (sp >> r) & 1:
                for c in range(C):
                    idx |= (1 << (r * C + c))
        out.add(idx)
    for bp in range(2 ** C):                # bars
        idx = 0
        for c in range(C):
            if (bp >> c) & 1:
                for r in range(R):
                    idx |= (1 << (r * C + c))
        out.add(idx)
    return sorted(out)


def idx_to_grid(idx, R, C):
    g = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            g[r, c] = (idx >> (r * C + c)) & 1
    return g


def fig_bas_grid(R, C, ckpt, qmmd_json, qkl_json, label, savepath):
    target, p_g, nq, kl_g = gqcbm_probs(ckpt)
    p_m, kl_m = qcbm_sel_probs(qmmd_json)
    p_k, kl_k = qcbm_sel_probs(qkl_json)

    valid = valid_bas_indices(R, C)
    n_pat = len(valid)

    # Layout: 4 rows × n_pat cols of grids; one column per pattern
    # Row 1: target (uniform 1/n_pat) — grids in black
    # Row 2-4: GQCBM, QCBM-MMD, QCBM-KL — opacity ∝ probability
    methods = [('Target', '#000000', target, None),
               ('GQCBM', '#2E5EAA', p_g, kl_g)]
    if p_m is not None:
        methods.append(('QCBM-SEL+MMD', '#E07B00', p_m, kl_m))
    if p_k is not None:
        methods.append(('QCBM-SEL+KL', '#888888', p_k, kl_k))
    n_rows = len(methods)

    label_col = 3.0   # in grid units, accommodates "QCBM-SEL+MMD, D_KL=, leak"
    fig = plt.figure(figsize=(0.7 * n_pat + 3.0, 0.95 * n_rows + 0.3))
    gs = gridspec.GridSpec(n_rows, n_pat + 1,
                           width_ratios=[label_col] + [1] * n_pat,
                           wspace=0.18, hspace=0.30)

    # Find max prob across methods on valid set for shared opacity scale
    pmax = max(float(np.max([probs[v] for v in valid]))
               for _, _, probs, _ in methods)

    for ri, (mname, color, probs, kl) in enumerate(methods):
        # Left annotation column
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis('off')
        leak = float(1.0 - sum(probs[v] for v in valid))
        annot = f'{mname}'
        if kl is not None and not math.isnan(kl):
            annot += f'\n$D_{{KL}}={kl:.4f}$'
        annot += f'\nleak P(invalid) = {leak:.3f}'
        ax_lbl.text(0.0, 0.5, annot, va='center', ha='left',
                    fontsize=9, color=color, fontweight='bold')

        for j, v in enumerate(valid):
            ax = fig.add_subplot(gs[ri, j + 1])
            grid = idx_to_grid(v, R, C)
            p = float(probs[v])
            # Foreground colour = method colour modulated by probability
            ax.set_facecolor('#f5f5f5')
            # Draw black filled cells with alpha = p / pmax
            from matplotlib.colors import to_rgba
            base = to_rgba(color)
            alpha = max(0.05, p / pmax)
            for rr in range(R):
                for cc in range(C):
                    if grid[rr, cc] > 0.5:
                        ax.add_patch(plt.Rectangle((cc, R - 1 - rr), 1, 1,
                                                   color=(base[0], base[1], base[2], alpha),
                                                   linewidth=0))
            ax.set_xlim(0, C); ax.set_ylim(0, R)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#bbbbbb'); spine.set_linewidth(0.5)
            ax.set_title(f'{p:.3f}', fontsize=6, pad=1)
            if ri == 0:
                ax.set_xlabel(f'$x{{=}}{v}$', fontsize=6, labelpad=1)
                ax.xaxis.set_label_position('top')

    fig.suptitle(rf'BAS ${R}{{\times}}{C}$ ({n_pat} valid patterns): '
                 r'probability mass per valid pattern. '
                 r'Cell opacity $\propto p(x)$, normalised across methods.',
                 fontsize=10, y=1.02)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath + '.pdf', bbox_inches='tight')
    plt.savefig(savepath + '.png', dpi=180, bbox_inches='tight')
    plt.close()
    print(f'saved {savepath}.{{pdf,png}}')


if __name__ == '__main__':
    # optional: `python make_bas_figures.py bas4` (or bas9) re-renders one figure
    only = set(sys.argv[1:])
    if not only or 'bas4' in only:
        fig_bas_grid(R=2, C=2,
                     ckpt='ckpts/extras_bas4.pt',
                     qmmd_json='results/archive/qcbmpl_v2_mmd_bas4_4q.json',  # Table 2 runs
                     qkl_json='results/archive/qcbmpl_v2_kl_bas4_4q.json',
                     label='BAS 2x2',
                     savepath='paper/figs/dist/bas4_4q')
    if not only or 'bas9' in only:
        fig_bas_grid(R=3, C=3,
                     ckpt='ckpts/gqe_bas9_v2_ent.pt',
                     qmmd_json='results/qcbmpl_v3_mmd_bas9_9q.json',
                     qkl_json=None,
                     label='BAS 3x3',
                     savepath='paper/figs/dist/bas9_9q')
    print('Done.')
