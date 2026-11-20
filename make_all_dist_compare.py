"""Distribution overlay figures for every target in Tables 1 and 2.
For each target: target vs GQCBM vs QCBM-SEL+MMD (and +KL when SEL-format
JSON is available). Writes one figure per target into paper/figs/dist/.
Also writes a meta-grid combining all 14 onto a single PDF.
"""
import os, sys, json, math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch, cudaq, cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
cudaq.set_target('nvidia', option='fp64')

from gqe_single_target import build_target
from circuit_ops import circuit_kernel
from qcbm_pennylane import circuit_probs as qcbm_probs
from pool_select import select_pool as _select_pool  # shared implementation


mpl.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
os.makedirs('paper/figs/dist', exist_ok=True)
C_TGT, C_GQCBM, C_QMMD, C_QKL = 'black', '#2E5EAA', '#E07B00', '#888888'


# ─────────────────────────────────────────────────────────────────────────────
# GQCBM reconstruction (pool variants resolved by pool_select.select_pool)
# ─────────────────────────────────────────────────────────────────────────────


def gqcbm_dist(ckpt_path):
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
        probs = np.ones(2 ** nq) / (2 ** nq)
    else:
        st = cudaq.get_state(circuit_kernel, nq, fc, fw)
        probs = cp.asnumpy(cp.abs(cp.array(st, dtype=cp.complex128)) ** 2)
    return target, probs, nq, float(d.get('best_kl', float('nan')))


# ─────────────────────────────────────────────────────────────────────────────
# QCBM-SEL reconstruction (uses qcbm_pennylane's SEL circuit)
# ─────────────────────────────────────────────────────────────────────────────
def qcbm_sel_dist(json_path, name_filter=None):
    if not os.path.exists(json_path):
        return None, None, None
    d = json.load(open(json_path))
    if not isinstance(d, list):
        d = [d]
    if name_filter:
        d = [r for r in d if r.get('name') == name_filter]
        if not d:
            return None, None, None
    rec = d[-1]
    if 'best_params' not in rec or rec['best_params'] is None:
        return None, None, None
    probs = qcbm_probs(rec['n_qubits'], rec['n_layers'],
                      np.array(rec['best_params']))
    return probs, float(rec.get('best_kl', float('nan'))), rec['n_qubits']


def _is_partial(json_path, name_filter=None):
    d = json.load(open(json_path))
    d = d if isinstance(d, list) else [d]
    if name_filter:
        d = [r for r in d if r.get('name') == name_filter]
    return bool(d and d[-1].get('partial'))


# ─────────────────────────────────────────────────────────────────────────────
# Target registry — (label, GQCBM ckpt, QCBM-MMD JSON+name, QCBM-KL JSON+name)
# ─────────────────────────────────────────────────────────────────────────────
TARGETS = [
    # Continuous
    ('gauss_5q',      'ckpts/seed_5q_42.pt',
     'results/qcbmpl_gauss_5q_s42.json',  None,
     'results/qcbm_fairkl_5q_s500.json',  None),
    ('gauss_10q',     'ckpts/gqe_q10_small_v2.pt',
     'results/qcbmpl_v3_mmd_gauss_10q.json', None,
     'results/qcbm_fairkl_10q_s500.json', None),
    ('gauss_15q',     'ckpts/seed_15q_42.pt',
     'results/qcbmpl_gauss_15q_s42.json', None,
     'results/qcbm_fairkl_15q_s500.json', None),
    ('bimodal_6q',    'ckpts/seed_6q_789.pt',
     'results/qcbmpl_bimodal_6q_s42.json', None,
     'results/qcbm_cvg_bimodal_6q.json',  None),
    ('exp_5q',        'ckpts/extras_exp5.pt',
     'results/qcbmpl_exp_5q.json',        None,
     'results/qcbm_cvg_exp_5q.json',      None),
    ('lognormal_5q',  'ckpts/gqe_lognormal_5q.pt',
     'results/qcbmpl_mmd_lognormal.json', 'lognormal_5q',
     'results/qcbmpl_kl_lognormal.json',  'lognormal_5q'),
    ('lognormal_10q', 'ckpts/gqe_lognormal_10q.pt',
     'results/qcbmpl_mmd_lognormal.json', 'lognormal_10q',
     'results/qcbmpl_kl_lognormal.json',  'lognormal_10q'),
    # Discrete (structured)
    ('ghz5_5q',       'ckpts/gqe_ghz5.pt',
     'results/qcbmpl_ghz5_5q.json',       None,
     'results/qcbm_cvg_ghz5_5q.json',     None),
    ('bas4_4q',       'ckpts/extras_bas4.pt',
     'results/archive/qcbmpl_v2_mmd_bas4_4q.json', None,   # Table 2 runs
     'results/archive/qcbmpl_v2_kl_bas4_4q.json',  None),
    ('ghz10_10q',     'ckpts/gqe_hamMMD_ghz10_10q.pt',
     'results/qcbmpl_v3_mmd_ghz10_10q.json', None,
     'results/qcbm_ghz10_10q.json',       None),
    ('cardinality_5q','ckpts/gqe_sym_card_5q.pt',
     'results/qcbmpl_v3_mmd_cardinality_5q.json', None,
     'results/qcbmpl_kl_cardinality.json','cardinality_5q'),
    ('cardinality_10q','ckpts/gqe_sym_card_10q_full.pt',
     'results/qcbmpl_v3_mmd_cardinality_10q.json', None,
     # Table 2 run; partial (hit walltime), marked with a dagger in the legend
     'results/archive/qcbmpl_v3_kl_cardinality_10q.json', None),
    ('bas9_9q',       'ckpts/gqe_bas9_v2_ent.pt',
     'results/qcbmpl_v3_mmd_bas9_9q.json', None,
     None,                                 None),
]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_overlay(ax, target, p_gqcbm, gqcbm_kl, p_qmmd, qmmd_kl,
                 p_qkl, qkl_kl, nq, title, qkl_tag=''):
    x = np.arange(2 ** nq)
    if nq >= 12:  # too dense for bars → line
        ax.fill_between(x, target, alpha=0.18, color=C_TGT, label='Target')
        ax.plot(x, p_gqcbm, color=C_GQCBM, lw=1.0,
                label=f'GQCBM (KL={gqcbm_kl:.4f})')
        if p_qmmd is not None:
            ax.plot(x, p_qmmd, color=C_QMMD, lw=1.0, ls='--',
                    label=f'QCBM-MMD (KL={qmmd_kl:.4f})')
        if p_qkl is not None:
            ax.plot(x, p_qkl, color=C_QKL, lw=0.9, ls=':',
                    label=f'QCBM-KL (KL={qkl_kl:.4f}{qkl_tag})')
    else:
        w = 0.8 / 4
        ax.bar(x - 1.5 * w, target, w, color=C_TGT, alpha=0.6, label='Target')
        ax.bar(x - 0.5 * w, p_gqcbm, w, color=C_GQCBM, alpha=0.85,
               label=f'GQCBM (KL={gqcbm_kl:.4f})')
        if p_qmmd is not None:
            ax.bar(x + 0.5 * w, p_qmmd, w, color=C_QMMD, alpha=0.85,
                   label=f'QCBM-MMD (KL={qmmd_kl:.4f})')
        if p_qkl is not None:
            ax.bar(x + 1.5 * w, p_qkl, w, color=C_QKL, alpha=0.7,
                   label=f'QCBM-KL (KL={qkl_kl:.4f}{qkl_tag})')
    ax.set_xlabel(r'Basis index $x$')
    ax.set_ylabel(r'$p(x)$')
    ax.set_title(title, loc='left', fontsize=10)
    ax.legend(loc='upper right', fontsize=7, frameon=True)


# targets whose paper/figs/dist/ figure comes from a dedicated script
OWN_SCRIPT = {'ghz10_10q': 'make_fig_ghz10.py', 'cardinality_10q': 'make_fig_dicke10.py'}


def main(only=None):
    panels = []  # (label, target, p_gqcbm, ...) for grid
    for label, gck, qmmd_json, qmmd_name, qkl_json, qkl_name in TARGETS:
        if only and label not in only:
            continue
        if not os.path.exists(gck):
            print(f'  SKIP {label}: no GQCBM ckpt {gck}')
            continue
        print(f'\n=== {label} ===')
        target, p_gqcbm, nq, kl_g = gqcbm_dist(gck)
        p_qmmd, kl_m, _ = qcbm_sel_dist(qmmd_json, qmmd_name) if qmmd_json else (None, None, None)
        p_qkl, kl_k, _ = qcbm_sel_dist(qkl_json, qkl_name) if qkl_json else (None, None, None)
        qkl_tag = r'$^\dagger$' if (qkl_json and _is_partial(qkl_json, qkl_name)) else ''
        print(f'  KL: GQCBM={kl_g:.4f}, QCBM-MMD={kl_m}, QCBM-KL={kl_k}')
        # individual figure (GHZ-10's SI figure is make_fig_ghz10.py; keep it in the grid only)
        if label in OWN_SCRIPT:
            print(f'  {label}: individual figure is made by {OWN_SCRIPT[label]}; not overwritten')
            panels.append((label, target, p_gqcbm, kl_g, p_qmmd, kl_m,
                           p_qkl, kl_k, nq, qkl_tag))
            continue
        fig, ax = plt.subplots(figsize=(8.5, 3.5))
        plot_overlay(ax, target, p_gqcbm, kl_g, p_qmmd, kl_m, p_qkl, kl_k,
                     nq, f'{label} ($n={nq}$)', qkl_tag=qkl_tag)
        plt.tight_layout()
        plt.savefig(f'paper/figs/dist/{label}.pdf', bbox_inches='tight')
        plt.savefig(f'paper/figs/dist/{label}.png', dpi=180, bbox_inches='tight')
        plt.close()
        panels.append((label, target, p_gqcbm, kl_g, p_qmmd, kl_m,
                       p_qkl, kl_k, nq, qkl_tag))
    print(f'\nWrote {len(panels)} individual figures → paper/figs/dist/')
    if only:  # partial re-render: leave the meta-grid untouched
        return

    # 4×4 meta-grid (last cell empty if 13 panels)
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 2.7))
    for i, panel in enumerate(panels):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        plot_overlay(ax, *panel[1:8], panel[8],
                     panel[0].replace('_', '-'), qkl_tag=panel[9])
    for j in range(n, rows * cols):
        ax = axes[j // cols][j % cols] if rows > 1 else axes[j % cols]
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('paper/figs/fig_all_dist_compare.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_all_dist_compare.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved meta-grid → paper/figs/fig_all_dist_compare.{pdf,png}')


if __name__ == '__main__':
    # optional: labels to re-render, e.g. `python make_all_dist_compare.py cardinality_10q`
    main(set(sys.argv[1:]) or None)
