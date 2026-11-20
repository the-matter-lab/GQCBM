"""Export the plotted data of Fig. S6, S7, S9 and S11 as CSV, on CPU (no CUDA-Q).

The figure scripts (make_bas_figures.py, make_all_dist_compare.py,
make_fig_exp10_bars.py) evaluate circuits with CUDA-Q on an NVIDIA GPU. This
script re-implements the two circuit kernels in numpy and writes the arrays
those figures plot, so the figures can be replotted without a GPU:

  GQCBM     circuit_ops.circuit_kernel: H on every qubit, then exp_pauli(c, P)
            = exp(i c P) for each Pauli term of theta * pool[op].
  QCBM-SEL  qcbm_pennylane.sel_kernel: per layer RZ.RY.RZ on each qubit, then a
            CNOT ring w -> (w+1) mod n.
  State index: qubit q is bit q of the basis index (little-endian).

Every exported series is checked: its forward KL against the target must match
the value stored in the checkpoint / result JSON (rtol 1e-4), or the script
exits non-zero and writes nothing for that figure.

Only the default 1+2-body pool and row_col_sym_pool_v2 are implemented, which is
what these four figures use. Fig. 2 and Fig. S8 are NOT exported here: their
GQCBM series did not reproduce on CPU; export them with the CUDA-Q scripts.
The one exception is the Fig. 2 QCBM-SEL+KL column (job `Fig2`), which this
script rewrites in place, leaving the GPU-exported columns as they are.

Usage (from the repository root, checkpoints in ckpts/, see download_ckpts.py):
    python scripts/export_level2_cpu.py          # all four figures
    python scripts/export_level2_cpu.py S6 S9    # selected figures only
    python scripts/export_level2_cpu.py Fig2     # Fig. 2 QCBM-SEL+KL column only
"""
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, 'paper', 'figure_data')
PAULI = {'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
         'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
PAIRS = [('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]


# ── operator pools (each op = list of Pauli terms {qubit: 'X'|'Y'|'Z'}) ──────
def pool_default(n):
    ops = [[{q: p}] for q in range(n) for p in 'XYZ']
    for a, b in itertools.combinations(range(n), 2):
        ops += [[{a: p, b: p}] for p in 'XYZ']
    return ops


def pool_row_col_sym(R, C):
    rows = [[r * C + c for c in range(C)] for r in range(R)]
    cols = [[r * C + c for r in range(R)] for c in range(C)]
    ops = []
    for line in rows + cols:
        ops += [[{q: p} for q in line] for p in 'XYZ']
        if len(line) >= 2:
            ops += [[{i: p1, j: p2} for i, j in itertools.combinations(line, 2)]
                    for p1, p2 in PAIRS]
        if len(line) >= 3:
            ops += [[{i: p, j: p, k: p} for i, j, k in itertools.combinations(line, 3)]
                    for p in 'XYZ']
    return ops


def pool_row_col_sym_v2(R, C):
    ops = pool_row_col_sym(R, C)
    diag = [(r1 * C + c1, r2 * C + c2)
            for r1, r2 in itertools.combinations(range(R), 2)
            for c1, c2 in itertools.product(range(C), repeat=2) if c1 != c2]
    ops += [[{i: p1, j: p2} for i, j in diag] for p1, p2 in PAIRS]
    return ops


def select_pool(n, cfg):
    special = [k for k in ('symmetric_pool', 'row_col_sym_pool', 'bas_full_pool',
                           'bas_rich_pool', 'add_all_3body', 'add_nn_3body')
               if cfg.get(k)]
    if cfg.get('row_col_sym_pool_v2'):
        R, C = (int(s) for s in cfg['row_col_sym_pool_v2'].split(','))
        return pool_row_col_sym_v2(R, C)
    if special:
        raise NotImplementedError(f'pool {special} not ported to CPU')
    return pool_default(n)


# ── statevector helpers ──────────────────────────────────────────────────────
def apply_1q(psi, U, q):
    return np.moveaxis(np.tensordot(U, np.moveaxis(psi, q, 0), axes=(1, 0)), 0, q)


def exp_pauli(psi, term, c):
    """exp(i c P) psi for a Pauli string P (P^2 = I)."""
    ppsi = psi
    for q, p in term.items():
        ppsi = apply_1q(ppsi, PAULI[p], q)
    return math.cos(c) * psi + 1j * math.sin(c) * ppsi


def probs_little_endian(psi):
    n = psi.ndim
    return (np.abs(psi) ** 2).transpose(range(n)[::-1]).reshape(-1)


def gqcbm_probs(ckpt):
    d = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = d.get('config', {}) or {}
    target = np.asarray(d['target_arr'], dtype=float)
    n = int(round(math.log2(len(target))))
    pool = select_pool(n, cfg)
    psi = np.full([2] * n, 1 / math.sqrt(2 ** n), dtype=complex)
    for idx, th in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
        if abs(float(th)) < 1e-8:
            continue
        for term in pool[int(idx)]:
            psi = exp_pauli(psi, term, float(th))
    return target, probs_little_endian(psi), float(d['best_kl'])


def sel_probs(json_path):
    rec = json.load(open(json_path))
    rec = rec[-1] if isinstance(rec, list) else rec
    n, L, p = rec['n_qubits'], rec['n_layers'], np.asarray(rec['best_params'], float)
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    ry = lambda t: np.array([[math.cos(t / 2), -math.sin(t / 2)],
                             [math.sin(t / 2), math.cos(t / 2)]], dtype=complex)
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for layer in range(L):
        for w in range(n):
            b = (layer * n + w) * 3
            psi = apply_1q(psi, rz(p[b]), w)
            psi = apply_1q(psi, ry(p[b + 1]), w)
            psi = apply_1q(psi, rz(p[b + 2]), w)
        for w in range(n):  # CNOT control w, target (w+1) % n
            t = (w + 1) % n
            sl = [slice(None)] * n
            sl[w] = 1
            sub = psi[tuple(sl)]
            psi = psi.copy()
            psi[tuple(sl)] = np.flip(sub, axis=t if t < w else t - 1)
    return probs_little_endian(psi), float(rec['best_kl'])


def kl(target, p):
    m = target > 0
    return float(np.sum(target[m] * np.log(target[m] / (p[m] + 1e-10))))


def check(label, target, p, stored):
    got = kl(target, p)
    ok = math.isclose(got, stored, rel_tol=1e-4, abs_tol=1e-7)
    print(f'  {label:14s} KL recomputed={got:.6f} stored={stored:.6f} '
          f'{"ok" if ok else "MISMATCH"}')
    return ok


# ── figure exports ───────────────────────────────────────────────────────────
def valid_bas_indices(R, C):  # same as make_bas_figures.valid_bas_indices
    out = set()
    for sp in range(2 ** R):
        out.add(sum(1 << (r * C + c) for r in range(R) if (sp >> r) & 1 for c in range(C)))
    for bp in range(2 ** C):
        out.add(sum(1 << (r * C + c) for c in range(C) if (bp >> c) & 1 for r in range(R)))
    return sorted(out)


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  wrote paper/figure_data/{name} ({len(rows)} rows)')


def export_bas(name, R, C, ckpt, qmmd_json, qkl_json):
    target, p_g, kl_g = gqcbm_probs(ckpt)
    series = [('target', target, None), ('GQCBM', p_g, kl_g)]
    ok = check('GQCBM', target, p_g, kl_g)
    for label, js in (('QCBM-SEL+MMD', qmmd_json), ('QCBM-SEL+KL', qkl_json)):
        if js:
            p, k = sel_probs(js)
            ok &= check(label, target, p, k)
            series.append((label, p, k))
    if not ok:
        return False
    valid = valid_bas_indices(R, C)
    rows = []
    for label, p, k in series:
        k_out = '' if k is None else f'{k:.6g}'
        rows += [[label, v, f'{p[v]:.10g}', k_out] for v in valid]
        rows.append([label, 'leakage', f'{1.0 - sum(p[v] for v in valid):.10g}', k_out])
    write(name, ['method', 'pattern', 'prob', 'stored_kl'], rows)
    return True


def export_dist(name, ckpt, sel_jsons):
    target, p_g, kl_g = gqcbm_probs(ckpt)
    cols, ok = [('gqcbm_kl', p_g)], check('GQCBM', target, p_g, kl_g)
    for col, js in sel_jsons:
        p, k = sel_probs(js)
        ok &= check(col, target, p, k)
        cols.append((col, p))
    if not ok:
        return False
    rows = [[x, f'{target[x]:.10g}'] + [f'{p[x]:.10g}' for _, p in cols]
            for x in range(len(target))]
    write(name, ['basis_index', 'target'] + [c for c, _ in cols], rows)
    return True


def export_fig2_sel_kl():
    """Replace only the qcbm_sel_kl column of fig2_bimodal20_3way.csv (Fig. 2).

    The other columns (GQCBM, QCBM-SEL+MMD) come from scripts/export_fig2_s8_gpu.py
    and are left untouched. Target: qcbm_pennylane.py TARGETS 'bimodal_20q'
    (c1=2/7, c2=5/7, sigma=0.125), checked against the CSV's binned target.
    """
    js = ('results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/'
          'qcbm_aligned_kl_bimodal_20q.json')
    n_states = 2 ** 20
    idx = np.arange(n_states, dtype=float)
    target = (np.exp(-0.5 * ((idx - 2 / 7 * n_states) / (0.125 * n_states)) ** 2) +
              np.exp(-0.5 * ((idx - 5 / 7 * n_states) / (0.125 * n_states)) ** 2))
    target /= target.sum()
    path = os.path.join(OUT, 'fig2_bimodal20_3way.csv')
    rows = list(csv.DictReader(open(path)))
    binned_t = target.reshape(512, -1).sum(axis=1)
    if not np.allclose(binned_t, [float(r['target']) for r in rows], rtol=1e-3, atol=1e-12):
        print('  target does not match the CSV target column')
        return False
    p, k = sel_probs(js)
    if not check('QCBM-SEL+KL', target, p, k):
        return False
    pb = p.reshape(512, -1).sum(axis=1)
    for r, v in zip(rows, pb):
        r['qcbm_sel_kl'] = f'{v:.10g}'
    write('fig2_bimodal20_3way.csv', list(rows[0].keys()), [list(r.values()) for r in rows])
    return True


def main():
    os.chdir(REPO_ROOT)
    jobs = [
        ('Fig. Fig2', export_fig2_sel_kl),
        ('Fig. S6', lambda: export_bas('fig_s6_bas4_patterns.csv', 2, 2, 'ckpts/extras_bas4.pt',
                                       'results/archive/qcbmpl_v2_mmd_bas4_4q.json',
                                       'results/archive/qcbmpl_v2_kl_bas4_4q.json')),
        # QCBM-SEL+KL JSON for GHZ-10 has no best_params, so the figure omits it.
        ('Fig. S7', lambda: export_dist('fig_s7_ghz10_dist.csv', 'ckpts/gqe_hamMMD_ghz10_10q.pt',
                                        [('qcbm_sel_mmd', 'results/qcbmpl_v3_mmd_ghz10_10q.json')])),
        ('Fig. S9', lambda: export_bas('fig_s9_bas9_patterns.csv', 3, 3, 'ckpts/gqe_bas9_v2_ent.pt',
                                       'results/qcbmpl_v3_mmd_bas9_9q.json', None)),
        ('Fig. S11', lambda: export_dist('fig_s11_exp10_bars.csv', 'ckpts/gqe_exp_10q_g20.pt',
                                         [('qcbm_sel_mmd', 'results/sweep_L4_mmd_exp_10q.json')])),
    ]
    only = set(sys.argv[1:])  # optional, e.g. `S6` exports Fig. S6 only
    failed = []
    for fig, job in jobs:
        if only and fig.split()[-1] not in only:
            continue
        print(fig)
        if not job():
            failed.append(fig)
    if failed:
        print('FAILED (nothing written for these):', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
