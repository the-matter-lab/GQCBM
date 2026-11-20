"""Export the plotted data of Fig. 2 (bimodal 20q, 3-way overlay) and
Fig. S8 (cardinality 10q distribution overlay) using the same CUDA-Q GPU
reconstruction as the figure scripts.

Reuses the figure scripts' own functions where they can be imported without
side effects:
  * make_all_dist_compare.gqcbm_dist / qcbm_sel_dist  (Fig. S8 and the GQCBM
    curve of Fig. 2 -- make_fig1_bimodal20_3way.py imports gqcbm_dist too)
  * make_fig1_bimodal20_3way.py is NOT importable (it plots at import time);
    its load_sel() and coarse_grain() are re-stated here verbatim in logic
    (rec[-1] of the JSON, sel_probs(n_qubits, n_layers, best_params);
    reshape(512, 2^20/512).sum(axis=1)).

Also writes a diagnostics JSON: forward KL(target || p) recomputed on the full
distribution for every series, the value the figure legend shows (stored
best_kl), and -- for the symmetric-pool checkpoint -- the KL under alternative
Pauli-term orderings inside each summed pool operator.

Usage (GPU node only):
  python -u scripts/export_fig2_s8_gpu.py --out_dir DIR [--tag TAG] [--only fig2|s8]

Fig. S8's QCBM-SEL+KL series reads the Table 2 run
(results/archive/qcbmpl_v3_kl_cardinality_10q.json, partial: hit walltime).
"""
import argparse, csv, json, math, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch, cudaq, cupy as cp

from make_all_dist_compare import gqcbm_dist, qcbm_sel_dist
from qcbm_pennylane import kl_of
from circuit_ops import circuit_kernel
from pool_select import select_pool
from gqe_single_target import build_target

cudaq.set_target('nvidia', option='fp64')


def kl_noeps(target, p):
    m = (target > 0) & (p > 0)
    return float(np.sum(target[m] * np.log(target[m] / p[m])))


def fmt(x):
    return f'{float(x):.10g}'


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  wrote {path}  ({len(rows)} rows)', flush=True)


def gqcbm_probs_ordered(ckpt_path, order):
    """Same as make_all_dist_compare.gqcbm_dist but with a chosen ordering of
    the Pauli terms inside each (possibly summed) pool operator.
    order in {'native', 'reversed', 'sorted', 'pool_build'}."""
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = d.get('config', {}) or {}
    nq = int(cfg['n_qubits'])
    pool = select_pool(nq, cfg)
    fc, fw = [], []
    for idx, ang in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
        if abs(float(ang)) < 1e-8:
            continue
        terms = []
        for term in float(ang) * pool[int(idx)]:
            c = term.get_coefficient().real
            w = term.to_string(nq).split(']')[-1].strip().ljust(nq, 'I')
            terms.append((c, w))
        if order == 'reversed':
            terms = terms[::-1]
        elif order == 'sorted':
            terms = sorted(terms, key=lambda t: t[1])
        elif order == 'pool_build':
            # construction order of build_op_pool_symmetric: qubit index
            # ascending (1-body), (i<j) lexicographic (2-body)
            def key(t):
                return [k for k, ch in enumerate(t[1]) if ch != 'I']
            terms = sorted(terms, key=key)
        for c, w in terms:
            fc.append(c); fw.append(cudaq.pauli_word(w))
    st = cudaq.get_state(circuit_kernel, nq, fc, fw)
    return cp.asnumpy(cp.abs(cp.array(st, dtype=cp.complex128)) ** 2)


def first_op_terms(ckpt_path, k=6):
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = d['config']; nq = int(cfg['n_qubits'])
    pool = select_pool(nq, cfg)
    idx = int(d['best_ops'][0])
    return [t.to_string(nq).split(']')[-1].strip().ljust(nq, 'I')
            for t in pool[idx]][:k]


def series_record(name, target, p, stored, source):
    return {'series': name, 'source': source,
            'sum_p': float(p.sum()),
            'kl_recomputed': kl_of(target, p),
            'kl_recomputed_noeps': kl_noeps(target, p),
            'kl_stored_legend': stored,
            'diff': kl_of(target, p) - stored}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--only', choices=('fig2', 's8'), default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    diag = {'tag': args.tag, 'python': sys.version,
            'cudaq_version': getattr(cudaq, '__version__', '?'),
            'torch_version': torch.__version__, 'figures': {}}
    print(diag, flush=True)

    if args.only != 's8':
        export_fig2(args, diag)
    if args.only != 'fig2':
        export_s8(args, diag)
    with open(os.path.join(args.out_dir, f'diag_{args.tag}.json'), 'w') as f:
        json.dump(diag, f, indent=1)
    print('DONE', flush=True)


def export_fig2(args, diag):
    # ── Fig. 2 : bimodal 20q ─────────────────────────────────────────────
    G2 = 'ckpts/gqe_bimodal_20q.pt'
    SKL2 = 'results/qcbm_n20m2_kl_bimodal_20q.json'
    SMMD2 = 'results/qcbm_n20m2_mmd_bimodal_20q.json'
    target, p_g, nq, kl_g = gqcbm_dist(G2)
    p_kl, kl_kl, _ = qcbm_sel_dist(SKL2)
    p_mmd, kl_mmd, _ = qcbm_sel_dist(SMMD2)
    # independent target check: rebuild from config
    cfg = torch.load(G2, map_location='cpu', weights_only=False)['config']
    t2 = build_target(cfg['target_kind'], 2 ** nq,
                      **json.loads(cfg.get('target_kw', '{}') or '{}'))
    recs = [series_record('gqcbm_kl', target, p_g, kl_g, G2),
            series_record('qcbm_sel_kl', target, p_kl, kl_kl, SKL2),
            series_record('qcbm_sel_mmd', target, p_mmd, kl_mmd, SMMD2)]
    N_BINS = 512
    bs = (2 ** nq) // N_BINS
    cg = lambda p: p.reshape(N_BINS, bs).sum(axis=1)
    tb, gb, kb, mb = cg(target), cg(p_g), cg(p_kl), cg(p_mmd)
    for r, pb in zip(recs, (gb, kb, mb)):
        r['kl_binned_512'] = kl_of(tb, pb)
        r['sum_binned'] = float(pb.sum())
    rows = [[i, fmt(tb[i]), fmt(gb[i]), fmt(kb[i]), fmt(mb[i])]
            for i in range(N_BINS)]
    write_csv(os.path.join(args.out_dir, 'fig2_bimodal20_3way.csv'),
              ['bin', 'target', 'gqcbm_kl', 'qcbm_sel_kl', 'qcbm_sel_mmd'],
              rows)
    np.save(os.path.join(args.out_dir, 'fig2_gqcbm_probs_full.npy'), p_g)
    diag['figures']['fig2'] = {
        'n_qubits': nq, 'bin_size': bs, 'sum_target': float(target.sum()),
        'target_maxabs_vs_rebuild': float(np.max(np.abs(target - t2))),
        'series': recs}
    print(json.dumps(diag['figures']['fig2'], indent=1), flush=True)


def export_s8(args, diag):
    # ── Fig. S8 : cardinality 10q ────────────────────────────────────────
    G8 = 'ckpts/gqe_sym_card_10q_full.pt'
    SMMD8 = 'results/qcbmpl_v3_mmd_cardinality_10q.json'
    SKL8 = 'results/archive/qcbmpl_v3_kl_cardinality_10q.json'  # Table 2 run
    target, p_g, nq, kl_g = gqcbm_dist(G8)
    p_mmd, kl_mmd, _ = qcbm_sel_dist(SMMD8)
    p_kl, kl_kl, _ = qcbm_sel_dist(SKL8, 'cardinality_10q')
    recs = [series_record('gqcbm_kl', target, p_g, kl_g, G8),
            series_record('qcbm_sel_mmd', target, p_mmd, kl_mmd, SMMD8),
            series_record('qcbm_sel_kl', target, p_kl, kl_kl, SKL8)]
    order_kls = {}
    for order in ('native', 'reversed', 'sorted', 'pool_build'):
        po = gqcbm_probs_ordered(G8, order)
        order_kls[order] = {'kl': kl_of(target, po),
                            'maxabs_vs_native_dist': float(np.max(np.abs(po - p_g)))}
    rows = [[x, fmt(target[x]), fmt(p_g[x]), fmt(p_mmd[x]), fmt(p_kl[x])]
            for x in range(2 ** nq)]
    write_csv(os.path.join(args.out_dir, 'fig_s8_cardinality10_dist.csv'),
              ['basis_index', 'target', 'gqcbm_kl', 'qcbm_sel_mmd',
               'qcbm_sel_kl'], rows)
    np.save(os.path.join(args.out_dir, 'figs8_gqcbm_probs_full.npy'), p_g)
    diag['figures']['fig_s8'] = {
        'n_qubits': nq, 'sum_target': float(target.sum()), 'series': recs,
        'gqcbm_term_order_kl': order_kls,
        'first_op_native_term_order': first_op_terms(G8)}
    print(json.dumps(diag['figures']['fig_s8'], indent=1), flush=True)


if __name__ == '__main__':
    main()
