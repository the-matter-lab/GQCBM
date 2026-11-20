"""Uniform circuit-statistics counter for GQCBM and QCBM-SEL.

Same decomposition convention for both methods (native gate set: {R_x, R_y, R_z, CX}):

  Pauli rotation exp(-i theta P) of weight w (= number of non-identity qubits)
  via the standard ladder:
      CX count   = 2 (w - 1)             # ladder up + down
      depth      = 2 w - 1                # sequential ladder
      single ops = 1                      # one central R_z (basis change ignored)

  For a token whose pool element is a sum of pairwise-commuting Paulis,
  Sigma_i P_i, we apply the terms independently — CX and depth ADD over terms
  (sequential composition; commuting parallelisation is a lower bound and is
  NOT used here, for fair comparison with the QCBM-SEL ring).

  QCBM-SEL: L layers, each layer = n Rot gates + n CNOTs in a ring.
      CX count = n * L
      depth    = (3 + n) * L              # 3 for Rot (parallel across qubits),
                                          #   n for sequential CX ring
      single ops = 3 n L

Outputs results/circuit_stats.json — used by §3.2 / §3.4 figures.
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --sel-only: recompute only the QCBM-SEL rows (pure JSON reads, CPU) and keep the
# GQCBM / GQCBM+MMD rows already in results/circuit_stats.json (those need CUDA-Q).
SEL_ONLY = '--sel-only' in sys.argv
if not SEL_ONLY:
    import torch, cudaq, cupy as cp
    cudaq.set_target('nvidia', option='fp64')
    from gqe_single_target import build_target
    from circuit_ops import circuit_kernel
    from qcbm_pennylane import kl_of
    from pool_select import select_pool  # shared; re-exported for scripts/eval_uniform.py


def pauli_weight(term, nq):
    w = term.to_string(nq).split(']')[-1].strip().ljust(nq, 'I')
    return sum(1 for c in w if c != 'I')


def gqcbm_stats(ckpt_path):
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = d.get('config', {}) or {}
    if 'target_arr' in d:
        nq = int(round(math.log2(len(d['target_arr']))))
    else:
        nq = int(cfg['n_qubits'])
    pool = select_pool(nq, cfg)
    cx = depth = single = 0
    n_used = 0
    for idx, ang in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
        if abs(float(ang)) < 1e-8:
            continue
        n_used += 1
        for term in pool[int(idx)]:
            w = pauli_weight(term, nq)
            if w == 0:
                continue
            if w == 1:
                single += 1
                depth += 1
            else:
                cx += 2 * (w - 1)
                depth += 2 * w - 1
                single += 1
    if cfg.get('loss_metric') == 'mmd':
        # best_kl in MMD-trained checkpoints tracks the minimum KL glimpsed
        # across any sampled circuit during training, not necessarily that
        # of the saved best_ops/best_theta circuit -- recompute fresh from
        # the actual saved circuit (same fix as Table 1/2's GQCBM+MMD
        # column).
        if 'target_arr' in d:
            target = (d['target_arr'].numpy() if hasattr(d['target_arr'], 'numpy')
                      else np.array(d['target_arr']))
        else:
            tkw = json.loads(cfg.get('target_kw', '{}') or '{}')
            target = build_target(cfg['target_kind'], 2 ** nq, **tkw)
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
        best_kl = kl_of(target, probs)
    else:
        best_kl = float(d.get('best_kl', float('nan')))
    return {'cx': cx, 'depth': depth, 'single': single,
            'tokens_used': n_used, 'best_kl': best_kl}


def qcbm_sel_stats(json_path):
    d = json.load(open(json_path))
    rec = d[-1] if isinstance(d, list) else d
    n, L = rec['n_qubits'], rec['n_layers']
    return {'cx': n * L, 'depth': (3 + n) * L, 'single': 3 * n * L,
            'n_layers': L,
            'best_kl': float(rec.get('best_kl', float('nan')))}


# ─────────────────────────────────────────────────────────────────────────────
# Registry — best ckpt / json per (target, method)
# QCBM-SEL+KL rows point at the exact-gradient re-run (results/sel_kl_exact_gb200/,
# cell map in its README.md). n=20 uses the 4,000-step protocol (t1_n20_aligned);
# its JSON best_kl is the end-of-run value (~4,400 real steps after preemptions).
# ─────────────────────────────────────────────────────────────────────────────
ENTRIES = [
    # ───── Continuous (Table 1: n=10/15/20 — best ckpt per cell) ──────
    ('gauss_10q',     'GQCBM',        10, 'ckpts/gqe_q10_small_v2.pt'),
    ('gauss_10q',     'GQCBM+MMD',    10, 'ckpts/gqe_mmd10_gauss_10q.pt'),
    ('gauss_10q',     'QCBM-SEL+KL',  10, 
                     'results/sel_kl_exact_gb200/runs/t1_n10/job_8803242/results/qcbm_aligned_kl_gauss_10q.json'),
    ('gauss_10q',     'QCBM-SEL+MMD', 10, 'results/sweep_L4_mmd_gauss_10q.json'),
    ('gauss_15q',     'GQCBM',        15, 'ckpts/gqe_q15_v4.pt'),
    ('gauss_15q',     'GQCBM+MMD',    15, 'ckpts/gqe_mmd15_gauss_15q.pt'),
    ('gauss_15q',     'QCBM-SEL+KL',  15, 
                     'results/sel_kl_exact_gb200/runs/t1_n15/job_8828095/results/qcbm_aligned_kl_gauss_15q_s42.json'),
    ('gauss_15q',     'QCBM-SEL+MMD', 15, 'results/sweep_L4_mmd_gauss_15q_s42.json'),
    ('gauss_20q',     'GQCBM',        20, 'ckpts/gqe_gauss_20q.pt'),
    ('gauss_20q',     'GQCBM+MMD',    20, 'ckpts/gqe_mmd20_gauss_20q.pt'),
    ('gauss_20q',     'QCBM-SEL+KL',  20, 
                     'results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/qcbm_aligned_kl_gauss_20q.json'),
    ('gauss_20q',     'QCBM-SEL+MMD', 20, 'results/qcbm_n20m2_mmd_gauss_20q.json'),
    ('bimodal_10q',   'GQCBM',        10, 'ckpts/gqe_bimodal_10q_g30.pt'),
    ('bimodal_10q',   'GQCBM+MMD',    10, 'ckpts/gqe_mmd10_bimodal_10q.pt'),
    ('bimodal_10q',   'QCBM-SEL+KL',  10, 
                     'results/sel_kl_exact_gb200/runs/t1_n10/job_8803242/results/qcbm_aligned_kl_bimodal_10q.json'),
    ('bimodal_10q',   'QCBM-SEL+MMD', 10, 'results/sweep_L4_mmd_bimodal_10q.json'),
    ('bimodal_15q',   'GQCBM',        15, 'ckpts/gqe_bimodal_15q_g30.pt'),
    ('bimodal_15q',   'GQCBM+MMD',    15, 'ckpts/gqe_mmd15_bimodal_15q.pt'),
    ('bimodal_15q',   'QCBM-SEL+KL',  15, 
                     'results/sel_kl_exact_gb200/runs/t1_n15/job_8828095/results/qcbm_aligned_kl_bimodal_15q.json'),
    ('bimodal_15q',   'QCBM-SEL+MMD', 15, 'results/sweep_L4_mmd_bimodal_15q.json'),
    ('bimodal_20q',   'GQCBM',        20, 'ckpts/gqe_bimodal_20q.pt'),
    ('bimodal_20q',   'GQCBM+MMD',    20, 'ckpts/gqe_mmd20_bimodal_20q.pt'),
    ('bimodal_20q',   'QCBM-SEL+KL',  20, 
                     'results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/qcbm_aligned_kl_bimodal_20q.json'),
    ('bimodal_20q',   'QCBM-SEL+MMD', 20, 'results/qcbm_n20m2_mmd_bimodal_20q.json'),
    ('exp_10q',       'GQCBM',        10, 'ckpts/gqe_exp_10q_g20.pt'),
    ('exp_10q',       'GQCBM+MMD',    10, 'ckpts/gqe_mmd10_exp_10q.pt'),
    ('exp_10q',       'QCBM-SEL+KL',  10, 
                     'results/sel_kl_exact_gb200/runs/t1_n10/job_8803242/results/qcbm_aligned_kl_exp_10q.json'),
    ('exp_10q',       'QCBM-SEL+MMD', 10, 'results/sweep_L4_mmd_exp_10q.json'),
    ('exp_15q',       'GQCBM',        15, 'ckpts/gqe_exp_15q_g30.pt'),
    ('exp_15q',       'GQCBM+MMD',    15, 'ckpts/gqe_mmd15_exp_15q.pt'),
    ('exp_15q',       'QCBM-SEL+KL',  15, 
                     'results/sel_kl_exact_gb200/runs/t1_n15/job_8828095/results/qcbm_aligned_kl_exp_15q.json'),
    ('exp_15q',       'QCBM-SEL+MMD', 15, 'results/sweep_L4_mmd_exp_15q.json'),
    ('exp_20q',       'GQCBM',        20, 'ckpts/gqe_exp_20q.pt'),
    ('exp_20q',       'GQCBM+MMD',    20, 'ckpts/gqe_mmd20_exp_20q.pt'),
    ('exp_20q',       'QCBM-SEL+KL',  20, 
                     'results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/qcbm_aligned_kl_exp_20q.json'),
    ('exp_20q',       'QCBM-SEL+MMD', 20, 'results/qcbm_n20m2_mmd_exp_20q.json'),
    ('lognormal_10q', 'GQCBM',        10, 'ckpts/gqe_lognormal_10q.pt'),
    ('lognormal_10q', 'GQCBM+MMD',    10, 'ckpts/gqe_mmd10_lognormal_10q.pt'),
    ('lognormal_10q', 'QCBM-SEL+KL',  10, 
                     'results/sel_kl_exact_gb200/runs/t1_n10/job_8803242/results/qcbm_aligned_kl_lognormal_10q.json'),
    ('lognormal_10q', 'QCBM-SEL+MMD', 10, 'results/sweep_L4_mmd_lognormal_10q.json'),
    ('lognormal_15q', 'GQCBM',        15, 'ckpts/gqe_lognormal_15q_g50.pt'),
    ('lognormal_15q', 'GQCBM+MMD',    15, 'ckpts/gqe_mmd15_lognormal_15q.pt'),
    ('lognormal_15q', 'QCBM-SEL+KL',  15, 
                     'results/sel_kl_exact_gb200/runs/t1_n15/job_8828095/results/qcbm_aligned_kl_lognormal_15q.json'),
    ('lognormal_15q', 'QCBM-SEL+MMD', 15, 'results/sweep_L4_mmd_lognormal_15q.json'),
    ('lognormal_20q', 'GQCBM',        20, 'ckpts/gqe_lognormal_20q.pt'),
    ('lognormal_20q', 'GQCBM+MMD',    20, 'ckpts/gqe_mmd20_lognormal_20q.pt'),
    ('lognormal_20q', 'QCBM-SEL+KL',  20, 
                     'results/sel_kl_exact_gb200/runs/t1_n20_aligned/job_9015307/results/qcbm_aligned_kl_lognormal_20q.json'),
    ('lognormal_20q', 'QCBM-SEL+MMD', 20, 'results/qcbm_n20m2_mmd_lognormal_20q.json'),
    # ───── Discrete ──────
    ('ghz5_5q',       'GQCBM',        5,  'ckpts/gqe_ghz5.pt'),
    ('ghz5_5q',       'GQCBM+MMD',    5,  'ckpts/gqe_mmdloss2_ghz5_5q.pt'),
    ('ghz5_5q',       'QCBM-SEL+KL',  5,  
                     'results/sel_kl_exact_gb200/runs/t2_small/job_8803245/results/qcbm_cvg_ghz5_5q.json'),
    ('ghz5_5q',       'QCBM-SEL+MMD', 5,  'results/archive/qcbmpl_v2_mmd_ghz5_5q.json'),
    ('bas4_4q',       'GQCBM',        4,  'ckpts/extras_bas4.pt'),
    ('bas4_4q',       'GQCBM+MMD',    4,  'ckpts/gqe_mmdloss2_bas4_4q.pt'),
    ('bas4_4q',       'QCBM-SEL+KL',  4,  
                     'results/sel_kl_exact_gb200/runs/t2_small/job_8803245/results/qcbmpl_v2_kl_bas4_4q.json'),
    ('bas4_4q',       'QCBM-SEL+MMD', 4,  'results/archive/qcbmpl_v2_mmd_bas4_4q.json'),
    ('ghz10_10q',     'GQCBM',        10, 'ckpts/gqe_hamMMD_ghz10_10q.pt'),
    ('ghz10_10q',     'GQCBM+MMD',    10, 'ckpts/gqe_hamMMD_ghz10_10q.pt'),
    ('ghz10_10q',     'QCBM-SEL+KL',  10, 
                     'results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/qcbmpl_v3_kl_ghz10_10q.json'),
    ('ghz10_10q',     'QCBM-SEL+MMD', 10, 'results/qcbmpl_v3_mmd_ghz10_10q.json'),
    ('cardinality_5q','GQCBM',        5,  'ckpts/gqe_sym_card_5q.pt'),
    ('cardinality_5q','GQCBM+MMD',    5,  'ckpts/gqe_symMMD_card_5q.pt'),
    ('cardinality_5q','QCBM-SEL+KL',  5,  
                     'results/sel_kl_exact_gb200/runs/t2_small/job_8803245/results/qcbmpl_v2_kl_cardinality_5q.json'),
    ('cardinality_5q','QCBM-SEL+MMD', 5,  'results/archive/qcbmpl_v2_mmd_cardinality_5q.json'),
    ('cardinality_10q','GQCBM',       10, 'ckpts/gqe_sym_card_10q_full.pt'),
    ('cardinality_10q','GQCBM+MMD',   10, 'ckpts/gqe_symMMD_card_10q.pt'),
    ('cardinality_10q','QCBM-SEL+KL', 10, 
                     'results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/qcbmpl_v3_kl_cardinality_10q.json'),
    ('cardinality_10q','QCBM-SEL+MMD',10, 'results/qcbmpl_v3_mmd_cardinality_10q.json'),
    ('bas9_9q',       'GQCBM',        9,  'ckpts/gqe_bas9_v2_ent.pt'),
    ('bas9_9q',       'GQCBM+MMD',    9,  'ckpts/gqe_symMMD_bas9_9q.pt'),
    ('bas9_9q',       'QCBM-SEL+KL',  9,  
                     'results/sel_kl_exact_gb200/runs/t2_medium/job_8806630/results/qcbmpl_v3_kl_bas9_9q.json'),
    ('bas9_9q',       'QCBM-SEL+MMD', 9,  'results/qcbmpl_v3_mmd_bas9_9q.json'),
]


def main():
    out = []
    kept = {}
    if SEL_ONLY:
        for d in json.load(open('results/circuit_stats.json')):
            if d['method'] in ('GQCBM', 'GQCBM+MMD'):
                kept[(d['target'], d['method'])] = d
    for label, method, nq, src in ENTRIES:
        if SEL_ONLY and method in ('GQCBM', 'GQCBM+MMD'):
            if (label, method) in kept:
                out.append(kept[(label, method)])
                print(f'  {label:18s} {method:14s} kept from results/circuit_stats.json')
            else:
                print(f'  SKIP {label} {method}: not in results/circuit_stats.json')
            continue
        if not os.path.exists(src):
            print(f'  SKIP {label} {method}: {src} missing')
            continue
        try:
            if method in ('GQCBM', 'GQCBM+MMD'):
                s = gqcbm_stats(src)
            else:
                s = qcbm_sel_stats(src)
            entry = {'target': label, 'method': method, 'n_qubits': nq,
                     'source': src, **s}
            out.append(entry)
            kl_s = (f'{s["best_kl"]:.4f}' if s['best_kl'] == s['best_kl']
                    else 'n/a')
            print(f'  {label:18s} {method:14s} n={nq:>2d}  '
                  f'CX={s["cx"]:>5d}  depth={s["depth"]:>5d}  '
                  f'single={s["single"]:>5d}  KL={kl_s}')
        except Exception as e:
            print(f'  ERR {label} {method}: {type(e).__name__}: {e}')

    os.makedirs('results', exist_ok=True)
    with open('results/circuit_stats.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {len(out)} entries → results/circuit_stats.json')


if __name__ == '__main__':
    main()
