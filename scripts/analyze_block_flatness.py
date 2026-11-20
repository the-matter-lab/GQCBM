"""Why the n=20 GQCBM+KL distributions look step-like (Fig. 2, Discussion).

For each n=20 continuous target, re-simulates the Table 1 GQCBM+KL circuit on
CPU (numpy statevector, same kernel as scripts/export_level2_cpu.py; each KL is
checked against the checkpoint's best_kl) and reports:

  * flat_block_bits: the largest k such that p is flat (mean relative spread
    < 2 %) inside blocks of 2^k consecutive basis states;
  * kl_total = KL(target || p);
  * kl_flatness = KL(target || target averaged over those blocks), the KL any
    distribution that is flat on such blocks must pay;
  * per-qubit maximum |theta| of the X/Y terms acting on that qubit (pi = global
    phase, pi/2 = bit flip; only intermediate angles reshape the distribution);
  * for the exponential target, the per-bit tilt angle 2^(q-20) rad that an
    exact product-state fit needs, versus the smallest angle in the alphabet.

All four circuits reproduce their stored best_kl on CPU (checked; the field
`reproduces_stored_kl` records it).

    python scripts/analyze_block_flatness.py --ckpt-dir ckpts

Writes results/analysis/n20_block_flatness.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, REPO)
import export_level2_cpu as E  # noqa: E402
from make_fig_continuous_grid import build_target, FAMILIES  # noqa: E402

CKPTS = {'gauss': 'gqe_gauss_20q', 'bimodal': 'gqe_bimodal_20q',
         'exp': 'gqe_exp_20q', 'lognormal': 'gqe_lognormal_20q'}
N_QUBITS = 20


def kl(a, b):
    return E.kl(a, b)  # same convention as the stored best_kl (eps 1e-10)


def kl_exact(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='ckpts')
    args = ap.parse_args()
    n, N = N_QUBITS, 2 ** N_QUBITS
    fam = {k: (kind, kw) for _, k, kind, kw in FAMILIES}
    out = {}
    for key, name in CKPTS.items():
        d = torch.load(os.path.join(args.ckpt_dir, name + '.pt'),
                       map_location='cpu', weights_only=False)
        cfg = d['config']
        pool = E.select_pool(n, cfg)
        max_theta = [0.0] * n
        psi = np.full([2] * n, 1 / math.sqrt(N), dtype=complex)
        for i, th in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
            if abs(th) < 1e-8:
                continue
            for term in pool[int(i)]:
                for q, p in term.items():
                    if p in 'XY':
                        max_theta[q] = max(max_theta[q], abs(th))
                psi = E.exp_pauli(psi, term, float(th))
        p = E.probs_little_endian(psi)
        t = build_target(fam[key][0], N, fam[key][1])
        spread = {k: float(np.mean(p.reshape(-1, 2 ** k).std(1) / p.reshape(-1, 2 ** k).mean(1)))
                  for k in range(1, n)}
        flat = max([k for k, v in spread.items() if v < 0.02], default=0)
        tb = t.reshape(-1, 2 ** flat).mean(1).repeat(2 ** flat) if flat else t
        k_tot, k_flat = kl(t, p), kl_exact(t, tb)
        stored = float(d['best_kl'])
        rec = {'checkpoint': name, 'stored_best_kl': stored, 'kl_total': k_tot,
               'reproduces_stored_kl': math.isclose(k_tot, stored, rel_tol=1e-3),
               'flat_block_bits': flat, 'flat_block_states': 2 ** flat,
               'kl_flatness': k_flat,
               'kl_flatness_share': (k_flat / k_tot) if k_tot > 0 else None,
               'mean_rel_spread_by_block_bits': spread,
               'max_xy_theta_per_qubit_lsb_first': max_theta,
               'n_gates': cfg.get('n_gates'), 'angle_bins_K': cfg.get('angle_bins_K')}
        if key == 'exp':
            scale = 0.25 * N  # build_target default
            rec['exp_needed_tilt_rad_per_bit'] = {q: 2 ** q / scale / 4 for q in range(n)}
            rec['smallest_alphabet_angle_rad'] = min(abs(x) for x in d['best_theta'].tolist()
                                                     if abs(x) > 1e-8)
        out[key] = rec
        print(f'{name}: KL {k_tot:.4f} (stored {stored:.4f}); flat on 2^{flat} blocks; '
              f'flatness KL {k_flat:.4f} ({100 * k_flat / k_tot:.0f} %)')
    os.makedirs('results/analysis', exist_ok=True)
    path = 'results/analysis/n20_block_flatness.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
