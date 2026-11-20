"""Random-circuit baseline: sample N random circuits at fixed gate count G
using the same (operator pool, discrete angle alphabet) as GQCBM, evaluate
KL against the target via the same CUDAQ kernel, report best-of-N.

This is the apples-to-apples "no learning" control showing that GQCBM's
autoregressive policy genuinely contributes — not the operator pool alone.
"""
import argparse, json, math, os, sys, time
import numpy as np
import cudaq, cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqe_single_target import build_target
from circuit_ops import build_op_pool, circuit_kernel

cudaq.set_target('nvidia', option='fp64')

# 27-value angle alphabet (K=12), identical to main GQCBM experiments
ANGLE_VALUES = np.array([
    -3.1416, -1.5708, -0.7854, -0.3927, -0.1963, -0.0982, -0.0491,
    -0.0245, -0.0123, -0.0061, -0.0031, -0.0015, -0.0008, 0.0,
     0.0008,  0.0015,  0.0031,  0.0061,  0.0123,  0.0245,  0.0491,
     0.0982,  0.1963,  0.3927,  0.7854,  1.5708,  3.1416])


def kl_of(target, probs):
    m = target > 0
    return float(np.sum(target[m] * np.log(target[m] / (probs[m] + 1e-10))))


def probs_of(n_qubits, op_pool, ops, angles):
    fc, fw = [], []
    for idx, ang in zip(ops, angles):
        if abs(float(ang)) < 1e-8:
            continue
        scaled = float(ang) * op_pool[int(idx)]
        for term in scaled:
            coeff = term.get_coefficient().real
            w = term.to_string(n_qubits).split(']')[-1].strip().ljust(n_qubits, 'I')
            fc.append(coeff); fw.append(cudaq.pauli_word(w))
    if not fc:
        return np.ones(2 ** n_qubits) / (2 ** n_qubits)
    state = cudaq.get_state(circuit_kernel, n_qubits, fc, fw)
    return cp.asnumpy(cp.abs(cp.array(state, dtype=cp.complex128)) ** 2)


def run(n_qubits, kind, kw, G, N, seed=42):
    rng = np.random.default_rng(seed)
    pool = build_op_pool(n_qubits)
    n_pool = len(pool)
    target = build_target(kind, 2 ** n_qubits, **kw)
    best_kl, best_mean = float('inf'), 0.0
    kls = []
    t0 = time.time()
    log_every = max(1, N // 10)
    for s in range(N):
        ops    = rng.integers(0, n_pool, size=G)
        angles = rng.choice(ANGLE_VALUES, size=G)
        p = probs_of(n_qubits, pool, ops, angles)
        kl = kl_of(target, p)
        kls.append(kl)
        if kl < best_kl: best_kl = kl
        if (s + 1) % log_every == 0:
            print(f"  n={n_qubits} N={s+1}/{N}  best={best_kl:.4f}  "
                  f"mean={np.mean(kls):.4f}  t={time.time()-t0:.0f}s",
                  flush=True)
    return {'n_qubits': n_qubits, 'kind': kind, 'kw': kw, 'G': G, 'N': N,
            'best_kl': best_kl, 'mean_kl': float(np.mean(kls)),
            'std_kl': float(np.std(kls)),
            'elapsed': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_json', default='results/random_baseline.json')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)

    # Match the headline scaling Gaussian targets (n=5, 10, 15)
    cases = [
        (5,  'gaussian', {'center': 0.5, 'sigma': 0.20}, 20),
        (10, 'gaussian', {'center': 0.5, 'sigma': 0.10}, 30),
        (15, 'gaussian', {'center': 0.5, 'sigma': 0.10}, 40),
    ]
    Ns = [100, 1000, 10000]

    results = []
    for (n, kind, kw, G) in cases:
        for N in Ns:
            print(f"\n=== n={n}, G={G}, kind={kind}, N={N} ===")
            r = run(n, kind, kw, G, N, seed=42)
            results.append(r)
            json.dump(results, open(args.out_json, 'w'), indent=2)

    print(f"\n>>> Random-circuit baseline summary (vs GQCBM): ")
    print(f"{'n':>3} {'G':>3} {'best-of-N':>11} {'best KL':>10} {'GQCBM':>10}")
    GQCBM = {5: 0.0016, 10: 0.0064, 15: 0.0055}
    for r in results:
        ng = r['n_qubits']
        print(f"{ng:>3} {r['G']:>3} {r['N']:>11} {r['best_kl']:>10.4f} "
              f"{GQCBM[ng]:>10.4f}")


if __name__ == '__main__':
    main()
