"""Exact forward-KL gradient for the QCBM-SEL ansatz (replaces the legacy parameter-shift-on-KL gradient).

`qcbm_pennylane.kl_grad` applies the ±π/2 parameter-shift rule to the KL
*value*, which is only exact for first-harmonic quantities. The Born
probabilities p_z(θ_i) are first-harmonic in each SEL parameter (every
parameter appears in exactly one rotation gate, generator P with P²=I), but
KL(p) is nonlinear in p, so shifting KL directly returns a filtered
direction, not ∂KL/∂θ. The correct construction mirrors `mmd_grad`: obtain
∂p/∂θ_i exactly from the shift, then apply the chain rule with
∂KL/∂p_z = −π_z / p_z.

Cost is identical to the old implementation (1 + 2·n_params state-vector
evaluations per call). This module does NOT modify qcbm_pennylane.py; it
imports `circuit_probs` and `kl_of` from it so the two can be compared.

Self-test (the pilot job scripts/run_pilot_sel_kl_exact_gb200.sh runs it):
compares kl_grad_exact against a central finite difference of kl_of, and
reports the discrepancy of the legacy kl_grad for reference.

Result (SLURM job 301916, 2026-08-25, Gaussian target, random θ, seed 0):
    n=4 L=2 (24 params): max|exact-fd| = 6.0e-10, max|legacy-fd| = 1.35,
                         cos(legacy, fd) = 0.68, |grad| = 3.4
    n=6 L=2 (36 params): max|exact-fd| = 9.4e-11, max|legacy-fd| = 0.53,
                         cos(legacy, fd) = 0.51, |grad| = 1.3
i.e. the new implementation is exact to floating-point precision, while the
legacy kl_grad differs from the true gradient by O(1) with only ~0.5-0.7
cosine similarity -- a materially wrong descent direction, not a small bias.
"""
import math
import numpy as np

try:
    import cudaq
    try:
        cudaq.set_target('nvidia', option='fp64')
    except Exception:
        pass
except ImportError:  # allows importing the pure-numpy parts without cudaq
    cudaq = None

from qcbm_pennylane import circuit_probs, kl_of, kl_grad as kl_grad_legacy


def kl_grad_exact(n_qubits, n_layers, params, target):
    """Return (grad, p): exact ∂KL(target||p(θ))/∂θ and p(θ)."""
    p = circuit_probs(n_qubits, n_layers, params)
    mask = target > 0
    w = np.zeros_like(p)
    w[mask] = -target[mask] / (p[mask] + 1e-10)      # dKL/dp, same smoothing as kl_of
    n = len(params)
    grad = np.zeros(n)
    shift = math.pi / 2
    for i in range(n):
        pp = params.copy(); pp[i] += shift
        pm = params.copy(); pm[i] -= shift
        dp = (circuit_probs(n_qubits, n_layers, pp) -
              circuit_probs(n_qubits, n_layers, pm)) / 2.0   # exact dp/dθ_i
        grad[i] = w @ dp
    return grad, p


def _finite_difference(n_qubits, n_layers, params, target, eps=1e-5):
    n = len(params)
    g = np.zeros(n)
    for i in range(n):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (kl_of(target, circuit_probs(n_qubits, n_layers, pp)) -
                kl_of(target, circuit_probs(n_qubits, n_layers, pm))) / (2 * eps)
    return g


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for n_qubits, n_layers in [(4, 2), (6, 2)]:
        n_states = 2 ** n_qubits
        idx = np.arange(n_states)
        target = np.exp(-((idx - 0.5 * n_states) ** 2) / (2 * (0.15 * n_states) ** 2))
        target /= target.sum()
        params = rng.uniform(0, 2 * math.pi, n_layers * n_qubits * 3)

        g_exact, _ = kl_grad_exact(n_qubits, n_layers, params, target)
        g_fd = _finite_difference(n_qubits, n_layers, params, target)
        g_legacy, _ = kl_grad_legacy(n_qubits, n_layers, params, target)

        err_exact = np.max(np.abs(g_exact - g_fd))
        err_legacy = np.max(np.abs(g_legacy - g_fd))
        cos_legacy = float(g_legacy @ g_fd / (np.linalg.norm(g_legacy) * np.linalg.norm(g_fd) + 1e-30))
        print(f'n={n_qubits} L={n_layers} params={len(params)}  '
              f'|grad_fd|={np.linalg.norm(g_fd):.4e}  '
              f'max|exact-fd|={err_exact:.3e}  '
              f'max|legacy-fd|={err_legacy:.3e}  cos(legacy,fd)={cos_legacy:.4f}')
    print('DONE')
