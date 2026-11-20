"""Validate fast_mmd.py (O(N log N)/O(N)) against the dense O(N^2)/O(N^2)
MMDEvaluator reference for every kernel geometry, before trusting the fast
path at n_qubits=20.

DenseMMDEvaluator below is copied verbatim from gqe_single_target.py:543-604
(the MMDEvaluator class), not reimplemented from scratch -- this file avoids
importing gqe_single_target.py directly only because that module does an
unconditional `import cudaq, cupy` at the top, which is unavailable in the
login-node system-python environment this test runs in (see project
on HPC systems CUDA-Q needs a separate venv). The class body itself
is pure numpy and has no cudaq/cupy dependency, so copying it here is safe
and exact -- any future change to the real MMDEvaluator should be mirrored
here or this test stops proving what it claims to prove.
"""
import math
import time
import numpy as np

from fast_mmd import mmd2_fast, FastMMDEvaluator


class DenseMMDEvaluator:
    """Verbatim copy of gqe_single_target.py:543-604 MMDEvaluator."""

    def __init__(self, target_arr, sigmas, kernel='index', grid_shape=None):
        n = len(target_arr)
        if kernel == 'index':
            idx = np.arange(n, dtype=np.float64)
            dist_sq = (idx[:, None] - idx[None, :]) ** 2
        elif kernel == 'hamming':
            idx = np.arange(n, dtype=np.uint32)
            xor = idx[:, None] ^ idx[None, :]
            ham = np.zeros_like(xor, dtype=np.float64)
            v = xor.copy()
            while v.any():
                ham += (v & 1).astype(np.float64)
                v >>= 1
            dist_sq = ham ** 2
        elif kernel == 'grid':
            if grid_shape is None:
                raise ValueError("grid kernel needs grid_shape=(R,C)")
            R, C = grid_shape
            nq = int(round(math.log2(n)))
            if R * C != nq:
                raise ValueError(f"grid_shape {R}x{C} != n_qubits {nq}")
            idx = np.arange(n, dtype=np.uint32)
            rows = np.stack([
                (idx >> (r * C)) & ((1 << C) - 1) for r in range(R)
            ], axis=1)
            cols = np.zeros((n, C), dtype=np.uint32)
            for c in range(C):
                for r in range(R):
                    cols[:, c] |= ((idx >> (r * C + c)) & 1) << r
            row_dist = (rows[:, None, :] != rows[None, :, :]).sum(axis=2)
            col_dist = (cols[:, None, :] != cols[None, :, :]).sum(axis=2)
            dist_sq = ((row_dist + col_dist).astype(np.float64)) ** 2
        else:
            raise ValueError(f"unknown MMD kernel: {kernel}")
        K = np.zeros_like(dist_sq)
        for s in sigmas:
            K += np.exp(-dist_sq / (2.0 * s * s))
        self.K = K
        self.target = target_arr
        self.sigmas = list(sigmas)
        self.kernel = kernel

    def __call__(self, circuit_probs):
        diff = self.target - circuit_probs
        return float(diff @ self.K @ diff)


def random_prob(n_states, rng):
    v = rng.random(n_states) + 0.01
    return v / v.sum()


def check(label, n_qubits, kernel, sigmas, grid_shape=None, seed=0):
    n_states = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    target = random_prob(n_states, rng)
    probs = random_prob(n_states, rng)

    dense = DenseMMDEvaluator(target, sigmas, kernel=kernel, grid_shape=grid_shape)
    ref = dense(probs)

    fast = mmd2_fast(target, probs, sigmas, kernel=kernel, grid_shape=grid_shape)
    fast_cls = FastMMDEvaluator(target, sigmas, kernel=kernel, grid_shape=grid_shape)(probs)

    abs_err = abs(fast - ref)
    rel_err = abs_err / max(abs(ref), 1e-300)
    ok = abs_err < 1e-8 or rel_err < 1e-8
    ok_cls = abs(fast_cls - ref) < 1e-8 or abs(fast_cls - ref) / max(abs(ref), 1e-300) < 1e-8
    status = "PASS" if (ok and ok_cls) else "FAIL"
    print(f"[{status}] {label}: n_qubits={n_qubits} kernel={kernel} sigmas={sigmas} "
          f"dense={ref:.10e} fast_fn={fast:.10e} fast_cls={fast_cls:.10e} "
          f"abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    return ok and ok_cls


if __name__ == '__main__':
    all_ok = True
    for nq in (4, 6, 8):
        all_ok &= check(f"index n={nq} single-sigma", nq, 'index', [3.0])
        all_ok &= check(f"index n={nq} multi-sigma", nq, 'index', [0.5, 5.0, 40.0])
        all_ok &= check(f"hamming n={nq} single-sigma", nq, 'hamming', [1.0])
        all_ok &= check(f"hamming n={nq} multi-sigma", nq, 'hamming', [0.5, 1.5, 3.0])

    # grid: needs n_qubits = R*C, use 2x2 (n=4) and 3x3 (n=9)
    all_ok &= check("grid 2x2 single-sigma", 4, 'grid', [1.0], grid_shape=(2, 2))
    all_ok &= check("grid 2x2 multi-sigma", 4, 'grid', [0.5, 2.0], grid_shape=(2, 2))
    all_ok &= check("grid 3x3 single-sigma", 9, 'grid', [1.0], grid_shape=(3, 3))
    all_ok &= check("grid 3x3 multi-sigma", 9, 'grid', [0.5, 2.0, 5.0], grid_shape=(3, 3))

    # a couple of extra seeds on the largest dense-feasible size to be sure
    for seed in (1, 2, 3):
        all_ok &= check(f"index n=10 seed={seed}", 10, 'index', [1.0, 20.0], seed=seed)
        all_ok &= check(f"hamming n=10 seed={seed}", 10, 'hamming', [2.0], seed=seed)

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")

    # Benchmark at n_qubits=20 (no dense reference possible -- 8.8TB), just
    # confirm it runs and report time/memory-relevant sizes.
    print()
    print("=== n_qubits=20 feasibility check (fast path only, no dense ref) ===")
    n_qubits = 20
    n_states = 2 ** n_qubits
    rng = np.random.default_rng(42)
    target = random_prob(n_states, rng)
    probs = random_prob(n_states, rng)
    for kernel, sigmas, grid_shape in [
        ('index', [0.25, 60.0], None),
        ('hamming', [1.0, 3.0], None),
    ]:
        t0 = time.time()
        val = mmd2_fast(target, probs, sigmas, kernel=kernel, grid_shape=grid_shape)
        dt = time.time() - t0
        print(f"kernel={kernel} sigmas={sigmas}: mmd2={val:.6e}  wall={dt:.3f}s")
