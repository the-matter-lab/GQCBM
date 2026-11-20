"""O(N log N)-time, O(N)-memory MMD^2 for the translation-invariant kernels
already used by MMDEvaluator in gqe_single_target.py.

Why this exists
----------------
MMDEvaluator builds a dense N x N kernel matrix (N = 2^n_qubits) and computes
MMD^2 = (target-probs)^T K (target-probs) as a dense quadratic form. At
n_qubits=20 that matrix is 2^20 x 2^20 x 8 bytes ~= 8.8 TB -- infeasible on a
single node, which is why the paper currently reports "N/A" for MMD at n=20.

All three kernel geometries implemented by MMDEvaluator turn out to be
translation-invariant, i.e. K[i,j] depends only on a "distance" between i and
j that is itself invariant under a group shift:
  - kernel='index':   K[i,j] = f(i - j)            (shift group: (Z, +))
  - kernel='hamming': K[i,j] = f(popcount(i xor j)) (shift group: (Z_2^n, xor))
  - kernel='grid':    dist_sq(i,j) is *also* a pure function of (i xor j) --
    row_dist(i,j) depends only on which row-bit-block of (i xor j) is
    nonzero, and col_dist(i,j) only on which column bit positions of
    (i xor j) are nonzero (verified against gqe_single_target.py's
    MMDEvaluator.__init__ row/col construction). So grid is xor-invariant
    too, same fast path as hamming with a different kernel vector.

For a kernel of the form K[i,j] = f(i - j) (Toeplitz), the quadratic form
e^T K e (e = target - probs) is computed via one FFT-based linear
convolution of e against the length-(2N-1) kernel vector f(delta),
delta in [-(N-1), N-1] (`_mmd2_index`).

For a kernel of the form K[i,j] = kappa[i xor j] (i.e. translation-invariant
on the Boolean cube Z_2^n), e^T K e is computed via the Walsh-Hadamard
transform (WHT), which diagonalizes xor-convolution exactly like the DFT
diagonalizes ordinary convolution: WHT(kappa (xor-conv) e) = WHT(kappa) *
WHT(e) elementwise (`_fwht`, `_mmd2_xor`).

Both routes are O(N log N) time and O(N) memory -- at n_qubits=20 that's a
~2*10^6-element FFT / a length-2^20 FWHT, i.e. megabytes and milliseconds,
not terabytes.

Correctness: every function here is validated bit-for-bit (see
test_fast_mmd.py) against MMDEvaluator's dense O(N^2) computation for small
n_qubits, across all three kernels and multiple sigmas, before being trusted
for n_qubits=20.
"""
import math
import numpy as np
from scipy.signal import fftconvolve


def _fwht(a):
    """Unnormalized fast Walsh-Hadamard transform. Self-inverse up to a
    factor of len(a): fwht(fwht(a)) == len(a) * a. len(a) must be a power
    of two. O(N log N) time, O(N) memory (vectorised, no Python-level loop
    over N elements)."""
    a = np.asarray(a, dtype=np.float64).copy()
    n = a.shape[0]
    if n & (n - 1) != 0:
        raise ValueError(f"_fwht requires a power-of-two length, got {n}")
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        y = a[:, 1, :].copy()
        a[:, 0, :] = x + y
        a[:, 1, :] = x - y
        a = a.reshape(n)
        h *= 2
    return a


def index_kernel_vector(n_states, sigmas):
    """f(delta) = sum_sigma exp(-delta^2 / (2 sigma^2)) for
    delta = -(n_states-1) .. (n_states-1), as a length (2*n_states-1) array
    indexed so that index k holds delta = k - (n_states - 1)."""
    delta = np.arange(-(n_states - 1), n_states, dtype=np.float64)
    f = np.zeros_like(delta)
    for s in sigmas:
        f += np.exp(-(delta ** 2) / (2.0 * s * s))
    return f


def hamming_kernel_vector(n_states, sigmas):
    """kappa[d] = sum_sigma exp(-popcount(d)^2 / (2 sigma^2)) for
    d = 0 .. n_states-1. Same value MMDEvaluator's dense hamming kernel
    assigns to any pair (i,j) with i xor j == d."""
    d = np.arange(n_states, dtype=np.uint32)
    pc = np.zeros(n_states, dtype=np.float64)
    v = d.copy()
    while v.any():
        pc += (v & 1).astype(np.float64)
        v >>= 1
    kappa = np.zeros(n_states, dtype=np.float64)
    for s in sigmas:
        kappa += np.exp(-(pc ** 2) / (2.0 * s * s))
    return kappa


def grid_kernel_vector(n_states, sigmas, grid_shape):
    """kappa[d] for the row/col-grouped-Hamming 'grid' kernel, d = i xor j.
    Mirrors MMDEvaluator's row_dist/col_dist construction exactly, but on
    the single xor value d instead of on the full (i,j) grid, since both
    row_dist and col_dist depend only on d = i xor j."""
    R, C = grid_shape
    nq = int(round(math.log2(n_states)))
    if R * C != nq:
        raise ValueError(f"grid_shape {R}x{C} != n_qubits {nq}")
    d = np.arange(n_states, dtype=np.uint32)
    row_dist = np.zeros(n_states, dtype=np.int64)
    for r in range(R):
        mask = ((1 << C) - 1) << (r * C)
        row_dist += ((d & mask) != 0).astype(np.int64)
    col_dist = np.zeros(n_states, dtype=np.int64)
    for c in range(C):
        colmask = 0
        for r in range(R):
            colmask |= 1 << (r * C + c)
        col_dist += ((d & colmask) != 0).astype(np.int64)
    dist_sq = ((row_dist + col_dist).astype(np.float64)) ** 2
    kappa = np.zeros(n_states, dtype=np.float64)
    for s in sigmas:
        kappa += np.exp(-dist_sq / (2.0 * s * s))
    return kappa


def _mmd2_index(e, sigmas):
    n_states = e.shape[0]
    fk = index_kernel_vector(n_states, sigmas)          # length 2N-1
    # y[i] = sum_j f(i-j) e[j]; this is the 'valid'-mode linear convolution
    # of the length-(2N-1) kernel vector with the length-N signal.
    y = fftconvolve(fk, e, mode='valid')
    return float(np.dot(e, y))


def _mmd2_xor(e, kappa):
    n_states = e.shape[0]
    y = _fwht(_fwht(kappa) * _fwht(e)) / n_states
    return float(np.dot(e, y))


def mmd2_fast(target, probs, sigmas, kernel='index', grid_shape=None):
    """Drop-in fast replacement for MMDEvaluator(target, sigmas, kernel,
    grid_shape)(probs), i.e. (target-probs)^T K (target-probs). O(N log N)
    time, O(N) memory instead of O(N^2)/O(N^2)."""
    target = np.asarray(target, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    e = target - probs
    n_states = e.shape[0]
    if n_states & (n_states - 1) != 0:
        raise ValueError(f"mmd2_fast requires a power-of-two n_states, got {n_states}")
    if kernel == 'index':
        return _mmd2_index(e, sigmas)
    elif kernel == 'hamming':
        kappa = hamming_kernel_vector(n_states, sigmas)
        return _mmd2_xor(e, kappa)
    elif kernel == 'grid':
        kappa = grid_kernel_vector(n_states, sigmas, grid_shape)
        return _mmd2_xor(e, kappa)
    else:
        raise ValueError(f"unknown MMD kernel: {kernel}")


def _kernel_vector(n_states, sigmas, kernel, grid_shape=None):
    if kernel == 'index':
        return index_kernel_vector(n_states, sigmas)
    elif kernel == 'hamming':
        return hamming_kernel_vector(n_states, sigmas)
    elif kernel == 'grid':
        return grid_kernel_vector(n_states, sigmas, grid_shape)
    else:
        raise ValueError(f"unknown MMD kernel: {kernel}")


def _pairwise_kernel(a, b, kernel_vec, kernel, n_states):
    """Kernel matrix between two batches of basis-state indices a, b, by
    looking up the precomputed length-N (or 2N-1, for 'index') kernel
    vector at each pair's distance -- exact same kernel definition as the
    dense/fast population MMD, just evaluated only at the sampled pairs
    instead of over the full N x N grid."""
    a = np.asarray(a); b = np.asarray(b)
    if kernel == 'index':
        d = a[:, None].astype(np.int64) - b[None, :].astype(np.int64) + (n_states - 1)
        return kernel_vec[d]
    else:  # 'hamming' / 'grid': both indexed by i xor j
        d = a[:, None].astype(np.uint32) ^ b[None, :].astype(np.uint32)
        return kernel_vec[d]


def mmd2_sampled(target, probs, sigmas, kernel='index', grid_shape=None,
                  n_samples=500, rng=None, unbiased=True):
    """Finite-sample MMD^2 estimator (Gretton et al. 2012 U-/V-statistic),
    the style used by the original QCBM paper (Liu & Wang 2018): draw
    n_samples basis-state indices from `probs` (the circuit's Born
    distribution) and from `target`, then estimate
        MMD^2 = E[K(x,x')] + E[K(y,y')] - 2 E[K(x,y)]
    from the m x m pairwise kernel sums over the samples, instead of the
    exact O(N) / O(N log N) population quantity. Cost is O(n_samples^2),
    independent of N -- this is how the original paper avoided ever facing
    the dense-kernel-matrix blowup this project's exact evaluator hit at
    n_qubits=20, at the cost of a stochastic (not exact) MMD value: the
    estimator has O(1/sqrt(n_samples)) standard error, so it introduces
    sampling noise the exact/fast path (mmd2_fast, FastMMDEvaluator above)
    does not have. unbiased=True uses the U-statistic (excludes the i=i'/
    j=j' diagonal, unbiased but can go slightly negative near MMD^2=0);
    unbiased=False uses the V-statistic (biased high, always >= 0).
    """
    target = np.asarray(target, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n_states = target.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    x = rng.choice(n_states, size=n_samples, p=probs)    # model samples
    y = rng.choice(n_states, size=n_samples, p=target)   # target samples
    kv = _kernel_vector(n_states, sigmas, kernel, grid_shape)
    Kxx = _pairwise_kernel(x, x, kv, kernel, n_states)
    Kyy = _pairwise_kernel(y, y, kv, kernel, n_states)
    Kxy = _pairwise_kernel(x, y, kv, kernel, n_states)
    m = n_samples
    if unbiased:
        term_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
        term_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    else:
        term_xx = Kxx.sum() / (m * m)
        term_yy = Kyy.sum() / (m * m)
    term_xy = Kxy.sum() / (m * m)
    return float(term_xx + term_yy - 2.0 * term_xy)


class SampledMMDEvaluator:
    """Drop-in alternative to MMDEvaluator/FastMMDEvaluator with the same
    __call__(circuit_probs) interface, but returning the finite-sample
    estimator mmd2_sampled() instead of the exact population MMD^2. Useful
    to match the original QCBM paper's evaluation convention, or to check
    how much sampling noise a given n_samples introduces relative to the
    exact value (compare against MMDEvaluator/FastMMDEvaluator on the same
    target/probs)."""

    def __init__(self, target_arr, sigmas, kernel='index', grid_shape=None,
                 n_samples=500, seed=None, unbiased=True):
        self.target = np.asarray(target_arr, dtype=np.float64)
        self.sigmas = list(sigmas)
        self.kernel = kernel
        self.grid_shape = grid_shape
        self.n_samples = n_samples
        self.unbiased = unbiased
        self.rng = np.random.default_rng(seed)

    def __call__(self, circuit_probs):
        return mmd2_sampled(self.target, circuit_probs, self.sigmas,
                             kernel=self.kernel, grid_shape=self.grid_shape,
                             n_samples=self.n_samples, rng=self.rng,
                             unbiased=self.unbiased)


class FastMMDEvaluator:
    """Same public interface as MMDEvaluator (gqe_single_target.py), but
    O(N log N) time / O(N) memory instead of O(N^2)/O(N^2). Precomputes the
    length-N (or 2N-1, for 'index') kernel vector once, not the N x N
    matrix."""

    def __init__(self, target_arr, sigmas, kernel='index', grid_shape=None):
        n_states = len(target_arr)
        if n_states & (n_states - 1) != 0:
            raise ValueError(f"FastMMDEvaluator requires power-of-two n_states, got {n_states}")
        self.target = np.asarray(target_arr, dtype=np.float64)
        self.sigmas = list(sigmas)
        self.kernel = kernel
        self.grid_shape = grid_shape
        self.n_states = n_states
        if kernel == 'index':
            self._fk = index_kernel_vector(n_states, sigmas)
            self._kappa = None
        elif kernel in ('hamming', 'grid'):
            self._fk = None
            self._kappa = (hamming_kernel_vector(n_states, sigmas) if kernel == 'hamming'
                            else grid_kernel_vector(n_states, sigmas, grid_shape))
        else:
            raise ValueError(f"unknown MMD kernel: {kernel}")

    def __call__(self, circuit_probs):
        e = self.target - np.asarray(circuit_probs, dtype=np.float64)
        if self.kernel == 'index':
            y = fftconvolve(self._fk, e, mode='valid')
        else:
            y = _fwht(_fwht(self._kappa) * _fwht(e)) / self.n_states
        return float(np.dot(e, y))
