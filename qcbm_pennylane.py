"""PennyLane-style QCBM baseline, faithfully replicating the official
PennyLane "Quantum circuit Born machines" demo, but evaluated on CUDAQ.

Faithful to https://pennylane.ai/qml/demos/tutorial_qcbm :

  Ansatz   : StronglyEntanglingLayers
             per layer → every qubit gets Rot(φ,θ,ω) = RZ(φ)·RY(θ)·RZ(ω)
                         then a ring of CNOTs  CNOT(i, (i+1) mod n)
  Loss     : squared MMD with multi-bandwidth RBF kernel
             bandwidth(Gaussian/bimodal/exp) = [0.25, 60]
             bandwidth(BAS/GHZ/discrete)     = [0.25, 0.5, 1]
  Optimiser: Adam, lr = 0.1, 100 iterations   (PennyLane demo defaults)
  Gradient : exact ∂MMD²/∂θ via parameter-shift on the probability vector
             ∂MMD²/∂θ_i = 2 (p−π)ᵀ K (∂p/∂θ_i)
             ∂p/∂θ_i = (p(θ+π/2 eᵢ) − p(θ−π/2 eᵢ)) / 2

Everything is exact statevector (cudaq.get_state, fp64) — no shot noise,
same as PennyLane's analytic device. Same 20 benchmark distributions as
qcbm_baseline.py so the two can be compared directly.
"""
import argparse, json, math, os, time

import numpy as np
import cudaq
import cupy as cp

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqe_single_target import build_target, MMDEvaluator

cudaq.set_target('nvidia', option='fp64')


# ─────────────────────────────────────────────────────────────────────────────
# StronglyEntanglingLayers kernel  (PennyLane Rot = RZ(φ)·RY(θ)·RZ(ω))
# ─────────────────────────────────────────────────────────────────────────────
@cudaq.kernel
def sel_kernel(n_qubits: int, n_layers: int, params: list[float]):
    q = cudaq.qvector(n_qubits)
    for l in range(n_layers):
        for w in range(n_qubits):
            base = (l * n_qubits + w) * 3
            rz(params[base + 0], q[w])
            ry(params[base + 1], q[w])
            rz(params[base + 2], q[w])
        for w in range(n_qubits):
            x.ctrl(q[w], q[(w + 1) % n_qubits])


def circuit_probs(n_qubits: int, n_layers: int, params: np.ndarray) -> np.ndarray:
    state = cudaq.get_state(sel_kernel, n_qubits, n_layers,
                            [float(x) for x in params])
    amps = cp.array(state, dtype=cp.complex128)
    return cp.asnumpy(cp.abs(amps) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Squared MMD + exact parameter-shift gradient
# ─────────────────────────────────────────────────────────────────────────────
def mmd_loss(mmd_ev, target: np.ndarray, p: np.ndarray) -> float:
    return mmd_ev(p)


def mmd_grad(n_qubits, n_layers, params, mmd_ev, target):
    """Exact ∂MMD²/∂θ via parameter-shift on p, chained through MMD.

    ∂MMD²/∂θ_i = 2 (p−π)ᵀ K (∂p/∂θ_i),  ∂p/∂θ_i from ±π/2 shifts.
    Cost: 1 + 2·n_params statevector evaluations per call. mmd_ev is an
    MMDEvaluator (gqe_single_target.py) — .apply(diff) returns K @ diff via
    the dense matrix (n_states<=4096) or the fast FFT/Walsh-Hadamard path
    (n_states>4096), so this works unchanged up to n_qubits=20.
    """
    p = circuit_probs(n_qubits, n_layers, params)
    diff = p - target                       # (p − π)
    Kd = 2.0 * mmd_ev.apply(diff)            # 2 K (p − π)
    n = len(params)
    grad = np.zeros(n)
    shift = math.pi / 2
    for i in range(n):
        pp = params.copy(); pp[i] += shift
        pm = params.copy(); pm[i] -= shift
        dp = (circuit_probs(n_qubits, n_layers, pp) -
              circuit_probs(n_qubits, n_layers, pm)) / 2.0
        grad[i] = Kd @ dp
    return grad, p


def kl_grad(n_qubits, n_layers, params, target):
    """LEGACY: NOT the gradient of KL.

    Returns (KL(θ+π/2 eᵢ) − KL(θ−π/2 eᵢ)) / 2, i.e. the ±π/2 parameter shift
    applied to the KL value. That is exact only for first-harmonic functions of
    θᵢ (expectation values, Born probabilities); KL is nonlinear in p. It equals
    the true slope ∂KL/∂θᵢ averaged over the window [θᵢ−π/2, θᵢ+π/2], which
    stops pointing downhill once the KL basin is narrower than ±90°.
    Every QCBM-SEL+KL result produced before the --kl_grad flag existed used
    this function. Kept only to reproduce those runs (--kl_grad legacy).
    Cost: 1 + 2·n_params statevector evaluations per call.
    """
    p = circuit_probs(n_qubits, n_layers, params)
    n = len(params)
    grad = np.zeros(n)
    sh = math.pi / 2
    for i in range(n):
        pp = params.copy(); pp[i] += sh
        pm = params.copy(); pm[i] -= sh
        kp = kl_of(target, circuit_probs(n_qubits, n_layers, pp))
        km = kl_of(target, circuit_probs(n_qubits, n_layers, pm))
        grad[i] = (kp - km) / 2.0
    return grad, p


def kl_grad_exact(n_qubits, n_layers, params, target):
    """Exact ∂KL(target‖p(θ))/∂θ: shift p (first-harmonic, so the ±π/2 shift is
    exact for ∂p/∂θᵢ), then chain rule with ∂KL/∂p_z = −target_z/(p_z+1e-10),
    the same smoothing as kl_of. Mirrors mmd_grad. Identical to
    qcbm_kl_grad_exact.kl_grad_exact (finite-difference self-test: SLURM job
    301916, max error 6e-10); defined here so the trainer needs no circular import.
    Cost: 1 + 2·n_params statevector evaluations per call (same as kl_grad).
    """
    p = circuit_probs(n_qubits, n_layers, params)
    mask = target > 0
    w = np.zeros_like(p)
    w[mask] = -target[mask] / (p[mask] + 1e-10)
    n = len(params)
    grad = np.zeros(n)
    sh = math.pi / 2
    for i in range(n):
        pp = params.copy(); pp[i] += sh
        pm = params.copy(); pm[i] -= sh
        dp = (circuit_probs(n_qubits, n_layers, pp) -
              circuit_probs(n_qubits, n_layers, pm)) / 2.0
        grad[i] = w @ dp
    return grad, p


KL_GRADS = {'exact': kl_grad_exact, 'legacy': kl_grad}


# ─────────────────────────────────────────────────────────────────────────────
# Adam (numpy) — PennyLane uses optax.adam(lr=0.1)
#   loss_mode='mmd' (PennyLane default)  or  'kl' (decouple objective)
# ─────────────────────────────────────────────────────────────────────────────
def adam_optimise(n_qubits, n_layers, init, mmd_ev, target,
                  n_steps, lr, log_every=20,
                  step_ckpt=None, prior_best_mmd=float('inf'),
                  loss_mode='mmd', return_history=False,
                  early_stop_patience=None, kl_grad_mode='exact'):
    params = init.copy()
    n = len(params)
    m = np.zeros(n); v = np.zeros(n)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_mmd, best_params = float('inf'), params.copy()
    history = []      # list of (step, loss, kl, mmd) for plateau diagnosis
    last_improve = 0
    kl_grad_fn = KL_GRADS[kl_grad_mode]
    t_start = time.time()

    for step in range(1, n_steps + 1):
        if loss_mode == 'kl':
            grad, p = kl_grad_fn(n_qubits, n_layers, params, target)
        else:
            grad, p = mmd_grad(n_qubits, n_layers, params, mmd_ev, target)
        # p (and hence the loss below) belongs to the CURRENT params — snapshot
        # them before the Adam update, otherwise best_params stores a vector
        # one step past the argmin (select-on-A-report-B bias, worst at high lr:
        # ~3-5% KL inflation at lr=0.1 on the n=20 cells).
        params_at_p = params.copy()
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        params -= lr * mh / (np.sqrt(vh) + eps)

        kl_val = kl_of(target, p)
        mmd_val = mmd_loss(mmd_ev, target, p) if mmd_ev is not None else float('nan')
        loss = kl_val if loss_mode == 'kl' else mmd_val
        if loss < best_mmd:
            best_mmd, best_params = loss, params_at_p
            last_improve = step
        if step % log_every == 0:
            history.append({'step': step, 'loss': loss,
                            'kl': kl_val, 'mmd': mmd_val,
                            'best_loss': best_mmd,
                            'wall_s': time.time() - t_start})
            print(f"    step {step:5d}/{n_steps}  {loss_mode.upper()}="
                  f"{loss:.6e}  KL={kl_val:.4f}  MMD={mmd_val:.3e}  "
                  f"best={best_mmd:.6e}  "
                  f"{(time.time() - t_start) / step:.3f} s/step", flush=True)
            if step_ckpt is not None and best_mmd < prior_best_mmd:
                step_ckpt(best_mmd, best_params)
        if (early_stop_patience is not None and
                step - last_improve >= early_stop_patience):
            print(f"    [early stop] no improvement for "
                  f"{early_stop_patience} steps at step {step}", flush=True)
            break

    if return_history:
        return best_params, best_mmd, history
    return best_params, best_mmd


# ─────────────────────────────────────────────────────────────────────────────
# KL of the final distribution (for cross-comparison with qcbm_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────
def kl_of(target: np.ndarray, p: np.ndarray) -> float:
    mask = target > 0
    return float(np.sum(target[mask] *
                        np.log(target[mask] / (p[mask] + 1e-10))))


# ─────────────────────────────────────────────────────────────────────────────
# Run one benchmark
# ─────────────────────────────────────────────────────────────────────────────
def run_one(name, n_qubits, n_layers, target, mmd_ev,
            n_starts, n_steps, lr, seed, ckpt_path=None, loss_mode='mmd',
            early_stop_patience=None, kl_grad_mode='exact'):
    n_p = n_layers * n_qubits * 3
    rng = np.random.default_rng(seed)
    full_history = []  # collected across starts

    best_mmd, best_params = float('inf'), None
    prev_starts = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            prev = json.load(open(ckpt_path))
            prev = prev[0] if isinstance(prev, list) else prev
            if prev.get('name') == name and math.isfinite(
                    prev.get('best_mmd', float('inf'))):
                best_mmd = float(prev['best_mmd'])
                if prev.get('best_params') is not None:
                    best_params = np.array(prev['best_params'], dtype=float)
                prev_starts = int(prev.get('starts_done', 0))
                print(f"  [resume] carried best_mmd={best_mmd:.6e} "
                      f"from {prev_starts} prior starts", flush=True)
        except Exception as e:
            print(f"  [resume] could not read checkpoint: {e}", flush=True)

    t0 = time.time()
    carried = best_params.copy() if best_params is not None else None

    def _ckpt(bm, bp, cum):
        if not ckpt_path:
            return
        p_best = (circuit_probs(n_qubits, n_layers, bp)
                  if bp is not None else None)
        rec = {'name': name, 'n_qubits': n_qubits, 'n_layers': n_layers,
               'n_params': n_p, 'n_starts': n_starts, 'n_steps': n_steps,
               'starts_done': cum, 'best_mmd': float(bm),
               'best_kl': (kl_of(target, p_best)
                           if p_best is not None else float('nan')),
               'best_params': bp.tolist() if bp is not None else None,
               'elapsed': time.time() - t0, 'partial': True,
               'kl_grad': kl_grad_mode if loss_mode == 'kl' else None}
        with open(ckpt_path, 'w') as f:
            json.dump([rec], f, indent=2)

    for s in range(n_starts):
        if s == 0 and carried is not None:
            init = carried.copy()
            print(f"  start {s+1}/{n_starts} (WARM from carried) ...",
                  flush=True)
        else:
            # PennyLane initialises weights ~ U(0, 2π)
            init = rng.uniform(0.0, 2 * math.pi, n_p)
            print(f"  start {s+1}/{n_starts} (random) ...", flush=True)

        _cum0 = prev_starts + s
        step_cb = (lambda bm, bp, _c=_cum0: _ckpt(bm, bp, _c))
        _le = 10 if n_qubits >= 10 else 20
        bp, bm, hist = adam_optimise(
            n_qubits, n_layers, init, mmd_ev, target, n_steps, lr,
            log_every=_le, step_ckpt=step_cb, prior_best_mmd=best_mmd,
            loss_mode=loss_mode, return_history=True,
            early_stop_patience=early_stop_patience,
            kl_grad_mode=kl_grad_mode)
        full_history.extend(hist)
        if bm < best_mmd:
            best_mmd, best_params = bm, bp
        _ckpt(best_mmd, best_params, prev_starts + s + 1)
        print(f"  [ckpt] cumulative starts={prev_starts+s+1} "
              f"best_mmd={best_mmd:.6e}", flush=True)

    p_final = circuit_probs(n_qubits, n_layers, best_params)
    best_kl = kl_of(target, p_final)
    elapsed = time.time() - t0
    print(f"  [{name}] n_q={n_qubits} L={n_layers} params={n_p} "
          f"→ MMD={best_mmd:.6e}  KL={best_kl:.5f}  t={elapsed:.1f}s",
          flush=True)
    return {'name': name, 'n_qubits': n_qubits, 'n_layers': n_layers,
            'n_params': n_p, 'n_starts': n_starts, 'n_steps': n_steps,
            'starts_done': prev_starts + n_starts,
            'best_mmd': best_mmd, 'best_kl': best_kl,
            'best_params': best_params.tolist(),
            'history': full_history,
            'elapsed': elapsed, 'partial': False,
            'kl_grad': kl_grad_mode if loss_mode == 'kl' else None}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark suite — same 20 targets as qcbm_baseline.py
# PennyLane layer counts: 6q→4, 9q BAS→6.  Scaled sensibly elsewhere.
#   (name, n_qubits, n_layers, kind, kw)
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    ('bell_2q',          2,  4, 'ghz',        {}),
    ('ghz5_5q',          5,  4, 'ghz',        {}),
    ('bas4_4q',          4,  4, 'bas',        {}),
    ('gauss_5q_s42',     5,  4, 'gaussian',   {'center':0.5,'sigma':0.20}),
    ('gauss_5q_s123',    5,  4, 'gaussian',   {'center':0.5,'sigma':0.20}),
    ('gauss_5q_s456',    5,  4, 'gaussian',   {'center':0.5,'sigma':0.20}),
    ('gauss_5q_s789',    5,  4, 'gaussian',   {'center':0.5,'sigma':0.20}),
    ('gauss_5q_s1000',   5,  4, 'gaussian',   {'center':0.5,'sigma':0.20}),
    ('bimodal_6q_s42',   6,  4, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('bimodal_6q_s123',  6,  4, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('bimodal_6q_s456',  6,  4, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('bimodal_6q_s789',  6,  4, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('bimodal_6q_s1000', 6,  4, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('exp_5q',           5,  4, 'exponential',{}),
    ('gauss_10q',        10, 6, 'gaussian',   {'center':0.5,'sigma':0.10}),
    ('bimodal_10q',     10,  6, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('exp_10q',         10,  6, 'exponential',{}),
    ('gauss_15q_s42',    15, 8, 'gaussian',   {'center':0.5,'sigma':0.10}),
    ('gauss_15q_s123',   15, 8, 'gaussian',   {'center':0.5,'sigma':0.10}),
    ('gauss_15q_s456',   15, 8, 'gaussian',   {'center':0.5,'sigma':0.10}),
    ('bas9_9q',          9,  6, 'bas',        {}),
    ('ghz10_10q',        10, 6, 'ghz',        {}),
    ('lognormal_5q',     5,  4, 'lognormal',  {'mu':-0.7,'sigma':0.5}),
    ('lognormal_10q',    10, 6, 'lognormal',  {'mu':-0.7,'sigma':0.5}),
    ('cardinality_5q',   5,  4, 'cardinality',{'k':2}),
    ('cardinality_10q',  10, 6, 'cardinality',{'k':5}),
    # n=15 / n=20 continuous sweep (added 2026-05-28)
    ('bimodal_15q',     15,  8, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('exp_15q',         15,  8, 'exponential',{}),
    ('lognormal_15q',   15,  8, 'lognormal',  {'mu':-0.7,'sigma':0.5}),
    ('gauss_20q',       20, 10, 'gaussian',   {'center':0.5,'sigma':0.10}),
    ('bimodal_20q',     20, 10, 'bimodal',    {'c1':2/7,'c2':5/7,'sigma':0.125}),
    ('exp_20q',         20, 10, 'exponential',{}),
    ('lognormal_20q',   20, 10, 'lognormal',  {'mu':-0.7,'sigma':0.5}),
]

SEED_OFFSET = {'s42':42,'s123':123,'s456':456,'s789':789,'s1000':1000}

# PennyLane demo bandwidths
BW_CONTINUOUS = [0.25, 60.0]      # Gaussian-mixture demo
BW_DISCRETE   = [0.25, 0.5, 1.0]  # Bars-and-Stripes demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_json', default='results/qcbm_pl.json')
    ap.add_argument('--lr',       type=float, default=0.1)   # PennyLane default
    ap.add_argument('--n_steps',  type=int,   default=100)   # PennyLane default
    ap.add_argument('--n_starts', type=int,   default=1)     # PennyLane: 1 init
    ap.add_argument('--filter',   default='')
    ap.add_argument('--loss',     default='mmd', choices=['mmd', 'kl'],
                    help="objective: mmd (PennyLane default) or kl")
    ap.add_argument('--early_stop_patience', type=int, default=None,
                    help="Stop if no improvement for N steps")
    ap.add_argument('--kl_grad', default='exact', choices=['exact', 'legacy'],
                    help="KL gradient for --loss kl. exact (default, fix for "
                         "see qcbm_kl_grad_exact.py) or legacy (parameter shift on the KL "
                         "value; reproduces every pre-fix QCBM-SEL+KL result)")
    ap.add_argument('--skip_mmd_eval', action='store_true',
                    help="Do not build the dense MMD kernel matrix. Required "
                         "for n>=18 where 2^n × 2^n floats exceeds GPU/RAM. "
                         "MMD value will be NaN; only valid with --loss kl.")
    ap.add_argument('--n_layers', type=int, default=None,
                    help="Override n_layers in the SEL ansatz for the matched "
                         "BENCHMARKS entry. Used for the layer ablation sweep.")
    args = ap.parse_args()
    if args.skip_mmd_eval and args.loss != 'kl':
        raise ValueError("--skip_mmd_eval requires --loss kl")

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    all_results = []

    for (name, n_qubits, n_layers, kind, kw) in BENCHMARKS:
        if args.filter and args.filter not in name:
            continue
        if args.n_layers is not None:
            n_layers = args.n_layers
        seed = next((v for k, v in SEED_OFFSET.items()
                     if name.endswith(k)), 42)
        target = build_target(kind, 2 ** n_qubits, **kw)
        bw = BW_CONTINUOUS if kind in ('gaussian', 'bimodal',
                                       'exponential') else BW_DISCRETE
        mmd_ev = None if args.skip_mmd_eval else MMDEvaluator(target, sigmas=bw)

        print(f"\n{'='*62}\n  {name}  ({kind}, {n_qubits}q, {n_layers}L, "
              f"SEL, MMD bw={bw}, {args.n_starts}×{args.n_steps} @lr{args.lr})",
              flush=True)
        res = run_one(name, n_qubits, n_layers, target, mmd_ev,
                      n_starts=args.n_starts, n_steps=args.n_steps,
                      lr=args.lr, seed=seed, ckpt_path=args.out_json,
                      loss_mode=args.loss,
                      early_stop_patience=args.early_stop_patience,
                      kl_grad_mode=args.kl_grad)
        all_results.append(res)
        with open(args.out_json, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\n\nResults → {args.out_json}")
    print(f"\n{'Experiment':<22}{'nq':>3}{'L':>3}{'params':>7}"
          f"{'Best MMD':>13}{'Best KL':>10}")
    print('-' * 62)
    for r in all_results:
        print(f"  {r['name']:<20}{r['n_qubits']:>3}{r['n_layers']:>3}"
              f"{r['n_params']:>7}{r['best_mmd']:>13.4e}{r['best_kl']:>10.5f}")


if __name__ == '__main__':
    main()
