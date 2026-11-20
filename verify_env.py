"""Confirm fix: pad pauli_word with I to n_qubits length."""
import cudaq, cupy as cp
from itertools import combinations

try:
    cudaq.set_target('nvidia', option='fp64')
    print("target: nvidia fp64")
except Exception as e:
    cudaq.set_target('qpp-cpu')
    print(f"target: qpp-cpu")

n_qubits = 5

@cudaq.kernel
def circuit_kernel(n: int, coeffs: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n)
    for i in range(n):
        h(q[i])
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

def build_op_pool(n_qubits):
    ops = []
    for q in range(n_qubits):
        for p in (cudaq.spin.x, cudaq.spin.y, cudaq.spin.z):
            ops.append(p(q))
    for q1, q2 in combinations(range(n_qubits), 2):
        ops.append(cudaq.spin.x(q1) * cudaq.spin.x(q2))
        ops.append(cudaq.spin.y(q1) * cudaq.spin.y(q2))
        ops.append(cudaq.spin.z(q1) * cudaq.spin.z(q2))
    return ops

def kl_eval_new(op_pool, n_qubits, target_arr, op_idx_list, ang_list):
    fc, fw = [], []
    for idx, ang in zip(op_idx_list, ang_list):
        if abs(ang) < 1e-8:
            continue
        scaled = float(ang) * op_pool[int(idx)]
        for term in scaled:
            coeff = term.get_coefficient().real
            word_str = term.to_string(n_qubits).split(']')[-1].strip()
            word_str = word_str.ljust(n_qubits, 'I')   # pad to n_qubits
            fc.append(coeff)
            fw.append(cudaq.pauli_word(word_str))
    state = cudaq.get_state(circuit_kernel, n_qubits, fc, fw)
    amps = cp.array(state, dtype=cp.complex128)
    probs = cp.abs(amps) ** 2
    target_gpu = cp.asarray(target_arr)
    mask = target_gpu > 0
    return float(cp.sum(target_gpu[mask] * cp.log(target_gpu[mask] / (probs[mask] + 1e-10))))

import numpy as np
n_states = 2 ** n_qubits
idx = np.arange(n_states, dtype=np.float64)
c, s = 0.5 * n_states, 0.2 * n_states
target = np.exp(-0.5 * ((idx - c) / s) ** 2)
target /= target.sum()

op_pool = build_op_pool(n_qubits)
print(f"op_pool size: {len(op_pool)}")

# test with a few random ops/angles
ops = [0, 5, 10, 15, 20]
angs = [0.5, -0.3, 1.2, -0.8, 0.1]
try:
    kl = kl_eval_new(op_pool, n_qubits, target, ops, angs)
    print(f"kl_eval_new result: {kl:.6f}  SUCCESS")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"FAILED: {e}")
