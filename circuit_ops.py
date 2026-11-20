"""Shared circuit-construction utilities: operator pool + CUDAQ state kernel.

Used by gqe_single_target.py, random_baseline.py, and the paper's
figure/analysis scripts (compare_all.py, compute_circuit_stats.py,
make_all_dist_compare.py, make_bas_figures.py, export_figure_data.py) to
build the Pauli operator pool and evaluate a circuit's output state on CUDAQ.
"""
from itertools import combinations

import cudaq


@cudaq.kernel
def circuit_kernel(n_qubits: int, coeffs: list[float],
                   words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits)
    for i in range(n_qubits):
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
