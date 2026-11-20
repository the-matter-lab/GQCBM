"""Sub-grid Curriculum Transfer (SCT): inject token sequences from a
solved smaller-qubit ckpt into the replay buffer of a larger-qubit run,
with qubit indices remapped to a sub-grid.  build_op_pool ordering:

    single-qubit (3 * n_qubits indices):
        idx 3*q + 0 = X_q, +1 = Y_q, +2 = Z_q
    two-qubit  (3 * C(n,2) indices, starting at 3*n_qubits):
        for pair (q1<q2) in combinations(range(n_qubits), 2):
            idx 3*n + 3*pair_idx + 0 = X_{q1}X_{q2}, +1 = YY, +2 = ZZ
"""
import math, torch
from itertools import combinations


def _pair_to_idx(q1, q2, n_qubits):
    """Position of (q1,q2) in combinations(range(n_qubits), 2)."""
    if q1 > q2:
        q1, q2 = q2, q1
    return q1 * n_qubits - q1 * (q1 + 1) // 2 + (q2 - q1 - 1)


def remap_op_idx(src_idx, src_n, dst_n, qmap):
    """Remap an op-pool index from src_n-qubit pool to dst_n-qubit pool
    via qubit mapping qmap[src_q] = dst_q."""
    if src_idx < 3 * src_n:
        src_q = src_idx // 3
        pauli = src_idx % 3
        return 3 * qmap[src_q] + pauli
    two_idx = src_idx - 3 * src_n
    pair_idx = two_idx // 3
    pauli = two_idx % 3
    src_pairs = list(combinations(range(src_n), 2))
    sq1, sq2 = src_pairs[pair_idx]
    dq1, dq2 = qmap[sq1], qmap[sq2]
    dst_pair_idx = _pair_to_idx(dq1, dq2, dst_n)
    return 3 * dst_n + 3 * dst_pair_idx + pauli


def theta_to_ang_idx(thetas, angle_values):
    """Snap each continuous angle to the nearest discrete bin."""
    out = []
    for t in thetas:
        idx = min(range(len(angle_values)),
                  key=lambda i: abs(angle_values[i] - float(t)))
        out.append(idx)
    return out


def load_seed(ckpt_path, qmap, dst_n_qubits, dst_angle_values, dst_n_gates):
    """Load a smaller-qubit ckpt, remap circuit to a sub-grid of a larger
    n-qubit system, and pad to dst_n_gates with zero-angle no-ops.
    Returns (ops_idx_tensor, ang_idx_tensor)."""
    d = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_n = int(round(math.log2(len(d['target_arr']))))
    assert len(qmap) == src_n, f"qmap len {len(qmap)} != src_n {src_n}"
    assert max(qmap) < dst_n_qubits, f"qmap target {max(qmap)} >= dst_n {dst_n_qubits}"

    src_ops = d['best_ops'].tolist()
    src_thetas = d['best_theta'].tolist()
    dst_ops = [remap_op_idx(o, src_n, dst_n_qubits, qmap) for o in src_ops]
    dst_ang_idx = theta_to_ang_idx(src_thetas, dst_angle_values)
    # Pad to dst_n_gates with X_0 at angle 0 (treated as identity downstream)
    zero_ang = min(range(len(dst_angle_values)),
                   key=lambda i: abs(dst_angle_values[i]))
    while len(dst_ops) < dst_n_gates:
        dst_ops.append(0)
        dst_ang_idx.append(zero_ang)
    dst_ops = dst_ops[:dst_n_gates]
    dst_ang_idx = dst_ang_idx[:dst_n_gates]
    return (torch.tensor(dst_ops, dtype=torch.long),
            torch.tensor(dst_ang_idx, dtype=torch.long))
