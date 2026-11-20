"""GQE single-target online RL — paper-faithful with replay buffer.

Paradigm (Nakaji 2024 arXiv:2401.09253):
    Vocabulary = (Pauli_op, angle) tuples — angle is BAKED INTO the token.
    No separate op-head / angle-head. One categorical head over the joint vocab.
    Pure autoregressive GPT + replay buffer with logit-matching loss.

Angle set (log-spaced, covers full range):
    {±π/2^k for k=0..K} ∪ {0}   — 2K+3 values
    K=9 → 21 values: ±180°, ±90°, ±45°, ..., ±0.35°, 0°

Token encoding:
    vocab_size = n_ops · n_angle_vals + 1   (+1 for BOS=0)
    token id (1..vocab-1) → (op_idx, ang_idx):
        op_idx  = (token - 1) // n_angle_vals
        ang_idx = (token - 1)  %  n_angle_vals
        theta   = angle_values[ang_idx]

Loss (logit matching from replay buffer, paper §3.2):
    Buffer keeps top-K circuits sorted by KL ascending.
    w_i = exp(-KL_i / kl_temp) / Z
    L = -Σ_i w_i · (1/T) Σ_t log p(token_t | ctx) − ent_coef · H
    Rationale: absolute KL drives weights — good circuits always reinforced,
    independent of the current batch (unlike GRPO group-relative advantage).

No dataset, no conditioning encoder, no SPSA, no diffusion.
"""
import argparse, json, math, os, time
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import cudaq
import cupy as cp

from fast_mmd import index_kernel_vector, hamming_kernel_vector, grid_kernel_vector, _fwht
from scipy.signal import fftconvolve

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuit_ops import build_op_pool, circuit_kernel


def build_op_pool_with_nn_3body(n_qubits):
    """Standard 1+2-body pool plus nearest-neighbor matching 3-body (XXX, YYY, ZZZ).
    Adds (n_qubits-2)*3 operators on consecutive triples (i, i+1, i+2).
    """
    ops = build_op_pool(n_qubits)
    for i in range(n_qubits - 2):
        ops.append(cudaq.spin.x(i) * cudaq.spin.x(i + 1) * cudaq.spin.x(i + 2))
        ops.append(cudaq.spin.y(i) * cudaq.spin.y(i + 1) * cudaq.spin.y(i + 2))
        ops.append(cudaq.spin.z(i) * cudaq.spin.z(i + 1) * cudaq.spin.z(i + 2))
    return ops


def build_op_pool_symmetric(n_qubits):
    """Permutation-symmetric SUM operators for Dicke / symmetric-state prep.
    Each token is a single GLOBAL multi-qubit operator (sum of products,
    not a single tensor product). Motivated by Global Variational QC for
    Symmetric State Preparation (arXiv:2312.05060): only S_alpha = sum_i
    sigma_alpha^(i) and S_{alpha,beta} = sum_{i<j} sigma_alpha^(i)
    sigma_beta^(j) are needed for arbitrary symmetric state prep.

    Returns 3 single-body + 6 two-body symmetric sums = 9 ops total.
    Vocabulary stays small (9 ops * 27 angles = 243 tokens vs 14176 with
    all 1+2+3-body for n=10).
    """
    from itertools import combinations as _comb
    P = (cudaq.spin.x, cudaq.spin.y, cudaq.spin.z)
    ops = []
    # 1-body symmetric: S_X, S_Y, S_Z
    for p in P:
        op = p(0)
        for i in range(1, n_qubits):
            op = op + p(i)
        ops.append(op)
    # 2-body symmetric: S_{XX}, S_{YY}, S_{ZZ}, S_{XY}, S_{XZ}, S_{YZ}
    pairs = [(P[0], P[0]), (P[1], P[1]), (P[2], P[2]),
             (P[0], P[1]), (P[0], P[2]), (P[1], P[2])]
    for p1, p2 in pairs:
        all_terms = list(_comb(range(n_qubits), 2))
        i, j = all_terms[0]
        op = p1(i) * p2(j)
        for i, j in all_terms[1:]:
            op = op + p1(i) * p2(j)
        ops.append(op)
    return ops


def build_op_pool_row_col_sym(n_qubits, grid_shape):
    """Row/column-symmetric SUM operators for 2D-grid structured targets
    like BAS, whose symmetry is row-permutation × col-permutation (not
    the full S_n that --symmetric_pool assumes). For an R*C grid:
        single-body  row/col sums (R+C) * 3 Paulis            = 3(R+C)
        two-body     row/col sums (R+C) * 6 Pauli pairs        = 6(R+C)
        three-body   row/col sums (R+C) * 3 Paulis (if size>=3)= 3(R+C)
    For 3x3 BAS: 18 + 36 + 18 = 72 ops, vocab 72*27+1 = 1945.
    Each row/col-sum token is a single global rotation within that
    row or column, exactly preserving the stripes or bars subspace.
    """
    from itertools import combinations as _comb
    R, C = grid_shape
    assert R * C == n_qubits, f"grid {R}x{C} != n_qubits {n_qubits}"
    P = (cudaq.spin.x, cudaq.spin.y, cudaq.spin.z)
    pairs = [(P[0], P[0]), (P[1], P[1]), (P[2], P[2]),
             (P[0], P[1]), (P[0], P[2]), (P[1], P[2])]
    rows = [[r * C + c for c in range(C)] for r in range(R)]
    cols = [[r * C + c for r in range(R)] for c in range(C)]
    ops = []

    def sum_single(qs, p):
        op = p(qs[0])
        for q in qs[1:]:
            op = op + p(q)
        return op

    def sum_pair(qs, p1, p2):
        pairs_ = list(_comb(qs, 2))
        i, j = pairs_[0]
        op = p1(i) * p2(j)
        for i, j in pairs_[1:]:
            op = op + p1(i) * p2(j)
        return op

    def sum_triple_same(qs, p):
        triples = list(_comb(qs, 3))
        i, j, k = triples[0]
        op = p(i) * p(j) * p(k)
        for i, j, k in triples[1:]:
            op = op + p(i) * p(j) * p(k)
        return op

    for line in rows + cols:
        for p in P:
            ops.append(sum_single(line, p))
        for p1, p2 in pairs:
            if len(line) >= 2:
                ops.append(sum_pair(line, p1, p2))
        if len(line) >= 3:
            for p in P:
                ops.append(sum_triple_same(line, p))
    return ops


def build_op_pool_row_col_sym_v2(n_qubits, grid_shape):
    """row_col_sym_pool augmented with DIAGONAL (off-axis) pair sums.
    Without diagonal pairs the pool cannot coherently mix the
    stripes-subspace and bars-subspace of a BAS-like target, so BAS-9 hits a
    plateau at ~0.29 with the basic row_col_sym pool. Adds 6 ops:
        D_alphabeta = Σ_{r1<r2, c1!=c2} sigma_alpha(r1,c1) sigma_beta(r2,c2)
    for (alpha, beta) in {XX, YY, ZZ, XY, XZ, YZ}. For 3x3 BAS this adds
    6 ops onto the existing 72, total 78 ops (vocab 78*27+1 = 2107).
    """
    from itertools import combinations as _comb, product as _prod
    R, C = grid_shape
    P = (cudaq.spin.x, cudaq.spin.y, cudaq.spin.z)
    pairs_xx = [(P[0], P[0]), (P[1], P[1]), (P[2], P[2]),
                (P[0], P[1]), (P[0], P[2]), (P[1], P[2])]

    ops = build_op_pool_row_col_sym(n_qubits, grid_shape)

    # Diagonal qubit-pair list (different row AND different column)
    diag_pairs = []
    for r1, r2 in _comb(range(R), 2):
        for c1, c2 in _prod(range(C), repeat=2):
            if c1 != c2:
                diag_pairs.append((r1 * C + c1, r2 * C + c2))
    for p1, p2 in pairs_xx:
        i, j = diag_pairs[0]
        op = p1(i) * p2(j)
        for i, j in diag_pairs[1:]:
            op = op + p1(i) * p2(j)
        ops.append(op)

    return ops


def build_op_pool_bas_full(n_qubits, grid_shape):
    """row_col_sym_v2 (78 ops on 3x3) + full S_n symmetric (9 ops) = 87 ops.
    Combines row/col mixed-symmetry with global permutation-symmetric ops.
    """
    ops = build_op_pool_row_col_sym_v2(n_qubits, grid_shape)
    ops.extend(build_op_pool_symmetric(n_qubits))
    return ops


def build_op_pool_bas_rich(n_qubits, grid_shape):
    """row_col_sym_v2 + all 3-body XXX/YYY/ZZZ on every triple. For 3x3:
    78 + C(9,3)*3 = 78 + 252 = 330 ops. Rich pool: everything 1+2+3-body
    that respects row × col × full permutation symmetries combined.
    """
    from itertools import combinations as _comb
    P = (cudaq.spin.x, cudaq.spin.y, cudaq.spin.z)
    ops = build_op_pool_row_col_sym_v2(n_qubits, grid_shape)
    for i, j, k in _comb(range(n_qubits), 3):
        for p in P:
            ops.append(p(i) * p(j) * p(k))
    return ops


def build_op_pool_with_all_3body(n_qubits):
    """1+2-body pool plus *all* matching 3-body (XXX, YYY, ZZZ) on every
    triple. Adds C(n,3) * 3 operators (n=9 → 84*3=252 new ops, n=10 → 360).
    Motivated by permutation-symmetric targets (Dicke / cardinality) and 2D
    structured targets (BAS rows + cols) that need 3-body correlations not
    captured by the consecutive-triple variant.
    """
    from itertools import combinations as _comb
    ops = build_op_pool(n_qubits)
    for i, j, k in _comb(range(n_qubits), 3):
        ops.append(cudaq.spin.x(i) * cudaq.spin.x(j) * cudaq.spin.x(k))
        ops.append(cudaq.spin.y(i) * cudaq.spin.y(j) * cudaq.spin.y(k))
        ops.append(cudaq.spin.z(i) * cudaq.spin.z(j) * cudaq.spin.z(k))
    return ops


def kl_eval_one(op_pool, n_qubits, target_arr, op_idx_list, ang_list, mmd_evaluator=None):
    """Compatible with new cudaq API (get_coefficient + to_string + pauli_word pad).
    Returns (kl, mmd) tuple. mmd is float('nan') if mmd_evaluator is None.
    """
    fc, fw = [], []
    for idx, ang in zip(op_idx_list, ang_list):
        if abs(ang) < 1e-8:
            continue
        scaled = float(ang) * op_pool[int(idx)]
        for term in scaled:
            coeff = term.get_coefficient().real
            word_str = term.to_string(n_qubits).split(']')[-1].strip()
            word_str = word_str.ljust(n_qubits, 'I')
            fc.append(coeff)
            fw.append(cudaq.pauli_word(word_str))
    state = cudaq.get_state(circuit_kernel, n_qubits, fc, fw)
    amps = cp.array(state, dtype=cp.complex128)
    probs = cp.abs(amps) ** 2
    target_gpu = cp.asarray(target_arr)
    mask = target_gpu > 0
    kl = float(cp.sum(target_gpu[mask] * cp.log(target_gpu[mask] / (probs[mask] + 1e-10))))
    # Reverse KL D(p||q): mode-seeking; heavily penalises probability mass
    # placed on basis states the target forbids (q(x)=0).
    pmask = probs > 1e-15
    revkl = float(cp.sum(probs[pmask] *
                         cp.log(probs[pmask] / (target_gpu[pmask] + 1e-10))))
    # Leak: total probability mass on forbidden states (q(x)=0). Combined
    # with forward KL gives a sparse-target-aware loss D(q||p) + λ · leak.
    invalid_mask = target_gpu == 0
    leak = float(cp.sum(probs[invalid_mask])) if bool(invalid_mask.any()) else 0.0
    # Fidelity loss 1 - |<psi_target | psi_c>|^2. Assumes a real-positive
    # pure-state target whose amplitudes are sqrt(q(x)); valid for BAS,
    # GHZ, and Dicke / cardinality-constrained targets. Unlike KL or leak,
    # this rewards constructive interference between valid-state amplitudes,
    # which the discrete token search must explicitly find.
    target_amp = cp.sqrt(target_gpu)
    overlap_amp = cp.conj(target_amp) @ amps
    fidelity_loss = float(1.0 - cp.abs(overlap_amp) ** 2)
    if mmd_evaluator is not None:
        mmd = mmd_evaluator(cp.asnumpy(probs))
    else:
        mmd = float('nan')
    return kl, mmd, revkl, leak, fidelity_loss


# =============================================================================
# Angle set builder
# =============================================================================
def build_angle_set(K=9):
    """Log-spaced angles: {±π/2^k for k=0..K} ∪ {0}.
    Returns sorted list of 2*(K+1)+1 = 2K+3 float values.
    K=9 → 21 values covering ±180° down to ±0.35°.
    """
    vals = {0.0}
    for k in range(K + 1):
        v = math.pi / (2 ** k)
        vals.add(v)
        vals.add(-v)
    return sorted(vals)   # ascending: -π, ..., 0, ..., +π


# =============================================================================
# Model — one vocab over (op, angle) tuples, one head
# =============================================================================
class GQESingleTarget(nn.Module):
    def __init__(self, n_ops, angle_values, n_gates,
                 n_embd=384, n_layer=6, n_head=8, dropout=0.0):
        super().__init__()
        self.n_ops = n_ops
        self.n_gates = n_gates
        self.n_embd = n_embd

        angle_t = torch.tensor(angle_values, dtype=torch.float32)
        self.register_buffer('angle_values', angle_t)
        self.n_angle_vals = len(angle_values)

        self.vocab_size = n_ops * self.n_angle_vals + 1    # 0 = BOS
        self.n_tokens = self.vocab_size - 1                # non-BOS tokens

        self.tok_emb = nn.Embedding(self.vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_gates + 1, n_embd)

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embd)

        self.head = nn.Linear(n_embd, self.n_tokens)

    # token id ↔ (op_idx, ang_idx). Token ids are 1..vocab-1 (0 is BOS).
    def decode(self, token_ids):
        """token_ids: any shape int tensor, values in [1, vocab). Returns (op_idx, ang_idx)."""
        x = token_ids - 1
        op_idx = x // self.n_angle_vals
        ang_idx = x % self.n_angle_vals
        return op_idx, ang_idx

    def _forward(self, seq):
        """seq: (B, T) int (BOS-prefixed). Returns logits (B, T, n_tokens)."""
        B, T = seq.shape
        pos = torch.arange(T, device=seq.device)
        x = self.tok_emb(seq) + self.pos_emb(pos).unsqueeze(0)
        causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.transformer(x, mask=causal)
        h = self.norm(h)
        return self.head(h)

    def sample(self, batch_size, temp=1.0):
        """Autoregressive sampling over the joint (op, angle) vocab.

        Returns:
            ops:    (B, n_gates) int — 0..n_ops-1
            theta:  (B, n_gates) float — angles
            logp:   (B, n_gates) — log p(token_t | ...)
            ent:    scalar — mean per-step entropy across positions
        """
        device = self.angle_values.device
        B = batch_size
        seq = torch.zeros(B, 1, dtype=torch.long, device=device)   # BOS
        ops_l, angs_l, logp_l = [], [], []
        ent_sum = 0.0

        for t in range(self.n_gates):
            logits = self._forward(seq)[:, -1] / max(temp, 1e-6)   # (B, n_tokens)
            dist = Categorical(logits=logits)
            tok = dist.sample()                                    # (B,) in [0, n_tokens)
            logp = dist.log_prob(tok)
            ent_sum = ent_sum + dist.entropy().mean()

            tok_id = tok + 1                                       # to vocab id (skip BOS=0)
            op_idx, ang_idx = self.decode(tok_id)
            ops_l.append(op_idx)
            angs_l.append(ang_idx)
            logp_l.append(logp)

            seq = torch.cat([seq, tok_id.unsqueeze(-1)], dim=1)

        ops = torch.stack(ops_l, dim=1)
        ang_idx = torch.stack(angs_l, dim=1)
        theta = self.angle_values[ang_idx]
        logp = torch.stack(logp_l, dim=1)
        ent = ent_sum / self.n_gates
        return ops, ang_idx, theta, logp, ent

    def sft_loss(self, ops, ang_idx, kls, kl_temp=0.1, ent_coef=0.0):
        """Logit-matching CE loss weighted by exp(-KL / kl_temp).

        ops:     (B, n_gates) int64
        ang_idx: (B, n_gates) int64  (0..n_angle_vals-1)
        kls:     (B,) float32
        Returns: (loss scalar, entropy scalar)
        """
        device = self.angle_values.device
        B, T = ops.shape
        ops = ops.to(device)
        ang_idx = ang_idx.to(device)
        kls = kls.to(device)

        # Reconstruct vocab token ids (1-indexed, BOS=0)
        tok_ids = ops * self.n_angle_vals + ang_idx + 1   # (B, T)

        # Input: [BOS, tok_1, ..., tok_{T-1}]  →  predict [tok_1, ..., tok_T]
        bos = torch.zeros(B, 1, dtype=torch.long, device=device)
        inp = torch.cat([bos, tok_ids[:, :-1]], dim=1)    # (B, T)
        logits = self._forward(inp)                        # (B, T, n_tokens)

        # CE loss per token, shift tok_ids to 0-indexed
        targets = tok_ids - 1                              # (B, T) in [0, n_tokens)
        ce = F.cross_entropy(
            logits.reshape(B * T, self.n_tokens),
            targets.reshape(B * T),
            reduction='none').reshape(B, T).mean(dim=1)   # (B,)

        # Softmax weights over buffer batch: w_i ∝ exp(-KL_i / kl_temp)
        log_w = -kls / kl_temp
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / (w.sum() + 1e-8)

        loss = (w * ce).sum()

        # Optional entropy regularization (from the same forward pass)
        if ent_coef > 0.0:
            lp = F.log_softmax(logits, dim=-1)
            ent = -(lp.exp() * lp).sum(dim=-1).mean()
            loss = loss - ent_coef * ent
        else:
            ent = torch.tensor(0.0, device=device)

        return loss, ent


# =============================================================================
# Replay buffer
# =============================================================================
class ReplayBuffer:
    """Sorted FIFO buffer: keeps top `capacity` circuits by KL ascending.
    Entries added with pinned=True are retained regardless of rank
    (used for Sub-grid Curriculum Transfer seeding)."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.ops: list     = []   # (n_gates,) int64 cpu tensors
        self.ang_idx: list = []   # (n_gates,) int64 cpu tensors
        self.kls: list     = []   # float
        self.pinned: list  = []   # bool — never evicted by ranking

    def add_batch(self, ops_cpu, ang_idx_cpu, kls_list, pinned=False):
        """ops_cpu: (B, n_gates), ang_idx_cpu: (B, n_gates), kls_list: list[float]."""
        for b in range(ops_cpu.shape[0]):
            self.ops.append(ops_cpu[b].clone())
            self.ang_idx.append(ang_idx_cpu[b].clone())
            self.kls.append(float(kls_list[b]))
            self.pinned.append(pinned)
        # Sort key: pinned first (False > True so (not pinned)=0 sorts first),
        # then by ascending KL. Truncate to capacity.
        order = sorted(range(len(self.kls)),
                       key=lambda i: (not self.pinned[i], self.kls[i]))
        order = order[:self.capacity]
        self.ops     = [self.ops[i]     for i in order]
        self.ang_idx = [self.ang_idx[i] for i in order]
        self.kls     = [self.kls[i]     for i in order]
        self.pinned  = [self.pinned[i]  for i in order]

    def sample_batch(self, batch_size):
        """Uniform random sample. Returns (ops, ang_idx, kls) tensors."""
        n = len(self.ops)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        ops     = torch.stack([self.ops[i]     for i in idx])
        ang_idx = torch.stack([self.ang_idx[i] for i in idx])
        kls     = torch.tensor([self.kls[i]    for i in idx], dtype=torch.float32)
        return ops, ang_idx, kls

    def __len__(self):
        return len(self.ops)

    @property
    def best_kl(self):
        return self.kls[0] if self.kls else float('inf')


# =============================================================================
# Target builders
# =============================================================================
def build_target(kind, n_states, **kw):
    idx = np.arange(n_states, dtype=np.float64)
    n_qubits = int(round(math.log2(n_states)))
    if kind == 'gaussian':
        c = kw.get('center', 0.5) * n_states
        s = kw.get('sigma', 0.10) * n_states
        arr = np.exp(-0.5 * ((idx - c) / s) ** 2)
    elif kind == 'bimodal':
        c1 = kw.get('c1', 0.25) * n_states
        c2 = kw.get('c2', 0.75) * n_states
        s = kw.get('sigma', 0.06) * n_states
        arr = (np.exp(-0.5 * ((idx - c1) / s) ** 2) +
               np.exp(-0.5 * ((idx - c2) / s) ** 2))
    elif kind == 'exponential':
        scale = kw.get('scale', 0.25) * n_states
        arr = np.exp(-idx / scale)
    elif kind == 'lognormal':
        # Discretised log-normal over the normalised index t=(x+1)/n_states,
        # q(x) ∝ (1/t) exp(-(ln t - mu)^2 / (2 sigma^2)).  Relevant to
        # financial option pricing (log-normal asset prices).
        mu = kw.get('mu', -0.7)
        sigma = kw.get('sigma', 0.5)
        t = (idx + 1.0) / n_states
        arr = np.exp(-0.5 * ((np.log(t) - mu) / sigma) ** 2) / t
    elif kind == 'cardinality':
        # Cardinality-constrained: uniform over basis states whose Hamming
        # weight equals k.  Relevant to portfolio selection (choose exactly
        # k assets).
        k = int(kw.get('k', n_qubits // 2))
        arr = np.zeros(n_states)
        for x in range(n_states):
            if bin(x).count('1') == k:
                arr[x] = 1.0
    elif kind == 'delta':
        which = kw.get('which', n_states - 1)
        arr = np.zeros(n_states); arr[int(which)] = 1.0
    elif kind == 'ghz':
        # |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2 → measurement: 0.5 each
        # At n_qubits=2 this is the Bell state |Φ+⟩.
        arr = np.zeros(n_states)
        arr[0] = 1.0
        arr[n_states - 1] = 1.0
    elif kind == 'uniform':
        # Uniform over all 2^n basis states. NOTE: the ansatz starts from
        # H^{⊗n}|0⟩, whose Born distribution is already exactly uniform, so
        # the all-zero-angle circuit is a trivial exact solution (KL = 0).
        # Added 2026-08-25 as a requested sanity/baseline target.
        arr = np.ones(n_states)
    elif kind == 'w_state':
        # |W_n⟩ = (1/√n) Σ_i |0...010...0⟩ (single excitation at qubit i)
        # measurement: each single-excitation basis state has prob 1/n
        # basis state indices: {2^i : i = 0..n_qubits-1}
        arr = np.zeros(n_states)
        for i in range(n_qubits):
            arr[1 << i] = 1.0
    elif kind == 'bas':
        # Bars and Stripes on √n × √n grid (e.g., 9 qubits → 3×3, 14 patterns)
        n_side = int(round(math.sqrt(n_qubits)))
        if n_side * n_side != n_qubits:
            raise ValueError(f"BAS requires square grid; got n_qubits={n_qubits}")
        arr = np.zeros(n_states)
        # Stripes: each row uniform → 2^n_side patterns
        for sp in range(2 ** n_side):
            state_idx = 0
            for row in range(n_side):
                if (sp >> row) & 1:
                    for col in range(n_side):
                        state_idx |= (1 << (row * n_side + col))
            arr[state_idx] = 1.0
        # Bars: each column uniform → 2^n_side patterns
        for bp in range(2 ** n_side):
            state_idx = 0
            for col in range(n_side):
                if (bp >> col) & 1:
                    for row in range(n_side):
                        state_idx |= (1 << (row * n_side + col))
            arr[state_idx] = 1.0
    else:
        raise ValueError(f"unknown target kind: {kind}")
    arr = arr / arr.sum()
    return arr


# =============================================================================
# MMD evaluator (RBF kernel, multi-bandwidth, pre-computed kernel matrix)
# =============================================================================
class MMDEvaluator:
    """MMD² with multi-bandwidth RBF kernel between target and circuit dist.

    kernel='index' (default): K[i,j] = Σ_σ exp(-(i-j)²/(2σ²)). Standard for
        smooth densities indexed by an integer ordering.
    kernel='hamming': K[i,j] = Σ_σ exp(-H(i,j)²/(2σ²)) where H(i,j) is the
        bitwise Hamming distance. Geometry-aware for structured discrete
        targets (GHZ, BAS, Dicke / cardinality-constrained) whose natural
        symmetry is permutation of qubits, not index proximity.
    kernel='grid': row/column-grouped Hamming for 2D-grid-structured targets
        (BAS).

    All three kernels are translation-invariant (index: K[i,j]=f(i-j);
    hamming/grid: K[i,j]=f(i xor j)), so MMD² = diff^T K diff never needs the
    dense n_states×n_states matrix: for n_states <= _DENSE_MAX it's built
    directly (bit-identical to the original implementation, kept for small
    problems where the O(n_states²) matrix is cheap and free of any new
    code-path risk); above that it dispatches to the O(n_states log n_states)
    time / O(n_states) memory path (FFT convolution for 'index', Walsh-
    Hadamard transform for 'hamming'/'grid' — see fast_mmd.py). This is what
    makes n_qubits=20 (n_states=2^20, 8.8 TB dense) tractable: the fast path
    uses tens of MB and runs in well under a second. Fast-vs-dense agreement
    is validated to float64 precision (~1e-16 relative) in test_fast_mmd.py
    across all three kernels before being trusted here.
    """
    _DENSE_MAX = 4096  # dense N×N matrix here is <= 134 MB; use fast path above this

    def __init__(self, target_arr, sigmas, kernel='index', grid_shape=None):
        n = len(target_arr)
        self.target = target_arr
        self.sigmas = list(sigmas)
        self.kernel = kernel
        self.grid_shape = grid_shape
        self.n_states = n
        self._fast = n > self._DENSE_MAX
        if self._fast:
            if n & (n - 1) != 0:
                raise ValueError(f"fast MMD path requires power-of-two n_states, got {n}")
            if kernel == 'index':
                self._fk = index_kernel_vector(n, sigmas)
                self._kappa = None
            elif kernel == 'hamming':
                self._fk = None
                self._kappa = hamming_kernel_vector(n, sigmas)
            elif kernel == 'grid':
                if grid_shape is None:
                    raise ValueError("grid kernel needs grid_shape=(R,C)")
                self._fk = None
                self._kappa = grid_kernel_vector(n, sigmas, grid_shape)
            else:
                raise ValueError(f"unknown MMD kernel: {kernel}")
            return
        if kernel == 'index':
            idx = np.arange(n, dtype=np.float64)
            dist_sq = (idx[:, None] - idx[None, :]) ** 2
        elif kernel == 'hamming':
            idx = np.arange(n, dtype=np.uint32)
            # H(i,j) = popcount(i XOR j); vectorised
            xor = idx[:, None] ^ idx[None, :]
            ham = np.zeros_like(xor, dtype=np.float64)
            v = xor.copy()
            while v.any():
                ham += (v & 1).astype(np.float64)
                v >>= 1
            dist_sq = ham ** 2
        elif kernel == 'grid':
            # Row/column-grouped Hamming for 2D-grid-structured targets (BAS).
            # Reshape each n-bit basis index into a (R, C) grid; the distance
            # counts how many rows differ + how many columns differ, treating
            # each row/column as an atomic unit. Captures the row/col
            # permutation symmetry of bars-and-stripes patterns.
            if grid_shape is None:
                raise ValueError("grid kernel needs grid_shape=(R,C)")
            R, C = grid_shape
            nq = int(round(math.log2(n)))
            if R * C != nq:
                raise ValueError(f"grid_shape {R}x{C} != n_qubits {nq}")
            idx = np.arange(n, dtype=np.uint32)
            rows = np.stack([
                (idx >> (r * C)) & ((1 << C) - 1) for r in range(R)
            ], axis=1)                              # shape (n_states, R)
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

    def apply(self, diff):
        """Return K @ diff (the matrix-vector product, not the scalar
        quadratic form) — used by callers that need it for a gradient
        (e.g. qcbm_pennylane.py's mmd_grad: d(diff^T K diff)/dp = 2 K diff),
        not just the MMD^2 value itself."""
        if self._fast:
            if self.kernel == 'index':
                return fftconvolve(self._fk, diff, mode='valid')
            return _fwht(_fwht(self._kappa) * _fwht(diff)) / self.n_states
        return self.K @ diff

    def __call__(self, circuit_probs):
        diff = self.target - circuit_probs
        return float(np.dot(diff, self.apply(diff)))


# =============================================================================
# Batch KL eval (CUDAQ, sequential)
# =============================================================================
def kl_eval_batch(op_pool, n_qubits, target_arr, ops, theta, mmd_evaluator=None):
    B = ops.shape[0]
    kls, mmds, revkls, leaks, fids = [], [], [], [], []
    ops_np = ops.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
    _printed_err = False
    for b in range(B):
        try:
            kl, mmd, revkl, leak, fid = kl_eval_one(
                op_pool, n_qubits, target_arr,
                ops_np[b].tolist(), theta_np[b].tolist(),
                mmd_evaluator=mmd_evaluator)
            kls.append(kl)
            mmds.append(mmd)
            revkls.append(revkl)
            leaks.append(leak)
            fids.append(fid)
        except Exception as e:
            if not _printed_err:
                import traceback
                print(f"  [kl_eval error] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                _printed_err = True
            kls.append(99.0)
            mmds.append(float('nan'))
            revkls.append(99.0)
            leaks.append(1.0)
            fids.append(1.0)
    return (torch.tensor(kls, dtype=torch.float32),
            torch.tensor(mmds, dtype=torch.float32),
            torch.tensor(revkls, dtype=torch.float32),
            torch.tensor(leaks, dtype=torch.float32),
            torch.tensor(fids, dtype=torch.float32))


# =============================================================================
# Training
# =============================================================================
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        cudaq.set_target('nvidia', option='fp64')
    except RuntimeError:
        cudaq.set_target('qpp-cpu')

    n_qubits = args.n_qubits
    n_states = 2 ** n_qubits
    if args.symmetric_pool:
        n_ops = 9  # 3 single-body sym + 6 two-body sym
    elif args.row_col_sym_pool or args.row_col_sym_pool_v2 \
            or args.bas_full_pool or args.bas_rich_pool:
        shape_str = (args.bas_rich_pool or args.bas_full_pool
                     or args.row_col_sym_pool_v2 or args.row_col_sym_pool)
        R, C = (int(s) for s in shape_str.split(','))
        line_ops = 3
        if C >= 2: line_ops += 6
        if C >= 3: line_ops += 3
        line_ops_col = 3
        if R >= 2: line_ops_col += 6
        if R >= 3: line_ops_col += 3
        n_ops = R * line_ops + C * line_ops_col
        if args.row_col_sym_pool_v2 or args.bas_full_pool or args.bas_rich_pool:
            n_ops += 6   # diagonal off-axis pair sums (XX/YY/ZZ/XY/XZ/YZ)
        if args.bas_full_pool:
            n_ops += 9   # global S_n symmetric (S_α and S_{αβ})
        if args.bas_rich_pool:
            n_ops += (n_qubits * (n_qubits - 1) * (n_qubits - 2) // 6) * 3  # all-3-body
    else:
        n_ops = n_qubits * 3 + (n_qubits * (n_qubits - 1) // 2) * 3
        if args.add_nn_3body:
            n_ops += (n_qubits - 2) * 3
        if args.add_all_3body:
            n_ops += (n_qubits * (n_qubits - 1) * (n_qubits - 2) // 6) * 3
    print(f"n_qubits={n_qubits} n_states={n_states} n_ops={n_ops} "
          f"n_gates={args.n_gates} angle_bins_K={args.angle_bins_K} "
          f"nn_3body={args.add_nn_3body} all_3body={args.add_all_3body}")
    print(f"vocab = n_ops · n_angle_vals + 1 = {n_ops*(2*args.angle_bins_K+3) + 1}")

    target_kw = json.loads(args.target_kw) if args.target_kw else {}
    target_arr = build_target(args.target_kind, n_states, **target_kw)
    print(f"Target: {args.target_kind} {target_kw}")
    print(f"  peak prob: {target_arr.max():.6f} at idx {target_arr.argmax()}")
    print(f"  entropy:   {-np.sum(target_arr * np.log(target_arr + 1e-12)):.3f} "
          f"(uniform={np.log(n_states):.3f})")

    if args.symmetric_pool:
        op_pool = build_op_pool_symmetric(n_qubits)
    elif args.bas_full_pool:
        R, C = (int(s) for s in args.bas_full_pool.split(','))
        op_pool = build_op_pool_bas_full(n_qubits, (R, C))
    elif args.bas_rich_pool:
        R, C = (int(s) for s in args.bas_rich_pool.split(','))
        op_pool = build_op_pool_bas_rich(n_qubits, (R, C))
    elif args.row_col_sym_pool_v2:
        R, C = (int(s) for s in args.row_col_sym_pool_v2.split(','))
        op_pool = build_op_pool_row_col_sym_v2(n_qubits, (R, C))
    elif args.row_col_sym_pool:
        R, C = (int(s) for s in args.row_col_sym_pool.split(','))
        op_pool = build_op_pool_row_col_sym(n_qubits, (R, C))
    elif args.add_all_3body:
        op_pool = build_op_pool_with_all_3body(n_qubits)
    elif args.add_nn_3body:
        op_pool = build_op_pool_with_nn_3body(n_qubits)
    else:
        op_pool = build_op_pool(n_qubits)
    assert len(op_pool) == n_ops, f"op_pool size {len(op_pool)} != n_ops {n_ops}"

    # MMD evaluator: enabled when --mmd_sigmas provided. MMDEvaluator uses a dense
    # n_states² kernel matrix only up to 4096 states; above that it auto-switches to
    # an O(n_states log n_states)-time/O(n_states)-memory path (fast_mmd.py), so this
    # is now just a hard safety cap (default 2^20 = n_qubits 20), not a memory limit.
    mmd_eval = None
    if args.mmd_sigmas:
        mmd_sigmas = [float(s) for s in args.mmd_sigmas.split(',')]
        if n_states > args.mmd_max_n_states:
            print(f"  MMD: SKIPPED (n_states={n_states} > {args.mmd_max_n_states})")
        else:
            grid_shape = None
            if args.mmd_kernel == 'grid':
                if not args.mmd_grid_shape:
                    raise ValueError("--mmd_kernel grid requires --mmd_grid_shape 'R,C'")
                R, C = (int(s) for s in args.mmd_grid_shape.split(','))
                grid_shape = (R, C)
            mmd_eval = MMDEvaluator(target_arr, mmd_sigmas,
                                    kernel=args.mmd_kernel, grid_shape=grid_shape)
            path = "fast O(N log N)" if mmd_eval._fast else f"dense {n_states}×{n_states}"
            print(f"  MMD: sigmas={mmd_sigmas}, kernel={args.mmd_kernel}, {path}")

    angle_values = build_angle_set(args.angle_bins_K)
    print(f"  angle set (K={args.angle_bins_K}): {len(angle_values)} values, "
          f"range [{min(angle_values):.4f}, {max(angle_values):.4f}]")

    model = GQESingleTarget(
        n_ops=n_ops, angle_values=angle_values, n_gates=args.n_gates,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head).to(device)
    print(f"  vocab_size={model.vocab_size}")
    print(f"  model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.log_dir, exist_ok=True)
    buf = ReplayBuffer(capacity=args.buf_capacity)
    log = []
    best_kl = float('inf')
    best_mmd = float('inf')
    best_score = float('inf')   # optimisation metric: KL by default, MMD if --loss_metric mmd
    best_ops = best_theta = None

    # ── Resume from an existing ckpt:
    #     load transformer weights, pin its best circuit in the buffer,
    #     pre-load best-tracker so subsequent improvements continue cleanly.
    if args.resume_from and os.path.exists(args.resume_from):
        prev = torch.load(args.resume_from, map_location=device,
                          weights_only=False)
        try:
            model.load_state_dict(prev['model'])
            print(f"  [resume] loaded model state from {args.resume_from}",
                  flush=True)
        except Exception as e:
            print(f"  [resume] could NOT load model state: {e}", flush=True)
        if 'best_ops' in prev and 'best_theta' in prev:
            prev_ops = prev['best_ops'].cpu() if hasattr(prev['best_ops'], 'cpu') \
                       else torch.tensor(prev['best_ops'])
            prev_theta = prev['best_theta'].cpu() if hasattr(prev['best_theta'], 'cpu') \
                         else torch.tensor(prev['best_theta'])
            try:
                kl_p, mmd_p, revkl_p, leak_p, fid_p = kl_eval_one(
                    op_pool, n_qubits, target_arr,
                    prev_ops.tolist(), prev_theta.tolist(),
                    mmd_evaluator=mmd_eval)
                best_kl = kl_p
                best_mmd = mmd_p if mmd_p == mmd_p else best_mmd
                best_ops = prev_ops.clone() if hasattr(prev_ops, 'clone') \
                           else torch.tensor(prev_ops)
                best_theta = prev_theta.clone() if hasattr(prev_theta, 'clone') \
                             else torch.tensor(prev_theta)
                score_map = {'kl': kl_p, 'mmd': mmd_p, 'revkl': revkl_p,
                             'fidelity': fid_p,
                             'kl_leak': kl_p + args.leak_lambda * leak_p}
                best_score = score_map.get(args.loss_metric, kl_p)
                # snap angles to nearest discrete index for buffer storage
                av_list = list(angle_values)
                ang_idx_prev = torch.tensor(
                    [min(range(len(av_list)),
                         key=lambda j: abs(av_list[j] - float(t)))
                     for t in prev_theta.tolist()], dtype=torch.long)
                buf.add_batch(prev_ops.unsqueeze(0).long(),
                              ang_idx_prev.unsqueeze(0),
                              [best_score], pinned=True)
                print(f"  [resume] best_kl={best_kl:.6f} best_mmd={best_mmd} "
                      f"best_score={best_score:.6f}; injected pinned to buffer",
                      flush=True)
            except Exception as e:
                print(f"  [resume] could not inject best circuit: {e}",
                      flush=True)

    # ── Sub-grid Curriculum Transfer: inject seed circuits from smaller-qubit
    #    ckpt(s), remapped to sub-grids of the current n-qubit system. The
    #    seeds are pinned so the buffer never evicts them.
    if args.seed_buffer_ckpts:
        from seed_buffer import load_seed
        seed_paths = [s for s in args.seed_buffer_ckpts.split(',') if s]
        seed_maps = [list(map(int, m.split(',')))
                     for m in args.seed_qubit_maps.split(';') if m]
        assert len(seed_paths) == len(seed_maps), \
            "seed_buffer_ckpts and seed_qubit_maps must have same length"
        for path, qmap in zip(seed_paths, seed_maps):
            seed_ops, seed_ang_idx = load_seed(
                path, qmap, n_qubits, angle_values, args.n_gates)
            seed_theta = torch.tensor(
                [angle_values[i] for i in seed_ang_idx.tolist()],
                dtype=torch.float32)
            kl_s, mmd_s, revkl_s, leak_s, fid_s = kl_eval_one(
                op_pool, n_qubits, target_arr,
                seed_ops.tolist(), seed_theta.tolist(),
                mmd_evaluator=mmd_eval)
            score_s = {'kl': kl_s, 'mmd': mmd_s, 'revkl': revkl_s,
                       'kl_leak': kl_s + args.leak_lambda * leak_s,
                       'fidelity': fid_s,
                      }[args.loss_metric]
            print(f"  [SCT seed] {path} qmap={qmap} "
                  f"KL={kl_s:.4f} leak={leak_s:.4f} fid_loss={fid_s:.4f} score={score_s:.4f}",
                  flush=True)
            buf.add_batch(seed_ops.unsqueeze(0), seed_ang_idx.unsqueeze(0),
                          [score_s], pinned=args.seed_pinned)

    for step in range(args.max_iters):
        t0 = time.time()

        # ── 1. Sample new circuits (temp fixed — buffer provides curriculum) ──
        ops, ang_idx, theta, logp, ent = model.sample(
            batch_size=args.num_samples, temp=args.temp_start)

        # ── 2. Evaluate KL / reverse-KL / fidelity (and optionally MMD) ──
        kls, mmds, revkls, leaks, fids = kl_eval_batch(
            op_pool, n_qubits, target_arr, ops, theta, mmd_eval)

        # Optimisation metric. Stage-1 (leak-first warm-up) optionally
        # overrides for the first --leak_first_steps steps to push the
        # policy onto the target support before any KL refinement.
        if args.leak_first_steps > 0 and step < args.leak_first_steps:
            score = leaks
        elif args.loss_metric == 'kl_leak':
            score = kls + args.leak_lambda * leaks
        elif args.loss_metric == 'kl_mmd':
            # forward KL + λ·MMD² hybrid: kernel-aware exploration on top of KL
            score = kls + args.kl_mmd_lambda * mmds
        else:
            score = {'kl': kls, 'mmd': mmds, 'revkl': revkls,
                     'fidelity': fids}[args.loss_metric]

        # Track both metrics for reporting
        kl_min_step = kls.min().item()
        best_kl = min(best_kl, kl_min_step)
        mmd_min_step = float('nan')
        if mmd_eval is not None and not torch.isnan(mmds).all():
            mmd_min_step = float(torch.nanquantile(mmds, 0.0).item())
            best_mmd = min(best_mmd, mmd_min_step)

        # Save the best circuit by the chosen optimisation metric
        score_min_step = score.min().item()
        if score_min_step < best_score:
            best_score = score_min_step
            b = int(score.argmin().item())
            best_ops  = ops[b].detach().cpu().clone()
            best_theta = theta[b].detach().cpu().clone()

        # ── 3. Add to replay buffer (keyed by the optimisation metric) ──
        buf.add_batch(ops.detach().cpu(), ang_idx.detach().cpu(), score.tolist())

        # ── 4. Train from buffer (logit matching, adaptive kl_temp) ──
        # kl_temp adapts to current buffer best: broad early, sharp later.
        # --kl_temp_fixed > 0 disables the adaptive rule (controlled study:
        # adaptive T=best_kl*0.5 goes pathologically flat on stuck sparse
        # targets, killing the good/bad sample discrimination).
        # Precedence: std-based > fixed > original adaptive (all opt-in,
        # defaults preserve original behaviour for backward compat).
        # std rule: T tracks the *spread* of batch KLs, not the absolute
        # best_kl — stays discriminative even when stuck at high KL
        # (fixes the vicious cycle: stuck→best_kl big→T big→flat→stuck).
        if args.kl_temp_std_coef > 0:
            # use the spread of the optimisation metric (KL or MMD), so the
            # temperature auto-scales to whichever loss_metric is active
            kl_temp_now = max(args.kl_temp_std_coef * float(score.std()),
                              args.kl_temp_min)
        elif args.kl_temp_fixed > 0:
            kl_temp_now = args.kl_temp_fixed
        else:
            kl_temp_now = max(buf.best_kl * 0.5, args.kl_temp_min)

        loss = torch.tensor(float('nan'))
        buf_ent = ent
        if len(buf) >= args.buf_min_size:
            buf_ops, buf_ang, buf_kls = buf.sample_batch(args.buf_batch_size)
            loss, buf_ent = model.sft_loss(
                buf_ops, buf_ang, buf_kls,
                kl_temp=kl_temp_now, ent_coef=args.ent_coef)

            if torch.isfinite(loss):
                # cosine lr schedule: lr(t) = lr_min + 0.5(lr_max-lr_min)(1+cos(π t/T))
                if args.lr_decay == 'cosine':
                    lr_now = (args.lr_min + 0.5 * (args.lr - args.lr_min) *
                              (1.0 + math.cos(math.pi * step / args.max_iters)))
                    for pg in opt.param_groups:
                        pg['lr'] = lr_now
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                print(f"[step {step}] non-finite loss, skip", flush=True)

        dt = time.time() - t0
        loss_val = float(loss.item()) if torch.isfinite(loss) else float('nan')
        rec = {
            'step': step,
            'kl_min': kl_min_step,
            'kl_mean': float(kls.mean().item()),
            'best_kl': best_kl,
            'mmd_min': mmd_min_step,
            'best_mmd': best_mmd if best_mmd != float('inf') else float('nan'),
            'buf_size': len(buf),
            'buf_best_kl': buf.best_kl,
            'kl_temp': kl_temp_now,
            'ent': float(ent.item()),
            'loss': loss_val,
            'dt': dt,
        }
        log.append(rec)

        if step % args.log_every == 0 or step == args.max_iters - 1:
            mmd_str = f" MMD={mmd_min_step:.4f}/best={best_mmd:.4f}" if mmd_eval else ""
            print(f"[step {step:5d}] KL min/mean={kl_min_step:.4f}/{kls.mean():.4f} "
                  f"best={best_kl:.4f}{mmd_str} T_kl={kl_temp_now:.4f} "
                  f"H={ent.item():.2f} loss={loss_val:.4f} dt={dt:.1f}s")

        if (step + 1) % args.save_every == 0:
            torch.save({
                'model': model.state_dict(),
                'config': vars(args),
                'angle_values': angle_values,
                'best_kl': best_kl,
                'best_ops': best_ops, 'best_theta': best_theta,
                'step': step + 1,
            }, args.save_ckpt)
            with open(os.path.join(args.log_dir, 'log.json'), 'w') as f:
                json.dump(log, f, indent=2)

    print(f"\n=== Training done. Best-during-train KL={best_kl:.4f} ===")
    print(f"Final sample-and-rank N={args.n_final_samples} ...")
    model.eval()
    with torch.no_grad():
        ops, _, theta, _, _ = model.sample(
            batch_size=args.n_final_samples, temp=args.temp_final)
    kls, mmds, revkls, leaks, fids = kl_eval_batch(
        op_pool, n_qubits, target_arr, ops, theta, mmd_eval)
    # Pick the final-batch best circuit by the chosen optimisation metric
    if args.loss_metric == 'kl_leak':
        final_score = kls + args.leak_lambda * leaks
    elif args.loss_metric == 'kl_mmd':
        final_score = kls + args.kl_mmd_lambda * mmds
    else:
        final_score = {'kl': kls, 'mmd': mmds, 'revkl': revkls,
                       'fidelity': fids}[args.loss_metric]
    fb = int(final_score.argmin().item())
    final_kl = float(kls[fb].item())
    print(f"  final best KL: {final_kl:.4f}  (mean={kls.mean().item():.4f}, "
          f"median={kls.median().item():.4f})")
    if mmd_eval is not None:
        final_mmd = float(mmds[fb].item())
        print(f"  final best MMD: {final_mmd:.6f}  (mean={mmds.mean().item():.6f}, "
              f"median={mmds.median().item():.6f})")
    else:
        final_mmd = float('nan')

    final_score_val = (final_mmd if args.loss_metric == 'mmd' else final_kl)
    if final_score_val < best_score:
        best_score = final_score_val
        best_ops = ops[fb].detach().cpu().clone()
        best_theta = theta[fb].detach().cpu().clone()
    best_kl = min(best_kl, final_kl)
    if mmd_eval is not None:
        best_mmd = min(best_mmd, final_mmd)

    torch.save({
        'model': model.state_dict(),
        'config': vars(args),
        'angle_values': angle_values,
        'target_arr': target_arr,
        'best_kl': best_kl,
        'best_mmd': best_mmd,
        'best_ops': best_ops,
        'best_theta': best_theta,
        'final_kls': kls.cpu().numpy(),
        'final_mmds': mmds.cpu().numpy(),
    }, args.save_ckpt)
    with open(os.path.join(args.log_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved: {args.save_ckpt}")
    print(f"OVERALL BEST KL: {best_kl:.4f}")
    if mmd_eval is not None:
        print(f"OVERALL BEST MMD: {best_mmd:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_qubits', type=int, default=10)
    p.add_argument('--n_gates', type=int, default=60)
    p.add_argument('--angle_bins_K', type=int, default=9,
                   help="Log-spaced angle set K: {±π/2^k, k=0..K} ∪ {0} → 2K+3 values "
                        "(paper Table 1 default K=9; small-n exploratory runs used K=12)")
    p.add_argument('--add_nn_3body', action='store_true',
                   help="Add nearest-neighbor matching 3-body ops (XXX, YYY, ZZZ on (i,i+1,i+2))")
    p.add_argument('--add_all_3body', action='store_true',
                   help="Add ALL matching 3-body ops (XXX/YYY/ZZZ on every triple). "
                        "Adds C(n,3)*3 ops — for permutation-symmetric (Dicke) and "
                        "2D-structured (BAS row+column) targets")
    p.add_argument('--symmetric_pool', action='store_true',
                   help="REPLACE pool with permutation-symmetric SUM ops "
                        "(S_alpha and S_{alpha,beta}, 9 ops total). "
                        "Each token is one global multi-qubit rotation. "
                        "For Dicke / symmetric-state targets (arXiv:2312.05060)")
    p.add_argument('--row_col_sym_pool', type=str, default='',
                   help="REPLACE pool with row/column-symmetric SUM ops for 2D grid "
                        "(e.g., '3,3' for 3x3 BAS). Per row/col: 1+2+3-body symmetric sums. "
                        "Respects the row-S_R x col-S_C symmetry of BAS-like targets")
    p.add_argument('--row_col_sym_pool_v2', type=str, default='',
                   help="row_col_sym_pool + DIAGONAL off-axis 2-body sums (6 extra ops). "
                        "Adds cross-row+cross-col coupling missing from v1, needed to "
                        "coherently superpose the stripes-subspace and bars-subspace of BAS")
    p.add_argument('--bas_full_pool', type=str, default='',
                   help="v2 pool + full S_n symmetric ops (9 ops). Combines row x col "
                        "symmetry with global permutation symmetry. 3x3: 87 ops")
    p.add_argument('--bas_rich_pool', type=str, default='',
                   help="v2 pool + ALL 3-body XXX/YYY/ZZZ Pauli on every triple. "
                        "3x3: 78 + C(9,3)*3 = 330 ops. Most expressive BAS pool")
    p.add_argument('--leak_first_steps', type=int, default=0,
                   help="Two-stage training: for the first N steps, score = leak only "
                        "(force policy onto target support); then switch to "
                        "--loss_metric. 0 = disabled.")
    p.add_argument('--n_embd', type=int, default=128)   # paper default
    p.add_argument('--n_layer', type=int, default=4)    # paper default
    p.add_argument('--n_head', type=int, default=4)     # paper default

    p.add_argument('--max_iters', type=int, default=400)
    p.add_argument('--num_samples', type=int, default=20,
                   help="Trajectories per step (paper GQE used N=50)")
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--temp_start', type=float, default=1.0,
                   help="Sampling temperature (fixed; replay buffer drives curriculum)")
    p.add_argument('--ent_coef', type=float, default=0.02,
                   help="Entropy regularization coefficient in sft_loss "
                        "(paper Table 1 default β=0.02; small-n exploratory runs used β=0.01)")

    # Replay buffer
    p.add_argument('--buf_capacity', type=int, default=1000,
                   help="Max circuits kept in replay buffer (sorted by KL)")
    p.add_argument('--buf_batch_size', type=int, default=64,
                   help="Circuits sampled from buffer per gradient step")
    p.add_argument('--buf_min_size', type=int, default=50,
                   help="Buffer must reach this size before training starts")
    p.add_argument('--kl_temp_min', type=float, default=0.01,
                   help="Floor for adaptive kl_temp = max(buf_best_kl*0.5, kl_temp_min)")
    p.add_argument('--kl_temp_fixed', type=float, default=0.0,
                   help="If >0, use this constant kl_temp instead of the "
                        "adaptive best_kl*0.5 rule (controlled study)")
    p.add_argument('--kl_temp_std_coef', type=float, default=0.0,
                   help="If >0, kl_temp = max(coef*std(batch_KL), kl_temp_min)"
                        " — spread-based adaptive T (overrides fixed/adaptive)")

    p.add_argument('--n_final_samples', type=int, default=512)
    p.add_argument('--temp_final', type=float, default=0.3,
                   help="Sampling temp for final evaluation (lower = more concentrated)")

    p.add_argument('--target_kind', type=str, default='gaussian',
                   choices=['gaussian', 'bimodal', 'exponential', 'delta', 'ghz', 'bas',
                            'w_state', 'lognormal', 'cardinality', 'uniform'])
    p.add_argument('--target_kw', type=str, default='{"center":0.5,"sigma":0.10}')

    p.add_argument('--mmd_sigmas', type=str, default='',
                   help="Comma-separated RBF bandwidths, e.g. '0.25,60'. Empty = no MMD eval")
    p.add_argument('--mmd_kernel', choices=['index', 'hamming', 'grid'], default='index',
                   help="MMD kernel geometry. 'index' = (i-j)^2 (smooth densities); "
                        "'hamming' = bitstring Hamming distance (GHZ family); "
                        "'grid' = row/column-grouped Hamming (BAS; needs --mmd_grid_shape)")
    p.add_argument('--mmd_grid_shape', type=str, default='',
                   help="Grid shape 'R,C' for --mmd_kernel grid, e.g. '3,3' for 3x3 BAS")
    p.add_argument('--loss_metric',
                   choices=['kl', 'mmd', 'revkl', 'kl_leak', 'fidelity', 'kl_mmd'],
                   default='kl',
                   help="Optimisation metric for buffer/loss/best-tracking. "
                        "'mmd' requires --mmd_sigmas (MMD ablation); "
                        "'revkl' = reverse KL D(p||q); "
                        "'kl_leak' = D(q||p) + lambda * P(invalid); "
                        "'fidelity' = 1 - |<psi_target | psi_c>|^2 (pure-state targets); "
                        "'kl_mmd' = D(q||p) + lambda * MMD^2 (needs --mmd_sigmas)")
    p.add_argument('--kl_mmd_lambda', type=float, default=1.0,
                   help="Lambda for kl_mmd hybrid loss (only used with --loss_metric kl_mmd)")
    p.add_argument('--leak_lambda', type=float, default=1.0,
                   help="Weight on the leak-onto-forbidden-states penalty (only for --loss_metric kl_leak)")
    p.add_argument('--seed_buffer_ckpts', type=str, default='',
                   help="Comma-separated smaller-qubit ckpt paths for Sub-grid "
                        "Curriculum Transfer (e.g. 'ckpts/extras_bas4.pt,ckpts/extras_bas4.pt')")
    p.add_argument('--seed_qubit_maps', type=str, default='',
                   help="Semicolon-separated qubit mappings, one per seed ckpt "
                        "(e.g. '0,1,3,4;1,2,4,5;3,4,6,7;4,5,7,8' for 3x3 sub-grids)")
    p.add_argument('--seed_pinned', action='store_true',
                   help="If set, seed circuits are pinned in the buffer (never evicted)")
    p.add_argument('--resume_from', type=str, default='',
                   help="Resume training from a previous ckpt: loads transformer "
                        "weights and pins the ckpt's best circuit in the buffer "
                        "(use to continue past dev wall on a still-descending run)")
    p.add_argument('--lr_decay', choices=['none', 'cosine'], default='none',
                   help="Learning-rate schedule. 'cosine' anneals from --lr "
                        "down to --lr_min over max_iters (fine-tune at plateau)")
    p.add_argument('--lr_min', type=float, default=1e-5,
                   help="Floor for --lr_decay cosine schedule")
    p.add_argument('--mmd_max_n_states', type=int, default=2**20,
                   help="Skip MMD if n_states exceeds this. MMDEvaluator auto-switches "
                        "to an O(n_states log n_states)-time/O(n_states)-memory path "
                        "above 4096 states (fast_mmd.py), so this ceiling now exists "
                        "only as a hard cap for pathologically large runs (default "
                        "2^20 = n_qubits 20), not the old dense-matrix memory limit")

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--save_ckpt', type=str, default='ckpts/gqe_single_target.pt')
    p.add_argument('--log_dir', type=str, default='gqe_experiments/single_target')

    args = p.parse_args()
    os.makedirs(os.path.dirname(args.save_ckpt) or '.', exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
