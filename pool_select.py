"""Single source of truth for reconstructing the operator pool a GQCBM
checkpoint was trained with, from its saved ``config`` dict.

compute_circuit_stats.py and make_all_dist_compare.py import it, so every
script rebuilds a checkpoint's pool the same way.

Order matters: flags are checked in the same precedence the training script
uses, and the default 1+2-body pool is returned when no special-pool flag is
set. NOTE: for the three checkpoints whose config dict was overwritten by a
mismatched resume chain (gqe_bas9_v2_ent.pt, gqe_ghz5.pt,
gqe_hamMMD_ghz10_10q.pt) the *pool* flags survived intact (they were passed
identically by the chain), so pool reconstruction is still correct even
though other config fields (K, n_embd, n_layer, ent_coef) are not.
"""
from gqe_single_target import (
    build_op_pool_symmetric,
    build_op_pool_row_col_sym,
    build_op_pool_row_col_sym_v2,
    build_op_pool_with_all_3body,
    build_op_pool_with_nn_3body,
    build_op_pool_bas_full,
    build_op_pool_bas_rich,
)
from circuit_ops import build_op_pool


def select_pool(nq, cfg):
    """Return the operator pool (list of cudaq SpinOperators) for a
    checkpoint config ``cfg`` on ``nq`` qubits."""
    cfg = cfg or {}
    if cfg.get('symmetric_pool'):
        return build_op_pool_symmetric(nq)
    if cfg.get('row_col_sym_pool_v2'):
        R, C = (int(s) for s in cfg['row_col_sym_pool_v2'].split(','))
        return build_op_pool_row_col_sym_v2(nq, (R, C))
    if cfg.get('row_col_sym_pool'):
        R, C = (int(s) for s in cfg['row_col_sym_pool'].split(','))
        return build_op_pool_row_col_sym(nq, (R, C))
    if cfg.get('bas_full_pool'):
        R, C = (int(s) for s in cfg['bas_full_pool'].split(','))
        return build_op_pool_bas_full(nq, (R, C))
    if cfg.get('bas_rich_pool'):
        R, C = (int(s) for s in cfg['bas_rich_pool'].split(','))
        return build_op_pool_bas_rich(nq, (R, C))
    if cfg.get('add_all_3body'):
        return build_op_pool_with_all_3body(nq)
    if cfg.get('add_nn_3body'):
        return build_op_pool_with_nn_3body(nq)
    return build_op_pool(nq)
