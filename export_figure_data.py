"""Export the underlying numerical data for every main-text and supplementary
figure into paper/figure_data/ as plain CSV files, for archival and reuse.

Main-text figures:
  fig_qcbm_compare  — 6q bimodal: learned vs target distribution + trajectory
  fig_cross_dist    — 4 learned-vs-target distribution overlays
  fig_phase         — 15q training dynamics (KL / entropy / temperature)
  scaling_kl_cnot   — KL and CNOT counts vs qubit count
Supplementary figures:
  fig_s10_2_beta    — entropy-coefficient sweep
  fig_s10_3_random  — random-circuit baseline
  fig_s10_4_mmd     — MMD-loss comparison
  fig_s10_5_budget  — QCBM training-budget trajectories
"""
import csv, json, math, os, re, sys
import numpy as np
import torch, cudaq, cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqe_single_target import build_target
from circuit_ops import build_op_pool, circuit_kernel

cudaq.set_target('nvidia', option='fp64')
OUT = 'paper/figure_data'
os.makedirs(OUT, exist_ok=True)


def write_csv(name, header, rows):
    with open(f'{OUT}/{name}', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  wrote {OUT}/{name}  ({len(rows)} rows)')


def learned_probs(ckpt):
    d = torch.load(f'ckpts/{ckpt}.pt', map_location='cpu', weights_only=False)
    nq = int(round(math.log2(len(d['target_arr']))))
    ops, th = d['best_ops'].tolist(), d['best_theta'].tolist()
    pool = build_op_pool(nq)
    fc, fw = [], []
    for idx, ang in zip(ops, th):
        if abs(float(ang)) < 1e-8:
            continue
        for term in float(ang) * pool[int(idx)]:
            c = term.get_coefficient().real
            w = term.to_string(nq).split(']')[-1].strip().ljust(nq, 'I')
            fc.append(c); fw.append(cudaq.pauli_word(w))
    if not fc:
        probs = np.ones(2 ** nq) / (2 ** nq)
    else:
        st = cudaq.get_state(circuit_kernel, nq, fc, fw)
        probs = cp.asnumpy(cp.abs(cp.array(st, dtype=cp.complex128)) ** 2)
    target = (d['target_arr'].numpy() if hasattr(d['target_arr'], 'numpy')
              else np.array(d['target_arr']))
    return target, probs


# ── Main fig: qcbm_compare (6q bimodal) ──────────────────────────────────────
def export_qcbm_compare():
    print('[fig_qcbm_compare]')
    target, probs = learned_probs('seed_6q_42')
    write_csv('fig_qcbm_compare_distribution.csv',
              ['basis_index', 'target_prob', 'gqcbm_prob'],
              [[i, f'{t:.8e}', f'{p:.8e}'] for i, (t, p)
               in enumerate(zip(target, probs))])
    log = json.load(open('gqe_experiments/seed_6q_42/log.json'))
    write_csv('fig_qcbm_compare_trajectory.csv',
              ['step', 'best_kl', 'best_mmd'],
              [[r['step'], r['best_kl'], r.get('best_mmd', '')]
               for r in log])


# ── Main fig: cross_dist (4 panels) ──────────────────────────────────────────
def export_cross_dist():
    print('[fig_cross_dist]')
    panels = [('gauss_5q', 'seed_5q_42'), ('exp_5q', 'extras_exp5'),
              ('ghz_5q', 'gqe_ghz5'),     ('gauss_15q', 'seed_15q_42')]
    for tag, ckpt in panels:
        target, probs = learned_probs(ckpt)
        write_csv(f'fig_cross_dist_{tag}.csv',
                  ['basis_index', 'target_prob', 'gqcbm_prob'],
                  [[i, f'{t:.8e}', f'{p:.8e}'] for i, (t, p)
                   in enumerate(zip(target, probs))])


# ── Main fig: phase (15q dynamics) ───────────────────────────────────────────
def export_phase():
    print('[fig_phase]')
    log = json.load(open('gqe_experiments/seed_15q_42/log.json'))
    write_csv('fig_phase_15q.csv',
              ['step', 'best_kl', 'kl_min', 'policy_entropy', 'kl_temp'],
              [[r['step'], r['best_kl'], r.get('kl_min', ''),
                r.get('ent', ''), r.get('kl_temp', '')] for r in log])


# ── Main fig: scaling_kl_cnot ────────────────────────────────────────────────
def export_scaling():
    print('[scaling_kl_cnot]')
    rows = [
        [5,  0.0016, 24,  0.1430, 20],
        [10, 0.0064, 42,  0.5735, 60],
        [15, 0.0055, 56,  0.6523, 120],
    ]
    write_csv('fig_scaling_kl_cnot.csv',
              ['n_qubits', 'gqcbm_kl', 'gqcbm_cnots',
               'qcbm_sel_kl', 'qcbm_sel_cnots'], rows)


# ── Supp fig S10.2: beta sweep ───────────────────────────────────────────────
def export_beta():
    print('[fig_s10_2_beta]')
    rows = []
    for b in ['0.01', '0.02', '0.03', '0.05', '0.10', '0.15']:
        f = f'gqe_experiments/beta_{b}/log.json'
        if os.path.exists(f):
            r = json.load(open(f))
            rows.append([b, min(x['best_kl'] for x in r)])
    write_csv('fig_s10_2_beta_sweep.csv', ['beta', 'best_kl'], rows)


# ── Supp fig S10.3: random baseline ──────────────────────────────────────────
def export_random():
    print('[fig_s10_3_random]')
    gqcbm = {5: 0.0016, 10: 0.0064, 15: 0.0055}
    rows = []
    if os.path.exists('results/random_baseline.json'):
        for r in json.load(open('results/random_baseline.json')):
            nq = r['n_qubits']
            rows.append([nq, r['G'], r['N'], f"{r['best_kl']:.6f}",
                         f"{r['mean_kl']:.6f}", gqcbm.get(nq, '')])
    write_csv('fig_s10_3_random_baseline.csv',
              ['n_qubits', 'G', 'N_random', 'random_best_kl',
               'random_mean_kl', 'gqcbm_kl'], rows)


# ── Supp fig S10.4: MMD comparison ───────────────────────────────────────────
def export_mmd():
    """Both columns read from real sources (never hardcoded literals):
    GQCBM from the *.pt checkpoint's stored best_mmd (an MMD-objective
    training run, ckpts/gqe_mmdloss2_*.pt); QCBM-SEL from the matching
    *.json sweep/restart result's best_mmd. Gauss-15 is omitted: no
    GQCBM+MMD checkpoint was ever trained at that scale (verified absent
    from ckpts/), not because of a kernel-matrix memory limit (a dense
    32,768x32,768 kernel is only ~8.6 GB, well within reach; that
    infeasibility claim was specific to n=20, not n=15).
    """
    print('[fig_s10_4_mmd]')
    targets = [
        ('Bell',      2,  'gqe_mmdloss2_bell_2q',    'results/archive/qcbmpl_v2_mmd_bell_2q.json'),
        ('GHZ-5',     5,  'gqe_mmdloss2_ghz5_5q',    'results/archive/qcbmpl_v2_mmd_ghz5_5q.json'),
        ('BAS-4',     4,  'gqe_mmdloss2_bas4_4q',    'results/archive/qcbmpl_v2_mmd_bas4_4q.json'),
        ('Gauss-5',   5,  'gqe_mmdloss2_gauss_5q',   'results/archive/qcbmpl_v2_mmd_gauss_5q_s42.json'),
        ('Bimodal-6', 6,  'gqe_mmdloss2_bimodal_6q', 'results/archive/qcbmpl_v2_mmd_bimodal_6q_s42.json'),
        ('Exp-5',     5,  'gqe_mmdloss2_exp_5q',     'results/archive/qcbmpl_v2_mmd_exp_5q.json'),
        # Gauss-10 uses the same sweep_L4 source as the main-text Table 1
        # QCBM-SEL+MMD column (complete, 20 restarts), not the partial v3 run.
        ('Gauss-10',  10, 'gqe_mmdloss2_gauss_10q',  'results/sweep_L4_mmd_gauss_10q.json'),
    ]
    rows = []
    for name, n, ckpt, jf in targets:
        d = torch.load(f'ckpts/{ckpt}.pt', map_location='cpu', weights_only=False)
        gqcbm_mmd = float(d['best_mmd'])
        jd = json.load(open(jf))
        jr = jd[0] if isinstance(jd, list) else jd
        assert jr.get('partial', False) is False, f'{jf} is a partial/incomplete run'
        qcbm_sel_mmd = float(jr['best_mmd'])
        rows.append([name, n, qcbm_sel_mmd, gqcbm_mmd])
    write_csv('fig_s10_4_mmd_comparison.csv',
              ['target', 'n_qubits', 'qcbm_sel_mmd_loss_mmd',
               'gqcbm_posthoc_mmd'],
              [[t, n, f'{q:.3e}', f'{g:.3e}'] for t, n, q, g in rows])


# ── Supp fig S10.5: QCBM budget trajectories ─────────────────────────────────
def export_budget():
    print('[fig_s10_5_budget]')
    rows = []
    for n, lr in [(5, '0.01'), (5, '0.03'), (10, '0.01'),
                  (10, '0.03'), (15, '0.01'), (15, '0.03')]:
        fn = f'slurm_logs/qcbm_nc_{n}q_lr{lr}.log'
        if not os.path.exists(fn):
            continue
        for line in open(fn):
            m = re.search(
                r'step\s+(\d+)/\d+\s+KL=([0-9.eE+-]+)\s+best=([0-9.eE+-]+)',
                line)
            if m:
                rows.append([n, lr, int(m.group(1)),
                             m.group(2), m.group(3)])
    write_csv('fig_s10_5_qcbm_budget.csv',
              ['n_qubits', 'learning_rate', 'step',
               'current_kl', 'best_kl'], rows)


def write_readme():
    txt = """Figure data — Generative Quantum Circuit Born Machines
========================================================

Plain-CSV numerical data underlying every figure in the main text and
Supplementary Information. One CSV per figure (or panel).

Main text
  fig_qcbm_compare_distribution.csv  6q bimodal: target vs GQCBM probability
  fig_qcbm_compare_trajectory.csv    6q bimodal: best KL / MMD vs step
  fig_cross_dist_gauss_5q.csv        5q Gaussian: target vs GQCBM probability
  fig_cross_dist_exp_5q.csv          5q exponential: target vs GQCBM
  fig_cross_dist_ghz_5q.csv          5q GHZ: target vs GQCBM
  fig_cross_dist_gauss_15q.csv       15q Gaussian: target vs GQCBM
  fig_phase_15q.csv                  15q training dynamics
  fig_scaling_kl_cnot.csv            KL and CNOT counts vs qubit count

Supplementary Information (Section S10 ablations)
  fig_s10_2_beta_sweep.csv           entropy-coefficient sweep
  fig_s10_3_random_baseline.csv      random-circuit baseline
  fig_s10_4_mmd_comparison.csv       MMD-loss comparison
  fig_s10_5_qcbm_budget.csv          QCBM training-budget trajectories

All probabilities sum to one. KL = forward Kullback-Leibler divergence.
"""
    with open(f'{OUT}/README.txt', 'w') as f:
        f.write(txt)
    print(f'  wrote {OUT}/README.txt')


if __name__ == '__main__':
    export_qcbm_compare()
    export_cross_dist()
    export_phase()
    export_scaling()
    export_beta()
    export_random()
    export_mmd()
    export_budget()
    write_readme()
    print('\nAll figure data exported to paper/figure_data/')
