"""Print the numeric tables of the paper from the files in this repository (CPU only).

    python make_tables.py            # all tables
    python make_tables.py continuous # one of: continuous, discrete, n20_protocol, ngates

Sources
  Table 1 (tab:continuous), Table 2 (tab:discrete)
      results/circuit_stats.json, one row per (target, model). It is written by
      compute_circuit_stats.py, which re-simulates every G-QCBM checkpoint and every
      V-QCBM result file and recomputes KL(target || model) and the compiled CX count.
      Exception: V-QCBM+KL at n=20 is the best KL at real step 4,000, read from the run
      log (make_fig_continuous_grid.step4000_kl; see results/sel_kl_exact_gb200/README.md).
  SI tab:n20_protocol
      multi-cycle: results/sel_kl_exact_gb200/runs/t1_n20/job_8803244/results/*.json
      single-trajectory: step-4,000 values from runs/t1_n20_aligned/job_9015307/logs/.
      Steps and wall-clock are in the same folders (logs/summary.txt, logs/*.log).
  SI tab:abl_ngates
      paper/figs/fig_ngates_sweep.csv (best_kl of ckpts/gqe_gauss_{10,15,20}q_g{20,30,50}.pt
      and ckpts/gqe_gauss_20q.pt; rebuild with `python make_ngates_sweep.py`, needs torch + ckpts).

Values are printed with four decimals; "~0" means |KL| < 1e-6, as in the paper.
"""
import csv
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

FAMILIES = [('gauss', 'Gaussian'), ('bimodal', 'Bimodal'), ('exp', 'Exponential'),
            ('lognormal', 'Log-normal')]
DISCRETE = [('ghz5_5q', 'GHZ-5', 5), ('ghz10_10q', 'GHZ-10', 10),
            ('cardinality_5q', 'Dicke D^5_2', 5), ('cardinality_10q', 'Dicke D^10_5', 10),
            ('bas4_4q', 'BAS 2x2', 4), ('bas9_9q', 'BAS 3x3', 9)]
# circuit_stats.json method names -> paper names
MODELS = [('GQCBM', 'G-QCBM+KL'), ('GQCBM+MMD', 'G-QCBM+MMD'),
          ('QCBM-SEL+KL', 'V-QCBM+KL'), ('QCBM-SEL+MMD', 'V-QCBM+MMD')]


def fmt(kl):
    return '~0' if abs(kl) < 1e-6 else f'{kl:.4f}'


def stats():
    return {(r['target'], r['method']): r for r in json.load(open('results/circuit_stats.json'))}


def step4000(key):
    from make_fig_continuous_grid import step4000_kl  # imports matplotlib only
    return step4000_kl(key)


def table_continuous():
    st = stats()
    print('Table 1 (tab:continuous): KL(target || model)')
    print(f"{'Target':12s} {'n':>3s} " + ' '.join(f'{m:>11s}' for _, m in MODELS))
    for key, name in FAMILIES:
        for n in (10, 15, 20):
            cells = []
            for meth, _ in MODELS:
                if meth == 'QCBM-SEL+KL' and n == 20:
                    kl = step4000(key)
                else:
                    kl = st[(f'{key}_{n}q', meth)]['best_kl']
                cells.append(fmt(kl))
            print(f'{name:12s} {n:3d} ' + ' '.join(f'{c:>11s}' for c in cells))
    print()


def table_discrete():
    st = stats()
    print('Table 2 (tab:discrete): KL / CX')
    print(f"{'Target':14s} {'n':>3s} " + ' '.join(f'{m:>18s}' for _, m in MODELS))
    for key, name, n in DISCRETE:
        cells = []
        for meth, _ in MODELS:
            r = st[(key, meth)]
            if key == 'ghz10_10q' and meth == 'GQCBM':
                cells.append('--')  # no G-QCBM+KL run trained on GHZ-10 (paper: "--")
                continue
            cells.append(f"{fmt(r['best_kl'])} / {r['cx']}")
        print(f'{name:14s} {n:3d} ' + ' '.join(f'{c:>18s}' for c in cells))
    print()


def table_n20_protocol():
    st = stats()
    base = 'results/sel_kl_exact_gb200/runs/t1_n20/job_8803244/results/qcbm_n20m2_kl_{}_20q.json'
    print('SI tab:n20_protocol: V-QCBM+KL at n=20')
    print(f"{'Target':12s} {'multi-cycle':>12s} {'single-traj':>12s} {'Delta':>8s} {'G-QCBM+KL':>10s}")
    for key, name in FAMILIES:
        rec = json.load(open(base.format(key)))
        rec = rec[-1] if isinstance(rec, list) else rec
        mc, st4 = rec['best_kl'], step4000(key)
        g = st[(f'{key}_20q', 'GQCBM')]['best_kl']
        print(f'{name:12s} {mc:12.4f} {st4:12.4f} {100 * (mc / st4 - 1):+7.1f}% {g:10.4f}')
    print()


def table_ngates():
    rows = list(csv.DictReader(open('paper/figs/fig_ngates_sweep.csv')))
    print('SI tab:abl_ngates: G-QCBM+KL best KL, single-mode Gaussian')
    print(f"{'Protocol':22s} {'n=10':>8s} {'n=15':>8s} {'n=20':>8s}")
    for proto, label in [('single_shot_3000iter', 'Single-shot'),
                         ('chain_plateau_multicycle', 'Multi-cycle')]:
        for g in (20, 30, 50):
            vals = {int(r['n_qubits']): float(r['best_kl']) for r in rows
                    if r['protocol'] == proto and int(r['n_gates']) == g}
            if vals:
                cells = [f'{vals[n]:.4f}' if n in vals else '---' for n in (10, 15, 20)]
                print(f'{label + f" G={g}":22s} ' + ' '.join(f'{c:>8s}' for c in cells))
    print()


TABLES = {'continuous': table_continuous, 'discrete': table_discrete,
          'n20_protocol': table_n20_protocol, 'ngates': table_ngates}

if __name__ == '__main__':
    for name in (sys.argv[1:] or TABLES):
        TABLES[name]()
