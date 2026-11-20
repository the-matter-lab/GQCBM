"""Split each n=20 forward KL into a between-bin and a within-bin part (Fig. 2 caption).

Chain rule over the 512 display bins of Fig. 2:
    KL(t || p) = KL(T || P) + sum_b T_b KL(t|b || p|b),
where T, P are the bin masses. The between-bin part is computed from the binned
curves in paper/figure_data/fig_continuous_grid.csv; within = total - between, with
the total being the Table 1 value of that series. For V-QCBM+KL the binned curve
uses the end-of-run parameters while the total is the step-4,000 value (they differ
by at most 3e-4), and the G-QCBM bimodal checkpoint was trained on centres rounded to
0.2857/0.7143; both show up as within-bin parts of about -1e-4 for G-QCBM.

    python scripts/analyze_kl_bin_split.py
Writes results/analysis/n20_kl_bin_split.json.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import make_fig_continuous_grid as G  # noqa: E402

OUT = 'results/analysis/n20_kl_bin_split.json'


def main():
    rows = [r for r in csv.DictReader(open(G.CSV_PATH)) if r['n_qubits'] == '20']
    stored = G.gqcbm_stored_kl()
    out = []
    for _, fam, _, _ in G.FAMILIES:
        rs = [r for r in rows if r['family'] == fam]
        t = np.array([float(r['target']) for r in rs])
        totals = {'G-QCBM+KL': ('gqcbm_kl', stored[(f'{fam}_20q', 'GQCBM')]),
                  'V-QCBM+KL': ('qcbm_sel_kl', G.step4000_kl(fam)),
                  'V-QCBM+MMD': ('qcbm_sel_mmd', G.stored_kl(G.sel_mmd_json(fam, 20)))}
        for name, (col, total) in totals.items():
            p = np.array([float(r[col]) for r in rs])
            m = t > 0
            between = float(np.sum(t[m] * np.log(t[m] / p[m])))
            within = total - between
            out.append(dict(family=fam, method=name, kl_total=total, kl_between_bins=between,
                            kl_within_bins=within, within_fraction=within / total, n_bins=len(rs)))
            print(f'{fam:10s} {name:11s} total {total:.4f}  between {between:.4f}  '
                  f'within {within:.4f} ({100 * within / total:.0f}%)')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
