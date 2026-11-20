"""4x3 grid of the continuous targets (rows: Gaussian, bimodal, exponential,
log-normal; columns: n = 10, 15, 20), each panel in the style of Fig. 2:
target + GQCBM+KL + QCBM-SEL+KL + QCBM-SEL+MMD, coarse-grained into 512 bins.

Runs on CPU (no CUDA-Q).

  python make_fig_continuous_grid.py                    # rebuild CSV, then plot
  python make_fig_continuous_grid.py --from-csv         # plot from the CSV only
  python make_fig_continuous_grid.py --ckpt-dir DIR     # where GQCBM ckpts are
  python make_fig_continuous_grid.py --include-older    # also draw older (non-
                                                        # Table 1) n=15 checkpoints

Data sources, one per series and cell (the same runs as Table 1 / Fig. 3):
  QCBM-SEL+KL   results/sel_kl_exact_gb200/ (exact-gradient re-run; n=20 is the
                4,000-step protocol). Circuit re-simulated from best_params.
  QCBM-SEL+MMD  the result JSONs in compute_circuit_stats.ENTRIES.
  GQCBM+KL      the checkpoints in compute_circuit_stats.ENTRIES, where available:
                - Hugging Face: gqe_exp_10q_g20, gqe_gauss_20q (CPU re-simulation).
                - Hugging Face: gqe_q10_small_v2, gqe_exp_20q, gqe_lognormal_10q, gqe_lognormal_20q
                  (CPU re-simulation).
                - gqe_q15_v4 (= seed_15q_42): per-state export
                  paper/figure_data/fig_cross_dist_gauss_15q.csv.
                - gqe_bimodal_20q: binned CUDA-Q export in fig2_bimodal20_3way.csv
                  (its CPU re-simulation does not reproduce).
                Missing, so the panel shows only the stored KL from
                results/circuit_stats.json: bimodal n=10 and n=15, exponential n=15,
                log-normal n=15 (gqe_bimodal_{10,15}q_g30, gqe_exp_15q_g30,
                gqe_lognormal_15q_g50).
Every re-simulated series must reproduce its stored best_kl (rtol 1e-4), or the
script stops and writes nothing.

Writes paper/figure_data/fig_continuous_grid.csv and
paper/figs/fig_continuous_grid.{pdf,png}.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, 'scripts'))
sys.path.insert(0, os.path.join(HERE, 'figures'))  # nature_style, gqcbm_palette

CSV_PATH = 'paper/figure_data/fig_continuous_grid.csv'
OUT = 'paper/figs/fig_continuous_grid'
N_BINS = 512
RUNS = 'results/sel_kl_exact_gb200/runs/'

FAMILIES = [  # (row label, key, target kind, target kwargs as in qcbm_pennylane.py)
    ('Gaussian', 'gauss', 'gaussian', {'center': 0.5, 'sigma': 0.10}),
    ('Bimodal', 'bimodal', 'bimodal', {'c1': 2 / 7, 'c2': 5 / 7, 'sigma': 0.125}),
    ('Exponential', 'exp', 'exponential', {}),
    ('Log-normal', 'lognormal', 'lognormal', {'mu': -0.7, 'sigma': 0.5}),
]
NQ = (10, 15, 20)


def sel_kl_json(key, n):
    if n == 10:
        return f'{RUNS}t1_n10/job_8803242/results/qcbm_aligned_kl_{key}_10q.json'
    if n == 15:
        name = 'gauss_15q_s42' if key == 'gauss' else f'{key}_15q'
        return f'{RUNS}t1_n15/job_8828095/results/qcbm_aligned_kl_{name}.json'
    return f'{RUNS}t1_n20_aligned/job_9015307/results/qcbm_aligned_kl_{key}_20q.json'


def sel_mmd_json(key, n):
    if n == 20:
        return f'results/qcbm_n20m2_mmd_{key}_20q.json'
    return f'results/sweep_L4_mmd_{key}_{n}q{"_s42" if (key, n) == ("gauss", 15) else ""}.json'


GQCBM_SOURCE = {  # cell -> 'ckpt:<name>' | 'fig2csv'; absent = not available
    ('exp', 10): 'ckpt:gqe_exp_10q_g20',
    ('gauss', 10): 'ckpt:gqe_q10_small_v2',
    ('exp', 20): 'ckpt:gqe_exp_20q',
    ('lognormal', 10): 'ckpt:gqe_lognormal_10q',
    ('lognormal', 20): 'ckpt:gqe_lognormal_20q',
    ('gauss', 20): 'ckpt:gqe_gauss_20q',
    ('bimodal', 20): 'fig2csv',
    # Table 1 checkpoints on Hugging Face (scripts/download_ckpts.py). gqe_q15_v4 is token-identical to seed_15q_42.
    ('gauss', 15): 'ckpt:gqe_q15_v4',
    ('bimodal', 10): 'ckpt:gqe_bimodal_10q_g30',
    ('bimodal', 15): 'ckpt:gqe_bimodal_15q_g30',
    ('exp', 15): 'ckpt:gqe_exp_15q_g30',
    ('lognormal', 15): 'ckpt:gqe_lognormal_15q_g50',
}
# Older checkpoints of a cell, NOT the Table 1 circuit. Drawn dotted, only with
# --include-older. Empty since every cell has its Table 1 checkpoint (2026-11-16);
# formerly bimodal/exp/lognormal n=15 -> gqe_{bimodal,exp,lognormal}_15q.
GQCBM_OLDER = {}
INCLUDE_OLDER = '--include-older' in sys.argv


def step4000_kl(key):
    """QCBM-SEL+KL n=20 value reported in Table 1: best KL at real step 4,000, read
    from the log of the last job (it holds every attempt), with the method given in
    results/sel_kl_exact_gb200/README.md (each attempt's step counter restarts)."""
    import re
    log = (f'{RUNS}t1_n20_aligned/job_9015307/logs/qcbm_aligned_kl_{key}_20q.log')
    cum = s = 0
    for line in open(log):
        if line.startswith('== attempt'):
            cum, s = cum + s, 0
            continue
        m = re.search(r'step +(\d+)/', line)
        if m:
            s = int(m.group(1))
            if cum + s >= 4000:
                return float(re.search(r'best=(\S+)', line).group(1))
    raise RuntimeError(f'{log}: never reached step 4000')


def stored_kl(js):
    rec = json.load(open(js))
    rec = rec[-1] if isinstance(rec, list) else rec
    return float(rec['best_kl'])


def build_target(kind, n_states, kw):
    """Same formulas as gqe_single_target.build_target (continuous kinds only)."""
    idx = np.arange(n_states, dtype=np.float64)
    if kind == 'gaussian':
        arr = np.exp(-0.5 * ((idx - kw['center'] * n_states) / (kw['sigma'] * n_states)) ** 2)
    elif kind == 'bimodal':
        s = kw['sigma'] * n_states
        arr = (np.exp(-0.5 * ((idx - kw['c1'] * n_states) / s) ** 2) +
               np.exp(-0.5 * ((idx - kw['c2'] * n_states) / s) ** 2))
    elif kind == 'exponential':
        arr = np.exp(-idx / (kw.get('scale', 0.25) * n_states))
    elif kind == 'lognormal':
        t = (idx + 1.0) / n_states
        arr = np.exp(-0.5 * ((np.log(t) - kw['mu']) / kw['sigma']) ** 2) / t
    return arr / arr.sum()


def binned(p):
    return p.reshape(N_BINS, -1).sum(axis=1)


def gqcbm_stored_kl():
    return {(d['target'], d['method']): d['best_kl']
            for d in json.load(open('results/circuit_stats.json'))}


def build_csv(ckpt_dir):
    import torch
    import export_level2_cpu as E
    stored = gqcbm_stored_kl()
    fig2 = list(csv.DictReader(open('paper/figure_data/fig2_bimodal20_3way.csv')))
    rows, ok = [], True
    for _, key, kind, kw in FAMILIES:
        for n in NQ:
            target = build_target(kind, 2 ** n, kw)
            series = {}
            for col, js in (('qcbm_sel_kl', sel_kl_json(key, n)),
                            ('qcbm_sel_mmd', sel_mmd_json(key, n))):
                p, k = E.sel_probs(js)
                ok &= E.check(f'{key}_{n}q {col}', target, p, k)
                series[col] = binned(p)
            src = GQCBM_SOURCE.get((key, n))
            older = False
            if src is None and INCLUDE_OLDER and (key, n) in GQCBM_OLDER:
                src, older = GQCBM_OLDER[(key, n)], True
            if src and src.startswith('ckpt:'):
                d = torch.load(os.path.join(ckpt_dir, src[5:] + '.pt'),
                               map_location='cpu', weights_only=False)
                pool = E.select_pool(n, d.get('config', {}) or {})
                psi = np.full([2] * n, 1 / math.sqrt(2 ** n), dtype=complex)
                for i, th in zip(d['best_ops'].tolist(), d['best_theta'].tolist()):
                    if abs(float(th)) < 1e-8:
                        continue
                    for term in pool[int(i)]:
                        psi = E.exp_pauli(psi, term, float(th))
                p = E.probs_little_endian(psi)
                if older:
                    # Qualitative curves only: allow 0.1 % (gqe_bimodal_15q re-simulates
                    # to 0.053487 vs stored 0.053463 on CPU); label with the recomputed KL.
                    got = E.kl(target, p)
                    good = math.isclose(got, float(d['best_kl']), rel_tol=1e-3)
                    print(f'  {key}_{n}q gqcbm_kl (older) KL recomputed={got:.6f} '
                          f'stored={float(d["best_kl"]):.6f} {"ok (rtol 1e-3)" if good else "MISMATCH"}')
                    ok &= good
                    series['gqcbm_kl_value'] = got
                else:
                    # Check against the target the checkpoint was trained on: G-QCBM
                    # bimodal runs used rounded centres 0.2857/0.7143 (V-QCBM: 2/7, 5/7),
                    # which moves KL in the 4th digit (bimodal n=10: 0.004503 vs 0.004500).
                    t_ck = (np.asarray(d['target_arr'], float) if 'target_arr' in d else target)
                    ok &= E.check(f'{key}_{n}q gqcbm_kl', t_ck, p, float(d['best_kl']))
                    series['gqcbm_kl_value'] = float(d['best_kl'])
                series['gqcbm_kl'] = binned(p)
            elif src and src.startswith('csv:'):
                r_ = list(csv.DictReader(open(src[4:])))
                p = np.array([float(x['gqcbm_prob']) for x in r_])
                ok &= E.check(f'{key}_{n}q gqcbm_kl', target, p,
                              stored[(f'{key}_{n}q', 'GQCBM')])
                series['gqcbm_kl'] = binned(p)
            elif src == 'fig2csv':
                series['gqcbm_kl'] = np.array([float(r['gqcbm_kl']) for r in fig2])
            tb = binned(target)
            for b in range(N_BINS):
                g = series.get('gqcbm_kl')
                rows.append([key, n, b, 'older' if older else ('table1' if g is not None else ''),
                             series.get('gqcbm_kl_value', ''), f'{tb[b]:.10g}',
                             '' if g is None else f'{g[b]:.10g}',
                             f'{series["qcbm_sel_kl"][b]:.10g}',
                             f'{series["qcbm_sel_mmd"][b]:.10g}'])
    if not ok:
        sys.exit('KL check failed; nothing written')
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'n_qubits', 'bin', 'gqcbm_source', 'gqcbm_ckpt_kl', 'target', 'gqcbm_kl',
                    'qcbm_sel_kl', 'qcbm_sel_mmd'])
        w.writerows(rows)
    print(f'wrote {CSV_PATH} ({len(rows)} rows)')


def draw_cell(ax, key, n, rows, stored, fontsize=7.2, x_name=0.535, dy=0.085):
    """One distribution panel (target, G-QCBM+KL, V-QCBM+KL, V-QCBM+MMD, D_KL box).
    rows: this cell's CSV rows. Returns (has_gqcbm_curve, is_older_ckpt)."""
    import gqcbm_palette as P
    col = lambda k: np.array([float(r[k]) for r in rows])
    bin_size = 2 ** n // N_BINS
    xb = (np.arange(N_BINS) * bin_size + bin_size / 2) / 2 ** n  # fraction of 2^n
    ax.fill_between(xb, col('target'), color=P.TARGET_FILL, lw=0, step='mid', zorder=1)
    ax.step(xb, col('qcbm_sel_kl'), where='mid', **P.line_kw('sel_kl'))
    ax.step(xb, col('qcbm_sel_mmd'), where='mid', **P.line_kw('sel_mmd'))
    has_g = rows[0]['gqcbm_kl'] != ''
    is_old = rows[0].get('gqcbm_source') == 'older'
    if has_g:
        kw = P.line_kw('gqcbm_kl')
        if is_old:  # not the Table 1 circuit: dotted (dashed = MMD objective)
            kw.update(ls=':', lw=1.5)
        ax.step(xb, col('gqcbm_kl'), where='mid', **kw)
    # D_KL block: Table 1 sources. GQCBM+KL results/circuit_stats.json;
    # QCBM-SEL+KL at n=20 the step-4,000 best KL from the run log, at n<=15 the
    # stored best_kl; QCBM-SEL+MMD the stored best_kl.
    g_kl = stored.get((f'{key}_{n}q', 'GQCBM'))
    g_txt = P.sig2(g_kl) + ('' if (has_g and not is_old) else '*')
    sel_kl_v = step4000_kl(key) if n == 20 else stored_kl(sel_kl_json(key, n))
    brows = [('gqcbm_kl', g_txt), ('sel_kl', P.sig2(sel_kl_v)),
             ('sel_mmd', P.sig2(stored_kl(sel_mmd_json(key, n))))]
    if is_old:
        brows.append(('gqcbm_kl', P.sig2(float(rows[0]['gqcbm_ckpt_kl'])),
                      'older ckpt (dotted)', False))
    left, bottom = P.kl_block(ax, brows, x_name=x_name, dy=dy, fontsize=fontsize)
    ax.set_xlim(0, 1)
    series = [col(c) for c in ('target', 'gqcbm_kl', 'qcbm_sel_kl', 'qcbm_sel_mmd')
              if rows[0][c] != '']
    top = max(v.max() for v in series)
    under = max(v[xb >= left].max() for v in series)
    ax.set_ylim(0, max(1.05 * top, 1.05 * under / bottom))
    ax.grid(False)  # figures/nature_style.py turns the grid on by default
    return has_g, is_old


# Two renderings of the same grid: 'full' (12 in wide, SI/slides) and 'page'
# (7.1 in wide, the main-text candidate, so text prints at its nominal size).
STYLE = {
    'full': dict(out=OUT, size=(12.0, 10.6), base=9, title=10, ylab=9.5, leg=10, note=8.5,
                 kl=7.2, x_name=0.535, dy=0.085, ncol=4, wrap=False),
    'page': dict(out=OUT + '_7in', size=(7.1, 6.9), base=7, title=7.5, ylab=7, leg=7.5, note=6.5,
                 kl=6.0, x_name=0.42, dy=0.11, ncol=4, wrap=True),
}


def plot(style='full'):
    st = STYLE[style]
    try:
        import nature_style
        nature_style.apply()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import gqcbm_palette as P  # figures/gqcbm_palette.py (paper-wide palette)

    data = {}
    for r in csv.DictReader(open(CSV_PATH)):
        data.setdefault((r['family'], int(r['n_qubits'])), []).append(r)
    stored = gqcbm_stored_kl()

    mpl.rcParams.update({'font.size': st['base'], 'axes.linewidth': 0.8 if style == 'full' else 0.6})
    if style != 'full':
        mpl.rcParams.update({k: st['base'] - 0.5 for k in ('xtick.labelsize', 'ytick.labelsize')})
        mpl.rcParams.update({'axes.labelsize': st['base']})
    fig, axes = plt.subplots(4, 3, figsize=st['size'], sharex='col')
    any_old = any_none = False
    for i, (label, key, _, _) in enumerate(FAMILIES):
        for j, n in enumerate(NQ):
            ax = axes[i, j]
            has_g, is_old = draw_cell(ax, key, n, data[(key, n)], stored, fontsize=st['kl'],
                                      x_name=st['x_name'], dy=st['dy'])
            any_none |= not has_g
            any_old |= is_old
            if i == 0:
                ax.set_title(f'$n={n}$ ({2 ** n // N_BINS} states/bin)', fontsize=st['title'])
            if j == 0:
                ax.set_ylabel(f'{label}\nprobability per bin', fontsize=st['ylab'])
            if i == len(FAMILIES) - 1:
                ax.set_xlabel(f'basis-state index $x / 2^{{{n}}}$')
            else:
                ax.tick_params(labelbottom=False)
    fig.legend(handles=P.legend_handles(('gqcbm_kl', 'sel_kl', 'sel_mmd'), proposed=True),
               loc='upper center', ncol=st['ncol'], frameon=False, bbox_to_anchor=(0.5, 1.0),
               fontsize=st['leg'])
    notes = ['Box values: $D_{\\mathrm{KL}}$(target $\\|$ model) on the full, unbinned '
             'distribution, as in Table 1.']
    if any_none or any_old:
        notes.append('*: no G-QCBM+KL curve for the Table 1 checkpoint (not available); the value is '
                     'its stored $D_{\\mathrm{KL}}$.'
                     + (' Dotted blue: an older checkpoint of that cell, shown for the qualitative '
                        'trend, with its own $D_{\\mathrm{KL}}$ in the "older ckpt" row.' if any_old else ''))
    if st['wrap']:
        import textwrap
        notes = [l for t in notes for l in textwrap.wrap(t, 120)]
    fig.text(0.5, -0.005, '\n'.join(notes), ha='center', fontsize=st['note'], color='#444')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    out = st['out']
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200 if style == 'full' else 300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}.{{pdf,png}}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-csv', action='store_true')
    ap.add_argument('--ckpt-dir', default='ckpts')
    ap.add_argument('--include-older', action='store_true')
    a = ap.parse_args()
    if not a.from_csv:
        build_csv(a.ckpt_dir)
    plot('full')
    plot('page')
