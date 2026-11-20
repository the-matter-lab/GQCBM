"""Generate the 5 figures referenced in Supplementary §S10 (ablation studies).

  fig_s10_1_tsweep   kl_temp T-sweep on ghz10 and bas9
  fig_s10_2_beta     entropy coefficient sweep on 5q Gaussian
  fig_s10_3_random   random-circuit baseline vs trained GQCBM
  fig_s10_4_mmd      MMD-loss comparison QCBM-SEL+MMD vs GQCBM
  fig_s10_5_budget   QCBM-SEL+KL training-budget saturation

All outputs go to paper/figs/.  Pure matplotlib over already-collected data.
"""
import json, os, re
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl
import gqcbm_palette as P  # figures/gqcbm_palette.py (paper-wide palette)

C_GQCBM = '#2E5EAA'
C_QCBM  = '#E07B00'
C_REF   = '#666666'

mpl.rcParams.update({'font.size': 10, 'axes.linewidth': 0.9,
                     'xtick.major.width': 0.9, 'ytick.major.width': 0.9})
os.makedirs('paper/figs', exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig S10.1: kl_temp T-sweep (ghz10 + bas9)
# ─────────────────────────────────────────────────────────────────────────────
def fig_s10_1_tsweep():
    print('[S10.1] kl_temp T-sweep ghz10/bas9')
    fixed_T = [0.05, 0.10, 0.20, 0.30, 0.50]
    std_c   = [0.3, 0.5, 1.0]
    data = {
        'ghz10': {'fixed': [0.8959, 2.4744, 1.5105, 1.8566, 2.6948],
                  'std':   [2.4374, 2.5162, 3.3777],
                  'adapt': 2.6069, 'sel_kl': 0.0030},
        'bas9':  {'fixed': [1.9462, 1.9927, 2.1551, 2.0572, 2.1954],
                  'std':   [2.3815, 2.8059, 2.5257],
                  'adapt': 2.3021, 'sel_kl': 0.1017},
    }
    titles = ['(a) 10-qubit GHZ', r'(b) $3\times 3$ BAS ($n=9$)']
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, key, ttl in zip(axes, ['ghz10', 'bas9'], titles):
        d = data[key]
        ax.plot(fixed_T, d['fixed'], 'o-', color=C_GQCBM, ms=9, lw=1.8,
                mfc='white', mew=2, label=r'fixed $T$', zorder=5)
        ax.plot(std_c, d['std'], 's--', color=C_QCBM, ms=9, lw=1.8,
                mfc='white', mew=2,
                label=r'spread $c\cdot\mathrm{std}(D_{\mathrm{KL}})$', zorder=5)
        ax.axhline(d['adapt'], color=C_REF, ls=':', lw=1.5,
                   label=f"adaptive (default)={d['adapt']:.4f}")
        ax.axhline(d['sel_kl'], color='black', ls='-.', lw=1.0, alpha=0.6,
                   label=f"QCBM-SEL+KL ref ={d['sel_kl']:.4f}")
        # Mark best fixed-T
        bi = int(np.argmin(d['fixed']))
        ax.annotate(f"best={d['fixed'][bi]:.4f}",
                    xy=(fixed_T[bi], d['fixed'][bi]),
                    xytext=(10, 12), textcoords='offset points',
                    fontsize=9, color=C_GQCBM, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C_GQCBM, lw=1))
        ax.set_xscale('log')
        ax.set_xlabel(r'$T$ (fixed) or coefficient $c$')
        ax.set_ylabel(r'$D_{\mathrm{KL,best}}$')
        ax.set_title(ttl)
        ax.grid(True, which='major', linestyle=':', alpha=0.35)
        ax.legend(loc='lower right', fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig('paper/figs/fig_s10_1_tsweep.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_s10_1_tsweep.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved paper/figs/fig_s10_1_tsweep.{pdf,png}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig S10.2: β sweep (5q Gaussian)
# ─────────────────────────────────────────────────────────────────────────────
def fig_s10_2_beta():
    print('[S10.2] β entropy sweep 5q Gaussian')
    # Values from paper/figure_data/fig_s10_2_beta_sweep.csv (scripts/archive/run_beta_sweep.sh:
    # 5 qubits, K=12, N=50). The star marks the sweep minimum, read from the data; it is not
    # the code default (--ent_coef 0.02 in gqe_single_target.py).
    import csv as _csv
    rows = list(_csv.DictReader(open('paper/figure_data/fig_s10_2_beta_sweep.csv')))
    beta = [float(r['beta']) for r in rows]
    kl = [float(r['best_kl']) for r in rows]
    i_min = int(np.argmin(kl))
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.plot(beta, kl, 'o-', color=P.COLOR['gqcbm_kl'], ms=8, lw=1.8,
            mfc='white', mew=1.8, zorder=4)
    ax.scatter([beta[i_min]], [kl[i_min]], s=260, marker='*', color=P.COLOR['gqcbm_kl'],
               edgecolor='white', lw=0.8, zorder=10,
               label=fr'minimum, $\beta={beta[i_min]:.2f}$')
    ax.axvline(0.02, color='#999999', ls=':', lw=1.0, zorder=1,
               label=r'code default $\beta=0.02$')
    for b_, k in zip(beta, kl):
        ax.annotate(f'{k:.4f}', xy=(b_, k), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # one labelled tick per swept beta, no other x ticks (title dropped: the caption carries it)
    from matplotlib.ticker import NullLocator
    ax.set_xticks(beta, [f'{b_:.2f}' for b_ in beta], fontsize=8.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Entropy coefficient $\beta$')
    ax.set_ylabel(r'best $D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    ax.grid(False)  # paper convention
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('paper/figs/fig_s10_2_beta.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_s10_2_beta.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved paper/figs/fig_s10_2_beta.{pdf,png}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig S10.3: Random-circuit baseline
# ─────────────────────────────────────────────────────────────────────────────
def fig_s10_3_random():
    print('[S10.3] Random-circuit baseline')
    # Values from paper/figure_data/fig_s10_3_random_baseline.csv (written by random_baseline.py).
    # The trained G-QCBM+KL references: n=5 0.0016 = seed 123 of scripts/run_multiseed_5q.sh
    # (SLURM 39630, best of seeds 42/123/456/789/1000 -> 0.0031/0.0016/0.0020/0.0020/0.0022),
    # reproduced by rerun job 796452 (Hugging Face, ckpts/seed_5q_123_rerun.pt, best_kl 0.001642).
    import csv as _csv
    data, gqcbm = {}, {}
    with open('paper/figure_data/fig_s10_3_random_baseline.csv') as f:
        for r in _csv.DictReader(f):
            n = int(r['n_qubits'])
            data.setdefault(n, []).append((int(r['N_random']), float(r['random_best_kl'])))
            gqcbm[n] = float(r['gqcbm_kl'])
    # Palette: the trained model is G-QCBM+KL (blue, solid, on top); the random baseline is not a
    # paper method, so it is neutral grey. One panel per qubit count,
    # so no colour is needed to pair a random curve with its trained reference.
    C_G, T_G = P.COLOR['gqcbm_kl'], P.TEXT['gqcbm_kl']
    C_R = '#7A7A7A'
    MK = {5: 'o', 10: 's', 15: '^'}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, n in zip(axs, (5, 10, 15)):
        N, kl = zip(*sorted(data[n]))
        ax.plot(N, kl, marker=MK[n], color=C_R, ms=6, lw=1.0, mfc='white', mew=1.4, zorder=3)
        ax.axhline(gqcbm[n], color=C_G, lw=1.8, zorder=4)
        ax.text(0.04, gqcbm[n] * 1.25, f'{gqcbm[n]:.4f}', transform=ax.get_yaxis_transform(),
                color=T_G, fontsize=8, fontweight='bold', va='bottom')
        # gap at the largest N: random best / trained (the caption's 28-43x)
        ax.annotate('', xy=(N[-1], gqcbm[n] * 1.06), xytext=(N[-1], kl[-1] * 0.9),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=0.8, shrinkA=0, shrinkB=0))
        ax.text(N[-1] * 0.8, (kl[-1] * gqcbm[n]) ** 0.5, f'×{kl[-1] / gqcbm[n]:.0f}',
                ha='right', va='center', fontsize=8.5, color='#333333')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(50, 2.2e4); ax.set_xticks([1e2, 1e3, 1e4])
        ax.set_title(f'$n={n}$', fontsize=10)
        ax.set_xlabel(r'Random samples $N$')
        ax.grid(False)
    axs[0].set_ylabel(r'best $D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
    axs[0].set_ylim(1e-3, 1.5)
    h = [mpl.lines.Line2D([], [], color=C_R, marker='o', mfc='white', mew=1.4, lw=1.0, ms=6),
         mpl.lines.Line2D([], [], color=C_G, lw=1.8)]
    fig.legend(h, [r'random circuits, best of $N$', f"trained {P.LABEL['gqcbm_kl']}"],
               loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig('paper/figs/fig_s10_3_random.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_s10_3_random.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved paper/figs/fig_s10_3_random.{pdf,png}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig S10.4: MMD-loss scatter (GQCBM post-hoc MMD vs QCBM-SEL+MMD)
# ─────────────────────────────────────────────────────────────────────────────
def fig_s10_4_mmd():
    print('[S10.4] MMD-loss scatter')
    # Like-for-like: both methods trained with the MMD loss under matched
    # PennyLane bandwidths. Data comes EXCLUSIVELY from
    # paper/figure_data/fig_s10_4_mmd_comparison.csv, which
    # export_figure_data.export_mmd() regenerates from the real checkpoints
    # (ckpts/gqe_mmdloss2_*.pt) and QCBM-SEL result JSONs -- never hardcode
    # values here (a previous hardcoded list drifted up to 3000x from the
    # actual sources and silently flipped the Gauss-10q conclusion).
    import csv as _csv
    pts = []
    with open('paper/figure_data/fig_s10_4_mmd_comparison.csv') as f:
        for row in _csv.DictReader(f):
            t = row['target']
            name = (t + 'q') if t[-1].isdigit() else f"{t}-{row['n_qubits']}q"
            pts.append((name, float(row['qcbm_sel_mmd_loss_mmd']),
                        float(row['gqcbm_posthoc_mmd'])))
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    lo, hi = 1e-13, 1e-1
    # diagonal
    ax.plot([lo, hi], [lo, hi], 'k-', alpha=0.4, lw=1, zorder=1)
    # Regions shading
    ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                    color=C_QCBM, alpha=0.06, zorder=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                    color=C_GQCBM, alpha=0.06, zorder=0)

    # Distinct marker+color per target; legend on the side replaces inline labels
    style = {
        'Bell-2q':    ('s', '#1f77b4'),
        'GHZ-5q':     ('s', '#9467bd'),
        'BAS-4q':     ('o', '#2ca02c'),
        'Gauss-5q':   ('o', '#d62728'),
        'Bimodal-6q': ('o', '#ff7f0e'),
        'Exp-5q':     ('o', '#8c564b'),
        'Gauss-10q':  ('o', '#e377c2'),
    }
    for name, qm, gm in pts:
        marker, color = style[name]
        # floor sub-axis values (exact/near-exact states on either side) onto
        # the log axis: Bell-2q GQCBM ~2e-30; Gauss-5q/Exp-5q QCBM-SEL ~3e-28/2e-31
        gm_plot = max(gm, lo)
        qm_plot = max(qm, lo)
        ax.scatter([qm_plot], [gm_plot], s=130, marker=marker, color=color,
                   edgecolor='black', lw=0.9, zorder=5, label=name)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(r'QCBM-SEL+MMD  MMD$^{2}$  (MMD-loss-trained)')
    ax.set_ylabel(r'GQCBM+MMD  MMD$^{2}$  (MMD-loss-trained)')
    ax.set_title(r'Like-for-like MMD comparison (PennyLane bandwidths)')
    ax.grid(True, which='major', linestyle=':', alpha=0.35)
    ax.text(1e-9, 5e-4, 'GQCBM worse\n(above $y=x$)',
            fontsize=9, color=C_QCBM, ha='center', alpha=0.85)
    ax.text(5e-4, 1e-10, 'GQCBM better\n(below $y=x$)',
            fontsize=9, color=C_GQCBM, ha='center', alpha=0.85)
    ax.text(0.02, 0.02,
            'Values below $10^{-13}$ are floored onto the axis:\n'
            'GQCBM+MMD Bell-2q ($\\approx\\!2\\times10^{-30}$);\n'
            'QCBM-SEL+MMD Gauss-5q ($\\approx\\!3\\times10^{-28}$)\n'
            'and Exp-5q ($\\approx\\!2\\times10^{-31}$).',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    # Side legend
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=True, title='Target')
    plt.tight_layout()
    plt.savefig('paper/figs/fig_s10_4_mmd.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_s10_4_mmd.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved paper/figs/fig_s10_4_mmd.{pdf,png}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig S10.5: QCBM training-budget trajectories
# ─────────────────────────────────────────────────────────────────────────────
BUDGET_CSV = 'paper/figure_data/fig_s10_5_qcbm_budget.csv'
BUDGET_RUNS = 'results/sel_kl_exact_gb200/runs/figS4/job_8803247/results'
KL_PLOT_LOWER = 1e-8   # log-axis lower bound; KL <= this is numerical zero


def export_budget_csv():
    """Rebuild BUDGET_CSV from the exact-gradient re-run's JSON histories
    (qcbm_pennylane.py --kl_grad exact; results/sel_kl_exact_gb200/README.md).
    The history logs every 10 (n=10, 15) or 20 (n=5) steps; `kl` is the KL at that
    step and `best_loss` the best KL so far (the training loss is the KL)."""
    import csv as _csv
    rows = []
    for n in (5, 10, 15):
        for lr in ('0.01', '0.03'):
            rec = json.load(open(f'{BUDGET_RUNS}/qcbm_nc_{n}q_lr{lr}.json'))
            rec = rec[-1] if isinstance(rec, list) else rec
            assert rec.get('kl_grad') == 'exact' and not rec.get('partial'), (n, lr)
            for h in rec['history']:
                rows.append([n, lr, h['step'], f"{h['kl']:.6e}", f"{h['best_loss']:.6e}"])
    with open(BUDGET_CSV, 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['n_qubits', 'learning_rate', 'step', 'current_kl', 'best_kl'])
        w.writerows(rows)
    print(f'  wrote {BUDGET_CSV} ({len(rows)} rows)')


def fig_s10_5_budget():
    print('[S10.5] QCBM training-budget trajectories (exact-gradient re-run)')
    import csv as _csv
    if os.path.isdir(BUDGET_RUNS):
        export_budget_csv()
    data = {}
    for r in _csv.DictReader(open(BUDGET_CSV)):
        d = data.setdefault((int(r['n_qubits']), r['learning_rate']), ([], [], []))
        d[0].append(int(r['step'])); d[1].append(float(r['current_kl']))
        d[2].append(float(r['best_kl']))
    # GQCBM reference: the same values as fig_s10_3_random_baseline.csv `gqcbm_kl`.
    gqcbm = {}
    for r in _csv.DictReader(open('paper/figure_data/fig_s10_3_random_baseline.csv')):
        gqcbm[int(r['n_qubits'])] = float(r['gqcbm_kl'])
    # Both series are V-QCBM+KL and neither learning rate is the Table 1 protocol (lr 0.005), so they
    # get equal weight: both solid, same width, two slight hue shifts of the V-QCBM+KL orange at its
    # lightness (L* ~ 55, like #B87333). Exception to the palette for this figure.
    col = {'0.01': '#CA663E', '0.03': '#A27E28'}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=False)
    for ax, n in zip(axes, (5, 10, 15)):
        floored = False
        for lr in ('0.01', '0.03'):
            s_, kl, bst = data[(n, lr)]
            floored |= min(kl + bst) <= KL_PLOT_LOWER
            kl = np.maximum(kl, KL_PLOT_LOWER); bst = np.maximum(bst, KL_PLOT_LOWER)
            ax.plot(s_, kl, '-', color=col[lr], lw=0.8, alpha=0.3, zorder=2)
            ax.plot(s_, bst, '-', color=col[lr], lw=1.6, zorder=3,
                    label=P.LABEL['sel_kl'] + f', lr = {lr} (best)')
        # G-QCBM+KL reference: full-width DASHED line labelled "final (separate run)".
        # A solid full-width line read as a flat G-QCBM training curve; dashed marks it
        # as a reference level. (Exception to the dashed = MMD rule, by design for this
        # figure.) It is the final KL of a separately trained model, not a trajectory.
        xmax = max(max(data[(n, lr)][0]) for lr in ('0.01', '0.03'))
        ax.axhline(gqcbm[n], color=P.COLOR['gqcbm_kl'], ls='--', lw=1.5, zorder=4,
                   label=f"{P.LABEL['gqcbm_kl']}, final (separate run)")
        ax.text(xmax, gqcbm[n] * 1.35, f'{gqcbm[n]:.4f}', color=P.TEXT['gqcbm_kl'],
                fontsize=8, fontweight='bold', ha='right', va='bottom', zorder=5)
        if floored:
            ax.text(0.97, 0.04, r'$D_{\mathrm{KL}}\leq 10^{-8}$ (numerical zero)'
                    '\ndrawn at $10^{-8}$', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='#444')
        ax.set_yscale('log')
        ax.set_xlabel('Training step')
        ax.set_ylabel(r'$D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})$')
        ax.set_title(f'$n={n}$ Gaussian')
        ax.grid(False)  # paper convention
        # n=15 curves sit near the top, so its legend goes to the empty middle.
        ax.legend(loc='center right' if n == 15 else 'upper right', fontsize=8.5)
    plt.tight_layout()
    plt.savefig('paper/figs/fig_s10_5_budget.pdf', bbox_inches='tight')
    plt.savefig('paper/figs/fig_s10_5_budget.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved paper/figs/fig_s10_5_budget.{pdf,png}')


if __name__ == '__main__':
    # fig_s10_1_tsweep()   # ghz10/bas9 T-sweep — skip per user request
    # Optional: name figures to draw only those, e.g. `python make_supp_figures.py beta budget`.
    figs = {'beta': fig_s10_2_beta, 'random': fig_s10_3_random,
            'mmd': fig_s10_4_mmd, 'budget': fig_s10_5_budget}
    for name in (sys.argv[1:] or figs):
        figs[name]()
    print('Done.')
