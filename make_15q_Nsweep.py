"""15q Gaussian num_samples sweep — trajectory plot.
Reads gqe_experiments/15q_N{N}/log.json for N in {10,20,40,60,80,100}.
Outputs paper/figs/fig_15q_Nsweep.{pdf,png}.
"""
import os, json
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 10, 'axes.linewidth': 0.9})
os.makedirs('paper/figs', exist_ok=True)

Ns = [10, 20, 40, 60, 80, 100]
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(Ns)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel (a): KL vs step
for N, c in zip(Ns, colors):
    log = f'gqe_experiments/15q_N{N}/log.json'
    if not os.path.exists(log):
        continue
    l = json.load(open(log))
    steps = [r['step'] for r in l]
    bkls  = [r['best_kl'] for r in l]
    axes[0].plot(steps, bkls, color=c, lw=1.5,
                 label=f'N={N}')
axes[0].axhline(0.05, color='red', lw=0.7, ls='--', alpha=0.5)
axes[0].axhline(0.01, color='red', lw=0.7, ls=':', alpha=0.5)
axes[0].set_xlabel('Training step')
axes[0].set_ylabel(r'best $D_{\mathrm{KL}}$')
axes[0].set_yscale('log')
axes[0].set_title('(a) Convergence vs. training step')
axes[0].legend(loc='upper right', frameon=True, ncol=2, fontsize=9)
axes[0].grid(True, which='major', linestyle=':', alpha=0.35)

# Panel (b): KL vs total samples = step × N
for N, c in zip(Ns, colors):
    log = f'gqe_experiments/15q_N{N}/log.json'
    if not os.path.exists(log):
        continue
    l = json.load(open(log))
    samples = [r['step'] * N for r in l]
    bkls    = [r['best_kl'] for r in l]
    axes[1].plot(samples, bkls, color=c, lw=1.5, label=f'N={N}')
axes[1].axhline(0.05, color='red', lw=0.7, ls='--', alpha=0.5)
axes[1].axhline(0.01, color='red', lw=0.7, ls=':', alpha=0.5)
axes[1].set_xlabel(r'Total quantum-evaluation samples ($N\times$step)')
axes[1].set_ylabel(r'best $D_{\mathrm{KL}}$')
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_title('(b) Sample efficiency')
axes[1].legend(loc='upper right', frameon=True, ncol=2, fontsize=9)
axes[1].grid(True, which='major', linestyle=':', alpha=0.35)

fig.suptitle(r'15-qubit Gaussian: GQCBM convergence vs. per-step sample size $N$',
             y=1.02)
plt.tight_layout()
plt.savefig('paper/figs/fig_15q_Nsweep.pdf', bbox_inches='tight')
plt.savefig('paper/figs/fig_15q_Nsweep.png', dpi=200, bbox_inches='tight')
plt.close()
print('saved paper/figs/fig_15q_Nsweep.{pdf,png}')
