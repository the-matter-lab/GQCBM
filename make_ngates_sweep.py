"""Effect of n_gates on GQCBM forward-KL for the Gaussian target at
n=10, 15, 20 qubits.

Reads best_kl from:
  ckpts/gqe_gauss_{n}q_g{G}.pt  for G in {20, 30, 50}, n in {10, 15, 20}
  ckpts/gqe_gauss_20q.pt        (chain-plateau, G=50, multi-cycle)

Single-shot 3000-iter sweep across G is plotted as solid lines (one per n).
The chain-plateau result for n=20 is overlaid as a star marker to show
that the same gate budget can reach a much lower KL when training is
extended via multi-cycle plateau search.

Output: paper/figs/fig_ngates_sweep.{pdf,png}
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 10, 'axes.linewidth': 0.9})

NQ = [10, 15, 20]
GS = [20, 30, 50]
COLORS = {10: '#1F4E79', 15: '#C00000', 20: '#207020'}
MARKERS = {10: 'o', 15: 's', 20: '^'}


def best_kl(path):
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location='cpu', weights_only=False)
    return float(d.get('best_kl'))


fig, ax = plt.subplots(figsize=(6.2, 3.6))

for n in NQ:
    ys = [best_kl(f'ckpts/gqe_gauss_{n}q_g{G}.pt') for G in GS]
    xs = [G for G, y in zip(GS, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    ax.plot(xs, ys, color=COLORS[n], marker=MARKERS[n], ms=7,
            mfc='none', mew=1.4, lw=1.6, label=f'$n={n}$ (single-shot)')

# Chain-plateau overlay (n=20 only — only chain ckpt we have for Gaussian).
chain = best_kl('ckpts/gqe_gauss_20q.pt')
if chain is not None:
    ax.plot([50], [chain], color=COLORS[20], marker='*', ms=14, mfc=COLORS[20],
            mec='black', mew=0.6, lw=0,
            label=f'$n=20$ chain-plateau ({chain:.4f})')

ax.set_yscale('log')
ax.set_xlabel(r'Number of operator tokens $G$')
ax.set_ylabel(r'best forward $D_{\mathrm{KL}}(q\|p)$')
ax.set_xticks(GS)
ax.grid(True, which='major', linestyle=':', alpha=0.35)
ax.grid(True, which='minor', linestyle=':', alpha=0.15)
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
          framealpha=0.9, edgecolor='#666', fontsize=8.5)
ax.set_title(r'GQCBM on Gaussian: effect of operator-budget $G$ '
             r'on converged KL')

os.makedirs('paper/figs', exist_ok=True)
out = 'paper/figs/fig_ngates_sweep'
fig.tight_layout()
fig.savefig(f'{out}.pdf', bbox_inches='tight')
fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
print(f'saved {out}.{{pdf,png}}')

# Also emit a small CSV companion for the Supplementary table.
with open('paper/figs/fig_ngates_sweep.csv', 'w') as f:
    f.write('n_qubits,n_gates,best_kl,protocol\n')
    for n in NQ:
        for G in GS:
            y = best_kl(f'ckpts/gqe_gauss_{n}q_g{G}.pt')
            if y is not None:
                f.write(f'{n},{G},{y:.6f},single_shot_3000iter\n')
    if chain is not None:
        f.write(f'20,50,{chain:.6f},chain_plateau_multicycle\n')
print('wrote paper/figs/fig_ngates_sweep.csv')
