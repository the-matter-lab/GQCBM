"""10-qubit exponential — bar histogram + solid black target line.
Small-$n$ case where GQCBM beats QCBM-SEL+MMD by ${\\sim}7\\times$
on forward KL.

Sources:
  GQCBM:   ckpts/gqe_exp_10q_g20.pt                    KL=0.0011
  SEL+MMD: results/sweep_L4_mmd_exp_10q.json           KL=0.0080
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'figures'))
try:
    import nature_style; nature_style.apply()
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl
import cudaq

cudaq.set_target('nvidia', option='fp64')

from make_all_dist_compare import gqcbm_dist
from qcbm_pennylane import circuit_probs as sel_probs

GQCBM_CKPT   = 'ckpts/gqe_exp_10q_g20.pt'
SEL_MMD_JSON = 'results/sweep_L4_mmd_exp_10q.json'

target, p_g, nq, kl_g = gqcbm_dist(GQCBM_CKPT)
rec = json.load(open(SEL_MMD_JSON))
rec = rec[-1] if isinstance(rec, list) else rec
p_m = sel_probs(rec['n_qubits'], rec['n_layers'], np.asarray(rec['best_params']))
kl_m = float(rec['best_kl'])

print(f'GQCBM   KL={kl_g:.4f}')
print(f'SEL+MMD KL={kl_m:.4f}')
print(f'ratio   {kl_m/kl_g:.1f}x lower for GQCBM')

x = np.arange(2 ** nq)
W = 0.4

C_GQCBM = '#1F4E79'
C_QMMD  = '#E07B00'

mpl.rcParams.update({'font.size': 11, 'axes.linewidth': 0.9})
fig, ax = plt.subplots(figsize=(11, 4.5))

ax.bar(x - W/2, p_g, width=W, color=C_GQCBM, alpha=0.9, edgecolor='none',
       label=f'GQCBM  ($D_{{\\mathrm{{KL}}}}={kl_g:.4f}$)')
ax.bar(x + W/2, p_m, width=W, color=C_QMMD,  alpha=0.9, edgecolor='none',
       label=f'QCBM-SEL+MMD  ($D_{{\\mathrm{{KL}}}}={kl_m:.4f}$)')

ax.plot(x, target, color='black', lw=1.6, label='Target', zorder=10)

ax.set_xlim(-1.0, 2 ** nq)
ax.set_ylim(0, float(target.max()) * 1.35)
ax.set_xlabel(r'basis-state index $x \in \{0, \ldots, 2^{10}-1\}$')
ax.set_ylabel(r'$p(x)$')
ax.legend(loc='upper right', frameon=True, framealpha=0.94, edgecolor='#666',
          fontsize=10.5)
ax.set_title(r'10-qubit exponential: bar histogram + target line '
             r'($2^{10}=1024$ basis states)')
ax.grid(True, axis='y', linestyle=':', alpha=0.3)

plt.tight_layout()
os.makedirs('paper/figs', exist_ok=True)
out = 'paper/figs/fig_exp10_bars'
plt.savefig(f'{out}.pdf', bbox_inches='tight')
plt.savefig(f'{out}.png', dpi=180, bbox_inches='tight')
print(f'\nsaved {out}.{{pdf,png}}')
