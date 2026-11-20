"""Nature-journal-style matplotlib defaults.

Nature figure standards:
  * Sans-serif typeface (Helvetica / Arial).
  * Single-column 89 mm = 3.5 in, double-column 183 mm = 7.2 in.
  * No top / right spines by default; tick marks point outward.
  * Categorical palette tuned for both colour and grayscale legibility.

Apply at the top of any figure-generation script:

    import nature_style
    nature_style.apply()
"""
import matplotlib as mpl


def apply():
    mpl.rcParams.update({
        # Fonts — Helvetica / Arial / DejaVu Sans (last is always shipped).
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans',
                            'Liberation Sans', 'Bitstream Vera Sans'],
        'mathtext.fontset': 'stixsans',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 14,
        # Heavier strokes for legibility, Nature-like clean axes.
        'lines.linewidth': 1.8,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#222222',
        'axes.labelcolor': '#222222',
        'axes.titlecolor': '#222222',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        # Light grid that does not overpower the data.
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.color': '#dddddd',
        'grid.alpha': 0.9,
        # Margins
        'figure.constrained_layout.use': False,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        'savefig.dpi': 300,
        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.4,
        'legend.handlelength': 2.2,
        'legend.labelspacing': 0.3,
    })


# Nature-friendly categorical palette (CB-friendly, prints well in
# greyscale, distinguishable in the two-line + one-fill case we use most).
PALETTE = {
    'gqcbm':     '#0072B2',  # blue
    'qcbm_mmd':  '#D55E00',  # vermillion
    'qcbm_kl':   '#E69F00',  # orange
    'target':    '#000000',  # black
    'highlight': '#009E73',  # bluish green
    'shade':     '#CC79A7',  # reddish purple
}

# Figure widths in inches (Nature: 89 mm and 183 mm).
W_1COL = 3.5
W_2COL = 7.2
