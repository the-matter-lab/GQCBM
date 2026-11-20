"""Paper-wide plotting palette for the GQCBM manuscript (set by Fig. 2,
see the comments below).

Rule: hue encodes the method, lightness the training objective. Figures show
"proposed method vs. baselines": GQCBM+KL is drawn thickest and on top, and the
two QCBM-SEL baselines get equal visual weight.

Line colours are used for lines, fills and markers. Text drawn in a series'
colour uses TEXT, a tint halfway between the line colour and the darkest tint
of the same hue that reaches 4.5:1 contrast on white (the light line colours
alone are too pale for small text).
"""
import math

TARGET_FILL = '#E5E5E5'

COLOR = {
    'gqcbm_kl': '#1F4E9C',   # dark blue
    'gqcbm_mmd': '#8DB3E2',  # light blue
    'sel_kl': '#B87333',     # muted dark orange
    'sel_mmd': '#E3B07A',    # light orange
}

TEXT = {  # contrast on white: 8.0, 3.1, 5.3, 3.0 : 1
    'gqcbm_kl': '#1F4E9C',
    'gqcbm_mmd': '#7594BC',
    'sel_kl': '#995E29',
    'sel_mmd': '#C08B56',
}

# Line styles. MMD objectives are dashed. GQCBM+MMD's width/style is by analogy
# with QCBM-SEL+MMD (not separately decided).
LINE = {
    'gqcbm_kl': dict(color=COLOR['gqcbm_kl'], lw=1.8, ls='-', zorder=4),
    'gqcbm_mmd': dict(color=COLOR['gqcbm_mmd'], lw=1.0, ls='--', zorder=3),
    'sel_kl': dict(color=COLOR['sel_kl'], lw=1.0, ls='-', zorder=3),
    'sel_mmd': dict(color=COLOR['sel_mmd'], lw=1.0, ls='--', zorder=3),
}

# Display names (design choice): G-QCBM (generated circuit) vs.
# V-QCBM (fixed-ansatz, variationally trained; formerly "QCBM-SEL"). Data keys in
# results/*.json and CSVs keep the old names ('GQCBM', 'QCBM-SEL+KL', ...).
LABEL = {'gqcbm_kl': 'G-QCBM+KL', 'gqcbm_mmd': 'G-QCBM+MMD',
         'sel_kl': 'V-QCBM+KL', 'sel_mmd': 'V-QCBM+MMD'}
SHORT = dict(LABEL)

# Legend order: Target, GQCBM+KL, QCBM-SEL+KL, QCBM-SEL+MMD, then GQCBM+MMD.
ORDER = ('gqcbm_kl', 'sel_kl', 'sel_mmd', 'gqcbm_mmd')

KL_HEADER = r'$D_{\mathrm{KL}}(\mathrm{target}\,\|\,\mathrm{model})\downarrow$'


def line_kw(key, with_zorder=True):
    kw = dict(LINE[key])
    if not with_zorder:
        kw.pop('zorder')
    return kw


def sig2(v):
    """Two significant figures, keeping trailing zeros (0.6016 -> '0.60')."""
    if v == 0:
        return '0'
    d = max(0, 1 - int(math.floor(math.log10(abs(v)))))
    return f'{v:.{d}f}'


def kl_block(ax, rows, x_name=0.565, x_val=0.985, y0=0.965, dy=0.085, fontsize=7.5):
    """Top-right D_KL box. rows: list of (key, value_text) or (key, value_text,
    name, bold) with key in COLOR; by default the name is SHORT[key] and only
    GQCBM+KL is bold. Returns (left edge, bottom) in axes fraction, so callers can
    keep data clear of the box."""
    import matplotlib as mpl
    n = len(rows)
    bottom = y0 - (n + 0.55) * dy
    ax.add_patch(mpl.patches.FancyBboxPatch(
        (x_name - 0.02, bottom), x_val - x_name + 0.03, (n + 0.6) * dy,
        boxstyle='round,pad=0.005', transform=ax.transAxes, fc='white',
        ec='#BBBBBB', lw=0.5, alpha=0.9, zorder=5))
    ax.text(x_name, y0, KL_HEADER, transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, color='#333333', zorder=6)
    for i, row in enumerate(rows, start=1):
        key, val = row[0], row[1]
        name = row[2] if len(row) > 2 else SHORT[key]
        bold = row[3] if len(row) > 3 else key == 'gqcbm_kl'
        w = 'bold' if bold else 'normal'
        y = y0 - i * dy
        ax.text(x_name, y, name, transform=ax.transAxes, ha='left', va='top',
                fontsize=fontsize, color=TEXT[key], fontweight=w, zorder=6)
        ax.text(x_val, y, val, transform=ax.transAxes, ha='right', va='top',
                fontsize=fontsize, color=TEXT[key], fontweight=w, zorder=6)
    return x_name - 0.02, bottom


# Top-legend suffix for the proposed method (design choice): shown in the
# shared top legend only (Fig. 2 candidates), never in the in-panel D_KL block.
PROPOSED = ' (proposed)'


def legend_handles(keys, target=True, proposed=False):
    """proposed=True labels G-QCBM+KL as 'G-QCBM+KL (proposed)'."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=TARGET_FILL, edgecolor='none', label='Target')] if target else []
    hs += [Line2D([], [], label=LABEL[k] + (PROPOSED if proposed and k == 'gqcbm_kl' else ''),
                  **line_kw(k, with_zorder=False)) for k in keys]
    return hs
