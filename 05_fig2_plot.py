"""Figure 2 (reference-paper style): propionate and acetate time courses (cycle 1), propionate onset,
maximum propionate removal rate, peak acetate and TAN at the end of cycle 1.
Usage: python 05_fig2_plot.py <fig2_vfa.csv> <fig2_metrics.csv> <fig2_tan.csv> <out prefix>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.fontset': 'stix',
                     'mathtext.default': 'bf', 'legend.frameon': False, 'axes.facecolor': 'none',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
vfa = pd.read_csv(sys.argv[1]); met = pd.read_csv(sys.argv[2]); tan = pd.read_csv(sys.argv[3]); out = sys.argv[4]
tan = tan[tan.cycle == 1].copy(); tan['TAN_g'] = tan.TAN / 1000
C_CTRL, C_HC, INK = '#24C4DB', '#ED117B', '#000000'
LV = [0, 1, 2, 4]
XLAB = r'Added NH$_4^+$-N (g L$^{-1}$)'
W, H = 7.2, 6.45
fig = plt.figure(figsize=(W, H))
L0, PW, GAP = 0.55, 1.29, 0.48
R1Y, R2Y, R3Y, RH = 4.62, 2.62, 0.45, 1.42


def ax_at(col, y0, h=RH):
    return fig.add_axes([(L0 + col * (PW + GAP)) / W, y0 / H, PW / W, h / H])


def series_panels(row_y, col, ylabel, ylim, yt, show_x=True):
    axs = []
    for i, lv in enumerate(LV):
        ax = ax_at(i, row_y); axs.append(ax)
        for hc, c, mk, fc in [(0, C_CTRL, 'o', 'white'), (1, C_HC, 's', C_HC)]:
            s = vfa[(vfa.ammonia == lv) & (vfa.hydrochar == hc)].groupby('day')[col].agg(['mean', 'std', 'count'])
            s = s[s['count'] > 0].reset_index()
            ax.errorbar(s.day, s['mean'], yerr=s['std'].fillna(0), color=c, lw=1.1, elinewidth=0.6, capsize=1.3,
                        capthick=0.6, marker=mk, ms=3.0, mfc=fc, mec=c, mew=0.8, zorder=3 if hc else 2)
        ax.set_xlim(-3, 87); ax.set_ylim(*ylim); ax.set_xticks([0, 20, 40, 60, 80]); ax.set_yticks(yt)
        ax.set_xlabel('Time (d)')
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
        ax.text(0.95, 0.95, f'{lv} g N L$^{{-1}}$', transform=ax.transAxes, ha='right', va='top', fontsize=8.5)
    return axs


axA = series_panels(R1Y, 'propionate', 'Propionate (mM)', (-2, 42), [0, 10, 20, 30, 40])
axB = series_panels(R2Y, 'acetate', 'Acetate (mM)', (-1, 24), [0, 5, 10, 15, 20])
hdl = [Line2D([], [], color=C_CTRL, marker='o', mfc='white', mec=C_CTRL, ms=4, lw=1.1),
       Line2D([], [], color=C_HC, marker='s', mfc=C_HC, mec=C_HC, ms=4, lw=1.1)]
fig.legend(hdl, ['Control', 'Hydrochar'], loc='upper center', bbox_to_anchor=(0.5, (R1Y + RH + 0.33) / H),
           ncol=2, handlelength=2.0, fontsize=9, columnspacing=2.5)


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def boxes(ax, df, col, ylabel, ylim, yt):
    rng = np.random.default_rng(5); off = 0.2; tops = {}
    for hc, c, mk in [(0, C_CTRL, 'o'), (1, C_HC, 's')]:
        data = [df[(df.hydrochar == hc) & (df.ammonia == lv)][col].dropna().values for lv in LV]
        pos = np.arange(4) + (off if hc else -off)
        bp = ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK, lw=0.8),
                        capprops=dict(color=INK, lw=0.8), boxprops=dict(edgecolor=INK, lw=0.8))
        for bx in bp['boxes']:
            bx.set_facecolor(c); bx.set_alpha(0.9)
        for x0, vv in zip(pos, data):
            ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(vv)), vv, s=12, marker=mk, facecolor=c, edgecolor=INK,
                       linewidth=0.6, zorder=4)
        for lv, vv in zip(LV, data):
            tops.setdefault(lv, []).append(vv.max())
    span = ylim[1] - ylim[0]
    for i, lv in enumerate(LV):
        a = df[(df.hydrochar == 0) & (df.ammonia == lv)][col].dropna()
        b = df[(df.hydrochar == 1) & (df.ammonia == lv)][col].dropna()
        p = stats.ttest_ind(b, a, equal_var=False).pvalue
        y = max(tops[lv]) + span * 0.05; h = span * 0.025
        ax.plot([i - off, i - off, i + off, i + off], [y, y + h, y + h, y], color=INK, lw=0.7)
        ax.text(i, y + h + span * 0.005, stars(p), ha='center', va='bottom', fontsize=8 if p < 0.05 else 7)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(x) for x in LV]); ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(*ylim); ax.set_yticks(yt); ax.set_xlabel(XLAB); ax.set_ylabel(ylabel)


axC = ax_at(0, R3Y); boxes(axC, met, 'prop_onset', 'Propionate onset (d)', (0, 80), [0, 20, 40, 60, 80])
axD = ax_at(1, R3Y); boxes(axD, met, 'prop_maxrate', r'Max. removal rate (mM d$^{-1}$)', (0, 4.2), [0, 1, 2, 3, 4])
axE = ax_at(2, R3Y); boxes(axE, met, 'acetate_peak', 'Peak acetate (mM)', (0, 25), [0, 5, 10, 15, 20, 25])
axF = ax_at(3, R3Y); boxes(axF, tan, 'TAN_g', r'TAN (g N L$^{-1}$)', (0, 5.6), [0, 1, 2, 3, 4, 5])


def letter(ax, s, y_in):
    fig.text((ax.get_position().x0 * W - 0.44) / W, y_in / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')


letter(axA[0], 'A', R1Y + RH + 0.03)
letter(axB[0], 'B', R2Y + RH + 0.03)
for ax, s in zip([axC, axD, axE, axF], 'CDEF'):
    letter(ax, s, R3Y + RH + 0.03)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
