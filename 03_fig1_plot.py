"""Figure 1: cumulative CH4, lag (box), Rm (box), ammonia-induced increase in lag (linear fits), mu_max (box).
Usage: python 03_fig1_plot.py <ch4_per_bottle.csv> <fig1_kinetics.csv> <out prefix>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy import stats
plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'xtick.direction': 'out', 'ytick.direction': 'out', 'mathtext.fontset': 'stix',
                     'mathtext.default': 'bf', 'legend.frameon': False, 'axes.facecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
d = pd.read_csv(sys.argv[1]); d = d[d.cycle == 1]; r = pd.read_csv(sys.argv[2]); out = sys.argv[3]
C_CTRL, C_HC, INK = '#24C4DB', '#ED117B', '#000000'
LV = [0, 1, 2, 4]; XLAB = r'Added NH$_4^+$-N (g L$^{-1}$)'
W, H = 7.2, 4.5; fig = plt.figure(figsize=(W, H))
L0, PW, GAP = 0.55, 1.29, 0.48
def ax_at(col, y0, h): return fig.add_axes([(L0 + col * (PW + GAP)) / W, y0 / H, PW / W, h / H])
ROW1_Y, ROW1_H, ROW2_Y, ROW2_H = 2.55, 1.52, 0.45, 1.50
# ---------- A: cumulative CH4
axA = []
for i, lv in enumerate(LV):
    ax = ax_at(i, ROW1_Y, ROW1_H); axA.append(ax)
    for hc, col, mk, fc in [(0, C_CTRL, 'o', 'white'), (1, C_HC, 's', C_HC)]:
        s = d[(d.ammonia == lv) & (d.hydrochar == hc)].groupby('day').CH4_mL.agg(['mean', 'std']).reset_index()
        ax.errorbar(s.day, s['mean'], yerr=s['std'], color=col, lw=1.1, elinewidth=0.6, capsize=1.3, capthick=0.6,
                    marker=mk, ms=3.0, mfc=fc, mec=col, mew=0.8, zorder=3 if hc else 2)
    ax.set_xlim(-3, 87); ax.set_ylim(-3, 56); ax.set_xticks([0, 20, 40, 60, 80]); ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel('Time (d)')
    if i == 0: ax.set_ylabel('Cumulative CH$_4$ (mL)')
    else: ax.set_yticklabels([])
    ax.text(0.95, 0.95, f'{lv} g N L$^{{-1}}$', transform=ax.transAxes, ha='right', va='top', fontsize=8.5)
hdl = [Line2D([], [], color=C_CTRL, marker='o', mfc='white', mec=C_CTRL, ms=4, lw=1.1),
       Line2D([], [], color=C_HC, marker='s', mfc=C_HC, mec=C_HC, ms=4, lw=1.1)]
fig.legend(hdl, ['Control', 'Hydrochar'], loc='upper center', bbox_to_anchor=(0.5, (ROW1_Y + ROW1_H + 0.33) / H),
           ncol=2, handlelength=2.0, fontsize=9, columnspacing=2.5)
# ---------- box panels
def stars(p): return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
def boxes(ax, col, ylabel, ylim, yt):
    rng = np.random.default_rng(3); off = 0.2; tops = {}
    for hc, c, mk in [(0, C_CTRL, 'o'), (1, C_HC, 's')]:
        data = [r[(r.hydrochar == hc) & (r.ammonia == lv)][col].values for lv in LV]
        pos = np.arange(4) + (off if hc else -off)
        bp = ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK, lw=0.8),
                        capprops=dict(color=INK, lw=0.8), boxprops=dict(edgecolor=INK, lw=0.8))
        for b in bp['boxes']: b.set_facecolor(c); b.set_alpha(0.9)
        for x0, v in zip(pos, data):
            ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(v)), v, s=12, marker=mk, facecolor=c, edgecolor=INK, linewidth=0.6, zorder=4)
        for lv, v in zip(LV, data): tops.setdefault(lv, []).append(v.max())
    span = ylim[1] - ylim[0]
    for i, lv in enumerate(LV):
        a = r[(r.hydrochar == 0) & (r.ammonia == lv)][col]; b = r[(r.hydrochar == 1) & (r.ammonia == lv)][col]
        p = stats.ttest_ind(b, a, equal_var=False).pvalue; y = max(tops[lv]) + span * 0.05; h = span * 0.025
        ax.plot([i - off, i - off, i + off, i + off], [y, y + h, y + h, y], color=INK, lw=0.7)
        ax.text(i, y + h + span * 0.005, stars(p), ha='center', va='bottom', fontsize=8 if p < 0.05 else 7)
    ax.set_xticks(range(4)); ax.set_xticklabels([str(v) for v in LV]); ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(*ylim); ax.set_yticks(yt); ax.set_xlabel(XLAB); ax.set_ylabel(ylabel)
axB = ax_at(0, ROW2_Y, ROW2_H); boxes(axB, 'lam_fix', r'Lag phase $\lambda$ (d)', (0, 86), [0, 20, 40, 60, 80])
axC = ax_at(1, ROW2_Y, ROW2_H); boxes(axC, 'Rm_fix', r'$R_m$ (mL d$^{-1}$)', (0, 7.5), [0, 2, 4, 6])
axE = ax_at(3, ROW2_Y, ROW2_H); boxes(axE, 'mu_max4', r'$\mu_{\mathbf{max}}$ (d$^{-1}$)', (0, 0.36), [0, 0.1, 0.2, 0.3])
# ---------- D: ammonia-induced increase in lambda (lambda minus the same treatment's 0 g N/L mean), numeric x axis
axD = ax_at(2, ROW2_Y, ROW2_H)
dl = pd.read_csv(sys.argv[2].replace('fig1_kinetics.csv', 'fig1_ammonia_delay.csv'))
rng = np.random.default_rng(5)
for hc, c, mk, dx in [(0, C_CTRL, 'o', -0.09), (1, C_HC, 's', 0.09)]:
    s = r[r.hydrochar == hc]; inc = s.lam_fix - s[s.ammonia == 0].lam_fix.mean()
    k_, b_ = np.polyfit(s.ammonia, inc, 1); xs = np.linspace(0, 4, 50)
    axD.plot(xs, k_ * xs + b_, color=c, lw=1.2, zorder=2)
    axD.scatter(s.ammonia + dx + rng.uniform(-0.04, 0.04, len(s)), inc, s=12, marker=mk, facecolor=c, edgecolor=INK, linewidth=0.6, zorder=4)
sl0, sl1, ps = dl.slope_ctrl.iloc[0], dl.slope_hc.iloc[0], dl.slope_diff_P.iloc[0]
axD.text(1.55, 27, f'{sl0:.1f}', color='#1595a8', fontsize=7.2, ha='right', va='center')
axD.text(4.25, 14.5, f'{sl1:.1f}', color=C_HC, fontsize=7.2, ha='right', va='center')
axD.text(0.05, 0.95, 'Slope: ' + ('P < 0.001' if ps < 0.001 else f'P = {ps:.3f}'), transform=axD.transAxes, ha='left', va='top', fontsize=7.2)
axD.set_xlim(-0.4, 4.4); axD.set_ylim(-6, 62); axD.set_xticks(LV); axD.set_yticks([0, 20, 40, 60])
axD.set_xlabel(XLAB); axD.set_ylabel(r'Increase in $\lambda$ (d)')
# ---------- panel letters
def letter(ax, s, y_in):
    fig.text((ax.get_position().x0 * W - 0.44) / W, y_in / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')
letter(axA[0], 'A', ROW1_Y + ROW1_H + 0.03)
for ax, s in zip([axB, axC, axD, axE], 'BCDE'): letter(ax, s, ROW2_Y + ROW2_H + 0.03)
for ext in ('png', 'pdf', 'svg'): fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
