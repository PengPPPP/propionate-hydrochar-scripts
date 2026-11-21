r"""Figure 5: methanogens at MAG level.
A  clustered heatmap of methanogen MAGs (group means, log colour scale, values printed) with a row dendrogram
   (average linkage on z-scored log abundance) and a trophic-type track;
B  composition of hydrogenotrophic methanogens in each group;
C  response scatter: control vs hydrochar abundance of each MAG at each ammonia level (log-log);
D  five representative MAGs (replicates, means, Welch t).
Usage: python 13_fig5_plot.py <folder with fig5_*.csv> <out prefix>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'mathtext.fontset': 'stix', 'mathtext.default': 'bf', 'legend.frameon': False,
                     'axes.facecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
DIR, out = sys.argv[1], sys.argv[2]
tab = pd.read_csv(f'{DIR}/fig5_methanogen_means.csv', index_col=0)          # rows in dendrogram-leaf order
Z = np.load(f'{DIR}/fig5_linkage.npy')
orig = pd.read_csv(f'{DIR}/fig5_cluster_order.csv').Genome.tolist()
comp = pd.read_csv(f'{DIR}/fig5_hydrogenotrophs.csv', index_col=0)
sc = pd.read_csv(f'{DIR}/fig5_response.csv')
ps = pd.read_csv(f'{DIR}/fig5_methanogen_samples.csv', index_col=0)

C_CTRL, C_HC, INK = '#24C4DB', '#ED117B', '#000000'
TROC = {'Hydrogenotrophic': '#10b3c6', 'Acetoclastic': '#e2a400', 'Methanosarcinaceae': '#4d8f14', 'Methylotrophic': '#a6a6a6'}
TROLAB = {'Hydrogenotrophic': 'Hydrogenotrophic', 'Acetoclastic': r'Acetoclastic ($\mathbfit{Methanothrix}$)',
          'Methanosarcinaceae': 'Methanosarcinaceae', 'Methylotrophic': 'Methylotrophic'}
HCOL = {'MAG14': '#3b2a8c', 'MAG15': '#7e5fd0', 'Other Methanobacteriaceae': '#b49cec',      # ordinal violet ramp
        'Methanoculleus': '#c0601d', 'Methanoregulaceae': '#bdbdbd'}
HLAB = {'MAG14': 'MAG14', 'MAG15': 'MAG15', 'Other Methanobacteriaceae': 'Other Methanobacteriaceae',
        'Methanoculleus': r'$\mathbfit{Methanoculleus}$', 'Methanoregulaceae': 'Methanoregulaceae'}
NL = r' g N L$^{-1}$'
GROUPS = ['L', 'LC', 'M', 'MC', 'H', 'HC']
W, H = 7.2, 6.45
fig = plt.figure(figsize=(W, H))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


n = len(tab)
RH, Y0 = 0.175, 2.62
YT = Y0 + n * RH
yc = Y0 + (np.arange(n) + 0.5) * RH            # leaf k (k = 0 is the first leaf) drawn from the bottom up

# ================= A: dendrogram + track + heatmap ================================================
DX0, DXW = 0.10, 0.42
axd = add_ax(DX0, Y0, DXW, n * RH)
d = dendrogram(Z, orientation='left', no_labels=True, ax=axd, color_threshold=0, above_threshold_color='#4d4d4d')
assert [orig[i] for i in d['leaves']] == list(tab.index), 'leaf order mismatch'
for c in axd.collections:
    c.set_linewidth(0.8)
axd.set_ylim(0, 10 * n); axd.axis('off')
TX = DX0 + DXW + 0.04
HX = TX + 0.12
CW, GG = 0.355, 0.07
xs = []
x = HX
for k, gname in enumerate(GROUPS):
    xs.append(x); x += CW + (GG if k in (1, 3) else 0)
xs = np.array(xs); HX1 = xs[-1] + CW
axA = add_ax(TX, Y0, HX1 - TX, n * RH)
axA.set_xlim(TX, HX1); axA.set_ylim(Y0, YT); axA.axis('off')
ramp = LinearSegmentedColormap.from_list('blue', ['#eef5fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
norm = LogNorm(vmin=0.01, vmax=25)
for i, mag in enumerate(tab.index):
    r = tab.loc[mag]
    axA.add_patch(Rectangle((TX, yc[i] - RH / 2), 0.08, RH, facecolor=TROC[r.trophic], edgecolor='white', lw=0.5))
    for j, gname in enumerate(GROUPS):
        v = r[f'mean_{gname}']
        col = ramp(norm(min(max(v, 0.01), 25))) if v >= 0.01 else '#f6f6f6'
        axA.add_patch(Rectangle((xs[j], yc[i] - RH / 2), CW, RH, facecolor=col, edgecolor='white', lw=0.6))
        txt = '0' if v < 0.005 else (f'{v:.1f}' if v >= 1 else f'{v:.2f}')
        axA.text(xs[j] + CW / 2, yc[i], txt, ha='center', va='center', fontsize=5.6,
                 color='white' if v >= 2.5 else '#1a1a1a', fontweight='normal' if v < 1 else 'bold')
    if r.novel_genus:
        lab = r.family + '$^{\\dagger}$'
    elif any(ch.isdigit() for ch in r.genus):                     # GTDB placeholder genus (e.g. UBA6): roman
        lab = r.genus
    else:
        lab = r'$\mathbfit{' + r.genus.replace('_', r'\_') + '}$'
    axA.text(HX1 + 0.05, yc[i], f'{r["name"]}', ha='left', va='center', fontsize=6.8)
    axA.text(HX1 + 0.47, yc[i], lab, ha='left', va='center', fontsize=6.4, clip_on=False)
for j, gname in enumerate(GROUPS):
    axA.add_patch(Rectangle((xs[j], YT + 0.05), CW, 0.075, facecolor=C_HC if gname.endswith('C') else C_CTRL,
                            edgecolor='white', lw=0.3, clip_on=False))
for k, lv in enumerate([1, 2, 4]):
    axA.text((xs[2 * k] + xs[2 * k + 1] + CW) / 2, YT + 0.17, f'{lv}{NL}', ha='center', va='bottom', fontsize=7, clip_on=False)
# colour bar + legends under A
cax = add_ax(HX, Y0 - 0.20, 1.30, 0.07)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ramp), cax=cax, orientation='horizontal')
cb.set_ticks([0.01, 0.1, 1, 10]); cb.set_ticklabels(['≤0.01', '0.1', '1', '10']); cb.minorticks_off()
cb.ax.tick_params(labelsize=6.4, length=2, width=0.6, pad=1); cb.outline.set_linewidth(0.6)
cax.set_xlabel('Relative abundance (%)', fontsize=6.8, labelpad=1)
leg_ax = add_ax(HX + 1.45, Y0 - 0.68, 1.9, 0.60); leg_ax.axis('off')
leg_ax.legend(handles=[Patch(facecolor=TROC[k], label=TROLAB[k]) for k in TROC], loc='upper left', ncol=1, fontsize=6.3,
              handlelength=0.9, handletextpad=0.4, labelspacing=0.22, borderaxespad=0, title='Methanogen type',
              title_fontsize=6.5, alignment='left')
fig.text((HX1 + 0.05) / W, (Y0 - 0.70) / H, r'$^{\dagger}$Genus not assigned in GTDB', fontsize=6.2, ha='left', va='bottom')

# ================= B: hydrogenotroph composition ===================================================
RX, RW = 5.00, 2.06
BT = YT + 0.125
BH = 1.32
axB = add_ax(RX, BT - BH, RW, BH)
pos = np.array([0, 1, 2.5, 3.5, 5.0, 6.0])
ORDER = ['MAG14', 'MAG15', 'Other Methanobacteriaceae', 'Methanoculleus', 'Methanoregulaceae']   # bottom -> top
bottom = np.zeros(6)
for cat in ORDER:
    v = np.array([comp.loc[cat, [f'{gname}_{r}' for r in (1, 2, 3)]].mean() for gname in GROUPS])
    axB.bar(pos, v, 0.78, bottom=bottom, color=HCOL[cat], edgecolor='white', linewidth=0.6, zorder=2)
    bottom += v
for p_, gname, b_ in zip(pos, GROUPS, bottom):
    vals = comp.loc['Total hydrogenotrophs', [f'{gname}_{r}' for r in (1, 2, 3)]].values
    axB.errorbar(p_, b_, yerr=vals.std(ddof=1), color=INK, lw=0.8, capsize=2, capthick=0.8, zorder=3)
    axB.text(p_, b_ + vals.std(ddof=1) + 1.2, f'{b_:.0f}', ha='center', va='bottom', fontsize=6.4)
    axB.add_patch(Rectangle((p_ - 0.39, -6.2), 0.78, 3.0, facecolor=C_HC if gname.endswith('C') else C_CTRL,
                            edgecolor='none', clip_on=False))
for k, lv in enumerate([1, 2, 4]):
    axB.text(pos[2 * k:2 * k + 2].mean(), -8.0, f'{lv}{NL}', ha='center', va='top', fontsize=7)
axB.set_xlim(-0.6, 6.6); axB.set_ylim(0, 78); axB.set_xticks([]); axB.set_yticks([0, 20, 40, 60])
axB.set_ylabel('Hydrogenotrophs (%)')
axB.legend(handles=[Patch(facecolor=HCOL[c], edgecolor='white', label=HLAB[c]) for c in ORDER[::-1]],
           loc='upper left', ncol=2, fontsize=5.9, handlelength=0.9, handletextpad=0.3, labelspacing=0.2,
           columnspacing=0.6, borderaxespad=0.25)

# ================= C: response scatter ==============================================================
CT = BT - BH - 0.62
axC = add_ax(RX, Y0, RW, CT - Y0)
lo, hi = 0.01, 40
xx = np.array([lo, hi])
axC.plot(xx, xx, color='#9a9a9a', lw=0.8, zorder=1)
axC.plot(xx, xx * 2, color='#c4c4c4', lw=0.7, ls='--', zorder=1); axC.plot(xx, xx / 2, color='#c4c4c4', lw=0.7, ls='--', zorder=1)
MK = {'L': 'o', 'M': '^', 'H': 's'}
for _, r in sc.iterrows():
    xv, yv = max(r.ctrl, 0.012), max(r.char, 0.012)
    lw, ls, ec = (1.2, '-', INK) if r.q < 0.05 else (0.8, (0, (2, 1.2)), INK) if r.P < 0.05 else (0.4, '-', '#7a7a7a')
    axC.scatter(xv, yv, s=26, marker=MK[r.level], color=TROC[r.trophic], edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
LBL = {('MAG14', 'H'): (2.2, 20.7, 'right'), ('MAG3', 'H'): (0.35, 4.6, 'right'),
       ('MAG15', 'H'): (13, 1.75, 'center'), ('MAG4', 'M'): (14, 0.80, 'center'), ('MAG4', 'L'): (12, 0.33, 'center'),
       ('MAG1', 'L'): (1.1, 0.045, 'left')}
lvname = {'L': '1', 'M': '2', 'H': '4'}
for (nm, lv), (tx, ty, ha) in LBL.items():
    r = sc[(sc.name == nm) & (sc.level == lv)].iloc[0]
    xv, yv = max(r.ctrl, 0.012), max(r.char, 0.012)
    axC.annotate(f'{nm} ({lvname[lv]} g)', (xv, yv), xytext=(tx, ty), fontsize=5.8, ha=ha, va='center',
                 arrowprops=dict(arrowstyle='-', color='#6b6b6b', lw=0.5, shrinkA=0, shrinkB=2.5))
axC.set_xscale('log'); axC.set_yscale('log'); axC.set_xlim(lo, hi); axC.set_ylim(lo, hi)
axC.set_xticks([0.01, 0.1, 1, 10]); axC.set_xticklabels(['0.01', '0.1', '1', '10'])
axC.set_yticks([0.01, 0.1, 1, 10]); axC.set_yticklabels(['0.01', '0.1', '1', '10'])
axC.minorticks_off(); axC.tick_params(labelsize=6.8)
axC.set_xlabel('Control (%)'); axC.set_ylabel('Hydrochar (%)')
axC.text(0.03, 0.97, 'Higher with\nhydrochar', transform=axC.transAxes, ha='left', va='top', fontsize=6.2, color='#8a8a8a')
axC.text(0.97, 0.03, 'Higher in\ncontrol', transform=axC.transAxes, ha='right', va='bottom', fontsize=6.2, color='#8a8a8a')
axC.legend(handles=[Line2D([], [], marker=MK[k], ls='', mfc='#d9d9d9', mec='#555555', ms=4.4,
                           label=lvname[k] + (NL if k == 'H' else '')) for k in ['L', 'M', 'H']] +
                   [axC.scatter([], [], s=20, facecolor='white', edgecolor=INK, linewidth=1.2, label='q<0.05'),
                    axC.scatter([], [], s=20, facecolor='white', edgecolor=INK, linewidth=0.8, linestyle=(0, (2, 1.2)),
                                label='P<0.05')],
           loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=5, fontsize=5.8, handletextpad=0.05, columnspacing=0.55,
           borderaxespad=0.15, handlelength=1.0)

# ================= D: five representative MAGs ======================================================
show = [('MAG14', 26, [0, 10, 20]), ('MAG15', 10, [0, 4, 8]), ('MAG3', 3.6, [0, 1.5, 3]),
        ('MAG4', 5.5, [0, 2.5, 5]), ('MAG12', 25, [0, 10, 20])]
DY, DHt, DG = 0.52, 0.80, 0.30
DX0_ = 0.66
DWd = (7.05 - DX0_ - 4 * DG) / 5
rng = np.random.default_rng(5)
for k, (nm, ymax, yt) in enumerate(show):
    ax = add_ax(DX0_ + k * (DWd + DG), DY, DWd, DHt)
    r0 = tab[tab.name == nm].iloc[0]
    for suf, col, mk, dx in [('', C_CTRL, 'o', -0.14), ('C', C_HC, 's', 0.14)]:
        means = []
        for j, lv in enumerate(['L', 'M', 'H']):
            v = ps.loc[nm, [f'{lv}{suf}_{i}' for i in (1, 2, 3)]].values.astype(float)
            ax.scatter(j + dx + rng.uniform(-0.035, 0.035, 3), v, s=8, marker=mk, facecolor='white', edgecolor=col,
                       linewidth=0.8, zorder=3)
            means.append(v.mean())
        ax.plot(np.arange(3) + dx, means, color=col, lw=1.0, zorder=2)
        ax.scatter(np.arange(3) + dx, means, s=30, marker=mk, facecolor=col, edgecolor=INK, linewidth=0.6, zorder=4)
    for j, lv in enumerate(['L', 'M', 'H']):
        a = ps.loc[nm, [f'{lv}_{i}' for i in (1, 2, 3)]].values.astype(float)
        b = ps.loc[nm, [f'{lv}C_{i}' for i in (1, 2, 3)]].values.astype(float)
        p = stats.ttest_ind(b, a, equal_var=False).pvalue if np.ptp(np.r_[a, b]) > 0 else 1
        if stars(p):
            ax.text(j, max(a.max(), b.max()) + ymax * 0.03, stars(p), ha='center', va='bottom', fontsize=8)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, ymax); ax.set_yticks(yt); ax.set_xticks(range(3))
    ax.set_xticklabels(['1', '2', '4']); ax.tick_params(labelsize=6.8)
    ax.add_patch(Rectangle((0.0, 1.07), 0.06, 0.13, transform=ax.transAxes, facecolor=TROC[r0.trophic], edgecolor='none',
                           clip_on=False))
    tl = (r0.family + '$^{\\dagger}$') if r0.novel_genus else r'$\mathbfit{' + r0.genus + '}$'
    ax.text(0.10, 1.06, f'{nm}  ' + tl, transform=ax.transAxes, ha='left', va='bottom', fontsize=6.4)
fig.text(0.24 / W, (DY + DHt / 2) / H, 'Relative\nabundance (%)', rotation=90, ha='center', va='center', fontsize=7.5,
         fontweight='bold')
fig.text((DX0_ + (7.05 - DX0_) / 2) / W, 0.08 / H, r'Added NH$_4^+$-N (g L$^{-1}$)', ha='center', va='bottom', fontsize=8,
         fontweight='bold')
fig.legend(handles=[Line2D([], [], marker='o', ls='-', color=C_CTRL, mfc=C_CTRL, mec=INK, ms=4.5, lw=1.0, label='Control'),
                    Line2D([], [], marker='s', ls='-', color=C_HC, mfc=C_HC, mec=INK, ms=4.5, lw=1.0, label='Hydrochar')],
           loc='lower right', bbox_to_anchor=(7.05 / W, (DY + DHt + 0.30) / H), ncol=2, fontsize=7, handlelength=1.8,
           handletextpad=0.3, columnspacing=1.2, borderaxespad=0)


# ================= panel letters ===================================================================
def letter(x_in, y_in, s):
    fig.text(x_in / W, y_in / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')


letter(0.06, YT + 0.36, 'A')
letter(RX - 0.50, YT + 0.36, 'B')
letter(RX - 0.50, CT + 0.04, 'C')
letter(0.06, DY + DHt + 0.30, 'D')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
