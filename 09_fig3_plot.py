"""Figure 3: community structure (MAG-based).
A  PCoA (Bray-Curtis) with rounded hulls, centroid spiders, control->hydrochar shift arrows (labelled with the mean
   between-group distance) and an inset PERMANOVA R2 donut;
B  within-group vs control-hydrochar Bray-Curtis distances per ammonia level;
C  guild composition of every sample with a treatment strip;
D  ratio of hydrogenotrophic methanogens to Methanothrix (dumbbell).
Colour encodes treatment (control cyan, hydrochar pink); marker shape encodes ammonia level.
Usage: python 09_fig3_plot.py <folder with fig3_*.csv> <out prefix>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from shapely.geometry import MultiPoint
from scipy import stats

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'mathtext.fontset': 'stix', 'mathtext.default': 'bf', 'legend.frameon': False,
                     'axes.facecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
DIR, out = sys.argv[1], sys.argv[2]
pc = pd.read_csv(f'{DIR}/fig3_pcoa.csv', index_col=0)
expl = pd.read_csv(f'{DIR}/fig3_pcoa_variance.csv', index_col=0).iloc[:, 0]
perm = pd.read_csv(f'{DIR}/fig3_permanova.csv')
pairs = pd.read_csv(f'{DIR}/fig3_distances.csv')
gu = pd.read_csv(f'{DIR}/fig3_guilds.csv', index_col=0)
C_CTRL, C_HC, INK, ARROW = '#24C4DB', '#ED117B', '#000000', '#3d3d3d'
C_CTRL_TXT = '#0f8a9b'
MK = {1: 'o', 2: '^', 4: 's'}
NL = r' g N L$^{-1}$'
W, H = 7.2, 6.35
fig = plt.figure(figsize=(W, H))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


# layout grid (inches)
AX, AY, AW, AH = 0.58, 2.88, 3.05, 3.05          # A: PCoA
RX, RW = 4.38, 2.70                               # right column (B over D)
BH, DH = 1.28, 1.28
CX, CY, CW, CH = 0.58, 0.50, 4.75, 1.62           # C: stacked bars
rng = np.random.default_rng(7)

# ================= A: PCoA
axA = add_ax(AX, AY, AW, AH)
XL, YL = (-0.32, 0.50), (-0.26, 0.30)
sx, sy = AW / (XL[1] - XL[0]), AH / (YL[1] - YL[0])        # data units -> inches, so hull padding is round
cent = {}
for (lv, hc), s in pc.groupby(['ammonia', 'hydrochar']):
    col = C_HC if hc else C_CTRL
    P = s[['PC1', 'PC2']].values
    c = P.mean(0); cent[(lv, hc)] = c
    shape = MultiPoint([(x * sx, y * sy) for x, y in P]).convex_hull.buffer(0.10, resolution=24)
    hx, hy = shape.exterior.xy
    axA.add_patch(Polygon(np.column_stack([np.array(hx) / sx, np.array(hy) / sy]), closed=True,
                          facecolor=col, alpha=0.18, edgecolor=col, lw=0.9, zorder=1))
    for p_ in P:
        axA.plot([c[0], p_[0]], [c[1], p_[1]], color=col, lw=0.7, zorder=2)
    axA.scatter(P[:, 0], P[:, 1], s=34, marker=MK[lv], facecolor=col, edgecolor=INK, linewidth=0.6, zorder=4)
    axA.scatter([c[0]], [c[1]], s=10, marker='o', facecolor='white', edgecolor=col, linewidth=0.9, zorder=3)
bc = pairs[pairs.type == 'Between'].groupby('ammonia').d.mean()
for lv in [1, 2, 4]:
    a, b = cent[(lv, 0)], cent[(lv, 1)]
    axA.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=1.0, color=ARROW, shrinkA=4, shrinkB=4, zorder=5))
    mid = (a + b) / 2
    axA.text(mid[0] + 0.012, mid[1], f'{bc[lv]:.2f}', fontsize=7.5, ha='left', va='center', color=ARROW, zorder=6)
LABPOS = {(1, 1): (-0.235, 0.088), (2, 1): (-0.040, 0.128), (4, 1): (0.060, 0.163),
          (1, 0): (-0.245, -0.120), (2, 0): (0.005, -0.108), (4, 0): (0.232, -0.152)}
for (lv, hc), (x, y) in LABPOS.items():
    axA.text(x, y, f'{lv}{NL}', fontsize=7, ha='center', va='center', color=C_HC if hc else C_CTRL_TXT, zorder=6)
axA.set_xlim(*XL); axA.set_ylim(*YL)
axA.axhline(0, color='#bdbdbd', lw=0.6, ls='--', zorder=0); axA.axvline(0, color='#bdbdbd', lw=0.6, ls='--', zorder=0)
axA.set_xlabel(f'PCoA1 ({expl.iloc[0]:.1f}%)'); axA.set_ylabel(f'PCoA2 ({expl.iloc[1]:.1f}%)')
leg = [Line2D([], [], marker='s', ls='', mfc=C_CTRL, mec=INK, ms=6, label='Control'),
       Line2D([], [], marker='s', ls='', mfc=C_HC, mec=INK, ms=6, label='Hydrochar')] + \
      [Line2D([], [], marker=MK[lv], ls='', mfc='#d9d9d9', mec=INK, ms=5.5, label=f'{lv}{NL}') for lv in [1, 2, 4]]
axA.legend(handles=leg, loc='upper left', fontsize=7, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)
# inset PERMANOVA donut (upper right, clear of the data)
din = fig.add_axes([(AX + AW - 1.26) / W, (AY + AH - 1.12) / H, 1.08 / W, 1.08 / H])
dcol = ['#4a5a8a', C_HC, '#e2a400', '#e3e3e3']
din.pie(perm.R2, colors=dcol, startangle=90, counterclock=False, wedgeprops=dict(width=0.36, edgecolor='white', linewidth=1.2))
din.text(0, 0.0, r'R$^2$', ha='center', va='center', fontsize=7.5)
din.text(0, -1.30, 'PERMANOVA', ha='center', va='center', fontsize=6.0)
dlab = [r'NH$_4^+$', 'HC', r'NH$_4^+$×HC', 'Res.']
ang0 = 90.0
for r2, lab in zip(perm.R2, dlab):
    ang = ang0 - r2 * 180.0
    ang0 -= r2 * 360.0
    x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    ha = 'left' if x > 0.15 else ('right' if x < -0.15 else 'center')
    din.text(1.12 * x, 1.12 * y, f'{lab} {r2 * 100:.0f}%', ha=ha, va='center', fontsize=5.6, color=INK)
din.set_xlim(-1.9, 1.9); din.set_ylim(-1.45, 1.45); din.set_aspect('equal'); din.axis('off')

# ================= B: within vs between distances
axB = add_ax(RX, AY + AH - BH, RW, BH)
for k, (typ, col, mk) in enumerate([('Within', '#bdbdbd', 'o'), ('Between', '#c84fd0', 'D')]):
    data = [pairs[(pairs.ammonia == lv) & (pairs.type == typ)].d.values for lv in [1, 2, 4]]
    pos = np.arange(3) + (0.2 if k else -0.2)
    bp = axB.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                     medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK, lw=0.8),
                     capprops=dict(color=INK, lw=0.8), boxprops=dict(edgecolor=INK, lw=0.8))
    for bx in bp['boxes']:
        bx.set_facecolor(col); bx.set_alpha(0.9)
    for x0, v in zip(pos, data):
        axB.scatter(x0 + rng.uniform(-0.08, 0.08, len(v)), v, s=8, marker=mk, facecolor=col, edgecolor=INK, linewidth=0.5, zorder=4)
axB.set_xticks(range(3)); axB.set_xticklabels(['1', '2', '4']); axB.set_xlim(-0.6, 2.6)
axB.set_ylim(0, 0.47); axB.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axB.set_xlabel(r'Added NH$_4^+$-N (g L$^{-1}$)'); axB.set_ylabel('Bray–Curtis distance')
axB.legend(handles=[Patch(facecolor='#bdbdbd', edgecolor=INK, label='Within group'),
                    Patch(facecolor='#c84fd0', edgecolor=INK, label='Control vs hydrochar')],
           loc='upper left', fontsize=6.8, handlelength=1.1, borderaxespad=0.3, ncol=2, columnspacing=1.0)

# ================= D: hydrogenotrophs : Methanothrix (dumbbell)
axD = add_ax(RX, AY, RW, DH)
ratio = gu.loc['Hydrogenotrophic methanogens'] / gu.loc['Methanothrix']
for i, (lv, lab) in enumerate([(1, 'L'), (2, 'M'), (4, 'H')]):
    a = ratio[[f'{lab}_{r}' for r in (1, 2, 3)]].values
    b = ratio[[f'{lab}C_{r}' for r in (1, 2, 3)]].values
    axD.plot([i - 0.15, i + 0.15], [a.mean(), b.mean()], color='#8c8c8c', lw=1.6, zorder=1)
    for x0, v, col, mk in [(i - 0.15, a, C_CTRL, 'o'), (i + 0.15, b, C_HC, 's')]:
        axD.scatter(x0 + rng.uniform(-0.04, 0.04, 3), v, s=9, marker=mk, facecolor='white', edgecolor=col, linewidth=0.8, zorder=2)
        axD.scatter([x0], [v.mean()], s=46, marker=mk, facecolor=col, edgecolor=INK, linewidth=0.7, zorder=3)
    p = stats.ttest_ind(b, a, equal_var=False).pvalue
    st = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    axD.text(i, max(a.max(), b.max()) + 0.25, st, ha='center', va='bottom', fontsize=8 if p < 0.05 else 7)
axD.set_xticks(range(3)); axD.set_xticklabels(['1', '2', '4']); axD.set_xlim(-0.6, 2.6)
axD.set_ylim(1, 5); axD.set_yticks([1, 2, 3, 4, 5])
axD.set_xlabel(r'Added NH$_4^+$-N (g L$^{-1}$)')
axD.set_ylabel('Hydrogenotrophs to\n' + r'$\mathbfit{Methanothrix}$ ratio')
axD.legend(handles=[Line2D([], [], marker='o', ls='', mfc=C_CTRL, mec=INK, ms=5.5, label='Control'),
                    Line2D([], [], marker='s', ls='', mfc=C_HC, mec=INK, ms=5.5, label='Hydrochar')],
           loc='upper left', fontsize=6.8, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3)

# ================= C: guild composition of every sample
axC = add_ax(CX, CY, CW, CH)
ORDER = ['SPOB', 'Other mmc bacteria', 'Other bacteria', 'Methanothrix', 'Other archaea', 'Methanosarcina',
         'Hydrogenotrophic methanogens']
COL = {'SPOB': '#c84fd0', 'Other mmc bacteria': '#f2858a', 'Other bacteria': '#d4d4d4', 'Methanothrix': '#e2a400',
       'Other archaea': '#7d8ce0', 'Methanosarcina': '#4d8f14', 'Hydrogenotrophic methanogens': '#10b3c6'}
LABEL = {'SPOB': 'SPOB', 'Other mmc bacteria': 'Other bacteria with\nmmc pathway', 'Other bacteria': 'Other bacteria',
         'Methanothrix': r'$\mathbfit{Methanothrix}$', 'Other archaea': 'Other archaea',
         'Methanosarcina': r'$\mathbfit{Methanosarcina}$', 'Hydrogenotrophic methanogens': 'Hydrogenotrophic\nmethanogens'}
samples, xs, x = [], [], 0.0
for lab in ['L', 'M', 'H']:
    for g_ in [lab, lab + 'C']:
        for r in (1, 2, 3):
            samples.append(f'{g_}_{r}'); xs.append(x); x += 1
        x += 0.35
    x += 0.9
xs = np.array(xs)
comp = gu[samples].reindex(ORDER); comp = comp / comp.sum() * 100
bottom = np.zeros(len(samples))
for k in ORDER:
    axC.bar(xs, comp.loc[k].values, 0.86, bottom=bottom, color=COL[k], edgecolor='white', linewidth=0.6, zorder=2)
    bottom += comp.loc[k].values
for i, s in enumerate(samples):
    axC.add_patch(plt.Rectangle((xs[i] - 0.43, 101.5), 0.86, 5.0, color=C_HC if 'C_' in s else C_CTRL, lw=0, clip_on=False))
axC.set_xlim(xs[0] - 0.8, xs[-1] + 0.8); axC.set_ylim(0, 108); axC.set_yticks([0, 25, 50, 75, 100]); axC.set_xticks([])
axC.set_ylabel('Relative abundance (%)')
for bx, lv in zip([xs[0:6].mean(), xs[6:12].mean(), xs[12:18].mean()], [1, 2, 4]):
    axC.text(bx, -6, f'{lv}{NL}', ha='center', va='top', fontsize=8)
handles = [Patch(facecolor=COL[k], edgecolor='white', label=LABEL[k]) for k in ORDER[::-1]]
handles += [Patch(facecolor=C_CTRL, label='Control'), Patch(facecolor=C_HC, label='Hydrochar')]
axC.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=6.8, handlelength=1.1,
           borderaxespad=0.0, labelspacing=0.32)


# ================= panel letters (next to each panel's top-left, above the frame)
def letter(ax, s, dx=0.48):
    p = ax.get_position()
    fig.text((p.x0 * W - dx) / W, (p.y1 * H + 0.04) / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')


for ax, s in [(axA, 'A'), (axB, 'B'), (axC, 'C'), (axD, 'D')]:
    letter(ax, s)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
