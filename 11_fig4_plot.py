r"""Figure 4: syntrophic propionate-oxidising bacteria (SPOB) at MAG level.
A  abundance heatmap of the 12 SPOB MAGs in all 18 samples (log colour scale) with a lineage strip and a
   CheckM quality track;
B  hydrochar effect per ammonia level (colour = log2FC hydrochar/control, size = abundance, outline = significance);
C  presence of mmc-pathway steps and electron-disposal genes (rows shared with A and B);
D  SPOB lineage composition of each group;
E  four representative MAGs (replicates, means, Welch t).
Usage: python 11_fig4_plot.py <folder with fig4_*.csv> <out prefix>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'mathtext.fontset': 'stix', 'mathtext.default': 'bf', 'legend.frameon': False,
                     'axes.facecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
DIR, out = sys.argv[1], sys.argv[2]
ab = pd.read_csv(f'{DIR}/fig4_spob_abundance.csv', index_col=0)
eff = pd.read_csv(f'{DIR}/fig4_hydrochar_effect_spob.csv')
gm = pd.read_csv(f'{DIR}/fig4_gene_matrix.csv', index_col=0)
lin = pd.read_csv(f'{DIR}/fig4_lineages.csv', index_col=0)

C_CTRL, C_HC, INK = '#24C4DB', '#ED117B', '#000000'
LINC = {'Syntrophobacteraceae': '#1baf7a', 'Syntrophopropionicum': '#eda100',
        'Other Pelotomaculaceae': '#bdbdbd', 'DTU098': '#7a52d6'}            # validated (CVD, all pairs)
LINLAB = {'Syntrophobacteraceae': 'Syntrophobacteraceae', 'Syntrophopropionicum': r'$\mathbfit{Syntrophopropionicum}$',
          'Other Pelotomaculaceae': 'Other Pelotomaculaceae', 'DTU098': 'DTU098 (Pelotomaculaceae)'}
NL = r' g N L$^{-1}$'
W, H = 7.2, 6.75
fig = plt.figure(figsize=(W, H))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


n = len(ab)
RH = 0.178                                   # row height (in)
Y0 = 3.95                                    # bottom of the shared-row block
YT = Y0 + n * RH                             # top of the shared-row block
rows_y = YT - (np.arange(n) + 0.5) * RH      # row centres (in)

# ================= A: heatmap ======================================================================
HX = 0.86                                    # heatmap left edge
CWID, GIN, GOUT = 0.118, 0.035, 0.075
samples, xs, x = [], [], HX
for lv in ['L', 'M', 'H']:
    for gname in [lv, lv + 'C']:
        for r in (1, 2, 3):
            samples.append(f'{gname}_{r}'); xs.append(x); x += CWID
        x += GIN
    x += GOUT - GIN
xs = np.array(xs); HX1 = xs[-1] + CWID
axA = add_ax(HX, Y0, HX1 - HX, n * RH)
axA.set_xlim(HX, HX1); axA.set_ylim(Y0, YT); axA.axis('off')
ramp = LinearSegmentedColormap.from_list('blue', ['#eef5fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
ramp.set_under('#f6f6f6')
norm = LogNorm(vmin=0.01, vmax=4.0)
for i, mag in enumerate(ab.index):
    for j, s in enumerate(samples):
        v = ab.loc[mag, s]
        col = ramp(norm(min(v, 4.0))) if v >= 0.01 else '#f6f6f6'
        axA.add_patch(Rectangle((xs[j], rows_y[i] - RH / 2), CWID, RH, facecolor=col, edgecolor='white', lw=0.5))
# treatment strip + ammonia labels on top
for j, s in enumerate(samples):
    axA.add_patch(Rectangle((xs[j], YT + 0.05), CWID, 0.075, facecolor=C_HC if 'C_' in s else C_CTRL,
                            edgecolor='white', lw=0.3, clip_on=False))
for k, lv in enumerate([1, 2, 4]):
    x0_, x1_ = xs[6 * k], xs[6 * k + 5] + CWID
    axA.text((x0_ + x1_) / 2, YT + 0.17, f'{lv}{NL}', ha='center', va='bottom', fontsize=7)
# lineage strip and row labels
for i, mag in enumerate(ab.index):
    axA.add_patch(Rectangle((HX - 0.105, rows_y[i] - RH / 2), 0.075, RH, facecolor=LINC[ab.loc[mag, 'lineage']],
                            edgecolor='white', lw=0.5, clip_on=False))
    axA.text(HX - 0.13, rows_y[i], ab.loc[mag, 'name'], ha='right', va='center', fontsize=7)
# colour bar under the heatmap
cax = add_ax(HX + 0.10, Y0 - 0.24, 1.25, 0.075)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ramp), cax=cax, orientation='horizontal')
cb.set_ticks([0.01, 0.1, 1, 4]); cb.set_ticklabels(['≤0.01', '0.1', '1', '4']); cb.ax.tick_params(labelsize=6.5, length=2, width=0.6)
cb.outline.set_linewidth(0.6); cb.minorticks_off()
cax.set_title('Relative abundance (%)', fontsize=6.8, pad=2)

# CheckM quality track
QX, QW = HX1 + 0.10, 0.46
axQ = add_ax(QX, Y0, QW, n * RH)
for i, mag in enumerate(ab.index):
    axQ.barh(rows_y[i], ab.loc[mag, 'completeness'], height=RH * 0.62, color='#cfcfcf', edgecolor='none')
    axQ.barh(rows_y[i], ab.loc[mag, 'contamination'], height=RH * 0.62, color='#4d4d4d', edgecolor='none')
axQ.set_ylim(Y0, YT); axQ.set_xlim(0, 100); axQ.set_yticks([])
axQ.set_xticks([0, 50, 100]); axQ.tick_params(axis='x', labelsize=6.2, length=2, pad=1)
for sp_ in ['top', 'right', 'left']:
    axQ.spines[sp_].set_visible(False)
axQ.spines['bottom'].set_linewidth(0.6)
axQ.text(50, YT + 0.17, 'CheckM (%)', ha='center', va='bottom', fontsize=6.6, transform=axQ.transData, clip_on=False)
axQ.legend(handles=[Patch(facecolor='#cfcfcf', label='Compl.'), Patch(facecolor='#4d4d4d', label='Contam.')],
           loc='upper left', bbox_to_anchor=(-0.05, -0.085), fontsize=6.0, handlelength=0.8, handletextpad=0.3,
           labelspacing=0.2, borderaxespad=0)

# ================= B: hydrochar effect bubbles =====================================================
BX, BSP = QX + QW + 0.36, 0.27
bx = BX + BSP * np.arange(3)
axB = add_ax(BX - BSP / 2, Y0, 3 * BSP, n * RH)
axB.set_xlim(BX - BSP / 2, BX + 2.5 * BSP); axB.set_ylim(Y0, YT)
div = LinearSegmentedColormap.from_list('div', ['#0a7584', C_CTRL, '#ececec', C_HC, '#9c0950'])
dnorm = Normalize(-6, 6)
for k in range(3):
    axB.axvline(bx[k], color='#e6e6e6', lw=0.6, zorder=0)
for i in range(n):
    axB.axhline(rows_y[i], color='#f0f0f0', lw=0.5, zorder=0)
for i, mag in enumerate(ab.index):
    name = ab.loc[mag, 'name']
    for k, lv in enumerate(['L', 'M', 'H']):
        r = eff[(eff.name == name) & (eff.level == lv)]
        if r.empty:
            axB.scatter(bx[k], rows_y[i], s=3, color='#b0b0b0', zorder=2)
            continue
        r = r.iloc[0]
        size = 8 + 65 * np.sqrt(max(r.ctrl, r.char))
        lw, ls, ec = (1.3, '-', INK) if r.q < 0.05 else (0.8, (0, (2, 1.2)), INK) if r.P < 0.05 else (0.4, '-', '#8a8a8a')
        axB.scatter(bx[k], rows_y[i], s=size, color=div(dnorm(np.clip(r.log2FC, -6, 6))), edgecolor=ec,
                    linewidth=lw, linestyle=ls, zorder=3)
axB.set_xticks(bx); axB.set_xticklabels(['1', '2', '4'], fontsize=7); axB.set_yticks([])
axB.tick_params(axis='x', length=0, pad=2)
for sp_ in ['top', 'right', 'left', 'bottom']:
    axB.spines[sp_].set_visible(False)
axB.text(bx[1], YT + 0.17, 'Hydrochar effect', ha='center', va='bottom', fontsize=6.8)
axB.text(bx[1], YT + 0.05, r'NH$_4^+$-N (g L$^{-1}$)', ha='center', va='bottom', fontsize=5.8)
# colour bar + size/outline keys under B
cbx = add_ax(BX - 0.08, Y0 - 0.36, 0.70, 0.065)
cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap=div), cax=cbx, orientation='horizontal')
cb2.set_ticks([-6, 0, 6]); cb2.ax.tick_params(labelsize=6.2, length=2, width=0.6, pad=1); cb2.outline.set_linewidth(0.6)
cbx.set_title(r'log$_2$FC (HC/Ctrl)', fontsize=6.4, pad=2)
kax = add_ax(BX - 0.34, Y0 - 1.02, 1.20, 0.50); kax.axis('off'); kax.set_xlim(0, 1.2); kax.set_ylim(0, 1)
for xk, v in zip([0.06, 0.48, 0.90], [0.1, 1, 3]):
    kax.scatter(xk, 0.84, s=8 + 65 * np.sqrt(v), facecolor='white', edgecolor='#6b6b6b', linewidth=0.6)
    kax.text(xk + 0.05 + 0.012 * np.sqrt(8 + 65 * np.sqrt(v)), 0.84, f'{v:g}%', fontsize=5.8, va='center', ha='left')
kax.scatter(0.10, 0.48, s=34, facecolor='white', edgecolor=INK, linewidth=1.3)
kax.text(0.19, 0.48, 'q < 0.05', fontsize=5.8, va='center')
kax.scatter(0.68, 0.48, s=34, facecolor='white', edgecolor=INK, linewidth=0.8, linestyle=(0, (2, 1.2)))
kax.text(0.77, 0.48, 'P < 0.05', fontsize=5.8, va='center')
kax.scatter(0.10, 0.14, s=3, color='#b0b0b0'); kax.text(0.19, 0.14, 'Not tested (mean < 0.1%)', fontsize=5.8, va='center')

# ================= C: gene matrix ==================================================================
genes = ['pct/prpE', 'mmdA/pcc', 'epi', 'mcm', 'sucCD', 'sdh/frd', 'fum', 'mdh', 'oad/pyc', 'por', 'pta-ack/acd', 'fdh', 'hyd']
n_mmc = 11
CX0, GW, GGAP = BX + 2.5 * BSP + 0.30, 0.136, 0.08
gx = np.array([CX0 + (k + 0.5) * GW + (GGAP if k >= n_mmc else 0) for k in range(len(genes))])
axC = add_ax(CX0, Y0, gx[-1] + GW / 2 - CX0, n * RH)
axC.set_xlim(CX0, gx[-1] + GW / 2); axC.set_ylim(Y0, YT); axC.axis('off')
for i, mag in enumerate(ab.index):
    axC.add_patch(Rectangle((CX0, rows_y[i] - RH / 2), gx[-1] + GW / 2 - CX0, RH,
                            facecolor='#f7f7f7' if i % 2 else 'none', edgecolor='none', zorder=0))
    for k, gname in enumerate(genes):
        present = bool(gm.loc[mag, gname])
        if present:
            axC.scatter(gx[k], rows_y[i], s=30, color='#3a3a3a' if k < n_mmc else '#7a52d6', edgecolor='none', zorder=2)
        else:
            axC.scatter(gx[k], rows_y[i], s=26, facecolor='white', edgecolor='#b5b5b5', linewidth=0.7, zorder=2)
for k, gname in enumerate(genes):
    axC.text(gx[k], Y0 - 0.05, gname, rotation=90, ha='center', va='top', fontsize=6.4, style='italic')
for x0_, x1_, lab in [(gx[0], gx[n_mmc - 1], 'Propionate → acetate (mmc)'),
                      (gx[n_mmc], gx[-1], r'e$^{-}$ disposal')]:
    axC.plot([x0_ - 0.04, x1_ + 0.04], [YT + 0.10] * 2, color=INK, lw=0.8, clip_on=False)
    axC.text((x0_ + x1_) / 2, YT + 0.15, lab, ha='center', va='bottom', fontsize=6.6, clip_on=False)

# ================= D: lineage composition ==========================================================
DX, DY, DW, DH = 0.62, 0.50, 2.40, 2.12
axD = add_ax(DX, DY, DW, DH)
STACK = ['DTU098', 'Other Pelotomaculaceae', 'Syntrophopropionicum', 'Syntrophobacteraceae']   # bottom -> top
groups = ['L', 'LC', 'M', 'MC', 'H', 'HC']
pos = np.array([0, 1, 2.6, 3.6, 5.2, 6.2])
gmean = {gname: lin[[f'{gname}_{r}' for r in (1, 2, 3)]].mean(1) for gname in groups}
bottom = np.zeros(6)
for L_ in STACK:
    v = np.array([gmean[gname][L_] for gname in groups])
    axD.bar(pos, v, 0.78, bottom=bottom, color=LINC[L_], edgecolor='white', linewidth=0.6, zorder=2)
    bottom += v
tot = lin.loc['Total SPOB']
for p_, gname, b_ in zip(pos, groups, bottom):
    vals = tot[[f'{gname}_{r}' for r in (1, 2, 3)]].values
    axD.errorbar(p_, b_, yerr=vals.std(ddof=1), color=INK, lw=0.8, capsize=2, capthick=0.8, zorder=3)
    axD.text(p_, b_ + vals.std(ddof=1) + 0.25, f'{b_:.1f}', ha='center', va='bottom', fontsize=6.6)
    axD.add_patch(Rectangle((p_ - 0.39, -0.62), 0.78, 0.30, facecolor=C_HC if gname.endswith('C') else C_CTRL,
                            edgecolor='none', clip_on=False))
for k, lv in enumerate([1, 2, 4]):
    axD.text(pos[2 * k:2 * k + 2].mean(), -0.85, f'{lv}{NL}', ha='center', va='top', fontsize=7.5)
axD.set_xlim(-0.65, 6.85); axD.set_ylim(0, 12.5); axD.set_xticks([])
axD.set_yticks([0, 3, 6, 9, 12]); axD.set_ylabel('SPOB relative abundance (%)')
axD.legend(handles=[Patch(facecolor=LINC[L_], edgecolor='white', label=LINLAB[L_]) for L_ in STACK[::-1]],
           loc='upper right', fontsize=6.4, handlelength=1.0, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)

# ================= E: four representative MAGs =====================================================
show = [('MAG93', 'Syntrophobacteraceae', 4.4, [0, 2, 4]), ('MAG132', 'DTU098', 5.2, [0, 2, 4]),
        ('MAG128', 'Syntrophopropionicum', 4.4, [0, 2, 4]), ('MAG133', 'Syntrophopropionicum', 1.45, [0, 0.6, 1.2])]
EX, EW, EG, EH, EGY = 3.88, 1.42, 0.40, 0.78, 0.46
ey = [DY + EH + EGY, DY + EH + EGY, DY, DY]
ex = [EX, EX + EW + EG, EX, EX + EW + EG]
rng = np.random.default_rng(11)
axE = []
name2idx = {ab.loc[m, 'name']: m for m in ab.index}
for (nm, L_, ymax, yt), x0_, y0_ in zip(show, ex, ey):
    ax = add_ax(x0_, y0_, EW, EH); axE.append(ax)
    mag = name2idx[nm]
    for trt, suf, col, mk, dx in [('Control', '', C_CTRL, 'o', -0.14), ('Hydrochar', 'C', C_HC, 's', 0.14)]:
        means = []
        for k, lv in enumerate(['L', 'M', 'H']):
            v = ab.loc[mag, [f'{lv}{suf}_{r}' for r in (1, 2, 3)]].values.astype(float)
            ax.scatter(k + dx + rng.uniform(-0.035, 0.035, 3), v, s=9, marker=mk, facecolor='white', edgecolor=col,
                       linewidth=0.8, zorder=3)
            means.append(v.mean())
        ax.plot(np.arange(3) + dx, means, color=col, lw=1.0, zorder=2)
        ax.scatter(np.arange(3) + dx, means, s=34, marker=mk, facecolor=col, edgecolor=INK, linewidth=0.6, zorder=4)
    for k, lv in enumerate(['L', 'M', 'H']):
        a = ab.loc[mag, [f'{lv}_{r}' for r in (1, 2, 3)]].values.astype(float)
        b = ab.loc[mag, [f'{lv}C_{r}' for r in (1, 2, 3)]].values.astype(float)
        p = stats.ttest_ind(b, a, equal_var=False).pvalue if np.ptp(np.r_[a, b]) > 0 else 1
        if stars(p):
            ax.text(k, max(a.max(), b.max()) + ymax * 0.03, stars(p), ha='center', va='bottom', fontsize=8)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, ymax); ax.set_yticks(yt); ax.set_xticks(range(3))
    ax.set_xticklabels(['1', '2', '4'] if y0_ == DY else [])
    ax.tick_params(labelsize=6.8)
    # title above the frame: lineage swatch + MAG name + lineage
    ax.add_patch(Rectangle((0.0, 1.07), 0.05, 0.13, transform=ax.transAxes, facecolor=LINC[L_], edgecolor='none', clip_on=False))
    lab = r'$\mathbfit{Syntrophopropionicum}$' if L_ == 'Syntrophopropionicum' else L_
    ax.text(0.08, 1.06, f'{nm}  ' + lab, transform=ax.transAxes, ha='left', va='bottom', fontsize=6.9)
fig.text((EX - 0.36) / W, (DY + EH + EGY / 2) / H, 'Relative abundance (%)', rotation=90, ha='center', va='center',
         fontsize=8, fontweight='bold')
fig.text((EX + EW + EG / 2) / W, (DY - 0.30) / H, r'Added NH$_4^+$-N (g L$^{-1}$)', ha='center', va='top', fontsize=8,
         fontweight='bold')
fig.legend(handles=[Line2D([], [], marker='o', ls='-', color=C_CTRL, mfc=C_CTRL, mec=INK, ms=4.5, lw=1.0, label='Control'),
                    Line2D([], [], marker='s', ls='-', color=C_HC, mfc=C_HC, mec=INK, ms=4.5, lw=1.0, label='Hydrochar')],
           loc='lower center', bbox_to_anchor=((EX + EW + EG / 2) / W, (DY + 2 * EH + EGY + 0.24) / H), ncol=2,
           fontsize=7, handlelength=1.8, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)


# ================= panel letters ===================================================================
def letter(x_in, y_in, s):
    fig.text(x_in / W, y_in / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')


letter(DX - 0.48, YT + 0.36, 'A')
letter(BX - BSP / 2 - 0.20, YT + 0.36, 'B')
letter(CX0 - 0.16, YT + 0.36, 'C')
letter(DX - 0.48, DY + DH + 0.04, 'D')
letter(EX - 0.50, DY + DH + 0.04, 'E')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
