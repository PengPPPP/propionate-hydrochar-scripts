r"""Figure 6: metatranscriptome at 4 g N/L (control vs hydrochar), strict 2 x 2 grid.
Every panel frame sits on the same grid (columns share left/right edges, rows share top/bottom edges);
panel letters sit just outside the top-left corner of each frame.
A  mRNA share vs DNA share of every MAG (control), with control->hydrochar arrows for key MAGs;
B  DNA and mRNA shares of each guild (back-to-back bars, control vs hydrochar);
C  within-genome module expression of MAG132 (log2 fold change hydrochar/control), labels inside the frame;
D  which methanogens carry the community's mcrA, formate, H2 and acetate-cleavage transcripts (control vs
   hydrochar means), legend inside the frame.
Usage: python 16_fig6_plot.py <folder with fig6_*.csv> <out prefix>
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 8,
                     'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3,
                     'ytick.major.size': 3, 'mathtext.fontset': 'stix', 'mathtext.default': 'bf', 'legend.frameon': False,
                     'axes.facecolor': 'none', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
DIR, out = sys.argv[1], sys.argv[2]
act = pd.read_csv(f'{DIR}/fig6_mrna_share.csv', index_col=0)
gs = pd.read_csv(f'{DIR}/fig6_guild_mrna.csv', index_col=0)
mc = pd.read_csv(f'{DIR}/fig6_mcrA_sources.csv', index_col=0)
mods = pd.read_csv(f'{DIR}/fig6_mag132_modules.csv')

C_CTRL, C_HC, INK = '#24C4DB', '#ED117B', '#000000'
GCOL = {'SPOB': '#c84fd0', 'Other mmc bacteria': '#f2858a', 'Other bacteria': '#d4d4d4', 'Methanothrix': '#e2a400',
        'Other archaea': '#7d8ce0', 'Methanosarcina': '#4d8f14', 'Hydrogenotrophic methanogens': '#10b3c6'}
GLAB = {'SPOB': 'SPOB', 'Other mmc bacteria': 'Other bacteria\nwith mmc', 'Other bacteria': 'Other bacteria',
        'Methanothrix': r'$\mathbfit{Methanothrix}$', 'Other archaea': 'Other archaea',
        'Methanosarcina': r'$\mathbfit{Methanosarcina}$', 'Hydrogenotrophic methanogens': 'Hydrogenotrophic\nmethanogens'}
MAP = {'SPOB (known lineage, mmc pathway)': 'SPOB', 'other bacteria with mmc pathway': 'Other mmc bacteria',
       'bacteria with complete WL pathway': 'Other bacteria', 'other bacteria': 'Other bacteria',
       'hydrogenotrophic methanogen': 'Hydrogenotrophic methanogens', 'acetoclastic methanogen': 'Methanothrix',
       'mixotrophic methanogen (Methanosarcina)': 'Methanosarcina', 'methylotrophic methanogen': 'Other archaea',
       'other archaea': 'Other archaea'}

# ---------------- grid (inches) ----------------
W, H = 7.2, 5.45
L1, L2, PW = 0.60, 4.55, 2.55          # left frame edges of the two columns and the common frame width
B2, B1, PH = 0.48, 3.08, 2.10          # bottom frame edges of the two rows and the common frame height
LET_DX, LET_DY = 0.32, 0.05            # letter offset from each frame's top-left corner
fig = plt.figure(figsize=(W, H))


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ================= A: mRNA vs DNA ===================================================================
axA = add_ax(L1, B1, PW, PH)
lo, hi = 0.001, 40
xx = np.array([lo, hi])
axA.plot(xx, xx, color='#9a9a9a', lw=0.8, zorder=1)
for f in (10, 0.1):
    axA.plot(xx, xx * f, color='#c4c4c4', lw=0.7, ls='--', zorder=1)
act['g'] = act.guild.map(MAP)
for gname in ['Other bacteria', 'Other archaea', 'Other mmc bacteria', 'Methanothrix', 'Methanosarcina',
              'Hydrogenotrophic methanogens', 'SPOB']:
    s = act[(act.g == gname) & ((act.DNA_H > 0) | (act.RNA_H > 0))]
    axA.scatter(np.maximum(s.DNA_H, 0.0012), np.maximum(s.RNA_H, 0.0012), s=13, color=GCOL[gname], edgecolor='white',
                linewidth=0.3, zorder=2, alpha=0.95)
KEY = {'MAG132': (0.95, 26, 'center'), 'MAG3': (0.32, 16, 'right'),
       'MAG15': (11, 2.9, 'center'), 'MAG14': (22, 1.25, 'center'), 'MAG12': (22, 0.17, 'center'),
       'MAG19': (20, 0.011, 'center')}
for nm, (tx, ty, ha) in KEY.items():
    r = act[act.name == nm].iloc[0]
    p0 = (max(r.DNA_H, 0.0012), max(r.RNA_H, 0.0012)); p1 = (max(r.DNA_HC, 0.0012), max(r.RNA_HC, 0.0012))
    col = GCOL[MAP[r.guild]]
    axA.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6, lw=0.8, color='#3d3d3d', zorder=3,
                                  shrinkA=2, shrinkB=2.5))
    axA.scatter(*p0, s=26, facecolor='white', edgecolor=col, linewidth=1.1, zorder=4)
    axA.scatter(*p1, s=26, facecolor=col, edgecolor=INK, linewidth=0.6, zorder=5)
    axA.annotate(nm, p1, xytext=(tx, ty), fontsize=6.0, ha=ha, va='center', zorder=6,
                 arrowprops=dict(arrowstyle='-', color='#8a8a8a', lw=0.45, shrinkA=1, shrinkB=2.5))
axA.set_xscale('log'); axA.set_yscale('log'); axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
tk = [0.001, 0.01, 0.1, 1, 10]
axA.set_xticks(tk); axA.set_xticklabels(['0.001', '0.01', '0.1', '1', '10'])
axA.set_yticks(tk); axA.set_yticklabels(['0.001', '0.01', '0.1', '1', '10'])
axA.minorticks_off(); axA.tick_params(labelsize=6.8)
axA.set_xlabel('DNA share (%)', labelpad=2); axA.set_ylabel('mRNA share (%)', labelpad=1)
axA.text(0.04, 0.96, 'Active beyond\nabundance', transform=axA.transAxes, ha='left', va='top', fontsize=6.0, color='#8a8a8a')
axA.text(0.96, 0.04, 'Abundant but\nlow activity', transform=axA.transAxes, ha='right', va='bottom', fontsize=6.0, color='#8a8a8a')
axA.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='#555555', mew=1.1, ms=4.6, label='Control'),
                    Line2D([], [], marker='o', ls='', mfc='#9a9a9a', mec=INK, mew=0.6, ms=4.6, label='Hydrochar')],
           loc='upper left', bbox_to_anchor=(0.0, 0.80), fontsize=6.0, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.2)

# ================= B: guild DNA vs mRNA (back-to-back) ==============================================
axB = add_ax(L2, B1, PW, PH)
ORDER = ['SPOB', 'Hydrogenotrophic methanogens', 'Methanosarcina', 'Methanothrix', 'Other archaea',
         'Other mmc bacteria', 'Other bacteria']
yb = np.arange(len(ORDER))[::-1].astype(float)
bh_ = 0.34
for y, k in zip(yb, ORDER):
    r = gs.loc[k]
    for dy, col, dv, rv in [(bh_ / 2 + 0.01, C_CTRL, r.DNA_H, r.RNA_H), (-bh_ / 2 - 0.01, C_HC, r.DNA_HC, r.RNA_HC)]:
        axB.barh(y + dy, -dv, height=bh_, color=col, edgecolor='none', zorder=2)
        axB.barh(y + dy, rv, height=bh_, color=col, edgecolor='none', zorder=2)
    if r.P_RNA < 0.05:
        axB.text(max(r.RNA_H, r.RNA_HC) + 1.2, y, stars(r.P_RNA), ha='left', va='center', fontsize=8)
    axB.add_patch(Rectangle((-49.5, y - 0.19), 2.6, 0.38, facecolor=GCOL[k], edgecolor='none', clip_on=False))
    axB.text(-51.0, y, GLAB[k], ha='right', va='center', fontsize=6.6, clip_on=False, linespacing=1.0)
axB.axvline(0, color=INK, lw=0.8, zorder=3)
axB.set_xlim(-47, 47); axB.set_ylim(-0.6, len(ORDER) - 0.1)
axB.set_xticks([-40, -20, 0, 20, 40]); axB.set_xticklabels(['40', '20', '0', '20', '40']); axB.set_yticks([])
axB.tick_params(labelsize=6.8)
axB.set_xlabel('Share (%)', labelpad=2)
axB.text(0.25, 0.985, 'DNA', transform=axB.transAxes, ha='center', va='top', fontsize=7.5)
axB.text(0.75, 0.985, 'mRNA', transform=axB.transAxes, ha='center', va='top', fontsize=7.5)
axB.legend(handles=[Patch(facecolor=C_CTRL, label='Control'), Patch(facecolor=C_HC, label='Hydrochar')],
           loc='upper right', bbox_to_anchor=(1.0, 0.90), fontsize=6.4, handlelength=1.0, handletextpad=0.3,
           labelspacing=0.2, borderaxespad=0.3)

# ================= C: MAG132 modules (labels inside the frame) ======================================
axC = add_ax(L1, B2, PW, PH)
XL0, XL1, XLAB, XSEP = -7.9, 2.9, -7.7, -2.75
LAB = {'mmc pathway': 'mmc pathway', 'Formate dehydrogenase': 'Formate dehydrogenase',
       '[FeFe] hydrogenase': '[FeFe] hydrogenase', 'Electron transfer flavoprotein': 'Etf',
       'K+ uptake (trk/kdp)': r'K$^+$ uptake (trk)', 'Compatible solutes (proVWX)': 'Compatible solutes',
       'Na+/H+ antiporter (mnh/nhaA)': r'Na$^+$/H$^+$ antiporter', 'NH4+ transporter (amt)': r'NH$_4^+$ transporter',
       'NH4+ assimilation (glnA/gdhA/gltBD)': r'NH$_4^+$ assimilation'}
cats = ['Propionate oxidation', 'Electron disposal', 'Ammonia stress']
yv, spans, y = [], [], 0.0
for c in cats:                                              # header line, then one row per module, then a small gap
    spans.append((c, y)); y -= 0.85
    for _, r in mods[mods.category == c].iterrows():
        yv.append((y, r)); y -= 1
    y += 1 - 0.75
for yy, r in yv:
    col = C_HC if r.log2FC > 0 else C_CTRL
    axC.plot([0, r.log2FC], [yy, yy], color=col, lw=1.6, zorder=2, solid_capstyle='butt')
    ec, lw = (INK, 1.1) if r.q < 0.05 else ('#7a7a7a', 0.5)
    axC.scatter(r.log2FC, yy, s=32, color=col, edgecolor=ec, linewidth=lw, zorder=3)
    st = stars(r.q)
    if st:
        axC.text(r.log2FC + (0.22 if r.log2FC > 0 else -0.22), yy, st, ha='left' if r.log2FC > 0 else 'right',
                 va='center', fontsize=7.5)
    axC.text(XLAB + 0.25, yy, LAB[r.module], ha='left', va='center', fontsize=6.4)
for c, yh in spans:
    axC.text(XLAB, yh, c, ha='left', va='center', fontsize=6.2, color='#555555', style='italic')
axC.axvline(0, color=INK, lw=0.8, zorder=1)
axC.axvline(XSEP, color='#d0d0d0', lw=0.6, zorder=0)
axC.set_xlim(XL0, XL1); axC.set_ylim(y + 0.15, 0.45)
axC.set_xticks([-2, -1, 0, 1, 2]); axC.set_yticks([]); axC.tick_params(labelsize=6.8)
axC.set_xlabel(r'log$_2$ fold change (hydrochar/control)', labelpad=2)
axC.xaxis.set_label_coords((0 - XL0) / (XL1 - XL0), -0.10)

# ================= D: which methanogens carry mcrA, formate, H2 and acetate-cleavage transcripts ==========
axD = add_ax(L2, B2, PW, PH)
src = pd.read_csv(f'{DIR}/fig6_methanogen_sources.csv')
MORD = ['MAG3', 'Other Methanoculleus', 'MAG15', 'MAG14', 'Methanosarcina', 'Methanothrix', 'Others']   # bottom -> top
MCOL = {'MAG3': '#c0601d', 'Other Methanoculleus': '#eab08f', 'MAG15': '#7e5fd0', 'MAG14': '#3b2a8c',
        'Methanosarcina': '#4d8f14', 'Methanothrix': '#e2a400', 'Others': '#bdbdbd'}
MLAB = {'MAG3': r'MAG3 ($\mathbfit{M.\ bourgensis}$)', 'Other Methanoculleus': r'Other $\mathbfit{Methanoculleus}$',
        'MAG15': 'MAG15', 'MAG14': 'MAG14', 'Methanosarcina': r'$\mathbfit{Methanosarcina}$',
        'Methanothrix': r'$\mathbfit{Methanothrix}$', 'Others': 'Others'}
FUNS = [('mcrA', r'$\mathbfit{mcrA}$'), ('Formate', 'Formate'), ('H2', r'H$_2$'), ('Acetate', 'Acetate')]
BW, OFF = 0.36, 0.20
for i, (f, _) in enumerate(FUNS):
    for j, (col, gc) in enumerate([('ctrl', C_CTRL), ('hc', C_HC)]):
        x = i + (OFF if j else -OFF)
        v = src[src.function == f].set_index('category')[col].reindex(MORD).fillna(0)
        bottom = 0
        for c in MORD:
            if v[c] > 0:
                axD.bar(x, v[c], BW, bottom=bottom, color=MCOL[c], edgecolor='white', linewidth=0.5, zorder=2)
            bottom += v[c]
        axD.add_patch(Rectangle((x - BW / 2, 102), BW, 5.0, facecolor=gc, edgecolor='none'))
p14 = src[(src.function == 'H2') & (src.category == 'MAG14')].P.iloc[0]   # the one significant shift (n = 3, Welch t)
if p14 < 0.05:
    v = src[src.function == 'H2'].set_index('category').hc.reindex(MORD).fillna(0)
    axD.text(2 + OFF + BW / 2 + 0.04, v['MAG3':'MAG15'].sum() + v['MAG14'] / 2, stars(p14), ha='left', va='center', fontsize=8)
axD.set_xlim(-0.5, 6.7); axD.set_ylim(0, 110)
axD.set_xticks(range(len(FUNS))); axD.set_xticklabels([t for _, t in FUNS])
axD.tick_params(axis='x', length=0, labelsize=6.8); axD.tick_params(axis='y', labelsize=6.8)
axD.set_yticks([0, 25, 50, 75, 100]); axD.set_ylabel('Share of methanogen transcripts (%)', labelpad=1)
hdl = [Patch(facecolor=MCOL[k], edgecolor='white', label=MLAB[k]) for k in MORD[::-1]]
hdl += [Patch(facecolor=C_CTRL, label='Control'), Patch(facecolor=C_HC, label='Hydrochar')]
axD.legend(handles=hdl, loc='center left', bbox_to_anchor=(0.585, 0.5), fontsize=6.0, handlelength=0.9, handletextpad=0.35,
           labelspacing=0.42, borderaxespad=0)
# guard: the legend must clear the last bar and stay inside the frame
fig.canvas.draw()
_leg = axD.get_legend().get_window_extent()
_bar = axD.transData.transform((len(FUNS) - 1 + OFF + BW / 2, 0))[0]
_frm = axD.get_window_extent()
assert _leg.x0 > _bar + 2 and _leg.x1 < _frm.x1 - 2, f'panel D legend overlaps: {_leg}, bar edge {_bar}, frame {_frm}'


# ================= panel letters: just outside each frame's top-left corner ==========================
def letter(x0, top, s):
    fig.text((x0 - LET_DX) / W, (top + LET_DY) / H, s, fontsize=12, fontweight='bold', ha='left', va='bottom')


letter(L1, B1 + PH, 'A'); letter(L2, B1 + PH, 'B'); letter(L1, B2 + PH, 'C'); letter(L2, B2 + PH, 'D')
fig.text(L1 / W, (B2 + PH + 0.04) / H, 'MAG132 (DTU098)', fontsize=7, ha='left', va='bottom', fontweight='bold')
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=600 if ext == 'png' else None)
print('saved', out)
