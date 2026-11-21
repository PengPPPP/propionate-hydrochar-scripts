r"""Section 3.4 data: the 12 SPOB MAGs - per-sample abundance, hydrochar effect per ammonia level,
mmc-pathway / electron-transfer gene matrix, lineage composition and lineage statistics.
Usage: python 10_fig4_data.py <data dir> <omics dir> <out dir>

Hydrochar effect: Welch t test within each ammonia level for every MAG with mean abundance >= 0.1 %
(average of control and hydrochar means), BH-FDR across those MAGs (same rule as the DNA statistics
in the omics dir). log2FC uses a pseudo-count of 0.001 %.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

D, O, OUT = sys.argv[1:4]
rel = pd.read_csv(f'{O}/mag_abundance.csv', index_col=0)
g = pd.read_csv(f'{O}/mag_guilds.csv', index_col=0)
LV = [('L', 1), ('M', 2), ('H', 4)]
def cols(x): return [f'{x}_{r}' for r in (1, 2, 3)]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


# ---------- hydrochar effect for all MAGs (table S9) -------------------------------------------
rows = []
for lv, _ in LV:
    a, b = rel[cols(lv)], rel[cols(lv + 'C')]
    keep = (a.mean(1) + b.mean(1)) / 2 >= 0.1
    t = stats.ttest_ind(b[keep], a[keep], axis=1, equal_var=False)
    for mag, p_, q_ in zip(rel.index[keep], t.pvalue, bh(t.pvalue)):
        rows.append(dict(level=lv, MAG=mag, name='MAG' + g.loc[mag, 'thesis_name'][3:], guild=g.loc[mag, 'guild'],
                         family=g.loc[mag, 'family'], genus=g.loc[mag, 'genus'],
                         ctrl=a.loc[mag].mean(), char=b.loc[mag].mean(),
                         log2FC=np.log2((b.loc[mag].mean() + 1e-3) / (a.loc[mag].mean() + 1e-3)), P=p_, q=q_))
eff = pd.DataFrame(rows)
eff.to_csv(f'{OUT}/fig4_hydrochar_effect_all.csv', index=False, encoding='utf-8-sig')

# ---------- SPOB set, lineage, row order -------------------------------------------------------
sp = g[g.guild.str.startswith('SPOB')].copy()
sp['name'] = ['MAG' + s[3:] for s in sp.thesis_name]


def lineage(r):
    if r.family == 'Syntrophobacteraceae': return 'Syntrophobacteraceae'
    if r.genus == 'Syntrophopropionicum': return 'Syntrophopropionicum'
    if r.genus == 'DTU098': return 'DTU098'
    return 'Other Pelotomaculaceae'


LIN = ['Syntrophobacteraceae', 'Syntrophopropionicum', 'Other Pelotomaculaceae', 'DTU098']
sp['lineage'] = sp.apply(lineage, axis=1)
ab = rel.loc[sp.index]
amm = np.array([{'L': 1, 'M': 2, 'H': 4}[c[0]] for c in rel.columns])
sp['weighted_ammonia'] = (ab.values * amm).sum(1) / ab.values.sum(1)          # abundance-weighted ammonia level
sp['lin_rank'] = sp.lineage.map({k: i for i, k in enumerate(LIN)})
sp = sp.sort_values(['lin_rank', 'weighted_ammonia'])

# ---------- gene matrix (KO presence in each MAG) ----------------------------------------------
ko = {}
for mag, s in zip(sp.index, sp.short):
    f = f'{D}/metagenome/ko/{s}.txt'
    ko[mag] = set(pd.read_csv(f, sep='\t', usecols=['Ko']).Ko.dropna().astype(str)) if os.path.exists(f) else set()
STEPS = {'pct/prpE': 'mmc_S1_activation', 'mmdA/pcc': 'mmc_S2_carboxyltransfer', 'epi': 'mmc_S3_epimerase',
         'mcm': 'mmc_S4_mutase', 'sucCD': 'mmc_S5_succinylCoA', 'sdh/frd': 'mmc_S6_SDH_FRD', 'fum': 'mmc_S7_fumarase',
         'mdh': 'mmc_S8_malate', 'oad/pyc': 'mmc_S9_OAA_pyruvate', 'por': 'mmc_S10_pyruvate_acCoA',
         'pta-ack/acd': 'mmc_S11_acetate'}
# catalytic subunits only (KO definitions checked against KEGG, 2026-11-13)
ET = {'fdh': ['K00123', 'K05299', 'K22015', 'K00122', 'K22516'],                 # formate dehydrogenase, catalytic
      '[FeFe] hyd': ['K00532', 'K00533', 'K17997', 'K18332'],                      # hydA / HndD
      '[NiFe] hyd': ['K06281', 'K14090', 'K00436'],                                # hyaB, echE, hoxH
      'rnf': ['K03614', 'K03615']}                                                 # rnfD / rnfC
gm = pd.DataFrame(index=sp.index)
for lab, c in STEPS.items(): gm[lab] = sp[c].astype(bool)
for lab, kos in ET.items(): gm[lab] = [bool(ko[m] & set(kos)) for m in sp.index]
gm['hyd'] = gm['[FeFe] hyd'] | gm['[NiFe] hyd']                                # any hydrogenase catalytic subunit
gm.insert(0, 'name', sp.name)
gm.to_csv(f'{OUT}/fig4_gene_matrix.csv', encoding='utf-8-sig')

# ---------- per-sample abundance + annotation --------------------------------------------------
out = sp[['name', 'lineage', 'family', 'genus', 'species', 'completeness', 'contamination', 'mmc_complete']].join(ab)
out.to_csv(f'{OUT}/fig4_spob_abundance.csv', encoding='utf-8-sig')
e4 = eff[eff.MAG.isin(sp.index)].copy()
e4.to_csv(f'{OUT}/fig4_hydrochar_effect_spob.csv', index=False, encoding='utf-8-sig')

# ---------- lineage composition and statistics -------------------------------------------------
lin = ab.groupby(sp.lineage).sum().reindex(LIN)
lin.loc['Total SPOB'] = lin.sum()
lin.to_csv(f'{OUT}/fig4_lineages.csv', encoding='utf-8-sig')
share = lin.loc[LIN] / lin.loc['Total SPOB'] * 100
hc = np.array([1 if c.split('_')[0].endswith('C') else 0 for c in rel.columns])


def anova2(y):
    df = pd.DataFrame({'y': np.asarray(y, float), 'a': amm, 'h': hc}); gm_ = df.y.mean()
    A = df.groupby('a').y.mean(); B = df.groupby('h').y.mean(); AB = df.groupby(['a', 'h']).y.mean()
    ssa = 6 * ((A - gm_) ** 2).sum(); ssb = 9 * ((B - gm_) ** 2).sum()
    ssab = 3 * sum((AB[(i, j)] - A[i] - B[j] + gm_) ** 2 for i in A.index for j in B.index)
    sse = sum(((df[(df.a == i) & (df.h == j)].y - AB[(i, j)]) ** 2).sum() for i in A.index for j in B.index)
    return {nm: stats.f.sf((ss / d_) / (sse / 12), d_, 12) for nm, ss, d_ in
            [('ammonia', ssa, 2), ('hydrochar', ssb, 1), ('interaction', ssab, 2)]}


rows = []
for what, tab in [('abundance', lin), ('share_of_SPOB', share)]:
    for name, y in tab.iterrows():
        r = dict(metric=what, lineage=name, **{f'P_{k}': v for k, v in anova2(y).items()})
        for lv, _ in LV:
            a, b = y[cols(lv)], y[cols(lv + 'C')]
            r[f'{lv}_ctrl'], r[f'{lv}_char'] = a.mean(), b.mean()
            r[f'{lv}_P'] = stats.ttest_ind(b, a, equal_var=False).pvalue
        rows.append(r)
ls_ = pd.DataFrame(rows); ls_.to_csv(f'{OUT}/fig4_lineage_stats.csv', index=False, encoding='utf-8-sig')

# control-only trend with ammonia (one-way ANOVA across 1/2/4 g N/L) for single MAGs
tr = []
for mag in sp.index:
    for trt, suf in [('control', ''), ('hydrochar', 'C')]:
        ys = [rel.loc[mag, cols(lv + suf)].values for lv, _ in LV]
        tr.append(dict(name=sp.loc[mag, 'name'], treatment=trt, **{f'{lv}': y.mean() for (lv, _), y in zip(LV, ys)},
                       P_oneway=stats.f_oneway(*ys).pvalue if np.ptp(np.concatenate(ys)) > 0 else np.nan))
tr = pd.DataFrame(tr); tr.to_csv(f'{OUT}/fig4_mag_vs_ammonia.csv', index=False, encoding='utf-8-sig')

# ---------- print summary ----------------------------------------------------------------------
pd.set_option('display.width', 250)
print(sp[['name', 'lineage', 'species', 'completeness', 'contamination', 'mmc_complete', 'weighted_ammonia']].round(2).to_string())
print('\n== gene matrix'); print(gm.drop(columns='name').astype(int).set_index(gm.name).to_string())
print('\n== hydrochar effect (SPOB)')
print(e4[['level', 'name', 'genus', 'ctrl', 'char', 'log2FC', 'P', 'q']].round(4).to_string(index=False))
print('\n== lineage statistics'); print(ls_.round(3).to_string(index=False))
print('\n== single-MAG ammonia trend'); print(tr.round(3).to_string(index=False))
other = g[(g.phylum == 'Cloacimonadota') | g.family.isin(['Smithellaceae', 'Syntrophaceae'])]
print('\n== Cloacimonadota / Smithellaceae max group mean (%):',
      other[[f'mean_{x}' for x in ['L', 'LC', 'M', 'MC', 'H', 'HC']]].max().max().round(3),
      '| summed max:', other[[f'mean_{x}' for x in ['L', 'LC', 'M', 'MC', 'H', 'HC']]].sum().max().round(3))
