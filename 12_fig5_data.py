r"""Section 3.5 data: methanogen MAGs - group means, row clustering, hydrogenotroph composition, trophic-group and
lineage statistics, single-MAG ammonia trends, response-scatter table.
Usage: python 12_fig5_data.py <data dir> <omics dir> <out dir>
Needs fig4_hydrochar_effect_all.csv in <out dir> (written by 10_fig4_data.py).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list

D, O, OUT = sys.argv[1:4]
rel = pd.read_csv(f'{O}/mag_abundance.csv', index_col=0)
g = pd.read_csv(f'{O}/mag_guilds.csv', index_col=0)
eff = pd.read_csv(f'{OUT}/fig4_hydrochar_effect_all.csv')
gt = pd.read_excel(f'{D}/metagenome/gtdbtk_annotation.xlsx', sheet_name='Sheet1').set_index('user_genome')
GROUPS = ['L', 'LC', 'M', 'MC', 'H', 'HC']
LV = ['L', 'M', 'H']
def cols(x): return [f'{x}_{r}' for r in (1, 2, 3)]


TROPH = {'hydrogenotrophic methanogen': 'Hydrogenotrophic', 'acetoclastic methanogen': 'Acetoclastic',
         'mixotrophic methanogen (Methanosarcina)': 'Methanosarcinaceae', 'methylotrophic methanogen': 'Methylotrophic'}
mg = g[g.guild.isin(TROPH)].copy()
mg['name'] = ['MAG' + s[3:] for s in mg.thesis_name]
mg['trophic'] = mg.guild.map(TROPH)
mg['red'] = gt.reindex(mg.index).red_value
mg['gtdb_ani'] = gt.reindex(mg.index).fastani_ani
mg['novel_genus'] = mg.genus.isna()
gm = pd.DataFrame({k: rel.loc[mg.index, cols(k)].mean(1) for k in GROUPS})
show = mg[gm.max(1) >= 0.2].copy()                         # methanogen MAGs with a group mean >= 0.2 %
gms = gm.loc[show.index]

# ---------- row clustering on z-scored log10 group means ----------------------------------------
lg = np.log10(gms + 0.01)
z = lg.sub(lg.mean(1), axis=0).div(lg.std(1, ddof=0), axis=0)
Z = linkage(z.values, method='average', metric='euclidean')
order = leaves_list(Z)
show = show.iloc[order]
out = show[['name', 'trophic', 'family', 'genus', 'species', 'completeness', 'contamination', 'red', 'gtdb_ani',
            'novel_genus']].join(gms.add_prefix('mean_')).join(z.add_prefix('z_'))
out.to_csv(f'{OUT}/fig5_methanogen_means.csv', encoding='utf-8-sig')
rel.loc[show.index].rename(index=show.name).to_csv(f'{OUT}/fig5_methanogen_samples.csv', encoding='utf-8-sig')
np.save(f'{OUT}/fig5_linkage.npy', Z)                   # rows of Z refer to the unclustered order below
pd.Series(gms.index, name='Genome').to_csv(f'{OUT}/fig5_cluster_order.csv', index=False, encoding='utf-8-sig')


# ---------- hydrogenotroph composition per sample ------------------------------------------------
def hcat(r):
    if r.name_ == 'MAG14': return 'MAG14'
    if r.name_ == 'MAG15': return 'MAG15'
    if r.family == 'Methanobacteriaceae': return 'Other Methanobacteriaceae'
    if r.genus == 'Methanoculleus': return 'Methanoculleus'
    return 'Methanoregulaceae'


hy = mg[mg.trophic == 'Hydrogenotrophic'].copy(); hy['name_'] = hy.name
hy['cat'] = hy.apply(hcat, axis=1)
HC_ORDER = ['MAG14', 'MAG15', 'Other Methanobacteriaceae', 'Methanoculleus', 'Methanoregulaceae']
comp = rel.loc[hy.index].groupby(hy.cat).sum().reindex(HC_ORDER)
comp.loc['Total hydrogenotrophs'] = comp.sum()
comp.to_csv(f'{OUT}/fig5_hydrogenotrophs.csv', encoding='utf-8-sig')
print(hy[['name', 'family', 'genus', 'cat']].sort_values('cat').to_string())

# ---------- statistics ---------------------------------------------------------------------------
amm = np.array([{'L': 1, 'M': 2, 'H': 4}[c[0]] for c in rel.columns])
hc = np.array([1 if c.split('_')[0].endswith('C') else 0 for c in rel.columns])


def anova2(y):
    df = pd.DataFrame({'y': np.asarray(y, float), 'a': amm, 'h': hc}); gm_ = df.y.mean()
    A = df.groupby('a').y.mean(); B = df.groupby('h').y.mean(); AB = df.groupby(['a', 'h']).y.mean()
    ssa = 6 * ((A - gm_) ** 2).sum(); ssb = 9 * ((B - gm_) ** 2).sum()
    ssab = 3 * sum((AB[(i, j)] - A[i] - B[j] + gm_) ** 2 for i in A.index for j in B.index)
    sse = sum(((df[(df.a == i) & (df.h == j)].y - AB[(i, j)]) ** 2).sum() for i in A.index for j in B.index)
    return {nm: stats.f.sf((ss / d_) / (sse / 12), d_, 12) for nm, ss, d_ in
            [('ammonia', ssa, 2), ('hydrochar', ssb, 1), ('interaction', ssab, 2)]}


tro = rel.loc[mg.index].groupby(mg.trophic).sum()
fam = rel.loc[mg.index].groupby(mg.family).sum()
share = comp.loc[HC_ORDER] / comp.loc['Total hydrogenotrophs'] * 100
rows = []
for what, tab in [('trophic', tro), ('family', fam), ('hydrogenotroph_category', comp), ('share_of_hydrogenotrophs', share)]:
    for name, y in tab.iterrows():
        r = dict(level=what, item=name, **{f'P_{k}': v for k, v in anova2(y).items()})
        for lv in LV:
            a, b = y[cols(lv)], y[cols(lv + 'C')]
            r[f'{lv}_ctrl'], r[f'{lv}_char'] = a.mean(), b.mean()
            r[f'{lv}_P'] = stats.ttest_ind(b, a, equal_var=False).pvalue if np.ptp(np.r_[a, b]) > 0 else np.nan
        rows.append(r)
st = pd.DataFrame(rows); st.to_csv(f'{OUT}/fig5_stats.csv', index=False, encoding='utf-8-sig')

tr = []
for mag in show.index:
    for trt, suf in [('control', ''), ('hydrochar', 'C')]:
        ys = [rel.loc[mag, cols(lv + suf)].values for lv in LV]
        tr.append(dict(name=show.loc[mag, 'name'], treatment=trt, **{lv: y.mean() for lv, y in zip(LV, ys)},
                       P_oneway=stats.f_oneway(*ys).pvalue if np.ptp(np.concatenate(ys)) > 0 else np.nan))
tr = pd.DataFrame(tr); tr.to_csv(f'{OUT}/fig5_mag_vs_ammonia.csv', index=False, encoding='utf-8-sig')

sc = eff[eff.MAG.isin(show.index)].copy()
sc['trophic'] = sc.MAG.map(show.trophic)
sc.to_csv(f'{OUT}/fig5_response.csv', index=False, encoding='utf-8-sig')

# ---------- print ----------------------------------------------------------------------------------
pd.set_option('display.width', 260)
print(out[['name', 'trophic', 'genus', 'species', 'red', 'gtdb_ani'] + [f'mean_{k}' for k in GROUPS]].round(3).to_string())
print('\n== statistics'); print(st.round(3).to_string(index=False))
print('\n== single MAG trend'); print(tr.round(3).to_string(index=False))
print('\n== methanogen hydrochar effects (P<0.05)')
print(sc[sc.P < 0.05][['level', 'name', 'genus', 'ctrl', 'char', 'log2FC', 'P', 'q']].round(4).to_string(index=False))
