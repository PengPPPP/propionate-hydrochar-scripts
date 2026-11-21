r"""Data for Fig. 6D and Figs. S13-S14 / Tables S13-S14: how hydrochar changes the community's response to ammonia (DNA, 18 metagenomes) and how electrons
and acetate move between the active partners at 4 g N/L (metatranscriptome, control E1-3 vs hydrochar H1-3).

1) Response of every MAG (mean abundance >= 0.1 %) to ammonia, with and without hydrochar:
   log2[(mean at 4 g N/L + 0.01) / (mean at 1 g N/L + 0.01)] in the control and in the hydrochar series;
   two-way ANOVA (ammonia 1/2/4 x hydrochar) on log2(abundance + 0.01), interaction P, BH across MAGs.
2) Screen of bacterial MAGs for the Wood-Ljungdahl (WL) pathway used in syntrophic acetate oxidation (methyl branch
   fhs/folD/metF, carbonyl branch acsB/acsCD/cooS, plus acetate activation) and their mRNA share at 4 g N/L.
3) Transcripts of formate, H2 and acetate genes in the active MAGs at 4 g N/L, per million KO-annotated community
   mRNA (reads proportional to RPKM x gene length, KO-annotated protein-coding genes only, as for Fig. 6).
   MAG132 hydrogenases are taken from its KEGG-website annotation (the community KO annotation misses them).
4) Which methanogens carry the community's mcrA, formate, H2 and acetate-cleavage transcripts (Fig. 6D).
Usage: python 15_fig6_flows_data.py <data dir> <omics dir> <out dir>
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

D, O, OUT = sys.argv[1:4]
g = pd.read_csv(f'{O}/mag_guilds.csv', index_col=0)
g['name'] = ['MAG' + s[3:] for s in g.thesis_name]
rel = pd.read_csv(f'{O}/mag_abundance.csv', index_col=0)
E, HH = ['E1', 'E2', 'E3'], ['H1', 'H2', 'H3']               # E = 4 g N/L control, H = 4 g N/L + hydrochar
S = E + HH
PS = 0.01                                                     # pseudocount, % relative abundance


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return stats.ttest_ind(b, a, equal_var=False).pvalue if np.ptp(np.r_[a, b]) > 0 else np.nan


def interaction_p(y, a, b):
    df = pd.DataFrame({'y': y, 'a': a, 'b': b}); gm = df.y.mean()
    A = df.groupby('a').y.mean(); B = df.groupby('b').y.mean(); C = df.groupby(['a', 'b']).y.mean(); n = df.groupby(['a', 'b']).size()
    ssab = sum(n[i, j] * (C[i, j] - A[i] - B[j] + gm) ** 2 for i, j in C.index)
    sse = sum((r.y - C[r.a, r.b]) ** 2 for r in df.itertuples()); dfe = len(df) - len(C); dfi = (len(A) - 1) * (len(B) - 1)
    return stats.f.sf((ssab / dfi) / (sse / dfe), dfi, dfe) if sse > 0 else np.nan


# ---------- 1) response to ammonia with and without hydrochar (DNA) ------------------------------
GRP = {'L': (1, 0), 'LC': (1, 1), 'M': (2, 0), 'MC': (2, 1), 'H': (4, 0), 'HC': (4, 1)}
cols = {k: [f'{k}_{i}' for i in (1, 2, 3)] for k in GRP}
rows = []
for m in g.index.intersection(rel.index):
    v = rel.loc[m]
    if v[sum(cols.values(), [])].mean() < 0.1:
        continue
    y, a, b = [], [], []
    for k, (am, hc) in GRP.items():
        for c in cols[k]:
            y.append(np.log2(v[c] + PS)); a.append(am); b.append(hc)
    mean = {k: v[cols[k]].mean() for k in GRP}
    rows.append(dict(Genome=m, name=g.name[m], guild=g.guild[m], family=g.family[m], genus=g.genus[m],
                     mean_all=v[sum(cols.values(), [])].mean(), **{f'mean_{k}': mean[k] for k in GRP},
                     resp_ctrl=np.log2((mean['H'] + PS) / (mean['L'] + PS)),
                     resp_hc=np.log2((mean['HC'] + PS) / (mean['LC'] + PS)), P_int=interaction_p(y, a, b)))
R = pd.DataFrame(rows)
R['q_int'] = bh(R.P_int.fillna(1))
rho, prho = stats.spearmanr(R.resp_ctrl, R.resp_hc)
R['spearman_rho'] = rho; R['spearman_P'] = prho
R.to_csv(f'{OUT}/si_ammonia_response.csv', index=False, encoding='utf-8-sig')
print(f'{len(R)} MAGs, Spearman rho = {rho:.2f} (P = {prho:.1e}); same sign: {(np.sign(R.resp_ctrl) == np.sign(R.resp_hc)).mean():.0%}')

# ---------- gene-level transcripts of KO-annotated genes ------------------------------------------
rp = pd.read_csv(f'{D}/metatranscriptome/RPKM.txt', sep='\t', usecols=['Start', 'End', 'Feature'] + [f'Mean {s}_sorted.bam' for s in S])
rp.columns = ['Start', 'End', 'Feature'] + S
rp['len'] = (rp.End - rp.Start).abs() + 1
rp['bin'] = rp.Feature.str.extract(r'_((?:LH|H|L|M)_\d+)_\d+_\d+$')[0]
GENE = re.compile(r'scaffold_\d+_[A-Z]+_\d+_\d+_\d+')
ko_of, kos_of_mag = {}, {}
for gid, s in zip(g.index, g.short):
    f = f'{D}/metagenome/ko/{s}.txt'
    kos_of_mag[gid] = set()
    if os.path.exists(f):
        k = pd.read_csv(f, sep='\t', usecols=['Ko', 'Feature']).dropna()
        kos_of_mag[gid] = set(k.Ko)
        for ko, feat in zip(k.Ko, k.Feature):
            for gene in GENE.findall(feat):
                ko_of[gene] = ko
cod = rp[rp.Feature.isin(ko_of)].copy()
cod['KO'] = cod.Feature.map(ko_of)
s2g = dict(zip(g.short, g.index)); cod['gid'] = cod.bin.map(s2g)
reads = cod[S].mul(cod.len, axis=0)
TOT = reads.sum()                                             # all KO-annotated community mRNA, per sample
mag_share = reads.groupby(cod.gid).sum() / TOT * 100

# ---------- 2) WL screen of bacterial MAGs ---------------------------------------------------------
WL = {'fhs': ['K01938'], 'folD': ['K01491', 'K01500'], 'metF': ['K00297', 'K25007'],
      'acsB': ['K14138'], 'acsCD': ['K00197', 'K00194'], 'cooS': ['K00198']}
ACT = {'ackA-pta': [['K00925'], ['K00625', 'K13788']], 'acs': [['K01895']], 'acd': [['K01905', 'K22224']]}
wl = []
for gid in g.index:
    if g.mcrA[gid] > 0 or g.domain[gid] != 'Bacteria':
        continue
    have = {st: int(bool(kos_of_mag[gid] & set(v))) for st, v in WL.items()}
    act = int(any(all(kos_of_mag[gid] & set(x) for x in v) for v in ACT.values()))
    if sum(have.values()) < 5 or not act:
        continue
    wl.append(dict(Genome=gid, name=g.name[gid], family=g.family[gid], genus=g.genus[gid], guild=g.guild[gid],
                   completeness=g.completeness[gid], **have, acetate_activation=act, WL_steps=sum(have.values()),
                   **{f'DNA_{k}': g[f'mean_{k}'][gid] for k in ['L', 'LC', 'M', 'MC', 'H', 'HC']},
                   mRNA_H=mag_share.reindex([gid])[E].mean(1).fillna(0).iloc[0],
                   mRNA_HC=mag_share.reindex([gid])[HH].mean(1).fillna(0).iloc[0]))
wl = pd.DataFrame(wl).sort_values('mRNA_HC', ascending=False)
wl.to_csv(f'{OUT}/si_wl_bacteria.csv', index=False, encoding='utf-8-sig')
# who transcribes the carbonyl branch (CODH/ACS) among bacteria at all
carb = cod[cod.KO.isin(['K14138', 'K00197', 'K00194', 'K00198']) & cod.gid.map(lambda x: g.domain.get(x) == 'Bacteria')]
print(f'WL screen: {len(wl)} bacterial MAGs with >=5/6 WL steps + acetate activation; max mRNA share '
      f'{wl[["mRNA_H", "mRNA_HC"]].max().max():.2f} %; bacterial CODH/ACS transcripts = '
      f'{(reads.loc[carb.index].sum() / TOT * 1e6).mean():.0f} per million community mRNA')

# ---------- 3) formate / H2 / acetate transcripts of the active partners ------------------------
GENES = {  # column: KOs (checked against KEGG: rest.kegg.jp/list)
    'fdh': ['K00123', 'K00124', 'K00126', 'K00127', 'K22015', 'K05299', 'K15022', 'K22516', 'K00125'],
    'fdhC': ['K21993'],                                                       # formate transporter
    'Hydrogenase': ['K00440', 'K00441', 'K00443', 'K14126', 'K14127', 'K14128', 'K13942']  # frhABG, mvhADG, hmd
                   + [f'K{k:05d}' for k in range(14086, 14092)]                # echA-F
                   + [f'K{k:05d}' for k in range(14092, 14126)],               # ehaA-R, ehbA-Q
    'ackA-pta': ['K00925', 'K00625', 'K13788'],
    'acs': ['K01895'],
    'CODH/ACS': ['K00192', 'K00195', 'K00193', 'K00197', 'K00194'],          # cdhA-E (acetate cleavage in acetoclasts)
    'mcrA': ['K00399'],
}
ko2col = {k: c for c, v in GENES.items() for k in v}
ROWS = ['MAG132', 'MAG3', 'MAG4', 'MAG15', 'MAG14', 'MAG10', 'MAG11', 'MAG12']
n2g = dict(zip(g.name, g.index))
sub = cod[cod.KO.isin(ko2col) & cod.gid.isin([n2g[n] for n in ROWS])]
val = (reads.loc[sub.index].groupby([sub.gid, sub.KO.map(ko2col)]).sum() / TOT * 1e6)
# MAG132 hydrogenases from the KEGG-website annotation ([FeFe] and [NiFe] catalytic subunits and HndABCD)
ann = pd.read_csv(f'{D}/metagenome/MAG132_kegg.txt', sep='\t', header=None, usecols=[0, 1]).dropna()
ann.columns = ['gene', 'KO']
HYD132 = ['K18330', 'K18331', 'K18332', 'K17997', 'K17998', 'K17999', 'K00532', 'K00533', 'K00534', 'K06441', 'K06281', 'K06282']
hg = rp[rp.Feature.isin(ann[ann.KO.isin(HYD132)].gene)]
val.loc[(n2g['MAG132'], 'Hydrogenase'), :] = (hg[S].mul(hg.len, axis=0).sum() / TOT * 1e6).values
dot = []
for n in ROWS:
    for c in GENES:
        v = val.loc[(n2g[n], c)] if (n2g[n], c) in val.index else pd.Series(0.0, index=S)
        dot.append(dict(name=n, gene=c, ctrl=v[E].mean(), hc=v[HH].mean(), mean=v.mean(),
                        log2FC=np.log2((v[HH].mean() + 1) / (v[E].mean() + 1)), P=welch(v[E], v[HH]), **v.to_dict()))
pd.DataFrame(dot).to_csv(f'{OUT}/si_key_gene_transcripts.csv', index=False, encoding='utf-8-sig')
# e-pilus gene pilA (marker discussed for direct interspecies electron transfer): total and main sources
pm = cod.KO == 'K02650'
pil = reads[pm].groupby(cod.gid[pm]).sum() / TOT * 1e6
src = (pil[E].mean(1) + pil[HH].mean(1)).sort_values(ascending=False)
pd.DataFrame({'name': [g.name[i] for i in src.index], 'guild': [g.guild[i] for i in src.index], 'family': [g.family[i] for i in src.index],
              'ctrl': pil.loc[src.index, E].mean(1).values, 'hc': pil.loc[src.index, HH].mean(1).values}).to_csv(
    f'{OUT}/si_pilA_sources.csv', index=False, encoding='utf-8-sig')
print(f'pilA: {pil[E].sum().mean():.0f} / {pil[HH].sum().mean():.0f} per million community mRNA (control / hydrochar); '
      f'from SPOB or methanogens: {pil.loc[[i for i in pil.index if g.guild[i].startswith("SPOB") or g.mcrA[i] > 0]].sum().mean():.0f}')

# ---------- 4) which methanogens carry formate, H2 and acetate-cleavage transcripts ------------
FUN = {'mcrA': GENES['mcrA'], 'Formate': GENES['fdh'] + GENES['fdhC'], 'H2': GENES['Hydrogenase'], 'Acetate': GENES['ackA-pta'] + GENES['CODH/ACS']}
CAT = {'MAG3': 'MAG3', 'MAG15': 'MAG15', 'MAG14': 'MAG14', 'MAG12': 'Methanothrix'}


def cat(gid):
    n = g.name[gid]
    if n in CAT: return CAT[n]
    if g.genus[gid] == 'Methanoculleus': return 'Other Methanoculleus'
    if g.genus[gid] == 'Methanosarcina' or g.family[gid] == 'Methanosarcinaceae': return 'Methanosarcina'
    return 'Others'


meth = set(g.index[g.mcrA > 0])
comp = []
for fun, kos in FUN.items():
    m = cod.KO.isin(kos) & cod.gid.isin(meth)
    x = reads[m].groupby(cod.gid[m].map(cat)).sum()
    pct = x / x.sum() * 100
    for c in pct.index:
        comp.append(dict(function=fun, category=c, ctrl=pct.loc[c, E].mean(), hc=pct.loc[c, HH].mean(),
                         P=welch(pct.loc[c, E], pct.loc[c, HH]), **pct.loc[c].to_dict()))
comp = pd.DataFrame(comp)
comp.to_csv(f'{OUT}/fig6_methanogen_sources.csv', index=False, encoding='utf-8-sig')
print(comp[['function', 'category', 'ctrl', 'hc', 'P']].round(3).to_string(index=False))
print('saved fig6_methanogen_sources.csv and si_*.csv to', OUT)
