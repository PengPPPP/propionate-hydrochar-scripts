r"""Section 3.6 data (metatranscriptome, 4 g N/L control 'H' = E1-3 vs hydrochar 'HC' = H1-3, n = 3).

Only protein-coding genes with a KO annotation are used, so that non-coding RNA that ends up inside MAG contigs
(residual rRNA, tmRNA next to smpB, etc.) does not inflate activity. Example: in MAG132 a 24-aa 'ORF' in the tmRNA
(ssrA) locus carried ~48 % of all reads, and in MAG126 one small contig carried >80 %.
1) mRNA share of every MAG: sum over its KO-annotated genes of RPKM x gene length (proportional to reads), as % of the
   same sum over all MAGs; DNA share; mRNA/DNA ratio; Welch t + BH (MAGs with mean mRNA share >= 0.1 %);
2) mRNA and DNA share of each guild (guilds as in Fig. 3);
3) contribution of each MAG to community mcrA mRNA (RPKM x length of K00399 genes);
4) within-genome module expression of MAG132: transcripts per million KO-annotated transcripts of MAG132 (RPKM-based),
   genes annotated with the KEGG-website annotation of MAG132 (covers the hydrogenases); Welch t + BH across modules.
Usage: python 14_fig6_data.py <data dir> <omics dir> <out dir>
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
short2g = dict(zip(g.short, g.index))
rel = pd.read_csv(f'{O}/mag_abundance.csv', index_col=0)
E, HH = ['E1', 'E2', 'E3'], ['H1', 'H2', 'H3']              # E = 4 g N/L control (H), H = 4 g N/L + hydrochar (HC)
S = E + HH


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return stats.ttest_ind(b, a, equal_var=False).pvalue if np.ptp(np.r_[a, b]) > 0 else np.nan


# ---------- gene-level RPKM, gene length, KO-annotated gene set ----------------------------------
rp = pd.read_csv(f'{D}/metatranscriptome/RPKM.txt', sep='\t', usecols=['Start', 'End', 'Feature'] + [f'Mean {s}_sorted.bam' for s in S])
rp.columns = ['Start', 'End', 'Feature'] + S
rp['len'] = (rp.End - rp.Start).abs() + 1
rp['bin'] = rp.Feature.str.extract(r'_((?:LH|H|L|M)_\d+)_\d+_\d+$')[0]
GENE = re.compile(r'scaffold_\d+_[A-Z]+_\d+_\d+_\d+')
ko_of = {}                                                     # gene -> KO (DNA-level KO tables of every MAG)
for s in g.short:
    f = f'{D}/metagenome/ko/{s}.txt'
    if os.path.exists(f):
        k = pd.read_csv(f, sep='\t', usecols=['Ko', 'Feature']).dropna()
        for ko, feat in zip(k.Ko, k.Feature):
            for gene in GENE.findall(feat):
                ko_of[gene] = ko
cod = rp[rp.Feature.isin(ko_of)].copy()
cod['KO'] = cod.Feature.map(ko_of)
reads = cod[S].mul(cod.len, axis=0)                            # proportional to read counts within a sample

# ---------- 1) mRNA share per MAG ---------------------------------------------------------------
share = reads.groupby(cod.bin).sum()
share = share / share.sum() * 100
share.index = [short2g[i] for i in share.index]
share = share.reindex(g.index).fillna(0)
act = pd.DataFrame({'name': g.name, 'guild': g.guild, 'family': g.family, 'genus': g.genus,
                    'DNA_H': rel[['H_1', 'H_2', 'H_3']].mean(1), 'DNA_HC': rel[['HC_1', 'HC_2', 'HC_3']].mean(1),
                    'RNA_H': share[E].mean(1), 'RNA_HC': share[HH].mean(1)})
act['P'] = [welch(share.loc[m, E], share.loc[m, HH]) for m in g.index]
keep = (act.RNA_H + act.RNA_HC) / 2 >= 0.1
act['q'] = np.nan; act.loc[keep, 'q'] = bh(act.loc[keep, 'P'].fillna(1))
act['RNA_DNA_H'] = act.RNA_H / act.DNA_H.where(act.DNA_H > 0)
act['RNA_DNA_HC'] = act.RNA_HC / act.DNA_HC.where(act.DNA_HC > 0)
act.join(share).to_csv(f'{OUT}/fig6_mrna_share.csv', encoding='utf-8-sig')

# ---------- 2) guild shares (mRNA and DNA) --------------------------------------------------------
MAP = {'SPOB (known lineage, mmc pathway)': 'SPOB', 'other bacteria with mmc pathway': 'Other mmc bacteria',
       'bacteria with complete WL pathway': 'Other bacteria', 'other bacteria': 'Other bacteria',
       'hydrogenotrophic methanogen': 'Hydrogenotrophic methanogens', 'acetoclastic methanogen': 'Methanothrix',
       'mixotrophic methanogen (Methanosarcina)': 'Methanosarcina', 'methylotrophic methanogen': 'Other archaea',
       'other archaea': 'Other archaea'}
gu = g.guild.map(MAP)
rna = share.groupby(gu).sum()
dna = rel[['H_1', 'H_2', 'H_3', 'HC_1', 'HC_2', 'HC_3']].groupby(gu).sum()
gs = pd.DataFrame({'RNA_H': rna[E].mean(1), 'RNA_HC': rna[HH].mean(1), 'DNA_H': dna[['H_1', 'H_2', 'H_3']].mean(1),
                   'DNA_HC': dna[['HC_1', 'HC_2', 'HC_3']].mean(1)})
gs['P_RNA'] = [welch(rna.loc[k, E], rna.loc[k, HH]) for k in gs.index]
gs.join(rna).to_csv(f'{OUT}/fig6_guild_mrna.csv', encoding='utf-8-sig')

# ---------- 3) mcrA mRNA by MAG -------------------------------------------------------------------
m_ = cod[cod.KO == 'K00399']
mc = reads.loc[m_.index].groupby(m_.bin).sum()
mc.index = [short2g[i] for i in mc.index]
pct = mc.div(mc.sum(), axis=1) * 100
mc = pd.DataFrame({'name': g.name.reindex(mc.index), 'family': g.family.reindex(mc.index), 'genus': g.genus.reindex(mc.index),
                   'guild': g.guild.reindex(mc.index)}).join(pct.add_suffix('_pct'))
mc['H_pct'] = pct[E].mean(1); mc['HC_pct'] = pct[HH].mean(1)
mc['P'] = [welch(pct.loc[m, E], pct.loc[m, HH]) for m in mc.index]
mc.to_csv(f'{OUT}/fig6_mcrA_sources.csv', encoding='utf-8-sig')

# ---------- 4) MAG132 within-genome module expression ---------------------------------------------
MODS = {  # module: KOs (definitions checked against KEGG)
    'mmc pathway': ['K17489', 'K03416', 'K17490', 'K01965', 'K01966', 'K11263', 'K18472', 'K01604', 'K05606', 'K01847',
                    'K01848', 'K01849', 'K01902', 'K01903', 'K00239', 'K00240', 'K00241', 'K00244', 'K00245', 'K01676',
                    'K01677', 'K01678', 'K01679', 'K00024', 'K00116', 'K01571', 'K01572', 'K01573', 'K01958', 'K01959',
                    'K01960', 'K00169', 'K00170', 'K00171', 'K00172', 'K03737'],
    'Formate dehydrogenase': ['K00123', 'K00124', 'K00127', 'K05299', 'K15022', 'K22015', 'K00122', 'K22516', 'K00125', 'K00126'],
    '[FeFe] hydrogenase': ['K18330', 'K18331', 'K18332', 'K17997', 'K17998', 'K17999', 'K00532', 'K00533', 'K00534', 'K06441'],
    'Electron transfer flavoprotein': ['K03521', 'K03522'],
    'K+ uptake (trk/kdp)': ['K03498', 'K03499', 'K01546', 'K01547', 'K01548'],
    'Compatible solutes (proVWX)': ['K02000', 'K02001', 'K02002'],
    'Na+/H+ antiporter (mnh/nhaA)': ['K05565', 'K05566', 'K05567', 'K05568', 'K05569', 'K05570', 'K05571', 'K03313'],
    'NH4+ transporter (amt)': ['K03320'],
    'NH4+ assimilation (glnA/gdhA/gltBD)': ['K01915', 'K00262', 'K00265', 'K00266'],
}
GROUPING = {'mmc pathway': 'Propionate oxidation', 'Formate dehydrogenase': 'Electron disposal',
            '[FeFe] hydrogenase': 'Electron disposal', 'Electron transfer flavoprotein': 'Electron disposal',
            'K+ uptake (trk/kdp)': 'Ammonia stress', 'Compatible solutes (proVWX)': 'Ammonia stress',
            'Na+/H+ antiporter (mnh/nhaA)': 'Ammonia stress', 'NH4+ transporter (amt)': 'Ammonia stress',
            'NH4+ assimilation (glnA/gdhA/gltBD)': 'Ammonia stress'}
b132 = rp[rp.bin == 'H_73'].set_index('Feature')[S]
ann = pd.read_csv(f'{D}/metagenome/MAG132_kegg.txt', sep='\t', header=None, usecols=[0, 1, 2])
ann.columns = ['gene', 'KO', 'definition']; ann = ann.dropna(subset=['KO'])
kog = b132.index.intersection(ann.gene)
tpm = b132.loc[kog] / b132.loc[kog].sum() * 1e6                # per million KO-annotated transcripts of MAG132
rows = []
for mod, kos in MODS.items():
    genes = tpm.index.intersection(ann[ann.KO.isin(kos)].gene)
    v = tpm.loc[genes].sum()
    rows.append(dict(module=mod, category=GROUPING[mod], n_genes=len(genes), H=v[E].mean(), HC=v[HH].mean(),
                     H_sd=v[E].std(ddof=1), HC_sd=v[HH].std(ddof=1),
                     log2FC=np.log2((v[HH].mean() + 1) / (v[E].mean() + 1)), P=welch(v[E], v[HH]),
                     **{s: v[s] for s in S}))
mods = pd.DataFrame(rows)
ok = mods.P.notna()
mods['q'] = np.nan; mods.loc[ok, 'q'] = bh(mods.loc[ok, 'P'])
mods.to_csv(f'{OUT}/fig6_mag132_modules.csv', index=False, encoding='utf-8-sig')
ann[ann.gene.isin(tpm.index) & ann.KO.isin(sum(MODS.values(), []))].to_csv(f'{OUT}/fig6_mag132_module_genes.csv', index=False,
                                                                          encoding='utf-8-sig')
# share of all reads of MAG132 that fall on KO-annotated genes (the rest is mostly the tmRNA locus)
frac = (b132.loc[kog].mul(rp.set_index('Feature').loc[kog, 'len'], axis=0).sum()
        / b132.mul(rp.set_index('Feature').loc[b132.index, 'len'], axis=0).sum())
pd.Series(frac, name='KO_annotated_read_fraction').to_csv(f'{OUT}/fig6_mag132_read_fraction.csv', encoding='utf-8-sig')

# ---------- print ----------------------------------------------------------------------------------
pd.set_option('display.width', 250)
act['avg'] = (act.RNA_H + act.RNA_HC) / 2
print('== most active MAGs (mRNA share)'); print(act.sort_values('avg', ascending=False).head(14)[
    ['name', 'guild', 'genus', 'DNA_H', 'DNA_HC', 'RNA_H', 'RNA_HC', 'RNA_DNA_H', 'RNA_DNA_HC', 'P', 'q']].round(3).to_string())
print('\n== guild shares'); print(gs.round(3).to_string())
print('\n== mcrA mRNA'); print(mc.sort_values('H_pct', ascending=False)[['name', 'genus', 'H_pct', 'HC_pct', 'P']].round(3).to_string())
print('\n== MAG132 modules'); print(mods[['module', 'n_genes', 'H', 'HC', 'log2FC', 'P', 'q']].round(4).to_string(index=False))
print('\n== MAG132 KO-annotated read fraction', frac.round(3).to_dict())
key = act[act.name.isin(['MAG132', 'MAG133', 'MAG126', 'MAG3', 'MAG14', 'MAG15', 'MAG12', 'MAG19', 'MAG17', 'MAG9', 'MAG10',
                         'MAG11', 'MAG95', 'MAG4'])]
print('\n== key MAGs'); print(key[['name', 'genus', 'DNA_H', 'DNA_HC', 'RNA_H', 'RNA_HC', 'RNA_DNA_H', 'RNA_DNA_HC', 'P', 'q']].round(3).to_string())
