"""Re-assign functional guilds from per-MAG KO content (DNA) and taxonomy.
Usage: python 07_mag_guilds.py <data dir> <omics dir>"""
import sys, os, numpy as np, pandas as pd
D, O = sys.argv[1], sys.argv[2]
m = pd.read_csv(f'{O}/mag_table.csv', index_col=0)
ko = {}
for g, s in zip(m.index, m.short):
    f = f'{D}/metagenome/ko/{s}.txt'
    ko[g] = set(pd.read_csv(f, sep='\t', usecols=['Ko']).Ko.dropna().astype(str)) if os.path.exists(f) else set()
m['n_KO'] = [len(ko[g]) for g in m.index]
def steps(defn):
    return {g: [bool(ko[g] & set(alts)) for alts in defn.values()] for g in m.index}
MMC = {'S1_activation': ['K01026', 'K01908', 'K01895', 'K01905', 'K22224', 'K00932', 'K18118'],
       'S2_carboxyltransfer': ['K17489', 'K03416', 'K17490', 'K01965', 'K01966', 'K11263', 'K18472', 'K01604'],
       'S3_epimerase': ['K05606'], 'S4_mutase': ['K01847', 'K01848', 'K01849'],
       'S5_succinylCoA': ['K01902', 'K01903', 'K01026', 'K18118'],
       'S6_SDH_FRD': ['K00239', 'K00240', 'K00244', 'K00245'], 'S7_fumarase': ['K01676', 'K01677', 'K01678', 'K01679'],
       'S8_malate': ['K00024', 'K00116', 'K00027', 'K00029'], 'S9_OAA_pyruvate': ['K01571', 'K01572', 'K01573', 'K01958', 'K01959', 'K01960', 'K01610', 'K01596'],
       'S10_pyruvate_acCoA': ['K00169', 'K00170', 'K00171', 'K00172', 'K03737', 'K00656'],
       'S11_acetate': ['K00625', 'K13788', 'K00925', 'K01905', 'K22224', 'K01895']}
WL = {'fhs': ['K01938'], 'folD': ['K01491', 'K01500'], 'metF': ['K00297', 'K25007'], 'acsE': ['K15023'],
      'acsB': ['K14138'], 'acsCD': ['K00197', 'K00194'], 'cooS': ['K00198']}
MCR = ['K00399']; HYDRO = {'fwd/fmd': ['K00200', 'K00201', 'K00202'], 'ftr': ['K00672'], 'mch': ['K01499'], 'mtd': ['K00319'], 'mer': ['K00320'], 'mtr': ['K00584', 'K00581']}
FDH = ['K00123', 'K00124', 'K00127', 'K22015', 'K05299', 'K15022', 'K00122', 'K22516']
mmc = steps(MMC); wl = steps(WL); hy = steps(HYDRO)
m['mmc_complete'] = [np.mean(mmc[g]) for g in m.index]
m['mmc_core_S2S3S4'] = [all(mmc[g][1:4]) for g in m.index]
m['WL_complete'] = [np.mean(wl[g]) for g in m.index]
m['mcrA'] = [bool(ko[g] & set(MCR)) for g in m.index]
m['hydrogenotrophic_complete'] = [np.mean(hy[g]) for g in m.index]
m['formate_dehydrogenase'] = [bool(ko[g] & set(FDH)) for g in m.index]
for k, v in MMC.items(): m[f'mmc_{k}'] = [mmc[g][list(MMC).index(k)] for g in m.index]
SPOB_LINEAGES = {'Syntrophobacteraceae', 'Pelotomaculaceae', 'Smithellaceae', 'Syntrophaceae'}
def guild(r):
    if r.domain == 'Archaea':
        if r.family == 'Methanotrichaceae': return 'acetoclastic methanogen'
        if r.family == 'Methanosarcinaceae': return 'mixotrophic methanogen (Methanosarcina)'
        if r.family in ('Methanobacteriaceae', 'Methanoculleaceae', 'Methanoregulaceae', 'Methanospirillaceae', 'Methanomicrobiaceae'): return 'hydrogenotrophic methanogen'
        if r.order in ('Methanomassiliicoccales',) or r.family in ('Methanomethylophilaceae', 'Methanomassiliicoccaceae'): return 'methylotrophic methanogen'
        return 'other archaea'
    if r.family in SPOB_LINEAGES and r.mmc_core_S2S3S4 and r.mmc_complete >= 0.8: return 'SPOB (known lineage, mmc pathway)'
    if r.mmc_core_S2S3S4 and r.mmc_complete >= 0.8: return 'other bacteria with mmc pathway'
    if r.WL_complete >= 0.85: return 'bacteria with complete WL pathway'
    return 'other bacteria'
m['guild'] = m.apply(guild, axis=1)
m.drop(columns=[c for c in m.columns if c.startswith('rel_')]).to_csv(f'{O}/mag_guilds.csv', encoding='utf-8-sig')
cols = [f'mean_{g}' for g in ['L', 'LC', 'M', 'MC', 'H', 'HC']]
print(m.groupby('guild')[cols].sum().round(2).assign(n=m.groupby('guild').size()).to_string())
print('\n== SPOB and mmc bacteria')
show = m[m.guild.str.contains('mmc')].sort_values('mean_HC', ascending=False)
print(show[['thesis_name', 'family', 'genus', 'mmc_complete', 'WL_complete', 'formate_dehydrogenase', 'completeness'] + cols].round(2).to_string())
print('\n== WL-complete bacteria')
print(m[m.guild == 'bacteria with complete WL pathway'][['thesis_name', 'family', 'genus', 'WL_complete'] + cols].round(2).to_string())
