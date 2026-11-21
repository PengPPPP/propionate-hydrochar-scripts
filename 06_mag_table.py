"""Build the MAG master table: ID mapping, GTDB taxonomy, CheckM quality, abundance in 18 metagenomes.
Usage: python 06_mag_table.py <data dir> <out dir>"""
import sys, pandas as pd, numpy as np
D, O = sys.argv[1], sys.argv[2]
ab = pd.read_excel(f'{D}/metagenome/abundance.xlsx', sheet_name='Sheet1').set_index('Genome')
unm = ab.loc['unmapped']; ab = ab.drop('unmapped')
smap = {'qqL': 'L', 'qqLH': 'LC', 'qqM': 'M', 'qqMH': 'MC', 'qqH': 'H', 'qqHH': 'HC'}
def rename(c):
    base = c.replace('_1.fastq', ''); rep = base[-1]; pre = base[:-1]
    return f'{smap[pre]}_{rep}'
ab.columns = [rename(c) for c in ab.columns]; unm.index = [rename(c) for c in unm.index]
order = [f'{g}_{r}' for g in ['L', 'LC', 'M', 'MC', 'H', 'HC'] for r in (1, 2, 3)]
ab = ab[order]; unm = unm[order]
rel = ab / (100 - unm) * 100          # % of reads mapped to MAGs
tax = pd.read_excel(f'{D}/metagenome/gtdbtk_annotation.xlsx', sheet_name='Sheet1')[['user_genome', 'classification']]
ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
t = tax.classification.str.split(';', expand=True); t.columns = ranks
t = t.apply(lambda s: s.str.replace(r'^[a-z]__', '', regex=True)); t.index = tax.user_genome
q = pd.read_csv(f'{D}/metagenome/genomeInfo.csv'); q['genome'] = q.genome.str.replace('.fa', '', regex=False)
q = q.set_index('genome')[['completeness', 'contamination', 'length', 'N50']]
mp = pd.read_excel(f'{D}/metagenome/name_map.xlsx'); mp['genome'] = 'Gp_qq' + mp.Sample.str.strip()
mp = mp.set_index('genome').Rename
m = pd.DataFrame(index=ab.index)
m['thesis_name'] = mp.reindex(m.index); m['short'] = m.index.str.replace('Gp_qq', '', regex=False)
m = m.join(t).join(q)
for g in ['L', 'LC', 'M', 'MC', 'H', 'HC']:
    m[f'mean_{g}'] = rel[[f'{g}_{r}' for r in (1, 2, 3)]].mean(axis=1)
m = m.join(rel.add_prefix('rel_'))
m.to_csv(f'{O}/mag_table.csv', encoding='utf-8-sig')
rel.to_csv(f'{O}/mag_abundance.csv', encoding='utf-8-sig')
print('MAGs', len(m), '| missing taxonomy', m.domain.isna().sum(), '| missing quality', m.completeness.isna().sum(), '| missing thesis name', m.thesis_name.isna().sum())
print('unmapped % per sample:', unm.round(1).to_dict())
print(m.groupby('domain').size().to_dict())
print('quality: completeness median %.1f, contamination median %.1f, >=90/<5: %d' % (m.completeness.median(), m.contamination.median(), ((m.completeness >= 90) & (m.contamination < 5)).sum()))
key = m[m.thesis_name.isin(['Bin12', 'Bin14', 'Bin132', 'Bin133', 'Bin95', 'Bin9', 'Bin3', 'Bin1'])]
print(key[['thesis_name', 'family', 'genus', 'species'] + [f'mean_{g}' for g in ['L', 'LC', 'M', 'MC', 'H', 'HC']]].round(2).to_string())
