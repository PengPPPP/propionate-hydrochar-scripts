"""Section 3.3 data: Bray-Curtis PCoA, PERMANOVA (ammonia x hydrochar), PERMDISP, pairwise distances,
guild composition per sample, hydrogenotroph:acetoclast ratio, two-way ANOVA per guild.
Usage: python 08_fig3_data.py <mag_abundance.csv> <mag_guilds.csv> <out dir>
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REL, GUILD, O = sys.argv[1:4]
rng = np.random.default_rng(20261113)
rel = pd.read_csv(REL, index_col=0)
g = pd.read_csv(GUILD, index_col=0)
samples = list(rel.columns)
grp = np.array([s.split('_')[0] for s in samples])
amm = np.array([{'L': 1, 'M': 2, 'H': 4}[x[0]] for x in grp])
hc = np.array([1 if x.endswith('C') else 0 for x in grp])

# ---------- Bray-Curtis, PCoA
D = squareform(pdist(rel.T.values, 'braycurtis')); n = len(D)
G = -0.5 * (np.eye(n) - 1 / n) @ (D ** 2) @ (np.eye(n) - 1 / n)
w, V = np.linalg.eigh(G); idx = np.argsort(w)[::-1]; w, V = w[idx], V[:, idx]
pos = w > 1e-10; expl = w[pos] / w[pos].sum() * 100
pc = pd.DataFrame(V[:, :2] * np.sqrt(w[:2]), index=samples, columns=['PC1', 'PC2'])
pc['group'], pc['ammonia'], pc['hydrochar'] = grp, amm, hc
pc.attrs['expl'] = expl[:2]
pc.to_csv(f'{O}/fig3_pcoa.csv', encoding='utf-8-sig')
pd.Series(expl[:5], index=[f'PC{i}' for i in range(1, 6)]).to_csv(f'{O}/fig3_pcoa_variance.csv', encoding='utf-8-sig')


# ---------- PERMANOVA (sequential SS, balanced design) with 9999 permutations
def ss_within(lbl):
    return sum((D[np.ix_(ix, ix)] ** 2).sum() / 2 / len(ix) for ix in (np.where(lbl == v)[0] for v in np.unique(lbl)))


def permanova(a, h):
    sst = (D ** 2).sum() / 2 / n
    ssa, ssh = sst - ss_within(a), sst - ss_within(h)
    cell = np.array([f'{x}{y}' for x, y in zip(a, h)]); ssr = ss_within(cell)
    ssab = sst - ssr - ssa - ssh
    dfa, dfr = len(np.unique(a)) - 1, n - len(np.unique(cell))
    F = np.array([ssa / dfa, ssh / 1, ssab / dfa]) / (ssr / dfr)
    return F, np.array([ssa, ssh, ssab, ssr]) / sst


F0, R2 = permanova(amm, hc); ge = np.zeros(3)
for _ in range(9999):
    p = rng.permutation(n); ge += permanova(amm[p], hc[p])[0] >= F0
P = (ge + 1) / 10000
perm = pd.DataFrame({'term': ['Ammonia', 'Hydrochar', 'Ammonia x Hydrochar', 'Residual'], 'R2': R2,
                     'pseudoF': list(F0) + [np.nan], 'P': list(P) + [np.nan]})
perm.to_csv(f'{O}/fig3_permanova.csv', index=False, encoding='utf-8-sig')
print(perm.round(4).to_string(index=False))

# ---------- PERMDISP (distance to centroid in full PCoA space, permutation F test)
Y = V[:, pos] * np.sqrt(w[pos])
cent = {c: Y[grp == c].mean(0) for c in np.unique(grp)}
dist_c = np.array([np.linalg.norm(Y[i] - cent[grp[i]]) for i in range(n)])


def fstat(lbl):
    return stats.f_oneway(*[dist_c[lbl == c] for c in np.unique(lbl)]).statistic


f0 = fstat(grp); cnt = sum(fstat(grp[rng.permutation(n)]) >= f0 for _ in range(9999))
print(f'PERMDISP: F={f0:.2f}, P={(cnt + 1) / 10000:.3f}')
pd.DataFrame({'term': ['PERMDISP (6 groups)'], 'F': [f0], 'P': [(cnt + 1) / 10000]}).to_csv(
    f'{O}/fig3_permdisp.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'sample': samples, 'group': grp, 'distance_to_centroid': dist_c}).to_csv(
    f'{O}/fig3_dispersion.csv', index=False, encoding='utf-8-sig')

# ---------- pairwise Bray-Curtis: within vs between (control-hydrochar) per ammonia level
pairs = []
for lv in [1, 2, 4]:
    c = np.where((amm == lv) & (hc == 0))[0]; h = np.where((amm == lv) & (hc == 1))[0]
    for i in c:
        for j in h:
            pairs.append(dict(ammonia=lv, type='Between', d=D[i, j]))
    for ix in (c, h):
        for a_ in range(3):
            for b_ in range(a_ + 1, 3):
                pairs.append(dict(ammonia=lv, type='Within', d=D[ix[a_], ix[b_]]))
pairs = pd.DataFrame(pairs); pairs.to_csv(f'{O}/fig3_distances.csv', index=False, encoding='utf-8-sig')
print(pairs.groupby(['ammonia', 'type']).d.mean().round(3).unstack().to_string())

# ---------- guild composition per sample
MAP = {'SPOB (known lineage, mmc pathway)': 'SPOB', 'other bacteria with mmc pathway': 'Other mmc bacteria',
       'bacteria with complete WL pathway': 'Other bacteria', 'other bacteria': 'Other bacteria',
       'hydrogenotrophic methanogen': 'Hydrogenotrophic methanogens', 'acetoclastic methanogen': 'Methanothrix',
       'mixotrophic methanogen (Methanosarcina)': 'Methanosarcina', 'methylotrophic methanogen': 'Other archaea',
       'other archaea': 'Other archaea'}
gu = rel.groupby(g.guild.map(MAP)).sum()
gu.to_csv(f'{O}/fig3_guilds.csv', encoding='utf-8-sig')
ratio = gu.loc['Hydrogenotrophic methanogens'] / gu.loc['Methanothrix']


def anova2(y):
    df = pd.DataFrame({'y': y.values, 'a': amm, 'h': hc}); gm = df.y.mean()
    A = df.groupby('a').y.mean(); B = df.groupby('h').y.mean(); AB = df.groupby(['a', 'h']).y.mean()
    ssa = 6 * ((A - gm) ** 2).sum(); ssb = 9 * ((B - gm) ** 2).sum()
    ssab = 3 * sum((AB[(i, j)] - A[i] - B[j] + gm) ** 2 for i in A.index for j in B.index)
    sse = sum(((df[(df.a == i) & (df.h == j)].y - AB[(i, j)]) ** 2).sum() for i in A.index for j in B.index)
    return {nm: stats.f.sf((ss / d_) / (sse / 12), d_, 12) for nm, ss, d_ in
            [('ammonia', ssa, 2), ('hydrochar', ssb, 1), ('interaction', ssab, 2)]}


rows = []
for name, y in list(gu.iterrows()) + [('H:A ratio', ratio)]:
    r = dict(guild=name, **{f'P_{k}': v for k, v in anova2(y).items()})
    for lv, lab in [(1, 'L'), (2, 'M'), (4, 'H')]:
        a = y[[f'{lab}_{i}' for i in (1, 2, 3)]]; b = y[[f'{lab}C_{i}' for i in (1, 2, 3)]]
        r[f'{lab}_ctrl'], r[f'{lab}_char'] = a.mean(), b.mean()
        r[f'{lab}_P'] = stats.ttest_ind(b, a, equal_var=False).pvalue
    rows.append(r)
gs = pd.DataFrame(rows); gs.to_csv(f'{O}/fig3_guild_stats.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(gs.round(3).to_string(index=False))
