"""Section 3.2 data: per-bottle propionate/acetate (cycle 1) from the raw VFA workbook, TAN, metrics and statistics.

Zero handling: a propionate 0 counts as depletion only if the previous measured value was <= 10 mM, otherwise it is
treated as a missing sample; an acetate 0 counts as a real zero only if propionate was measured on that day.
Propionate onset = time at which propionate fell, and stayed, below 25 mM (linear interpolation).
Usage: python 04_fig2_data.py <VFA workbook> <TAN workbook> <fig1_kinetics.csv> <out dir>
"""
import sys
import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

VFA, TANX, KIN, O = sys.argv[1:5]
GROUP = {1: 'N0', 2: 'N0', 3: 'N0', 4: 'N0C', 5: 'N0C', 6: 'N0C', 7: 'L', 8: 'L', 9: 'L', 16: 'LC', 17: 'LC', 18: 'LC',
         10: 'M', 11: 'M', 12: 'M', 19: 'MC', 20: 'MC', 21: 'MC', 13: 'H', 14: 'H', 15: 'H', 22: 'HC', 23: 'HC', 24: 'HC'}
# group -> (added NH4+-N g/L, hydrochar, day on which cycle 1 ended and substrate was renewed)
META = {'N0': (0, 0, 28), 'N0C': (0, 1, 24), 'L': (1, 0, 40), 'LC': (1, 1, 28),
        'M': (2, 0, 54), 'MC': (2, 1, 40), 'H': (4, 0, 82), 'HC': (4, 1, 58)}

# ---------- raw VFA blocks
wb = openpyxl.load_workbook(VFA, data_only=True)
raw = []
for sh in ['0', '1', '2', '4']:
    ws = wb[sh]
    days, species = None, None
    for row in ws.iter_rows(values_only=True):
        a = row[0]
        if days is None and isinstance(row[1], (int, float)):
            days = [x for x in row[1:] if isinstance(x, (int, float))]
            continue
        if row[1] in ('\u4e59\u9178', '\u4e19\u9178'):          # workbook labels: acetate, propionate
            species = 'acetate' if row[1] == '\u4e59\u9178' else 'propionate'
            continue
        if isinstance(a, int) and a in GROUP and species:
            for dday, val in zip(days, row[1:1 + len(days)]):
                raw.append(dict(bottle=a, species=species, day=dday,
                                raw=val if isinstance(val, (int, float)) else np.nan))
        if a == 0 or a == 'C':
            species = None
raw = pd.DataFrame(raw).drop_duplicates(['bottle', 'species', 'day'])

rows = []
for b, g in GROUP.items():
    amm, hc, end = META[g]
    p = raw[(raw.bottle == b) & (raw.species == 'propionate') & (raw.day < end)].sort_values('day')
    ac = raw[(raw.bottle == b) & (raw.species == 'acetate') & (raw.day < end)].set_index('day').raw
    prev = 30.0
    for _, r in p.iterrows():
        pv = np.nan if (r.raw == 0 and prev > 10) else r.raw
        if not np.isnan(pv):
            prev = pv
        av = np.nan if np.isnan(pv) else ac.get(r.day, np.nan)
        rows.append(dict(group=g, ammonia=amm, hydrochar=hc, bottle=b, day=r.day, propionate=pv, acetate=av))
    rows.append(dict(group=g, ammonia=amm, hydrochar=hc, bottle=b, day=0, propionate=30.0, acetate=np.nan))  # nominal start
v = pd.DataFrame(rows).sort_values(['bottle', 'day'])
v.to_csv(f'{O}/fig2_vfa.csv', index=False, encoding='utf-8-sig')

# ---------- per-bottle metrics
kin = pd.read_csv(KIN).set_index('bottle')
met = []
for b, s in v.groupby('bottle'):
    s = s.dropna(subset=['propionate'])
    t = s.day.values.astype(float)
    y = s.propionate.values
    below = y < 25
    onset = np.nan
    for i in range(1, len(t)):
        if below[i:].all() and not below[i - 1]:
            onset = t[i - 1] + (y[i - 1] - 25) * (t[i] - t[i - 1]) / (y[i - 1] - y[i])
            break
    rate = np.nan
    for i in range(len(t) - 2):
        k = -np.polyfit(t[i:i + 3], y[i:i + 3], 1)[0]
        rate = k if np.isnan(rate) else max(rate, k)
    done = t[np.argmax(y <= 1)] if (y <= 1).any() else np.nan
    acs = s.dropna(subset=['acetate'])
    pk = acs.acetate.max()
    pkd = acs.day.values[acs.acetate.values.argmax()]
    cons = 30 - acs.propionate.values
    ok = cons > 5
    conv = (1 - acs.acetate.values[ok] / cons[ok]).min() if ok.any() else np.nan
    met.append(dict(bottle=b, group=s.group.iloc[0], ammonia=s.ammonia.iloc[0], hydrochar=s.hydrochar.iloc[0],
                    prop_onset=onset, prop_maxrate=rate, prop_done=done, acetate_peak=pk, acetate_peak_day=pkd,
                    acetate_conv_min=conv, lam=kin.loc[b, 'lam_fix'], Rm=kin.loc[b, 'Rm_fix']))
met = pd.DataFrame(met)

# ---------- TAN at the end of cycles 1 and 2 (columns 9-11 = mg N/L)
ws = openpyxl.load_workbook(TANX, data_only=True).worksheets[0]
L2G = {'A': 'N0', 'B': 'N0C', 'C': 'L', 'D': 'M', 'E': 'H', 'F': 'LC', 'G': 'MC', 'H': 'HC'}
tan, cyc = [], None
for row in ws.iter_rows(values_only=True):
    if row[0] in ('\u7b2c\u4e00\u5468\u671f', '\u7b2c\u4e8c\u5468\u671f'):   # workbook labels: cycle 1, cycle 2
        cyc = 1 if row[0] == '\u7b2c\u4e00\u5468\u671f' else 2
        continue
    if cyc and row[0] in L2G and row[8] == row[0]:
        for k, val in enumerate(row[9:12]):
            if isinstance(val, (int, float)):
                g = L2G[row[0]]
                tan.append(dict(cycle=cyc, group=g, ammonia=META[g][0], hydrochar=META[g][1], rep=k + 1, TAN=val))
tan = pd.DataFrame(tan)
pH, T = 7.65, 37.0
tan['FAN_est'] = tan.TAN * 10 ** pH / (np.exp(6344 / (273 + T)) + 10 ** pH)   # free ammonia at the initial pH
tan.to_csv(f'{O}/fig2_tan.csv', index=False, encoding='utf-8-sig')
t1 = tan[tan.cycle == 1].copy()


# ---------- statistics
def anova2(df, y):
    df = df.dropna(subset=[y])
    gm = df[y].mean()
    A = df.groupby('ammonia')[y].mean()
    B = df.groupby('hydrochar')[y].mean()
    cell = df.groupby(['ammonia', 'hydrochar'])[y]
    n, m = cell.size(), cell.mean()
    ssa = sum(n[c] * (A[c[0]] - gm) ** 2 for c in m.index)
    ssb = sum(n[c] * (B[c[1]] - gm) ** 2 for c in m.index)
    ssab = sum(n[c] * (m[c] - A[c[0]] - B[c[1]] + gm) ** 2 for c in m.index)
    sse = sum(((df[(df.ammonia == c[0]) & (df.hydrochar == c[1])][y] - m[c]) ** 2).sum() for c in m.index)
    dfe = len(df) - len(m)
    out = {}
    for nm, ss, d_ in [('ammonia', ssa, len(A) - 1), ('hydrochar', ssb, 1), ('interaction', ssab, len(A) - 1)]:
        F = (ss / d_) / (sse / dfe)
        out[nm] = (F, stats.f.sf(F, d_, dfe))
    return out


pd.set_option('display.width', 250)
cols = ['prop_onset', 'prop_maxrate', 'prop_done', 'acetate_peak', 'acetate_peak_day', 'acetate_conv_min']
print(met.groupby(['ammonia', 'hydrochar'])[cols].agg(['mean', 'std']).round(2).to_string())
print(t1.groupby(['ammonia', 'hydrochar'])[['TAN', 'FAN_est']].agg(['mean', 'std']).round(0).to_string())
st = []
for y, df in [('prop_onset', met), ('prop_maxrate', met), ('acetate_peak', met), ('acetate_conv_min', met), ('TAN', t1)]:
    a = anova2(df, y)
    row = dict(metric=y, **{f'{k}_F': x[0] for k, x in a.items()}, **{f'{k}_P': x[1] for k, x in a.items()})
    for lv in [0, 1, 2, 4]:
        c = df[(df.ammonia == lv) & (df.hydrochar == 0)][y].dropna()
        h = df[(df.ammonia == lv) & (df.hydrochar == 1)][y].dropna()
        row[f'Welch_{lv}'] = stats.ttest_ind(h, c, equal_var=False).pvalue if len(c) > 1 and len(h) > 1 else np.nan
    st.append(row)
st = pd.DataFrame(st)
st.to_csv(f'{O}/fig2_stats.csv', index=False, encoding='utf-8-sig')
print(st.round(4).to_string(index=False))
r, pr = stats.pearsonr(met.prop_onset, met.lam)
print(f'propionate onset vs methane lag: r={r:.3f}, P={pr:.2g}, n={len(met)}')
met.to_csv(f'{O}/fig2_metrics.csv', index=False, encoding='utf-8-sig')
t2 = tan[tan.cycle == 2]
print('cycle-2 TAN:', t2.groupby(['ammonia', 'hydrochar']).TAN.mean().round(0).to_dict())
