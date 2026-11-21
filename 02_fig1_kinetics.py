"""Section 3.1 kinetics from raw per-bottle CH4 (cycle 1).
Gompertz with Pm fixed to each bottle's measured maximum (lam_fix, Rm_fix; used in Fig 1B-D),
free Gompertz for comparison, mu_max4 = max slope of ln(CH4) over 4 consecutive points between 1.5 mL and 80% of max (Fig 1E),
t10, final CH4, two-way ANOVA and per-level Welch t tests.
Usage: python 02_fig1_kinetics.py <ch4_per_bottle.csv> <out dir>"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
d = pd.read_csv(sys.argv[1]); O = sys.argv[2]
d = d[d.cycle == 1]
def gomp(t, Pm, Rm, lam): return Pm * np.exp(-np.exp(Rm * np.e / Pm * (lam - t) + 1))
def r2(y, yh): return 1 - ((y - yh)**2).sum() / ((y - y.mean())**2).sum()
rows = []
for (g, b), s in d.groupby(['group', 'bottle']):
    s = s.sort_values('day'); t = s.day.values.astype(float); y = s.CH4_mL.values; ymax = y.max()
    p0 = [ymax, max(np.diff(y).max() / 2, .1), t[np.argmax(y > .1 * ymax)]]
    res = dict(group=g, ammonia=s.ammonia.iloc[0], hydrochar=s.hydrochar.iloc[0], bottle=b, CH4_final=ymax)
    pf, _ = curve_fit(gomp, t, y, p0=[ymax * 1.1] + p0[1:], bounds=([ymax, .01, -5], [3 * ymax, 30, 120]), method='trf', maxfev=50000)
    res.update(Pm_free=pf[0], Rm_free=pf[1], lam_free=pf[2], R2_free=r2(y, gomp(t, *pf)))
    fx = lambda tt, Rm, lam: gomp(tt, ymax, Rm, lam)
    pp, _ = curve_fit(fx, t, y, p0=p0[1:], bounds=([.01, -5], [30, 120]), method='trf', maxfev=50000)
    res.update(Rm_fix=pp[0], lam_fix=pp[1], R2_fix=r2(y, fx(t, *pp)))
    m = (y >= 1.5) & (y <= 0.7 * ymax)
    k, c = np.polyfit(t[m], np.log(y[m]), 1); res.update(mu=k, mu_n=int(m.sum()), mu_R2=np.corrcoef(t[m], np.log(y[m]))[0, 1]**2)
    tt = np.interp(0.1 * ymax, y, t) if np.all(np.diff(y) >= -1) else np.nan
    for i in range(1, len(t)):
        if y[i-1] < .1 * ymax <= y[i]: tt = t[i-1] + (.1 * ymax - y[i-1]) * (t[i] - t[i-1]) / (y[i] - y[i-1]); break
    res['t10'] = tt
    best = np.nan
    for i in range(len(t) - 3):
        tt4, yy4 = t[i:i + 4], y[i:i + 4]
        if yy4.min() >= 1.5 and yy4.max() <= 0.8 * ymax and np.all(np.diff(yy4) > 0):
            k4 = np.polyfit(tt4, np.log(yy4), 1)[0]; best = k4 if np.isnan(best) else max(best, k4)
    res['mu_max4'] = best
    rows.append(res)
r = pd.DataFrame(rows).sort_values(['ammonia', 'hydrochar', 'bottle'])
r.to_csv(f'{O}/fig1_kinetics.csv', index=False, encoding='utf-8-sig')
def anova2(df, y):
    gm = df[y].mean(); A = df.groupby('ammonia')[y].mean(); B = df.groupby('hydrochar')[y].mean(); AB = df.groupby(['ammonia', 'hydrochar'])[y].mean()
    n = 3; la, lb = len(A), len(B)
    ssa = n * lb * ((A - gm)**2).sum(); ssb = n * la * ((B - gm)**2).sum()
    ssab = n * sum((AB[(i, j)] - A[i] - B[j] + gm)**2 for i in A.index for j in B.index)
    sse = sum(((df[(df.ammonia == i) & (df.hydrochar == j)][y] - AB[(i, j)])**2).sum() for i in A.index for j in B.index)
    dfe = len(df) - la * lb; out = {}
    for nm, ss, dfn in [('ammonia', ssa, la - 1), ('hydrochar', ssb, 1), ('interaction', ssab, la - 1)]:
        F = (ss / dfn) / (sse / dfe); out[nm] = (dfn, dfe, F, stats.f.sf(F, dfn, dfe))
    return out
pd.set_option('display.width', 250)
summ = r.groupby(['ammonia', 'hydrochar'])[['CH4_final', 'lam_fix', 'Rm_fix', 'R2_fix', 'lam_free', 'Rm_free', 'Pm_free', 'R2_free', 'mu', 'mu_R2', 'mu_max4', 't10']].agg(['mean', 'std']).round(3)
print(summ[['lam_fix', 'Rm_fix', 'mu_max4', 'CH4_final']].to_string())
r['ln_lam'] = np.log(r.lam_fix)                     # log scale: interaction = hydrochar changes the RELATIVE ammonia effect
st = []
for y in ['lam_fix', 'ln_lam', 'Rm_fix', 'lam_free', 'Rm_free', 'mu', 'mu_max4', 't10', 'CH4_final']:
    a = anova2(r, y); st.append(dict(metric=y, **{f'{k}_F': v[2] for k, v in a.items()}, **{f'{k}_P': v[3] for k, v in a.items()}))
    for lv in [0, 1, 2, 4]:
        c = r[(r.ammonia == lv) & (r.hydrochar == 0)][y]; h = r[(r.ammonia == lv) & (r.hydrochar == 1)][y]
        st[-1][f'Welch_{lv}'] = stats.ttest_ind(h, c, equal_var=False).pvalue
st = pd.DataFrame(st); st.to_csv(f'{O}/fig1_stats.csv', index=False, encoding='utf-8-sig')
print(st.round(4).to_string(index=False))

# ---------- ammonia-induced delay: lambda minus the mean lambda of the same treatment without added ammonia
# A smaller slope with hydrochar means hydrochar offsets the ammonia effect itself (specific alleviation),
# on top of the general shortening seen without added ammonia.
dl = []
for hc in (0, 1):
    s = r[r.hydrochar == hc]; base = s[s.ammonia == 0].lam_fix.mean()
    for lv in [0, 1, 2, 4]:
        v = s[s.ammonia == lv].lam_fix - base
        dl.append(dict(hydrochar=hc, ammonia=lv, delay_mean=v.mean(), delay_sd=v.std(), lam_rel=s[s.ammonia == lv].lam_fix.mean() / base))
dl = pd.DataFrame(dl)
for lv in [1, 2, 4]:                                   # share of the control's ammonia-induced delay removed by hydrochar
    c = dl[(dl.hydrochar == 0) & (dl.ammonia == lv)].delay_mean.iloc[0]; h = dl[(dl.hydrochar == 1) & (dl.ammonia == lv)].delay_mean.iloc[0]
    dl.loc[dl.ammonia == lv, 'removed_pct'] = (1 - h / c) * 100
X = np.c_[np.ones(len(r)), r.ammonia, r.hydrochar, r.ammonia * r.hydrochar]; yv = r.lam_fix.values   # ANCOVA, added N continuous
b = np.linalg.lstsq(X, yv, rcond=None)[0]; res = yv - X @ b; dfe = len(yv) - X.shape[1]
cov = (res @ res / dfe) * np.linalg.inv(X.T @ X); t = b[3] / np.sqrt(cov[3, 3])
dl['slope_ctrl'] = b[1]; dl['slope_hc'] = b[1] + b[3]
dl['slope_diff_P'] = 2 * stats.t.sf(abs(t), dfe)
dl.to_csv(f'{O}/fig1_ammonia_delay.csv', index=False, encoding='utf-8-sig')
print(dl.round(3).to_string(index=False))
