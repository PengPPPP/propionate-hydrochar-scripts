"""Rebuild per-bottle, per-cycle cumulative CH4 from raw GC sheet and compute kinetics.
Usage: python 01_gas_reconstruction.py <gas workbook> <out_dir>"""
import sys, numpy as np, pandas as pd, openpyxl
from scipy.optimize import curve_fit
src, out = sys.argv[1], sys.argv[2]
ws = openpyxl.load_workbook(src, data_only=True).worksheets[0]
days = [ws.cell(1, c).value for c in range(3, 66)]
def bottle(b):
    v = [ws.cell(57 + b, c).value for c in range(3, 66)]
    return np.array([x if isinstance(x, (int, float)) else np.nan for x in v], float)
# raw letter -> (ammonia g/L added, hydrochar, bottles, cycle boundaries in days since 8.01)
G = {'N0':  (0, 0, [1, 2, 3],    [0, 28, 40]),
     'N0C': (0, 1, [4, 5, 6],    [0, 24, 36]),
     'L':   (1, 0, [7, 8, 9],    [0, 40, 58]),
     'LC':  (1, 1, [16, 17, 18], [0, 28, 40]),
     'M':   (2, 0, [10, 11, 12], [0, 54, 90]),
     'MC':  (2, 1, [19, 20, 21], [0, 40, 62]),
     'H':   (4, 0, [13, 14, 15], [0, 82, 126]),   # cycle 2 incomplete in this file
     'HC':  (4, 1, [22, 23, 24], [0, 58, 102])}
def gomp(t, Pm, Rm, lam): return Pm * np.exp(-np.exp(Rm * np.e / Pm * (lam - t) + 1))
def cross(t, y, lv):
    for i in range(1, len(t)):
        if y[i-1] < lv <= y[i]: return t[i-1] + (lv - y[i-1]) * (t[i] - t[i-1]) / (y[i] - y[i-1])
    return np.nan
long, rows = [], []
for g, (amm, hc, bs, cb) in G.items():
    for cyc in (1, 2):
        s, e = cb[cyc - 1], cb[cyc]
        for b in bs:
            y_all = bottle(b)
            m = [(d - s, v) for d, v in zip(days, y_all) if s < d <= e and not np.isnan(v)]
            t = np.r_[0, [x[0] for x in m]]; y = np.r_[0, [x[1] for x in m]]
            for ti, yi in zip(t, y): long.append(dict(group=g, ammonia=amm, hydrochar=hc, bottle=b, cycle=cyc, day=ti, CH4_mL=yi))
            complete = not (g == 'H' and cyc == 2)
            ymax = y.max()
            try:
                p, _ = curve_fit(gomp, t, y, p0=[ymax, max(np.diff(y).max() / 2, .1), t[np.argmax(y > .1 * ymax)]],
                                 bounds=([ymax * .95, .01, -5], [ymax * 1.25, 30, 120]), method='trf', maxfev=50000)
            except Exception:
                p = [np.nan] * 3
            rows.append(dict(group=g, ammonia=amm, hydrochar=hc, bottle=b, cycle=cyc, complete=complete,
                             CH4_max=ymax, t_end=t.max(), t10=cross(t, y, .1 * ymax), t50=cross(t, y, .5 * ymax),
                             t90=cross(t, y, .9 * ymax), Rmax_obs=(np.diff(y) / np.diff(t)).max(),
                             Pm=p[0], Rm=p[1], lam=p[2]))
pd.DataFrame(long).to_csv(f'{out}/ch4_per_bottle.csv', index=False, encoding='utf-8-sig')
r = pd.DataFrame(rows); r.to_csv(f'{out}/kinetics_per_cycle.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
agg = r.groupby(['cycle', 'group'], sort=False)[['CH4_max', 't10', 't50', 't90', 'Rmax_obs', 'Rm', 'lam']].agg(['mean', 'std']).round(2)
print(agg.to_string())
