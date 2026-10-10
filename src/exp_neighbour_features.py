"""Quick experiment: do neighbour-row (lag/lead/rolling) features add signal?
Evaluated with leave-one-year-out CV (test = 2 unseen years) and repeated stratified CV."""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings('ignore')

B = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
te[B] = te[B].fillna(te[B].median())
y = tr.rainfall.values
tr['year'] = np.arange(len(tr)) // 365


def base(d):
    d = d.copy()
    d['dew_dep'] = d.temparature - d.dewpoint
    d['trange'] = d.maxtemp - d.mintemp
    d['cm_s'] = d.cloud / 100 - d.sunshine / 12
    d['sin'] = np.sin(2 * np.pi * d.day / 365); d['cos'] = np.cos(2 * np.pi * d.day / 365)
    d['wd_s'] = np.sin(np.deg2rad(d.winddirection)); d['wd_c'] = np.cos(np.deg2rad(d.winddirection))
    return d


def neigh(d, cols, lags=(1, 2), rolls=(3, 7)):
    d = d.copy()
    for c in cols:
        for k in lags:
            d[f'{c}_lag{k}'] = d[c].shift(k).bfill()
            d[f'{c}_lead{k}'] = d[c].shift(-k).ffill()
        d[f'{c}_d1'] = d[c] - d[c].shift(1).bfill()
        d[f'{c}_dn1'] = d[c].shift(-1).ffill() - d[c]
        for r in rolls:
            d[f'{c}_rm{r}'] = d[c].rolling(r, center=True, min_periods=1).mean()
    return d


def cv(X, name):
    X = X.values.astype(float)
    res = {}
    for mname, mk in [('LR', lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))),
                      ('LGB', lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=4, min_child_samples=30,
                                                         subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=2))]:
        oof = np.zeros(len(y)); a = []
        for trn, val in LeaveOneGroupOut().split(X, y, tr.year):
            m = mk().fit(X[trn], y[trn]); oof[val] = m.predict_proba(X[val])[:, 1]; a.append(roc_auc_score(y[val], oof[val]))
        s = []
        for trn, val in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(X, y):
            m = mk().fit(X[trn], y[trn]); s.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        res[mname] = (np.mean(a), np.mean(s))
    print(f"{name:28s} " + "  ".join(f"{k}: LOYO {v[0]:.4f} RSKF {v[1]:.4f}" for k, v in res.items()), flush=True)


b = base(tr); F0 = B + ['dew_dep', 'trange', 'cm_s', 'sin', 'cos', 'wd_s', 'wd_c']
cv(tr[B], 'raw')
cv(b[F0], 'base-eng')
key = ['cloud', 'sunshine', 'humidity', 'pressure', 'dewpoint', 'dew_dep']
for lags, rolls, nm in [((1,), (), 'lag1'), ((1, 2), (), 'lag12'), ((1,), (3,), 'lag1+rm3'), ((1, 2), (3, 7), 'lag12+rm37')]:
    n = neigh(b, key, lags, rolls)
    cv(n[[c for c in n.columns if c not in ('id', 'day', 'rainfall', 'year')]], 'neigh ' + nm)
n = neigh(b, ['pressure', 'dewpoint', 'temparature', 'windspeed'], (1,), ())
cv(n[[c for c in n.columns if c not in ('id', 'day', 'rainfall', 'year')]], 'neigh smooth-vars lag1')
