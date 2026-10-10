"""Independent seed-averaged ensemble for the final day.
OOF from 5x3 repeated stratified CV (+ leave-one-year-out check); test preds from full-train refits.
Original Rainfall.csv is only ever appended to training folds."""
import numpy as np, pandas as pd, warnings, time, os
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
OUT = 'outputs/ensemble'; os.makedirs(OUT, exist_ok=True)

B = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
og = pd.read_csv('data/Rainfall.csv'); og.columns = [c.strip() for c in og.columns]
og['rainfall'] = (og.rainfall.str.strip() == 'yes').astype(int); og = og.dropna().reset_index(drop=True)
te[B] = te[B].fillna(tr[B].median())
y = tr.rainfall.values; yo = og.rainfall.values
years = np.arange(len(tr)) // 365


def fe(d):
    d = d[['day'] + B].copy()
    d['dew_dep'] = d.temparature - d.dewpoint
    d['trange'] = d.maxtemp - d.mintemp
    d['cm_s'] = d.cloud / 100 - d.sunshine / 12
    d['hum_cloud'] = d.humidity * d.cloud / 100
    d['sin'] = np.sin(2 * np.pi * d.day / 365); d['cos'] = np.cos(2 * np.pi * d.day / 365)
    d['wd_s'] = np.sin(np.deg2rad(d.winddirection)); d['wd_c'] = np.cos(np.deg2rad(d.winddirection))
    return d.drop(columns=['day'])


Xa, Xt, Xo = fe(tr), fe(te), fe(og)
RAW = B; ENG = list(Xa.columns)
LIN = [c for c in ENG if c != 'winddirection']

M = {  # name: (cols, factory(seed), n_seeds, use_orig)
    'lr_raw': (RAW, lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), 1, False),
    'lr_eng': (LIN, lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), 1, False),
    'lr_eng_og': (LIN, lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), 1, True),
    'lr_spline': (LIN, lambda s: make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=0),
                                               SplineTransformer(n_knots=5, degree=3), StandardScaler(),
                                               LogisticRegression(C=0.01, max_iter=5000)), 1, False),
    'svm': (LIN, lambda s: make_pipeline(StandardScaler(), SVC(C=0.3, gamma=0.01, probability=False)), 1, False),
    'mlp': (LIN, lambda s: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1.0, learning_rate_init=3e-3, max_iter=400,
                                                                         early_stopping=False, random_state=s)), 4, False),
    'lgb': (ENG, lambda s: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.015, num_leaves=4, min_child_samples=40, subsample=0.6,
                                              subsample_freq=1, colsample_bytree=0.4, reg_lambda=10, verbose=-1, n_jobs=2, random_state=s), 3, False),
    'lgb_og': (ENG, lambda s: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.015, num_leaves=4, min_child_samples=40, subsample=0.6,
                                                 subsample_freq=1, colsample_bytree=0.4, reg_lambda=10, verbose=-1, n_jobs=2, random_state=s), 3, True),
    'xgb': (ENG, lambda s: xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=2, min_child_weight=8, subsample=0.6,
                                             colsample_bytree=0.4, reg_lambda=10, n_jobs=2, random_state=s, verbosity=0), 3, False),
    'cat': (ENG, lambda s: CatBoostClassifier(iterations=500, learning_rate=0.02, depth=3, l2_leaf_reg=10, rsm=0.5, verbose=0,
                                              thread_count=2, random_seed=s), 2, False),
    'et': (ENG, lambda s: ExtraTreesClassifier(500, min_samples_leaf=12, max_features=0.4, n_jobs=2, random_state=s), 1, False),
    'rf': (ENG, lambda s: RandomForestClassifier(500, min_samples_leaf=10, max_features=0.3, n_jobs=2, random_state=s), 1, False),
}


def score(m, X):
    is_svm = hasattr(m, 'steps') and isinstance(m.steps[-1][1], SVC)
    return m.decision_function(X) if is_svm else m.predict_proba(X)[:, 1]


def fit_pred(name, trn, Xev):
    cols, mk, ns, use_og = M[name]
    Xf = Xa.iloc[trn][cols].values; yf = y[trn]
    if use_og:
        Xf = np.vstack([Xf, Xo[cols].values]); yf = np.r_[yf, yo]
    return np.mean([score(mk(s).fit(Xf, yf), Xev[cols].values) for s in range(ns)], axis=0)


R = 3
rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=42).split(Xa, y))
OOF = {}; LOYO = {}; TEST = {}
for name in M:
    t0 = time.time()
    oof = np.zeros((R, len(y)))
    for i, (a, b) in enumerate(rskf):
        oof[i // 5, b] = fit_pred(name, a, Xa.iloc[b])
    lo = np.zeros(len(y))
    for a, b in LeaveOneGroupOut().split(Xa, y, years):
        lo[b] = fit_pred(name, a, Xa.iloc[b])
    OOF[name] = oof; LOYO[name] = lo
    TEST[name] = fit_pred(name, np.arange(len(y)), Xt)
    print(f"{name:10s} RSKF {np.mean([roc_auc_score(y, o) for o in oof]):.5f}  LOYO {np.mean([roc_auc_score(y[years == g], lo[years == g]) for g in range(6)]):.5f}"
          f"  {time.time() - t0:.0f}s", flush=True)

names = list(M)
rk = lambda v: rankdata(v) / len(v)


def blend_auc(sel, w=None):
    w = np.ones(len(sel)) if w is None else w
    return np.mean([roc_auc_score(y, sum(wi * rk(OOF[n][r]) for wi, n in zip(w, sel))) for r in range(R)])


def loyo_auc(sel):
    p = sum(np.concatenate([rk(LOYO[n][years == g]) for g in range(6)]) for n in sel)
    return np.mean([roc_auc_score(y[years == g], p[years == g]) for g in range(6)])


print(f"\nmean_all    RSKF {blend_auc(names):.5f}  LOYO {loyo_auc(names):.5f}")
# greedy forward selection (with replacement) on repeat-averaged rank AUC
sel = []
best = 0
for _ in range(12):
    cand = max(names, key=lambda n: blend_auc(sel + [n]))
    sc = blend_auc(sel + [cand])
    if sc <= best + 1e-5: break
    sel.append(cand); best = sc
print(f"hill_climb  RSKF {best:.5f}  LOYO {loyo_auc(sel):.5f}  -> {sel}")

# nested check of the hill climb: select on 4 folds' rows, score on the 5th
nested = []
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(Xa, y)):
    s2 = []; bb = 0
    for _ in range(12):
        fn = lambda S: np.mean([roc_auc_score(y[a], sum(rk(OOF[n][r][a]) for n in S)) for r in range(R)])
        c = max(names, key=lambda n: fn(s2 + [n])); v = fn(s2 + [c])
        if v <= bb + 1e-5: break
        s2.append(c); bb = v
    nested.append(np.mean([roc_auc_score(y[b], sum(rk(OOF[n][r][b]) for n in s2)) for r in range(R)]))
    base = np.mean([roc_auc_score(y[b], sum(rk(OOF[n][r][b]) for n in names)) for r in range(R)])
    print(f"  fold {f}: hill {nested[-1]:.5f}  mean_all {base:.5f}")
print(f"nested hill_climb {np.mean(nested):.5f}")

sub = pd.read_csv('data/sample_submission.csv')
for tag, S in [('mean_all', names), ('hill', sel)]:
    p = sum(rk(TEST[n]) for n in S) / len(S)
    sub.assign(rainfall=np.clip(p, 0.001, 0.999)).to_csv(f'{OUT}/submission_{tag}.csv', index=False)
pd.DataFrame({n: rk(TEST[n]) for n in names}).to_csv(f'{OUT}/test_ranks.csv', index=False)
pd.DataFrame({n: OOF[n].mean(0) for n in names}).assign(y=y).to_csv(f'{OUT}/oof.csv', index=False)
print('saved')
