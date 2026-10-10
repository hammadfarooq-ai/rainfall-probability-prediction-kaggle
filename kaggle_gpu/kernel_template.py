"""GPU models for the Rainfall challenge: TabPFN v2, RealMLP, TabM.
Same CV splits as the local src/build_ensemble.py so OOFs can be blended offline."""
import subprocess, sys, glob, io, time, traceback, warnings
for pkgs in (['tabpfn==2.0.9'], ['pytabkit']):
    print(subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, capture_output=True, text=True).stderr[-1500:], flush=True)
import numpy as np, pandas as pd, torch
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV, torch.__version__, flush=True)

ORIG_CSV = """__ORIG__"""
B = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
p = [f for f in glob.glob('/kaggle/input/**/train.csv', recursive=True)][0]
tr = pd.read_csv(p); te = pd.read_csv(p.replace('train.csv', 'test.csv'))
og = pd.read_csv(io.StringIO(ORIG_CSV)); og.columns = [c.strip() for c in og.columns]
og['rainfall'] = (og.rainfall.str.strip() == 'yes').astype(int); og = og.dropna().reset_index(drop=True)
te[B] = te[B].fillna(tr[B].median())
y = tr.rainfall.values; yo = og.rainfall.values
years = np.arange(len(tr)) // 365
print(tr.shape, te.shape, og.shape, flush=True)


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


def tabpfn(n):
    def mk(s):
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device=DEV, n_estimators=n, random_state=s, ignore_pretraining_limits=True)
    return mk


TOP = ['cloud', 'sunshine', 'humidity', 'dewpoint', 'pressure', 'windspeed', 'temparature', 'dew_dep']


def tabicl(n):
    def mk(s):
        from tabicl import TabICLClassifier
        return TabICLClassifier(n_estimators=n, random_state=s, device=DEV)
    return mk


def realmlp(s):
    from pytabkit import RealMLP_TD_Classifier
    return RealMLP_TD_Classifier(device=DEV, random_state=s, verbosity=0)


def tabm(s):
    from pytabkit import TabM_D_Classifier
    return TabM_D_Classifier(device=DEV, random_state=s, verbosity=0)


M = {  # name: (cols, factory, cv_seeds, test_seeds, use_orig, do_loyo)
    'tabpfn_raw': (RAW, tabpfn(16), 1, 3, False, True),
    'tabpfn_eng': (ENG, tabpfn(16), 1, 3, False, True),
    'tabpfn_raw_og': (RAW, tabpfn(16), 1, 3, True, True),
    'tabpfn_eng_og': (ENG, tabpfn(16), 1, 3, True, True),
    'realmlp': (ENG, realmlp, 1, 5, False, False),
    'realmlp_og': (ENG, realmlp, 1, 5, True, False),
    'tabm': (ENG, tabm, 1, 5, False, False),
    'tabm_og': (ENG, tabm, 1, 5, True, False),
}


def fit_pred(name, trn, Xev, ns):
    cols, mk, _, _, use_og, _ = M[name]
    Xf = Xa.iloc[trn][cols].reset_index(drop=True); yf = y[trn]
    if use_og:
        Xf = pd.concat([Xf, Xo[cols]], ignore_index=True); yf = np.r_[yf, yo]
    return np.mean([mk(s).fit(Xf, yf).predict_proba(Xev[cols].reset_index(drop=True))[:, 1] for s in range(ns)], axis=0)


R = 3
rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=42).split(Xa, y))
T0 = time.time()
for name in M:
    try:
        t0 = time.time(); _, _, cvs, tes, _, do_loyo = M[name]
        oof = np.zeros((R, len(y)))
        for i, (a, b) in enumerate(rskf):
            oof[i // 5, b] = fit_pred(name, a, Xa.iloc[b], cvs)
        msg = f"{name:14s} RSKF {np.mean([roc_auc_score(y, o) for o in oof]):.5f}"
        if do_loyo:
            lo = np.zeros(len(y))
            for a, b in LeaveOneGroupOut().split(Xa, y, years):
                lo[b] = fit_pred(name, a, Xa.iloc[b], cvs)
            msg += f"  LOYO {np.mean([roc_auc_score(y[years == g], lo[years == g]) for g in range(6)]):.5f}"
            np.save(f'loyo_{name}.npy', lo)
        np.save(f'oof_{name}.npy', oof)
        np.save(f'test_{name}.npy', fit_pred(name, np.arange(len(y)), Xt, tes))
        print(msg, f" {time.time() - t0:.0f}s", flush=True)
    except Exception:
        print(name, 'FAILED'); traceback.print_exc(); sys.stdout.flush()
    if time.time() - T0 > 3 * 3600:
        print('time budget hit'); break
print('done', time.time() - T0)
