# %% [cell 1]
import os, glob, json, math, time, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import display, Markdown

from scipy import stats
from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as e:
    optuna = None; print("optuna not available -> tuning disabled:", e)
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(max(1, os.cpu_count() or 1)); TORCH_OK = True
except Exception as e:
    TORCH_OK = False; print("torch not available -> the neural-net model is disabled:", e)

warnings.filterwarnings("ignore")
print({k: v.__version__ for k, v in dict(pandas=pd, numpy=np, lightgbm=lgb, xgboost=xgb, catboost=cb).items()}, "| torch:", TORCH_OK)

# %% [cell 2]
# ---- run profile: quick (sanity check, minutes) | standard (default) | max (longest). Env overrides always win.
FAST = os.environ.get("RAIN_FAST", "0") == "1"
PROFILE = os.environ.get("RAIN_PROFILE", "quick" if FAST else "standard")
PROF = {"quick":    dict(rep=1,  trials=0,   timeout=0,   abl=1),
        "standard": dict(rep=6,  trials=60,  timeout=110, abl=2),
        "max":      dict(rep=12, trials=200, timeout=300, abl=3)}[PROFILE]

class CFG:
    seed          = 42
    n_splits      = 5
    n_repeats     = int(os.environ.get("RAIN_REPEATS", PROF["rep"]))
    orig_rep      = 1                                                   # copies of the original data per fold for "plain" training (CV-tuned below)
    optuna_trials = int(os.environ.get("RAIN_TRIALS", PROF["trials"]))
    optuna_timeout= int(os.environ.get("RAIN_TIMEOUT", PROF["timeout"]))
    use_optuna    = (optuna is not None) and PROFILE != "quick" and int(os.environ.get("RAIN_TRIALS", PROF["trials"])) > 0
    tune_seed     = 1000
    n_jobs        = -1

def seed_everything(s=CFG.seed):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"] = str(s)
    if TORCH_OK: torch.manual_seed(s)
seed_everything()
ABL_REP = PROF["abl"]
print(f"profile={PROFILE} | repeats={CFG.n_repeats} | optuna={CFG.use_optuna} ({CFG.optuna_trials} trials x {CFG.optuna_timeout}s)")

# ---- plotting style (validated categorical palette, one-hue sequential, blue<->red diverging with a neutral midpoint)
INK, INK2, GRID, SURFACE, MID = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#f0efec"
RAIN, DRY, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_br", ["#e34948", MID, "#2a78d6"])
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "legend.frameon": False, "figure.dpi": 100, "font.size": 10})

OUT_DIR = Path(os.environ.get("RAIN_OUT") or ("/kaggle/working" if Path("/kaggle/working").exists() else "output"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [cell 3]
def locate(*names):
    dirs = [os.environ.get("RAIN_DATA_DIR"), "/kaggle/input/rainfall-probability-prediction-challenge", "/kaggle/input/playground-series-s5e3", "data", "."]
    for name in names:
        for d in dirs:
            if d and (Path(d) / name).exists(): return Path(d) / name
    for name in names:
        for pat in (f"/kaggle/input/**/{name}", f"**/{name}"):
            hits = glob.glob(pat, recursive=True)
            if hits: return Path(hits[0])
    return None

P_TRAIN, P_TEST, P_SAMPLE = locate("train.csv"), locate("test.csv"), locate("sample_submission.csv")
P_ORIG = locate("Rainfall.csv", "rainfall.csv")
if P_TRAIN is None or P_TEST is None:
    raise FileNotFoundError("train.csv / test.csv not found. Attach the competition data (Add Input) or put them in ./data")
print("train:", P_TRAIN, "\ntest :", P_TEST, "\nsample:", P_SAMPLE, "\norig :", P_ORIG)

TARGET = "rainfall"
BASE = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]

train, test = pd.read_csv(P_TRAIN), pd.read_csv(P_TEST)
sample = pd.read_csv(P_SAMPLE) if P_SAMPLE else None
print("train", train.shape, "| test", test.shape, "| rain rate: %.4f" % train[TARGET].mean())

orig = None
if P_ORIG is not None:
    orig = pd.read_csv(P_ORIG); orig.columns = orig.columns.str.strip()
    if orig[TARGET].dtype == object:
        orig[TARGET] = orig[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    orig = orig.dropna(subset=[TARGET]).reset_index(drop=True); orig[TARGET] = orig[TARGET].astype(int)
    orig["day"] = np.arange(1, len(orig) + 1)                              # true day-of-year (row order is chronological)
    orig[BASE] = orig[BASE].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    print("original", orig.shape, "| rain rate: %.4f" % orig[TARGET].mean())
else:
    print("Original dataset not attached -> pretraining is skipped; every model trains the plain way (still fully functional).")

train["is_test"] = 0; test["is_test"] = 1; test[TARGET] = np.nan
comp = pd.concat([train, test], ignore_index=True).sort_values("id").reset_index(drop=True)
comp["doy"] = comp["day"].clip(1, 365).astype(int); comp["year_idx"] = comp["id"] // 365
comp[BASE] = comp[BASE].interpolate(limit_direction="both")
if orig is not None:
    orig["doy"] = orig["day"].clip(1, 365).astype(int); orig["year_idx"] = -1
display(comp.drop(columns="is_test").head(4))

# %% [cell 4]
tr = comp[comp.is_test == 0]
fig = plt.figure(figsize=(15, 4.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1.3])
ax = fig.add_subplot(gs[0]); ax.grid(False)
rate = tr[TARGET].mean()
ax.pie([rate, 1 - rate], colors=[RAIN, DRY], startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor=SURFACE, linewidth=2))
ax.text(0, 0, f"{rate:.1%}\nrain", ha="center", va="center", fontsize=15, fontweight="bold"); ax.set_title("Target balance (train)")

ax = fig.add_subplot(gs[1])
FEATS_ = BASE + ["doy"]; y_ = tr[TARGET].astype(int).values
mi_ = mutual_info_classif(tr[FEATS_].values, y_, n_neighbors=7, random_state=0)
sig = pd.Series({c: roc_auc_score(y_, tr[c]) for c in FEATS_}).sub(0.5).abs().mul(2).sort_values()
ax.barh(sig.index, sig.values, color=RAIN, height=0.7); ax.set_title("Signal strength per feature  |2·AUC-1|"); ax.set_xlabel("separation power")

ax = fig.add_subplot(gs[2])
def adversarial(a, b, cols, seed=0):
    X = pd.concat([a[cols], b[cols]]).values; t = np.r_[np.zeros(len(a)), np.ones(len(b))]; oof = np.zeros(len(t))
    for tri, vai in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, t):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=30, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=-1).fit(X[tri], t[tri])
        oof[vai] = m.predict_proba(X[vai])[:, 1]
    return roc_auc_score(t, oof)

te_ = comp[comp.is_test == 1]
advs = {"train vs test": adversarial(tr, te_, FEATS_)}
if orig is not None: advs["train vs original"] = adversarial(tr, orig, FEATS_)
ax.bar(list(advs.keys()), list(advs.values()), color=[RAIN, GREEN][:len(advs)], width=0.5)
ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.set_ylim(0.4, 1.0); ax.set_title("Adversarial AUC (0.5 = indistinguishable)")
plt.tight_layout(); plt.show()
if orig is not None:
    print(f"train vs original adversarial AUC = {advs['train vs original']:.3f}  -> "
          + ("datasets differ somewhat: a pretrain->fine-tune split is worth testing" if advs['train vs original'] > 0.58
             else "datasets look similar: fine-tuning may only match plain pooling, but we still test both"))

# %% [cell 5]
def es(t): return 6.112 * np.exp(17.67 * t / (t + 243.5))                       # saturation vapour pressure (hPa, Magnus formula)

def static_features(d):
    d = d.copy(); ang = 2 * np.pi * d["doy"] / 365.0
    d["doy_sin1"], d["doy_cos1"] = np.sin(ang), np.cos(ang)
    d["doy_sin2"], d["doy_cos2"] = np.sin(2 * ang), np.cos(2 * ang)
    d["temp_range"] = d.maxtemp - d.mintemp; d["dew_dep"] = d.temparature - d.dewpoint; d["dew_dep_min"] = d.mintemp - d.dewpoint
    d["vp_deficit"] = es(d.temparature) - es(d.dewpoint); d["rh_calc"] = 100 * es(d.dewpoint) / es(d.temparature)
    d["hum_x_cloud"] = d.humidity * d.cloud / 100; d["cloud_m_sun"] = d.cloud / 100 - d.sunshine / 12; d["sun_per_cloud"] = d.sunshine / (d.cloud + 1)
    d["wind_u"] = d.windspeed * np.cos(np.deg2rad(d.winddirection)); d["wind_v"] = d.windspeed * np.sin(np.deg2rad(d.winddirection))
    return d

ANOM_VARS = ["cloud", "sunshine", "humidity", "dewpoint", "pressure", "temparature"]
def climatology_anomaly(d, cols=ANOM_VARS, win=15):
    out = {}
    for c in cols:
        m = d.groupby("doy")[c].mean().reindex(range(1, 366)).interpolate(limit_direction="both")
        ext = pd.concat([m.iloc[-win:], m, m.iloc[:win]]); sm = ext.rolling(win, center=True, min_periods=1).mean().iloc[win:-win]; sm.index = m.index
        out[f"{c}_anom"] = d[c] - d["doy"].map(sm)
    return pd.DataFrame(out, index=d.index)

SEQ_VARS = ["pressure", "temparature", "dewpoint", "humidity", "cloud", "sunshine", "windspeed", "dew_dep"]
def sequence_features(d):
    new = {}
    for c in SEQ_VARS:
        s = d[c]
        for k in (1, 2): new[f"{c}_lag{k}"] = s.shift(k).fillna(s); new[f"{c}_lead{k}"] = s.shift(-k).fillna(s)
    for c in ["pressure", "temparature", "dewpoint", "humidity", "cloud", "sunshine"]:
        new[f"{c}_d_prev"] = d[c] - new[f"{c}_lag1"]; new[f"{c}_d_next"] = new[f"{c}_lead1"] - d[c]
    for c in ["cloud", "sunshine", "humidity", "dewpoint", "pressure"]:
        for w in (3, 5, 7): new[f"{c}_rm{w}"] = d[c].rolling(w, center=True, min_periods=1).mean()
        new[f"{c}_rs5"] = d[c].rolling(5, center=True, min_periods=2).std().fillna(0)
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)

PAIR_DRIVERS = ["cloud", "sunshine", "humidity", "pressure", "dewpoint"]
ZREF = comp[PAIR_DRIVERS].agg(["mean", "std"])
def pair_features(d):
    import itertools
    z = (d[PAIR_DRIVERS] - ZREF.loc["mean"]) / ZREF.loc["std"]; new = {}
    for a, b in itertools.combinations(PAIR_DRIVERS, 2): new[f"{a}_x_{b}"] = z[a] * z[b]; new[f"{a}_m_{b}"] = z[a] - z[b]
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)

def build(frame):
    s0 = frame.copy(); s1 = static_features(s0); s2 = pd.concat([s1, climatology_anomaly(s1)], axis=1)
    s3 = sequence_features(s2); s4 = pair_features(s3)
    cols = dict(static=[c for c in s1.columns if c not in s0.columns], anom=[c for c in s2.columns if c not in s1.columns],
                seq=[c for c in s3.columns if c not in s2.columns], pairs=[c for c in s4.columns if c not in s3.columns])
    return s4, cols

FEAT, BLOCKS = build(comp)
FEAT_O = build(orig)[0] if orig is not None else None
BASE_COLS = BASE + ["doy"]
FEATURE_SETS = {
    "A base":          BASE_COLS,
    "B +static":       BASE_COLS + BLOCKS["static"],
    "C +anomaly":      BASE_COLS + BLOCKS["static"] + BLOCKS["anom"],
    "D +sequence":     BASE_COLS + BLOCKS["static"] + BLOCKS["anom"] + BLOCKS["seq"],
    "E +pairs (all)":  BASE_COLS + BLOCKS["static"] + BLOCKS["anom"] + BLOCKS["seq"] + BLOCKS["pairs"],
}
assert FEAT[FEATURE_SETS["E +pairs (all)"]].isna().sum().sum() == 0, "NaNs left in features"
print({k: len(v) for k, v in FEATURE_SETS.items()})

is_te = FEAT["is_test"].values == 1
Y_TR = FEAT.loc[~is_te, TARGET].astype(int).values
Y_OR = FEAT_O[TARGET].astype(int).values if FEAT_O is not None else None
GROUPS = FEAT.loc[~is_te, "year_idx"].values

def get_arrays(cols):
    Xtr = FEAT.loc[~is_te, cols].values.astype(np.float64)
    Xte = FEAT.loc[is_te, cols].values.astype(np.float64)
    Xor = FEAT_O[cols].values.astype(np.float64) if FEAT_O is not None else None
    return Xtr, Xte, Xor

# %% [cell 6]
def score_of(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

def run_plain_cv(make, params, cols, seed_base, n_repeats, n_splits=5, orig_rep=None):
    """Simple (non fine-tuning) repeated CV, used for the ablation and the original-data weight sweep."""
    rep = CFG.orig_rep if orig_rep is None else orig_rep
    Xtr, Xte, Xor = get_arrays(cols)
    oof = np.zeros((n_repeats, len(Xtr))); te = np.zeros(len(Xte)); fold_auc = []
    for r in range(n_repeats):
        for f, (a, b) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed_base + r).split(Xtr, Y_TR)):
            Xa, ya = Xtr[a], Y_TR[a]
            if rep > 0 and Xor is not None: Xa, ya = np.vstack([Xa] + [Xor] * rep), np.r_[ya, np.tile(Y_OR, rep)]
            m = make(seed=seed_base + 10 * r + f, **params).fit(Xa, ya)
            oof[r, b] = score_of(m, Xtr[b]); te += score_of(m, Xte) / (n_repeats * n_splits); fold_auc.append(roc_auc_score(Y_TR[b], oof[r, b]))
    rep_auc = [roc_auc_score(Y_TR, oof[r]) for r in range(n_repeats)]
    return dict(oof=oof.mean(0), test=te, auc=float(np.mean(rep_auc)), std=float(np.std(rep_auc)), fold_mean=float(np.mean(fold_auc)))

def make_lr(seed=0, C=0.03):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
def make_lgbm(seed=0, learning_rate=0.02, n_estimators=400, num_leaves=6, max_depth=3, min_child_samples=40, subsample=0.7, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.1):
    return lgb.LGBMClassifier(objective="binary", learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves, max_depth=max_depth,
                              min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
                              reg_lambda=reg_lambda, reg_alpha=reg_alpha, random_state=seed, n_jobs=-1, verbose=-1)

abl_rows = []
for fs_name, cols in FEATURE_SETS.items():
    for mname, make, params in [("LogReg", make_lr, dict(C=0.03)), ("LightGBM", make_lgbm, dict())]:
        r = run_plain_cv(make, params, cols, seed_base=CFG.seed + 500, n_repeats=ABL_REP)
        abl_rows.append(dict(feature_set=fs_name, n_feat=len(cols), model=mname, auc=r["auc"], std=r["std"]))
ABL = pd.DataFrame(abl_rows)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, mname, colr in zip(axes, ["LogReg", "LightGBM"], [RAIN, DRY]):
    d = ABL[ABL.model == mname].reset_index(drop=True)
    ax.errorbar(d.auc, np.arange(len(d)), xerr=d["std"], fmt="o", color=colr, ms=10, mec=SURFACE, mew=2, capsize=3)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([f"{r.feature_set} ({r.n_feat})" for r in d.itertuples()]); ax.invert_yaxis()
    ax.set_xlabel("CV ROC-AUC"); ax.set_title(mname)
fig.suptitle("Ablation: feature blocks", x=0.01, ha="left", fontsize=13, fontweight="bold"); plt.tight_layout(); plt.show()

def pick_best(model_name):
    d = ABL[ABL.model == model_name]; top = d.auc.max(); ok = d[d.auc >= top - 0.001]
    return ok.sort_values("n_feat").iloc[0]["feature_set"]
FS_LIN, FS_TREE = pick_best("LogReg"), pick_best("LightGBM")
FS = {"lin": FEATURE_SETS[FS_LIN], "tree": FEATURE_SETS[FS_TREE], "spl": FEATURE_SETS["B +static"]}
print(f"feature set for linear/kernel/NN models  : {FS_LIN} ({len(FS['lin'])})\nfeature set for tree models              : {FS_TREE} ({len(FS['tree'])})\nfeature set for spline/linear-leaf models: B +static ({len(FS['spl'])})")

# %% [cell 7]
if FEAT_O is not None:
    rows_ = []
    for rep in (0, 1, 2, 3):
        for mname, make, params in [("LogReg", make_lr, dict(C=0.03)), ("LightGBM", make_lgbm, dict())]:
            r = run_plain_cv(make, params, FS[{"LogReg": "lin", "LightGBM": "tree"}[mname]], seed_base=CFG.seed + 600, n_repeats=ABL_REP, orig_rep=rep)
            rows_.append(dict(orig_rep=rep, model=mname, auc=r["auc"]))
    REPDF = pd.DataFrame(rows_); m_ = REPDF.groupby("orig_rep").auc.mean()
    CFG.orig_rep = int(m_[m_ >= m_.max() - 0.0005].index.min())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for mname, colr in zip(["LogReg", "LightGBM"], [RAIN, DRY]):
        d = REPDF[REPDF.model == mname]; ax.plot(d.orig_rep, d.auc, "o-", color=colr, lw=2.5, ms=9, mec=SURFACE, mew=2, label=mname)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("copies of the original dataset per training fold"); ax.set_ylabel("CV AUC")
    ax.set_title(f"Original-data weight for PLAIN training -> orig_rep = {CFG.orig_rep}"); ax.legend()
    plt.tight_layout(); plt.show()
else:
    CFG.orig_rep = 0
print("orig_rep (plain-training weight) =", CFG.orig_rep)

# %% [cell 8]
from sklearn.inspection import permutation_importance

Xtr_full, _, _ = get_arrays(FS["tree"])
a_imp, b_imp = next(StratifiedKFold(5, shuffle=True, random_state=CFG.seed + 900).split(Xtr_full, Y_TR))
m_imp = make_lgbm(seed=CFG.seed).fit(Xtr_full[a_imp], Y_TR[a_imp])
try:
    pi = permutation_importance(m_imp, Xtr_full[b_imp], Y_TR[b_imp], scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=-1)
except Exception as e:
    print("parallel permutation importance unavailable -> falling back to single-threaded:", repr(e)[:100])
    pi = permutation_importance(m_imp, Xtr_full[b_imp], Y_TR[b_imp], scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=FS["tree"]).sort_values()
FS_TREE_PRUNED = [c for c, v in imp.items() if v > 0] or FS["tree"]                     # never prune down to nothing

r_full = run_plain_cv(make_lgbm, dict(), FS["tree"], seed_base=CFG.seed + 950, n_repeats=ABL_REP)
r_pruned = run_plain_cv(make_lgbm, dict(), FS_TREE_PRUNED, seed_base=CFG.seed + 950, n_repeats=ABL_REP)
USE_PRUNED = r_pruned["auc"] >= r_full["auc"] - 0.0003
if USE_PRUNED: FS["tree"] = FS_TREE_PRUNED

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
bottom = imp.head(18)
axes[0].barh(bottom.index, bottom.values, color=[DRY if v <= 0 else RAIN for v in bottom.values], height=0.7)
axes[0].axvline(0, color=INK2, lw=1); axes[0].set_title("Weakest 18 features by permutation importance"); axes[0].set_xlabel("AUC drop when shuffled")
axes[1].bar(["full set", "pruned set"], [r_full["auc"], r_pruned["auc"]], color=[DRY, RAIN], width=0.5)
axes[1].set_ylim(min(r_full["auc"], r_pruned["auc"]) - 0.004, max(r_full["auc"], r_pruned["auc"]) + 0.003)
axes[1].set_title(f"Full ({len(imp)} feat) vs pruned ({len(FS_TREE_PRUNED)} feat)\n-> {'PRUNED SET KEPT' if USE_PRUNED else 'full set kept (pruning did not help)'}")
plt.tight_layout(); plt.show()
print(f"tree feature set: {len(FS['tree'])} columns" + (" (pruned)" if USE_PRUNED else " (full, unpruned)"))

# %% [cell 9]
# ============================== gradient-boosting fine-tuning wrappers ==============================
def split_stratified_internal(X, y, frac, seed):
    """Carve a small internal validation slice OUT OF THE FOLD'S OWN TRAINING ROWS (never the CV validation fold)
    so the fine-tune stage can early-stop on real held-out signal instead of a fixed, guessed round count."""
    X, y = np.asarray(X), np.asarray(y)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return X, X[:0], y, y[:0]
    return train_test_split(X, y, test_size=frac, random_state=seed, stratify=y)

class LGBMFineTune:
    def __init__(self, seed=0, Xor=None, yor=None,
                 pre_n_estimators=200, pre_learning_rate=0.05, pre_num_leaves=8, pre_min_child_samples=15,
                 n_estimators=150, learning_rate=0.02, num_leaves=6, max_depth=3, min_child_samples=40,
                 subsample=0.7, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.1,
                 early_stop_frac=0.15, early_stop_rounds=25):
        self.seed, self.Xor, self.yor = seed, Xor, yor
        self.early_stop_frac, self.early_stop_rounds = early_stop_frac, early_stop_rounds
        self.pre = dict(n_estimators=pre_n_estimators, learning_rate=pre_learning_rate, num_leaves=pre_num_leaves,
                        min_child_samples=pre_min_child_samples, subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0)
        self.fin = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth,
                        min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree,
                        reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    def fit(self, X, y):
        kw = dict(n_jobs=THREADS, verbose=-1)
        if self.Xor is None or len(self.Xor) == 0:
            self.m = lgb.LGBMClassifier(random_state=self.seed, **kw, **self.fin).fit(X, y); return self
        m1 = lgb.LGBMClassifier(random_state=self.seed, **kw, **self.pre).fit(self.Xor, self.yor)
        Xf, Xv, yf, yv = split_stratified_internal(X, y, self.early_stop_frac, self.seed)
        if len(yv) < 6 or len(np.unique(yv)) < 2:                                    # too few rows to early-stop safely -> fixed rounds
            self.m = lgb.LGBMClassifier(random_state=self.seed + 1, **kw, **self.fin).fit(X, y, init_model=m1.booster_); return self
        fin = dict(self.fin); fin["n_estimators"] = max(fin["n_estimators"], 40) * 3      # generous cap; early stopping finds the real count
        self.m = lgb.LGBMClassifier(random_state=self.seed + 1, **kw, **fin).fit(
            Xf, yf, init_model=m1.booster_, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(self.early_stop_rounds, verbose=False)])
        return self
    def predict_proba(self, X): return self.m.predict_proba(X)

class XGBFineTune:
    def __init__(self, seed=0, Xor=None, yor=None,
                 pre_n_estimators=200, pre_learning_rate=0.05, pre_max_depth=3,
                 n_estimators=120, learning_rate=0.02, max_depth=3, min_child_weight=8, subsample=0.7,
                 colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.1, gamma=0.0,
                 early_stop_frac=0.15, early_stop_rounds=25):
        self.seed, self.Xor, self.yor = seed, Xor, yor
        self.early_stop_frac, self.early_stop_rounds = early_stop_frac, early_stop_rounds
        self.pre = dict(n_estimators=pre_n_estimators, learning_rate=pre_learning_rate, max_depth=pre_max_depth, min_child_weight=2,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
        self.fin = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                        subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha, gamma=gamma)
    def fit(self, X, y):
        kw = dict(tree_method="hist", eval_metric="auc", n_jobs=THREADS, verbosity=0)
        if self.Xor is None or len(self.Xor) == 0:
            self.m = xgb.XGBClassifier(random_state=self.seed, **kw, **self.fin).fit(X, y); return self
        m1 = xgb.XGBClassifier(random_state=self.seed, **kw, **self.pre).fit(self.Xor, self.yor)
        Xf, Xv, yf, yv = split_stratified_internal(X, y, self.early_stop_frac, self.seed)
        if len(yv) < 6 or len(np.unique(yv)) < 2:
            self.m = xgb.XGBClassifier(random_state=self.seed + 1, **kw, **self.fin).fit(X, y, xgb_model=m1.get_booster()); return self
        fin = dict(self.fin); fin["n_estimators"] = max(fin["n_estimators"], 40) * 3
        self.m = xgb.XGBClassifier(random_state=self.seed + 1, early_stopping_rounds=self.early_stop_rounds, **kw, **fin).fit(
            Xf, yf, xgb_model=m1.get_booster(), eval_set=[(Xv, yv)], verbose=False)
        return self
    def predict_proba(self, X): return self.m.predict_proba(X)

class CatFineTune:
    def __init__(self, seed=0, Xor=None, yor=None,
                 pre_iterations=200, pre_learning_rate=0.06, pre_depth=4,
                 iterations=150, learning_rate=0.03, depth=4, l2_leaf_reg=8.0, rsm=0.6, subsample=0.8,
                 early_stop_frac=0.15, early_stop_rounds=25):
        self.seed, self.Xor, self.yor = seed, Xor, yor
        self.early_stop_frac, self.early_stop_rounds = early_stop_frac, early_stop_rounds
        self.pre = dict(iterations=pre_iterations, learning_rate=pre_learning_rate, depth=pre_depth, l2_leaf_reg=2.0, rsm=0.9, subsample=0.9)
        self.fin = dict(iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg, rsm=rsm, subsample=subsample)
    def fit(self, X, y):
        kw = dict(bootstrap_type="Bernoulli", verbose=0, thread_count=THREADS, allow_writing_files=False)
        if self.Xor is None or len(self.Xor) == 0:
            self.m = cb.CatBoostClassifier(random_seed=self.seed, **kw, **self.fin).fit(X, y); return self
        m1 = cb.CatBoostClassifier(random_seed=self.seed, **kw, **self.pre).fit(self.Xor, self.yor)
        Xf, Xv, yf, yv = split_stratified_internal(X, y, self.early_stop_frac, self.seed)
        if len(yv) < 6 or len(np.unique(yv)) < 2:
            self.m = cb.CatBoostClassifier(random_seed=self.seed + 1, **kw, **self.fin).fit(X, y, init_model=m1); return self
        fin = dict(self.fin); fin["iterations"] = max(fin["iterations"], 40) * 3
        self.m = cb.CatBoostClassifier(random_seed=self.seed + 1, early_stopping_rounds=self.early_stop_rounds, **kw, **fin).fit(
            Xf, yf, init_model=m1, eval_set=(Xv, yv))
        return self
    def predict_proba(self, X): return self.m.predict_proba(X)

print("LGBMFineTune / XGBFineTune / CatFineTune defined (each with early-stopped, warm-started fine-tuning)")

# %% [cell 10]
# ============================== small PyTorch net: plain trainer + fine-tuner ==============================
NN_DEVICE = os.environ.get("RAIN_NN_DEVICE", "cpu")

if TORCH_OK:
    class _MLP(nn.Module):
        def __init__(self, d_in, hidden=48, depth=2, dropout=0.2):
            super().__init__()
            dims = [d_in] + [hidden] * depth
            self.body = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
            self.out = nn.Linear(hidden, 1); self.drop = nn.Dropout(dropout)
        def forward(self, x):
            for lin in self.body: x = self.drop(F.silu(lin(x)))
            return self.out(x).squeeze(-1)

    def _train_loop(net, Xs, ys, epochs, lr, wd, batch, noise, device):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        n = len(Xs); spe = max(1, n // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * spe))
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for s in range(spe):
                idx = perm[s * batch:(s + 1) * batch]
                if len(idx) == 0: continue
                xb, yb = Xs[idx], ys[idx]
                if noise > 0: xb = xb + noise * torch.randn_like(xb)
                loss = F.binary_cross_entropy_with_logits(net(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return net

    def _train_loop_val(net, Xs, ys, Xv, yv_raw, epochs, lr, wd, batch, noise, device, patience=8):
        """Like _train_loop, but after every epoch it checks AUC on a held-out slice (never the CV fold) and
        keeps the best-scoring weights instead of just whatever the last epoch produced — an early-stopping
        equivalent for the neural net's fine-tune stage."""
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        n = len(Xs); spe = max(1, n // batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * spe))
        yv_bin = yv_raw.cpu().numpy() > 0.5
        best_state, best_auc, bad = None, -1.0, 0
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, device=device)
            for s in range(spe):
                idx = perm[s * batch:(s + 1) * batch]
                if len(idx) == 0: continue
                xb, yb = Xs[idx], ys[idx]
                if noise > 0: xb = xb + noise * torch.randn_like(xb)
                loss = F.binary_cross_entropy_with_logits(net(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xv)).cpu().numpy()
            auc_v = roc_auc_score(yv_bin, pv) if len(np.unique(yv_bin)) == 2 else 0.5
            if auc_v > best_auc:
                best_auc = auc_v; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= patience: break
        if best_state is not None: net.load_state_dict(best_state)
        return net

    class TorchMLP:
        """Plain trainer (used for the 'plain' twin) and, via TorchMLPFineTune, a fine-tuner."""
        def __init__(self, seed=0, hidden=48, depth=2, dropout=0.2, wd=0.05, lr=3e-3, epochs=45, batch=64, noise=0.05, smooth=0.02):
            self.seed, self.hidden, self.depth, self.dropout = seed, hidden, depth, dropout
            self.wd, self.lr, self.epochs, self.batch, self.noise, self.smooth = wd, lr, epochs, batch, noise, smooth
        def _prep_fit(self, X):
            self.pre = make_pipeline(SimpleImputer(strategy="median"),
                                     QuantileTransformer(n_quantiles=min(200, max(10, len(X))), output_distribution="normal", random_state=self.seed))
            return torch.tensor(self.pre.fit_transform(np.asarray(X, np.float32)), dtype=torch.float32, device=NN_DEVICE)
        def fit(self, X, y):
            torch.manual_seed(self.seed)
            Xs = self._prep_fit(X); ys = torch.tensor(np.asarray(y), dtype=torch.float32, device=NN_DEVICE) * (1 - 2 * self.smooth) + self.smooth
            self.net = _MLP(Xs.shape[1], self.hidden, self.depth, self.dropout).to(NN_DEVICE)
            _train_loop(self.net, Xs, ys, self.epochs, self.lr, self.wd, self.batch, self.noise, NN_DEVICE)
            self.net.eval(); return self
        def predict_proba(self, X):
            xs = torch.tensor(self.pre.transform(np.asarray(X, np.float32)), dtype=torch.float32, device=NN_DEVICE)
            with torch.no_grad(): p = torch.sigmoid(self.net(xs)).cpu().numpy()
            return np.c_[1 - p, p]

    class TorchMLPFineTune(TorchMLP):
        """Pretrain on the original dataset, then continue training (fine-tune) on the fold's competition rows
        at a reduced learning rate, with the first layer optionally frozen."""
        def __init__(self, seed=0, Xor=None, yor=None, hidden=48, depth=2, dropout=0.2, wd=0.05, lr=3e-3, epochs=45,
                     pre_epochs=60, pre_lr=5e-3, ft_lr_mult=0.3, ft_epochs=20, freeze_first=True, batch=64, noise=0.05, smooth=0.02):
            super().__init__(seed=seed, hidden=hidden, depth=depth, dropout=dropout, wd=wd, lr=lr, epochs=epochs, batch=batch, noise=noise, smooth=smooth)
            self.Xor, self.yor = Xor, yor
            self.pre_epochs, self.pre_lr, self.ft_lr_mult, self.ft_epochs, self.freeze_first = pre_epochs, pre_lr, ft_lr_mult, ft_epochs, freeze_first
        def fit(self, X, y):
            torch.manual_seed(self.seed)
            if self.Xor is None or len(self.Xor) == 0:
                return super().fit(X, y)
            Xs_or = self._prep_fit(self.Xor)                                          # normalisation is fitted on the PRETRAIN data, kept fixed for fine-tuning
            ys_or = torch.tensor(np.asarray(self.yor), dtype=torch.float32, device=NN_DEVICE) * (1 - 2 * self.smooth) + self.smooth
            self.net = _MLP(Xs_or.shape[1], self.hidden, self.depth, self.dropout).to(NN_DEVICE)
            _train_loop(self.net, Xs_or, ys_or, self.pre_epochs, self.pre_lr, self.wd, self.batch, self.noise, NN_DEVICE)
            if self.freeze_first:
                for p in self.net.body[0].parameters(): p.requires_grad_(False)
            Xf, Xv, yf, yv = split_stratified_internal(X, y, 0.15, self.seed)          # internal carve from the fold's OWN training rows only
            Xf_t = torch.tensor(self.pre.transform(np.asarray(Xf, np.float32)), dtype=torch.float32, device=NN_DEVICE)
            yf_t = torch.tensor(yf, dtype=torch.float32, device=NN_DEVICE) * (1 - 2 * self.smooth) + self.smooth
            if len(yv) >= 6 and len(np.unique(yv)) == 2:
                Xv_t = torch.tensor(self.pre.transform(np.asarray(Xv, np.float32)), dtype=torch.float32, device=NN_DEVICE)
                yv_t = torch.tensor(yv, dtype=torch.float32, device=NN_DEVICE)
                _train_loop_val(self.net, Xf_t, yf_t, Xv_t, yv_t, self.ft_epochs, self.lr * self.ft_lr_mult, self.wd, self.batch, self.noise, NN_DEVICE)
            else:                                                                      # too few rows to carve a validation slice -> fixed-epoch fine-tune
                _train_loop(self.net, Xf_t, yf_t, self.ft_epochs, self.lr * self.ft_lr_mult, self.wd, self.batch, self.noise, NN_DEVICE)
            for p in self.net.parameters(): p.requires_grad_(True)
            self.net.eval(); return self
    print("TorchMLP / TorchMLPFineTune defined")
else:
    print("torch unavailable -> the neural-net model is skipped everywhere below")

# %% [cell 11]
THREADS = -1                                                       # single-fold-at-a-time on this box; each model may use all cores

def augment(Xa, ya, Xor, yor, rep):
    if rep <= 0 or Xor is None: return Xa, ya
    return np.vstack([Xa] + [Xor] * rep), np.r_[ya, np.tile(yor, rep)]

def _fold_fit(make, params, Xtr, ytr, Xor, yor, a, seed, needs_orig, orig_rep):
    if needs_orig:
        return make(seed=seed, Xor=Xor, yor=yor, **params).fit(Xtr[a], ytr[a])
    Xa, ya = augment(Xtr[a], ytr[a], Xor, yor, orig_rep)
    return make(seed=seed, **params).fit(Xa, ya)

def run_cv(make, params, cols, seed_base, n_repeats, n_splits=5, needs_orig=False, orig_rep=None):
    rep = CFG.orig_rep if orig_rep is None else orig_rep
    Xtr, Xte, Xor = get_arrays(cols)
    oof = np.zeros((n_repeats, len(Xtr))); te = np.zeros(len(Xte)); fold_auc = []
    for r in range(n_repeats):
        for f, (a, b) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed_base + r).split(Xtr, Y_TR)):
            m = _fold_fit(make, params, Xtr, Y_TR, Xor, Y_OR, a, seed_base + 10 * r + f, needs_orig, rep)
            pv = score_of(m, Xtr[b]); oof[r, b] = pv; te += score_of(m, Xte) / (n_repeats * n_splits); fold_auc.append(roc_auc_score(Y_TR[b], pv))
    rep_auc = [roc_auc_score(Y_TR, oof[r]) for r in range(n_repeats)]; o = oof.mean(0)
    return dict(oof=o, test=te, auc=float(np.mean(rep_auc)), std=float(np.std(rep_auc)), fold_mean=float(np.mean(fold_auc)), auc_avg=float(roc_auc_score(Y_TR, o)))

def quick_cv(make, params, cols, seed_base, needs_orig=False, orig_rep=None):
    return run_cv(make, params, cols, seed_base, 1, needs_orig=needs_orig, orig_rep=orig_rep)["fold_mean"]

# %% [cell 12]
S = lambda t, n, lo, hi, log=False: t.suggest_float(n, lo, hi, log=log)
I = lambda t, n, lo, hi: t.suggest_int(n, lo, hi)

def make_lr_spline(seed=0, C=0.02, n_knots=5):
    return make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=n_knots, degree=3), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
def make_svm(seed=0, C=1.0, gamma=0.005):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf", probability=True))
def make_knn(seed=0, n_neighbors=60, p=2):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", p=p))
def make_qda(seed=0, reg_param=0.5):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=reg_param))
def make_et(seed=0, min_samples_leaf=6, max_features=0.4, criterion="gini"):
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=min_samples_leaf, max_features=max_features, criterion=criterion, n_jobs=-1, random_state=seed)
def make_rf(seed=0, min_samples_leaf=8, max_features=0.3, max_samples=0.8):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=min_samples_leaf, max_features=max_features, max_samples=max_samples, n_jobs=-1, random_state=seed)
def make_lgbm_linleaf(seed=0, learning_rate=0.03, n_estimators=250, num_leaves=4, min_child_samples=60, linear_lambda=10.0, colsample_bytree=0.6, reg_lambda=5.0):
    return lgb.LGBMClassifier(objective="binary", learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves, max_depth=3,
                              min_child_samples=min_child_samples, subsample=0.7, subsample_freq=1, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                              linear_tree=True, linear_lambda=linear_lambda, random_state=seed, n_jobs=-1, verbose=-1)

def make_lgbm_plain(seed=0, **p): return make_lgbm(seed=seed, **p)                          # reuse from part 1 (defaults only, no pretrain stage)
def make_xgb_plain(seed=0, learning_rate=0.02, n_estimators=400, max_depth=3, min_child_weight=8, subsample=0.7, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.1, gamma=0.0):
    return xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample,
                             colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, reg_alpha=reg_alpha, gamma=gamma, tree_method="hist", eval_metric="auc",
                             random_state=seed, n_jobs=-1, verbosity=0)
def make_cat_plain(seed=0, iterations=400, learning_rate=0.04, depth=4, l2_leaf_reg=8.0, rsm=0.6, subsample=0.8):
    return cb.CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg, rsm=rsm, bootstrap_type="Bernoulli",
                                 subsample=subsample, random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False)

# name -> dict(plain=(make, default, space), finetune=(make, default, space) or None, fs=role)
CANDIDATES = {
 "LogReg":   dict(fs="lin",  plain=(make_lr,   dict(C=0.03),  lambda t: dict(C=S(t, "C", 1e-3, 2.0, True))), finetune=None),
 "LogReg-Spline": dict(fs="spl", plain=(make_lr_spline, dict(C=0.02, n_knots=5),
                    lambda t: dict(C=S(t, "C", 1e-3, 0.5, True), n_knots=I(t, "n_knots", 3, 7))), finetune=None),
 "SVM-RBF":  dict(fs="lin",  plain=(make_svm,  dict(C=1.0, gamma=0.005), lambda t: dict(C=S(t, "C", 0.05, 30, True), gamma=S(t, "gamma", 1e-4, 5e-2, True))), finetune=None),
 "kNN":      dict(fs="lin",  plain=(make_knn,  dict(n_neighbors=60, p=2), lambda t: dict(n_neighbors=I(t, "n_neighbors", 15, 200), p=I(t, "p", 1, 2))), finetune=None),
 "QDA":      dict(fs="lin",  plain=(make_qda,  dict(reg_param=0.5), lambda t: dict(reg_param=S(t, "reg_param", 0.0, 1.0))), finetune=None),
 "LightGBM-Lin": dict(fs="spl", plain=(make_lgbm_linleaf, dict(),
                    lambda t: dict(learning_rate=S(t, "learning_rate", 0.005, 0.05, True), n_estimators=I(t, "n_estimators", 100, 500),
                                   num_leaves=I(t, "num_leaves", 3, 8), min_child_samples=I(t, "min_child_samples", 30, 100),
                                   linear_lambda=S(t, "linear_lambda", 1e-2, 100, True), colsample_bytree=S(t, "colsample_bytree", 0.3, 0.9),
                                   reg_lambda=S(t, "reg_lambda", 1e-2, 50, True))), finetune=None),
 "ExtraTrees": dict(fs="tree", plain=(make_et, dict(min_samples_leaf=6, max_features=0.4, criterion="gini"),
                    lambda t: dict(min_samples_leaf=I(t, "min_samples_leaf", 1, 20), max_features=S(t, "max_features", 0.1, 0.8), criterion=t.suggest_categorical("criterion", ["gini", "entropy"]))), finetune=None),
 "RandForest": dict(fs="tree", plain=(make_rf, dict(min_samples_leaf=8, max_features=0.3, max_samples=0.8),
                    lambda t: dict(min_samples_leaf=I(t, "min_samples_leaf", 2, 25), max_features=S(t, "max_features", 0.1, 0.7), max_samples=S(t, "max_samples", 0.5, 1.0))), finetune=None),
 "LightGBM": dict(fs="tree",
     plain=(make_lgbm_plain, dict(), lambda t: dict(learning_rate=S(t, "learning_rate", 0.005, 0.05, True), n_estimators=I(t, "n_estimators", 150, 800),
                                                     num_leaves=I(t, "num_leaves", 3, 16), max_depth=I(t, "max_depth", 2, 5), min_child_samples=I(t, "min_child_samples", 15, 80),
                                                     subsample=S(t, "subsample", 0.5, 1.0), colsample_bytree=S(t, "colsample_bytree", 0.2, 0.9),
                                                     reg_lambda=S(t, "reg_lambda", 1e-2, 50, True), reg_alpha=S(t, "reg_alpha", 1e-3, 5, True))),
     finetune=(LGBMFineTune, dict(), lambda t: dict(pre_n_estimators=I(t, "pre_n_estimators", 80, 350), pre_learning_rate=S(t, "pre_learning_rate", 0.02, 0.1, True),
                                                     n_estimators=I(t, "n_estimators", 60, 300), learning_rate=S(t, "learning_rate", 0.005, 0.05, True),
                                                     num_leaves=I(t, "num_leaves", 3, 16), max_depth=I(t, "max_depth", 2, 5), min_child_samples=I(t, "min_child_samples", 15, 80),
                                                     reg_lambda=S(t, "reg_lambda", 1e-2, 50, True)))),
 "XGBoost": dict(fs="tree",
     plain=(make_xgb_plain, dict(), lambda t: dict(learning_rate=S(t, "learning_rate", 0.005, 0.05, True), n_estimators=I(t, "n_estimators", 150, 700), max_depth=I(t, "max_depth", 1, 4),
                                                    min_child_weight=S(t, "min_child_weight", 1, 30), subsample=S(t, "subsample", 0.5, 1.0), colsample_bytree=S(t, "colsample_bytree", 0.2, 0.9),
                                                    reg_lambda=S(t, "reg_lambda", 0.1, 50, True), gamma=S(t, "gamma", 0.0, 2.0))),
     finetune=(XGBFineTune, dict(), lambda t: dict(pre_n_estimators=I(t, "pre_n_estimators", 80, 350), pre_learning_rate=S(t, "pre_learning_rate", 0.02, 0.1, True),
                                                    n_estimators=I(t, "n_estimators", 50, 250), learning_rate=S(t, "learning_rate", 0.005, 0.05, True),
                                                    max_depth=I(t, "max_depth", 1, 4), min_child_weight=S(t, "min_child_weight", 1, 30), reg_lambda=S(t, "reg_lambda", 0.1, 50, True)))),
 "CatBoost": dict(fs="tree",
     plain=(make_cat_plain, dict(), lambda t: dict(iterations=I(t, "iterations", 200, 700), learning_rate=S(t, "learning_rate", 0.01, 0.08, True), depth=I(t, "depth", 2, 6),
                                                    l2_leaf_reg=S(t, "l2_leaf_reg", 1, 30, True), rsm=S(t, "rsm", 0.3, 1.0), subsample=S(t, "subsample", 0.5, 1.0))),
     finetune=(CatFineTune, dict(), lambda t: dict(pre_iterations=I(t, "pre_iterations", 80, 350), pre_learning_rate=S(t, "pre_learning_rate", 0.02, 0.1, True),
                                                    iterations=I(t, "iterations", 60, 300), learning_rate=S(t, "learning_rate", 0.01, 0.08, True), depth=I(t, "depth", 2, 6),
                                                    l2_leaf_reg=S(t, "l2_leaf_reg", 1, 30, True)))),
}
if TORCH_OK:
    CANDIDATES["TorchMLP"] = dict(fs="lin",
        plain=(TorchMLP, dict(), lambda t: dict(hidden=I(t, "hidden", 24, 96), depth=I(t, "depth", 1, 3), dropout=S(t, "dropout", 0.05, 0.5),
                                                wd=S(t, "wd", 1e-3, 0.3, True), lr=S(t, "lr", 5e-4, 1e-2, True), epochs=I(t, "epochs", 20, 70))),
        finetune=(TorchMLPFineTune, dict(), lambda t: dict(hidden=I(t, "hidden", 24, 96), depth=I(t, "depth", 1, 3), dropout=S(t, "dropout", 0.05, 0.5),
                                                            pre_epochs=I(t, "pre_epochs", 30, 90), pre_lr=S(t, "pre_lr", 1e-3, 1e-2, True),
                                                            ft_lr_mult=S(t, "ft_lr_mult", 0.1, 0.8), ft_epochs=I(t, "ft_epochs", 10, 35),
                                                            freeze_first=t.suggest_categorical("freeze_first", [True, False]))))
GRAYS = ["#52514e", "#8a8983", "#b5b4ad"]
MODEL_COLORS = {n: (CAT[i] if i < len(CAT) else GRAYS[(i - len(CAT)) % len(GRAYS)]) for i, n in enumerate(CANDIDATES)}
print(len(CANDIDATES), "candidate model families:", list(CANDIDATES))

# %% [cell 13]
# ============================== TabPFN v2: optional pretrained tabular foundation model (auto-enabled on a GPU) ==============================
# TabPFN is a transformer pretrained on millions of synthetic tabular tasks; on small data (this one has ~2,500 rows) it is a very different learner from
# tree ensembles, which is what makes it useful in a blend. Only the OPEN v2 weights are used (newer versions are gated behind a licence click-through).
# Needs: a GPU session + Internet ON (pip install + weight download) or a Kaggle Model with a *.ckpt attached.  Any failure -> TabPFN is skipped silently.
import subprocess, sys
WANT_TABPFN = os.environ.get("RAIN_TABPFN", "auto")
TABPFN_DEVICE = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"
TABPFN_OK, TABPFN_CKPT, TabPFNClassifier, TabPFN_V2 = False, None, None, None

def _import_tabpfn():
    try:
        from tabpfn import TabPFNClassifier as C
        from tabpfn.constants import ModelVersion as V
        return C, V.V2
    except Exception:
        return None, None

def _find_tabpfn_ckpt():
    hits = [h for h in glob.glob("/kaggle/input/**/*.ckpt", recursive=True) if "tabpfn" in h.lower() and "classif" in os.path.basename(h).lower()]
    return hits[0] if hits else None

class TabPFNWrap:
    def __init__(self, seed=0, n_estimators=8):
        self.seed, self.n_estimators = seed, n_estimators
    def fit(self, X, y):
        kw = dict(device=TABPFN_DEVICE, n_estimators=self.n_estimators, random_state=self.seed)
        if TABPFN_CKPT: kw["model_path"] = TABPFN_CKPT
        self.m = TabPFNClassifier.create_default_for_version(TabPFN_V2, **kw).fit(np.asarray(X, np.float32), np.asarray(y))
        return self
    def predict_proba(self, X):
        return self.m.predict_proba(np.asarray(X, np.float32))

def make_tabpfn(seed=0, n_estimators=8):
    return TabPFNWrap(seed=seed, n_estimators=n_estimators)

if WANT_TABPFN == "1" or (WANT_TABPFN == "auto" and TABPFN_DEVICE == "cuda"):
    TabPFNClassifier, TabPFN_V2 = _import_tabpfn()
    if TabPFNClassifier is None:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True, timeout=900)
            TabPFNClassifier, TabPFN_V2 = _import_tabpfn()
        except Exception as e:
            print("could not install TabPFN (Internet off?):", repr(e)[:120])
    if TabPFNClassifier is not None:
        TABPFN_CKPT = _find_tabpfn_ckpt()
        try:                                                                        # smoke test on tiny synthetic data before trusting it
            rng_ = np.random.default_rng(0); Xs_ = rng_.normal(size=(200, 6)); ys_ = (Xs_[:, 0] + rng_.normal(size=200) > 0).astype(int)
            p_ = TabPFNWrap(seed=0, n_estimators=2).fit(Xs_, ys_).predict_proba(Xs_[:20]); TABPFN_OK = bool(np.isfinite(p_).all())
        except Exception as e:
            print("TabPFN smoke test failed -> TabPFN disabled:", repr(e)[:200])
if TABPFN_OK:
    CANDIDATES["TabPFN"] = dict(fs="lin", plain=(make_tabpfn, dict(n_estimators=8), lambda t: dict(n_estimators=8)), finetune=None, tune=False, max_repeats=3)
    MODEL_COLORS = {n: (CAT[i] if i < len(CAT) else GRAYS[(i - len(CAT)) % len(GRAYS)]) for i, n in enumerate(CANDIDATES)}
print(f"TabPFN: {'ON (device=' + TABPFN_DEVICE + ', ckpt=' + str(TABPFN_CKPT) + ')' if TABPFN_OK else 'off (needs GPU + Internet or an attached model; see notes)'} | candidates: {len(CANDIDATES)}")

# %% [cell 14]
# ============================== AutoGluon: optional AutoML member (CPU-friendly, no GPU or licence needed) ==============================
# AutoGluon automatically trains and stacks many models (LightGBM/XGBoost/CatBoost/neural nets/...) under one time budget. It is a genuinely
# different, independent ensembling system from the hand-built one above, which is exactly what a blend benefits from. Capped to a few repeats
# and a modest per-fit time limit so it cannot blow up the runtime; any failure (not installed, Internet off, error) -> skipped silently.
import tempfile, shutil, warnings as _warnings
WANT_AG = os.environ.get("RAIN_AUTOGLUON", "1") == "1"
AG_OK, TabularPredictor = False, None

def _import_ag():
    try:
        from autogluon.tabular import TabularPredictor as P
        return P
    except Exception:
        return None

class AutoGluonWrap:
    def __init__(self, seed=0, time_limit=60, presets="medium_quality"):
        self.seed, self.time_limit, self.presets = seed, time_limit, presets
    def fit(self, X, y):
        X = np.asarray(X); self.cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=self.cols); df["target"] = np.asarray(y).astype(int)
        self._path = tempfile.mkdtemp(prefix="ag_")
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.m = TabularPredictor(label="target", eval_metric="roc_auc", positive_class=1, verbosity=0, path=self._path).fit(
                df, time_limit=self.time_limit, presets=self.presets, num_cpus="auto", num_gpus=0, verbosity=0)
        return self
    def predict_proba(self, X):
        df = pd.DataFrame(np.asarray(X), columns=self.cols)
        p = np.asarray(self.m.predict_proba(df, as_pandas=False, as_multiclass=False), dtype=float)
        return np.c_[1 - p, p]

def make_autogluon(seed=0, time_limit=60, presets="medium_quality"):
    return AutoGluonWrap(seed=seed, time_limit=time_limit, presets=presets)

if WANT_AG:
    TabularPredictor = _import_ag()
    if TabularPredictor is None:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "autogluon.tabular"], check=True, timeout=900)
            TabularPredictor = _import_ag()
        except Exception as e:
            print("could not install AutoGluon (Internet off?):", repr(e)[:120])
    if TabularPredictor is not None:
        try:                                                                        # smoke test on tiny synthetic data before trusting it
            rng_ = np.random.default_rng(0); Xs_ = rng_.normal(size=(120, 5)); ys_ = (Xs_[:, 0] + rng_.normal(size=120) > 0).astype(int)
            p_ = AutoGluonWrap(seed=0, time_limit=15).fit(Xs_, ys_).predict_proba(Xs_[:20]); AG_OK = bool(np.isfinite(p_).all())
        except Exception as e:
            print("AutoGluon smoke test failed -> AutoGluon disabled:", repr(e)[:200])
if AG_OK:
    CANDIDATES["AutoGluon"] = dict(fs="lin", plain=(make_autogluon, dict(time_limit=60, presets="medium_quality"), lambda t: dict()),
                                   finetune=None, tune=False, max_repeats=2)
    MODEL_COLORS = {n: (CAT[i] if i < len(CAT) else GRAYS[(i - len(CAT)) % len(GRAYS)]) for i, n in enumerate(CANDIDATES)}
print(f"AutoGluon: {'ON' if AG_OK else 'off (needs Internet to install; see notes)'} | candidates: {len(CANDIDATES)}")

# %% [cell 15]
mode_rows = []; t0 = time.time()
for name, c in CANDIDATES.items():
    cols = FS[c["fs"]]
    pm, pd_, _ = c["plain"]
    r_plain = run_cv(pm, pd_, cols, seed_base=CFG.seed + 800, n_repeats=ABL_REP)
    mode_rows.append(dict(model=name, mode="plain", auc=r_plain["auc"], std=r_plain["std"]))
    if c["finetune"] is not None and FEAT_O is not None:
        fm, fd_, _ = c["finetune"]
        r_ft = run_cv(fm, fd_, cols, seed_base=CFG.seed + 800, n_repeats=ABL_REP, needs_orig=True)
        mode_rows.append(dict(model=name, mode="finetune", auc=r_ft["auc"], std=r_ft["std"]))
MODE_CMP = pd.DataFrame(mode_rows)
print(f"mode comparison done in {time.time() - t0:.0f}s")

WINNER = {}
for name in CANDIDATES:
    d = MODE_CMP[MODE_CMP.model == name].set_index("mode")
    if "finetune" in d.index and d.loc["finetune", "auc"] > d.loc["plain", "auc"] + 1e-4:
        WINNER[name] = "finetune"
    else:
        WINNER[name] = "plain"

pivot = MODE_CMP.pivot(index="model", columns="mode", values="auc")
fig, ax = plt.subplots(figsize=(9, 5))
has_ft = "finetune" in pivot.columns
x = np.arange(len(pivot))
ax.bar(x - (0.18 if has_ft else 0), pivot["plain"], width=0.34, color=RAIN, label="plain")
if has_ft: ax.bar(x + 0.18, pivot["finetune"], width=0.34, color=DRY, label="pretrain -> fine-tune")
for i, name in enumerate(pivot.index):
    if WINNER[name] == "finetune":
        ax.text(i + 0.18, pivot.loc[name, "finetune"] + 0.0004, "winner", ha="center", fontsize=8, color=DRY, fontweight="bold")
    else:
        ax.text(i - (0.18 if has_ft else 0), pivot.loc[name, "plain"] + 0.0004, "winner", ha="center", fontsize=8, color=RAIN, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=20, ha="right"); ax.set_ylabel("CV ROC-AUC"); ax.set_ylim(pivot.min().min() - 0.005, pivot.max().max() + 0.006)
ax.set_title("Plain training vs pretrain -> fine-tune (per model, same CV folds)"); ax.legend()
plt.tight_layout(); plt.show()
display(MODE_CMP.pivot(index="model", columns="mode", values="auc").round(5))
print("chosen mode per model:", WINNER)

# %% [cell 16]
MODEL_SPECS = {}
for name, c in CANDIDATES.items():
    make, default, space = c[WINNER[name]]
    MODEL_SPECS[name] = dict(make=make, fs=c["fs"], default=default, space=space, needs_orig=(WINNER[name] == "finetune"),
                             tune=c.get("tune", True), max_repeats=c.get("max_repeats", 99))

BEST = {n: dict(sp["default"]) for n, sp in MODEL_SPECS.items()}
STUDIES, TUNE_LOG = {}, []
if CFG.use_optuna:
    t_all = time.time()
    for name, sp in MODEL_SPECS.items():
        if not sp.get("tune", True): continue
        cols = FS[sp["fs"]]
        def objective(trial, sp=sp, cols=cols):
            return quick_cv(sp["make"], sp["space"](trial), cols, CFG.seed + CFG.tune_seed, needs_orig=sp["needs_orig"])
        try:
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CFG.seed, n_startup_trials=6))
            if sp["default"]: study.enqueue_trial(sp["default"])
            t0 = time.time()
            study.optimize(objective, n_trials=CFG.optuna_trials, timeout=CFG.optuna_timeout, show_progress_bar=False)
            BEST[name] = {**sp["default"], **study.best_params}; STUDIES[name] = study
            base_val = study.trials[0].value if sp["default"] else study.best_value
            TUNE_LOG.append(dict(model=name, mode=WINNER[name], trials=len(study.trials), best=study.best_value, secs=round(time.time() - t0)))
            print(f"{name:12s} ({WINNER[name]:8s}) trials={len(study.trials):3d}  best={study.best_value:.4f}  ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"{name:12s} tuning failed -> keeping defaults ({repr(e)[:100]})")
    print(f"total tuning time {time.time() - t_all:.0f}s")
    if TUNE_LOG: display(pd.DataFrame(TUNE_LOG).round(4))
else:
    print("Optuna skipped (quick profile or disabled) -> using reasonable defaults")

# %% [cell 17]
RES = {}; t_all = time.time()
for name, sp in list(MODEL_SPECS.items()):
    t0 = time.time()
    try:
        RES[name] = run_cv(sp["make"], BEST[name], FS[sp["fs"]], seed_base=CFG.seed, n_repeats=min(CFG.n_repeats, sp.get("max_repeats", 99)), needs_orig=sp["needs_orig"])
        print(f"{name:12s} ({WINNER[name]:8s}) AUC={RES[name]['auc']:.5f} ± {RES[name]['std']:.5f}   (avg-OOF {RES[name]['auc_avg']:.5f})   {time.time() - t0:5.0f}s")
    except Exception as e:
        print(f"{name:12s} FAILED -> dropped from the ensemble ({repr(e)[:120]})")
MODEL_SPECS = {k: v for k, v in MODEL_SPECS.items() if k in RES}
print(f"total {time.time() - t_all:.0f}s | {len(RES)} models in the ensemble")

LB = pd.DataFrame({n: dict(auc=r["auc"], std=r["std"], mode=WINNER[n]) for n, r in RES.items()}).T
LB["auc"] = LB["auc"].astype(float); LB["std"] = LB["std"].astype(float); LB = LB.sort_values("auc", ascending=False)
display(LB)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
d = LB.sort_values("auc")
ax.errorbar(d.auc, np.arange(len(d)), xerr=d["std"].astype(float) * 1.96, fmt="none", ecolor=INK2, capsize=3, lw=1.4, zorder=1)
ax.scatter(d.auc, np.arange(len(d)), s=110, c=[MODEL_COLORS[n] for n in d.index], edgecolors=SURFACE, linewidths=2, zorder=3)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([f"{n} ({m})" for n, m in zip(d.index, d["mode"])])
for i, (n, r) in enumerate(d.iterrows()): ax.text(float(r.auc) + 0.0016 + 1.96 * float(r["std"]), i, f"{float(r.auc):.4f}", va="center", fontsize=9, color=INK)
ax.set_xlabel("CV ROC-AUC (mean, 95% band across repeats)"); ax.set_title("Model leaderboard (out-of-fold) — mode shown in brackets")
plt.tight_layout(); plt.show()

# %% [cell 18]
def rank01(a):
    a = np.asarray(a, float); return (rankdata(a) - 1) / (len(a) - 1)
def fast_auc(y, s):
    r = rankdata(s); n1 = int(np.sum(y)); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

NAMES = list(RES)
R_OOF = pd.DataFrame({n: rank01(RES[n]["oof"]) for n in NAMES})
R_TE = pd.DataFrame({n: rank01(RES[n]["test"]) for n in NAMES})

def hill_climb(R, y, max_iter=80, patience=8):
    names = list(R.columns); counts = dict.fromkeys(names, 0); cur = np.zeros(len(R)); k = 0
    best, best_counts, no_imp = -1, None, 0
    for _ in range(max_iter):
        sc = {n: fast_auc(y, (cur + R[n].values) / (k + 1)) for n in names}
        nb = max(sc, key=sc.get); counts[nb] += 1; cur += R[nb].values; k += 1
        if sc[nb] > best + 1e-6: best, best_counts, no_imp = sc[nb], dict(counts), 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    w = pd.Series(best_counts); return w / w.sum()

def logit_rank(R): r = R.clip(1e-3, 1 - 1e-3); return np.log(r / (1 - r))
def fit_single(R, y):
    best = max(R.columns, key=lambda n: fast_auc(y, R[n].values)); return lambda Rn: Rn[best].values
def fit_mean_all(R, y): return lambda Rn: Rn.mean(1).values
def fit_mean_pruned(R, y, tol=0.004):
    a = {n: fast_auc(y, R[n].values) for n in R.columns}; keep = [n for n in a if a[n] >= max(a.values()) - tol]
    return lambda Rn: Rn[keep].mean(1).values
def fit_hill(R, y):
    w = hill_climb(R, y); return lambda Rn: (Rn[w.index] * w.values).sum(1).values
def fit_hill_bag(R, y, n_boot=30, seed=0):
    rng = np.random.default_rng(seed); W = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].min() == y[ix].max(): continue
        W.append(hill_climb(R.iloc[ix].reset_index(drop=True), y[ix]).reindex(R.columns).fillna(0))
    w = pd.concat(W, axis=1).mean(1); w = w / w.sum(); return lambda Rn: (Rn[w.index] * w.values).sum(1).values
def fit_stack(R, y, C=0.05):
    m = LogisticRegression(C=C, max_iter=3000).fit(logit_rank(R), y); return lambda Rn: m.decision_function(logit_rank(Rn))
def fit_blend_optuna(R, y, n_trials=40, seed=11):
    """Continuous, Optuna-searched non-negative rank-blend weights — a smoother alternative to discrete hill-climbing.
    Evaluation is pure arithmetic on precomputed ranks, so this is cheap even nested inside CV."""
    names = list(R.columns)
    if optuna is None: return fit_mean_all(R, y)
    def objective(trial):
        raw = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in names])
        if raw.sum() <= 1e-9: return 0.5
        return fast_auc(y, (R[names].values * (raw / raw.sum())).sum(1))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    w = np.array([study.best_params[f"w_{n}"] for n in names]); w = w / (w.sum() or 1)
    return lambda Rn: (Rn[names].values * w).sum(1)

BLENDERS = {"best_single": fit_single, "mean_all": fit_mean_all, "mean_pruned": fit_mean_pruned, "hill_climb": fit_hill,
            "hill_bag": fit_hill_bag, "stack_LR": fit_stack, "blend_optuna": fit_blend_optuna}

def blend_cv(fit_fn, R, y, n_rep=5, seed=7):
    per_rep = []
    for r in range(n_rep):
        fa = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(R, y):
            pred = fit_fn(R.iloc[a], y[a]); fa.append(fast_auc(y[b], pred(R.iloc[b])))
        per_rep.append(np.mean(fa))
    return float(np.mean(per_rep)), float(np.std(per_rep))

rows = [dict(method=nm, **dict(zip(("auc", "std"), blend_cv(fn, R_OOF, Y_TR)))) for nm, fn in BLENDERS.items()]
BLEND_CV = pd.DataFrame(rows).set_index("method")
print("generic blenders evaluated (nested CV, nothing fit on its own test rows)")

# %% [cell 19]
TOP_FEATS = [c for c in ["cloud", "sunshine", "humidity", "pressure", "doy_sin1", "dew_dep"] if c in FEAT.columns]
Zx = FEAT.loc[~is_te, TOP_FEATS].reset_index(drop=True)
Zx = ((Zx - Zx.mean()) / Zx.std()).fillna(0.0).values
Zx_te = FEAT.loc[is_te, TOP_FEATS].reset_index(drop=True)
mu_, sd_ = FEAT.loc[~is_te, TOP_FEATS].mean(), FEAT.loc[~is_te, TOP_FEATS].std()
Zx_te = ((Zx_te - mu_) / sd_).fillna(0.0).values                                       # standardised on TRAIN stats only -> no test-distribution leakage
R_OOF_np = R_OOF.values

def fit_stack_feat_np(R_np, extra_np, y, C=0.1):
    return LogisticRegression(C=C, max_iter=3000).fit(np.hstack([logit_rank(pd.DataFrame(R_np)).values, extra_np]), y)

sf_rows = []
for rep in range(5):
    fa = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7 + rep).split(R_OOF_np, Y_TR):
        m_ = fit_stack_feat_np(R_OOF_np[a], Zx[a], Y_TR[a])
        Xb = np.hstack([logit_rank(pd.DataFrame(R_OOF_np[b])).values, Zx[b]]); fa.append(fast_auc(Y_TR[b], m_.decision_function(Xb)))
    sf_rows.append(np.mean(fa))
BLEND_CV.loc["stack_feat"] = [float(np.mean(sf_rows)), float(np.std(sf_rows))]
display(BLEND_CV.round(5).sort_values("auc"))

BEST_BLEND = BLEND_CV.auc.idxmax()
ALT_BLEND = BLEND_CV.drop(index=[BEST_BLEND]).auc.idxmax()
print("selected blender:", BEST_BLEND, "| alternative for 2nd submission:", ALT_BLEND)

# %% [cell 20]
def blend_predict(name, R_fit, y_fit, R_oof_apply, R_te_apply):
    """Fit the named blender on the full OOF data and return (oof_score, test_score). stack_feat is handled
    separately since it needs the extra feature arrays alongside the rank predictions."""
    if name == "stack_feat":
        m_ = fit_stack_feat_np(R_fit.values, Zx, y_fit)
        oof_s = m_.decision_function(np.hstack([logit_rank(R_oof_apply).values, Zx]))
        te_s = m_.decision_function(np.hstack([logit_rank(R_te_apply).values, Zx_te]))
        return oof_s, te_s
    pred = BLENDERS[name](R_fit, y_fit)
    return pred(R_oof_apply), pred(R_te_apply)

oof_score, te_score = blend_predict(BEST_BLEND, R_OOF, Y_TR, R_OOF, R_TE)
alt_oof, alt_te = blend_predict(ALT_BLEND, R_OOF, Y_TR, R_OOF, R_TE)

def platt(oof_s, te_s):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(oof_s.reshape(-1, 1), Y_TR)
    assert lr.coef_[0, 0] > 0, "calibration flipped the ranking"
    return lr.predict_proba(oof_s.reshape(-1, 1))[:, 1], lr.predict_proba(te_s.reshape(-1, 1))[:, 1]

P_OOF, P_TE = platt(oof_score, te_score)
ALT_P_OOF, ALT_P_TE = platt(alt_oof, alt_te)
print(f"OOF AUC  main blend ({BEST_BLEND}): {roc_auc_score(Y_TR, P_OOF):.5f} | alt ({ALT_BLEND}): {roc_auc_score(Y_TR, ALT_P_OOF):.5f} | best single: {LB.auc.iloc[0]:.5f}")
print(f"Brier {brier_score_loss(Y_TR, P_OOF):.4f}  logloss {log_loss(Y_TR, P_OOF):.4f}  (base-rate Brier {Y_TR.mean() * (1 - Y_TR.mean()):.4f})")

fig, ax = plt.subplots(figsize=(8, 3.8))
d = BLEND_CV.sort_values("auc")
ax.errorbar(d.auc, np.arange(len(d)), xerr=d["std"], fmt="o", color=RAIN, ms=10, mec=SURFACE, mew=2, capsize=3)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.index); ax.set_xlabel("nested blend-CV AUC"); ax.set_title("Blenders compared")
plt.tight_layout(); plt.show()

# %% [cell 21]
rng = np.random.default_rng(0)
fpr, tpr, _ = roc_curve(Y_TR, P_OOF)
grid = np.linspace(0, 1, 201); boots = []
for _ in range(250):
    ix = rng.integers(0, len(Y_TR), len(Y_TR))
    if len(np.unique(Y_TR[ix])) < 2: continue
    f_, t_, _ = roc_curve(Y_TR[ix], P_OOF[ix]); boots.append(np.interp(grid, f_, t_))
boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5], 0)
auc_boot = [roc_auc_score(Y_TR[i], P_OOF[i]) for i in [rng.integers(0, len(Y_TR), len(Y_TR)) for _ in range(250)]]
ci = np.percentile(auc_boot, [2.5, 97.5])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axes[0]
for n in LB.index: f_, t_, _ = roc_curve(Y_TR, R_OOF[n]); ax.plot(f_, t_, color=MODEL_COLORS[n], lw=1, alpha=0.5)
ax.fill_between(grid, lo, hi, color=RAIN, alpha=0.18, lw=0); ax.plot(fpr, tpr, color=RAIN, lw=3); ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
ax.set_title(f"ROC  AUC={roc_auc_score(Y_TR, P_OOF):.4f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f})"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")

ax = axes[1]
fp, mp = calibration_curve(Y_TR, P_OOF, n_bins=10, strategy="quantile")
ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1); ax.plot(mp, fp, "o-", color=RAIN, lw=2.5, ms=9, mec=SURFACE, mew=2)
ax.set_title("Reliability"); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")

ax = axes[2]
for cls, colr, lab in [(0, DRY, "no rain"), (1, RAIN, "rain")]:
    sns.kdeplot(P_OOF[Y_TR == cls], ax=ax, color=colr, fill=True, alpha=0.4, lw=2, label=lab, clip=(0, 1))
ax.set_title("Score separation"); ax.set_xlabel("predicted P(rain)"); ax.legend()
plt.tight_layout(); plt.show()

# %% [cell 22]
years = sorted(np.unique(GROUPS)); LOYO = pd.DataFrame(index=list(MODEL_SPECS) + ["blend (mean-rank)"], columns=years, dtype=float)
preds_year = {n: {} for n in MODEL_SPECS}
for name, sp in MODEL_SPECS.items():
    Xtr, _, Xor = get_arrays(FS[sp["fs"]])
    for yr_ in years:
        a, b = np.where(GROUPS != yr_)[0], np.where(GROUPS == yr_)[0]
        m = _fold_fit(sp["make"], BEST[name], Xtr, Y_TR, Xor, Y_OR, a, CFG.seed, sp["needs_orig"], CFG.orig_rep)
        p = score_of(m, Xtr[b]); preds_year[name][yr_] = p; LOYO.loc[name, yr_] = roc_auc_score(Y_TR[b], p)
for yr_ in years:
    b = np.where(GROUPS == yr_)[0]
    LOYO.loc["blend (mean-rank)", yr_] = roc_auc_score(Y_TR[b], np.mean([rank01(preds_year[n][yr_]) for n in MODEL_SPECS], 0))
LOYO["LOYO mean"] = LOYO[years].mean(1); LOYO["random-CV"] = [RES[n]["auc"] for n in MODEL_SPECS] + [np.nan]
LOYO = LOYO.sort_values("LOYO mean", ascending=False)

fig, ax = plt.subplots(figsize=(11, 4.8))
sns.heatmap(LOYO[years + ["LOYO mean", "random-CV"]], annot=True, fmt=".3f", cmap=SEQ_CMAP, linewidths=2, linecolor=SURFACE, annot_kws={"size": 9}, cbar_kws={"label": "AUC"}, ax=ax)
ax.set_title("Leave-one-year-out AUC vs random-CV AUC"); ax.set_xlabel("held-out year"); ax.grid(False)
plt.tight_layout(); plt.show()
print("mean gap  random-CV minus LOYO: %.4f" % (LOYO["random-CV"] - LOYO["LOYO mean"]).mean())

# %% [cell 23]
test_ids = FEAT.loc[is_te, "id"].values
sub = pd.DataFrame({"id": test_ids, TARGET: P_TE})
sub_alt = pd.DataFrame({"id": test_ids, TARGET: ALT_P_TE})
if sample is not None:
    sub = sample[["id"]].merge(sub, on="id", how="left"); sub_alt = sample[["id"]].merge(sub_alt, on="id", how="left")
for s in (sub, sub_alt):
    assert s.shape[0] == len(test) and s[TARGET].notna().all() and s[TARGET].between(0, 1).all() and s["id"].is_unique
sub.to_csv(OUT_DIR / "submission.csv", index=False)
sub_alt.to_csv(OUT_DIR / "submission_alt.csv", index=False)
pd.DataFrame({"id": FEAT.loc[~is_te, "id"].values, "y": Y_TR, "blend_oof": P_OOF, **{f"oof_{n}": RES[n]["oof"] for n in RES}}).to_csv(OUT_DIR / "oof_predictions.csv", index=False)
print("saved ->", OUT_DIR); display(sub.head())

summary = pd.DataFrame({"score": [LB.auc.iloc[0], BLEND_CV.loc[BEST_BLEND, "auc"], roc_auc_score(Y_TR, P_OOF), LOYO.loc["blend (mean-rank)", "LOYO mean"]]},
                       index=[f"best single model CV ({LB.index[0]}, {LB.iloc[0]['mode']})", f"blend nested-CV ({BEST_BLEND})", "blend OOF AUC (in-sample weights)", "leave-one-year-out (mean-rank blend)"])
display(summary.round(5))
n_ft = sum(1 for v in WINNER.values() if v == "finetune")
print(f"\nfine-tuning won for {n_ft} of {len(WINNER)} candidate models: " + ", ".join(n for n, v in WINNER.items() if v == "finetune") if n_ft else "\nfine-tuning did not beat plain training for any model this run -> every model used plain training (still a valid, CV-checked outcome).")

# %% [cell 24]
EXT_WEIGHT = float(os.environ.get("RAIN_EXT_WEIGHT", 0.15))                # kept small: every external notebook found scores below this pipeline

def find_external_submissions():
    found, skip = [], ("train.csv", "test.csv", "rainfall.csv", "sample_submission")
    for p in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        if any(k in p.lower() for k in skip): continue
        try: d = pd.read_csv(p)
        except Exception: continue
        if {"id", TARGET}.issubset(d.columns) and len(d) == len(test) and set(d["id"]) == set(test["id"]) and pd.to_numeric(d[TARGET], errors="coerce").notna().all():
            found.append((p, d[["id", TARGET]]))
    return found

ext = find_external_submissions(); own = sub.set_index("id")[TARGET]; kept = []
for p, d in ext:
    r_ = d.set_index("id")[TARGET].reindex(own.index).astype(float); c_ = stats.spearmanr(own.values, r_.values)[0]
    print(f"{'kept   ' if c_ >= 0.85 else 'ignored'} corr={c_:.3f}  {p}")
    if c_ >= 0.85: kept.append(pd.Series(rankdata(r_.values) / len(r_), index=own.index))
if kept:
    mix = (1 - EXT_WEIGHT) * pd.Series(rankdata(own.values) / len(own), index=own.index) + EXT_WEIGHT * pd.concat(kept, axis=1).mean(1)
    mixed_p = np.sort(own.values)[rankdata(mix.values, method="ordinal").astype(int) - 1]          # map back onto our calibrated probability scale (order = mix)
    sub_ext = pd.DataFrame({"id": own.index, TARGET: mixed_p}); sub_ext.to_csv(OUT_DIR / "submission_with_external.csv", index=False)
    print(f"blended {len(kept)} external file(s) at weight {EXT_WEIGHT} -> submission_with_external.csv  (Spearman with main: {stats.spearmanr(mixed_p, own.values)[0]:.4f})")
else:
    print("no usable external submission found -> skipped (normal unless you attached a notebook's output via Add Input)")
