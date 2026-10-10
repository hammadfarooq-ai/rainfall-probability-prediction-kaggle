"""Named candidate blends from foundation models (Kaggle GPU) + local models + old stack."""
import numpy as np, pandas as pd, glob, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
rk = lambda v: rankdata(v) / len(v)
loc = pd.read_csv('outputs/ensemble/oof.csv'); y = loc.y.values; years = np.arange(len(y)) // 365
tloc = pd.read_csv('outputs/ensemble/test_ranks.csv')
O = {c: rk(loc[c]) for c in loc.columns if c != 'y'}; T = {c: tloc[c].values for c in tloc.columns}
for f in sorted(glob.glob('outputs/gpu/oof_*.npy')):
    n = os.path.basename(f)[4:-4]
    O[n] = rk(np.load(f).mean(0)); T[n] = rk(np.load(f.replace('oof_', 'test_')))
O['old_stack'] = rk(pd.read_csv('outputs/stack_run3/oof_predictions.csv').blend_oof); T['old_stack'] = rk(pd.read_csv('outputs/stack_run3/submission.csv').rainfall)
ref = T['tabpfn_raw_og']
C = {
    'fm2': ['tabpfn_raw_og', 'tabicl_raw_og'],
    'fm4': ['tabpfn_raw_og', 'tabpfn_raw_og_big', 'tabicl_raw_og', 'tabicl_eng_og'],
    'fm2_lin_gbm': ['tabpfn_raw_og', 'tabicl_raw_og', 'lr_eng_og', 'lgb_og'],
    'fm2_old': ['tabpfn_raw_og', 'tabicl_raw_og', 'old_stack', 'old_stack'],
    'pfn_old': ['tabpfn_raw_og', 'old_stack'],
    'tabicl': ['tabicl_raw_og'],
    'pfn_big': ['tabpfn_raw_og_big'],
    'tabicl3': ['tabicl_raw_og', 'tabicl_eng_og', 'tabicl_raw'],
    'tabicl_eng': ['tabicl_eng_og'],
    'tabicl_old': ['tabicl_raw_og', 'old_stack'],
    'tabicl_lr': ['tabicl_raw_og', 'lr_eng_og'],
}
sub = pd.read_csv('data/sample_submission.csv')
for tag, S in C.items():
    o = sum(O[m] for m in S); p = sum(T[m] for m in S) / len(S)
    print(f"{tag:12s} OOF {roc_auc_score(y, o):.5f}  yearly-min {min(roc_auc_score(y[years == g], o[years == g]) for g in range(6)):.4f}"
          f"  spearman vs submitted TabPFN {pd.Series(p).corr(pd.Series(ref), method='spearman'):.4f}")
    sub.assign(rainfall=np.clip(p, 0.001, 0.999)).to_csv(f'outputs/ensemble/sub_{tag}.csv', index=False)

