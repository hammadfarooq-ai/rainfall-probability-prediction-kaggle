"""Weighted (incl. negative-weight) rank blends around TabICL."""
import numpy as np, pandas as pd, glob, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
rk = lambda v: rankdata(v) / len(v)
loc = pd.read_csv('outputs/ensemble/oof.csv'); y = loc.y.values
tloc = pd.read_csv('outputs/ensemble/test_ranks.csv')
O = {c: rk(loc[c]) for c in loc.columns if c != 'y'}; T = {c: tloc[c].values for c in tloc.columns}
for f in sorted(glob.glob('outputs/gpu/oof_*.npy')):
    n = os.path.basename(f)[4:-4]
    O[n] = rk(np.load(f).mean(0)); T[n] = rk(np.load(f.replace('oof_', 'test_')))
O['old_stack'] = rk(pd.read_csv('outputs/stack_run3/oof_predictions.csv').blend_oof); T['old_stack'] = rk(pd.read_csv('outputs/stack_run3/submission.csv').rainfall)
C = {
    'x_icl_m_old': {'tabicl_raw_og': 1.5, 'old_stack': -0.5},
    'x_icl_m_lr': {'tabicl_raw_og': 1.5, 'lr_eng_og': -0.5},
    'x_icl_m_old_lr': {'tabicl_raw_og': 2.0, 'old_stack': -0.5, 'lr_eng_og': -0.5},
    'x_icl_m_old_strong': {'tabicl_raw_og': 2.0, 'old_stack': -1.0},
    'tabicl_raw': {'tabicl_raw': 1.0},
}
sub = pd.read_csv('data/sample_submission.csv')
for tag, W in C.items():
    o = sum(w * O[m] for m, w in W.items()); p = rk(sum(w * T[m] for m, w in W.items()))
    print(f"{tag:20s} OOF {roc_auc_score(y, o):.5f}  spearman vs TabICL test {pd.Series(p).corr(pd.Series(T['tabicl_raw_og']), method='spearman'):.4f}")
    sub.assign(rainfall=np.clip(p, 0.001, 0.999)).to_csv(f'outputs/ensemble/sub_{tag}.csv', index=False)
