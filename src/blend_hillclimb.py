"""Hill-climb blend over local models, the earlier stack and the Kaggle-GPU models."""
import numpy as np, pandas as pd, glob, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
rk = lambda v: rankdata(v) / len(v)
loc = pd.read_csv('outputs/ensemble/oof.csv'); y = loc.y.values
tloc = pd.read_csv('outputs/ensemble/test_ranks.csv')
O = {c: rk(loc[c]) for c in loc.columns if c != 'y'}; T = {c: tloc[c].values for c in tloc.columns}
for f in sorted(glob.glob('outputs/gpu/oof_*.npy')):
    n = os.path.basename(f)[4:-4]
    O[n] = rk(np.load(f).mean(0)); T[n] = rk(np.load(f.replace('oof_', 'test_')))
O['old_stack'] = rk(pd.read_csv('outputs/stack_run3/oof_predictions.csv').blend_oof); T['old_stack'] = rk(pd.read_csv('outputs/stack_run3/submission.csv').rainfall)
names = list(O)
for n in names: print(f"{n:14s} {roc_auc_score(y, O[n]):.5f}")


def hill(idx, cands, k=15):
    sel = []; best = 0
    for _ in range(k):
        c = max(cands, key=lambda n: roc_auc_score(y[idx], sum(O[m][idx] for m in sel + [n])))
        v = roc_auc_score(y[idx], sum(O[m][idx] for m in sel + [c]))
        if v <= best + 1e-5: break
        sel.append(c); best = v
    return sel, best


sub = pd.read_csv('data/sample_submission.csv'); allidx = np.arange(len(y))
for tag, cands in [('noold', [n for n in names if n != 'old_stack']), ('all', names),
                   ('gpu', [n for n in names if n.startswith(('tabpfn', 'realmlp', 'tabm'))])]:
    if not cands: continue
    sel, best = hill(allidx, cands)
    nest = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(O[names[0]], y):
        s2, _ = hill(a, cands); nest.append(roc_auc_score(y[b], sum(O[m][b] for m in s2)))
    print(f"\n[{tag}] hill {best:.5f}  nested {np.mean(nest):.5f}  -> {sel}")
    p = sum(T[m] for m in sel) / len(sel)
    sub.assign(rainfall=np.clip(p, 0.001, 0.999)).to_csv(f'outputs/ensemble/sub_blend_{tag}.csv', index=False)
    print('  spearman vs old_stack test', round(pd.Series(p).corr(pd.Series(T['old_stack']), method='spearman'), 4))
