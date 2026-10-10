import pandas as pd, numpy as np, time, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
B=["pressure","maxtemp","temparature","mintemp","dewpoint","humidity","cloud","sunshine","winddirection","windspeed"]
def f(d):
    d=d.copy(); d['doy']=d['day'].clip(1,365)
    d['dew_dep']=d.temparature-d.dewpoint; d['trange']=d.maxtemp-d.mintemp
    d['cm_s']=d.cloud/100-d.sunshine/12
    return d
tr,te=f(tr),f(te); C=B+['doy','dew_dep','trange','cm_s']
X,y,Xt=tr[C].values,tr.rainfall.values,te[C].values
n=int(sys.argv[1]) if len(sys.argv)>1 else 2
oof=np.zeros(len(X)); t0=time.time()
for i,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)):
    m=TabPFNClassifier(device='cpu',n_estimators=n,ignore_pretraining_limits=True).fit(X[a],y[a])
    oof[b]=m.predict_proba(X[b])[:,1]; print(i,roc_auc_score(y[b],oof[b]),time.time()-t0,flush=True)
print('OOF AUC',roc_auc_score(y,oof))
np.save('outputs/legacy/tabpfn_oof.npy',oof)
