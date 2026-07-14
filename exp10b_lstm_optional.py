"""
OPTIONAL: true LSTM temporal baseline under rolling-origin (requires PyTorch).
Run on a machine with torch installed:  pip install torch  ;  python exp10b_lstm_optional.py
Mirrors exp10 (shuffled vs rolling-origin) with a recurrent model so the paper can
report an LSTM number directly. Uses sequences of the last SEQ windows per class.
"""
import os, numpy as np, pandas as pd
try:
    import torch, torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not installed. Run: pip install torch")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
OUT="results"; RNG=42; SEQ=8; torch.manual_seed(RNG); np.random.seed(RNG)
df=pd.read_parquet(os.path.join(OUT,"features_ds1.parquet"))
meta=["label","cls","bin_idx","t_start"]; Xcols=[c for c in df.columns if c not in meta]
base=df[Xcols].values.astype("float32")
classes=sorted(df["label"].unique()); y=np.array([classes.index(v) for v in df["label"]])
frac=np.empty(len(df)); seqs=np.zeros((len(df),SEQ,len(Xcols)),dtype="float32")
for c in df["cls"].unique():
    idx=df.index[df["cls"]==c].values; o=idx[np.argsort(df.loc[idx,"bin_idx"].values)]
    frac[o]=np.arange(len(o))/len(o)
    for j,ri in enumerate(o):
        seqs[ri]=np.stack([base[o[max(0,j-(SEQ-1)+t)]] for t in range(SEQ)])
class LSTM(nn.Module):
    def __init__(s,f,h,n): super().__init__(); s.l=nn.LSTM(f,h,batch_first=True); s.fc=nn.Linear(h,n)
    def forward(s,x): o,_=s.l(x); return s.fc(o[:,-1])
def train_eval(tr,te):
    sc=StandardScaler().fit(base[tr]); 
    def norm(a): return ((a-sc.mean_)/np.sqrt(sc.var_+1e-8)).astype("float32")
    Xtr=torch.tensor(norm(seqs[tr])); ytr=torch.tensor(y[tr])
    Xte=torch.tensor(norm(seqs[te]))
    m=LSTM(len(Xcols),64,len(classes)); opt=torch.optim.Adam(m.parameters(),1e-3)
    lossf=nn.CrossEntropyLoss()
    for ep in range(120):
        opt.zero_grad(); out=m(Xtr); loss=lossf(out,ytr); loss.backward(); opt.step()
    with torch.no_grad(): pred=m(Xte).argmax(1).numpy()
    return f1_score(y[te],pred,average="macro")
# shuffled 5-fold
from sklearn.model_selection import StratifiedKFold
sh=[]; 
for tri,tei in StratifiedKFold(5,shuffle=True,random_state=RNG).split(base,y):
    tr=np.zeros(len(df),bool); tr[tri]=True; te=~tr; sh.append(train_eval(tr,te))
# rolling-origin
ro=[]
for oo in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
    tr=frac<oo; te=(frac>=oo)&(frac<min(oo+0.15,1.0))
    if te.sum()<10: continue
    ro.append(train_eval(tr,te))
print(f"LSTM  shuffled={np.mean(sh):.3f}  rolling-origin={np.mean(ro):.3f}±{np.std(ro):.3f}  drop={np.mean(sh)-np.mean(ro):.3f}")
