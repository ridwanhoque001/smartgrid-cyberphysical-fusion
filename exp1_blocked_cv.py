"""
EXPERIMENT 1 — Leakage-aware evaluation of the fusion detector (dataset 1).

The paper's headline 0.96 comes from StratifiedKFold(shuffle=True), which can place
temporally adjacent windows (highly correlated) in both train and test -> optimistic.
Here we compare that against TIME-BLOCKED CV, which keeps contiguous time blocks intact
so no window in test is adjacent to a training window. The gap is the leakage.

Outputs: results/blocked_cv.txt, results/blocked_cv.png
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="results"; RNG=42
df=pd.read_parquet(os.path.join(OUT,"features_ds1.parquet"))
meta=["label","cls","bin_idx","t_start"]
Xcols=[c for c in df.columns if c not in meta]
X=df[Xcols].values; y=df["label"].values
phys=[c for c in Xcols if c.startswith("phys_")]; cyb=[c for c in Xcols if c.startswith("cyb_")]

def rf(): return RandomForestClassifier(n_estimators=300,random_state=RNG,n_jobs=-1,class_weight="balanced")

# ---- (a) random shuffled 5-fold (the optimistic protocol) ----
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RNG)
rand=cross_val_score(rf(),df[Xcols],y,cv=cv,scoring="f1_macro",n_jobs=1)

# ---- (b) time-blocked 5-fold ----
# Within each class, order windows by time (bin_idx) and cut into 5 contiguous blocks.
# Fold k = block k of every class held out; the rest train. Adjacent windows never split.
NB=5
block=np.empty(len(df),dtype=int)
for c in df["cls"].unique():
    idx=df.index[df["cls"]==c]
    order=idx[np.argsort(df.loc[idx,"bin_idx"].values)]
    edges=np.linspace(0,len(order),NB+1).astype(int)
    for k in range(NB):
        block[order[edges[k]:edges[k+1]]]=k

blk=[]
for k in range(NB):
    te=block==k; tr=~te
    m=rf(); m.fit(X[tr],y[tr]); p=m.predict(X[te])
    blk.append(f1_score(y[te],p,average="macro"))
blk=np.array(blk)

# single held-out final time block (last 25% of every class in time) for a headline number
te = np.zeros(len(df),bool)
for c in df["cls"].unique():
    idx=df.index[df["cls"]==c]; order=idx[np.argsort(df.loc[idx,"bin_idx"].values)]
    te[order[int(0.75*len(order)):]]=True
m=rf(); m.fit(X[~te],y[~te]); temporal_holdout=f1_score(y[te],m.predict(X[te]),average="macro")

with open(os.path.join(OUT,"blocked_cv.txt"),"w") as f:
    f.write("LEAKAGE-AWARE EVALUATION (dataset 1, fusion RF)\n")
    f.write(f"Random shuffled 5-fold : {rand.mean():.3f} +/- {rand.std():.3f}   (optimistic)\n")
    f.write(f"Time-blocked 5-fold    : {blk.mean():.3f} +/- {blk.std():.3f}   (leakage-controlled)\n")
    f.write(f"Single temporal hold-out (last 25%): {temporal_holdout:.3f}\n")
    f.write(f"Leakage gap (random - blocked): {rand.mean()-blk.mean():.3f}\n")
print(open(os.path.join(OUT,"blocked_cv.txt")).read())

plt.figure(figsize=(5,4))
labels=["Random\n5-fold","Time-blocked\n5-fold","Temporal\nhold-out"]
means=[rand.mean(),blk.mean(),temporal_holdout]; errs=[rand.std(),blk.std(),0]
bars=plt.bar(labels,means,yerr=errs,capsize=6,color=["#B0B0B0","#2E7D32","#1B5E20"])
plt.ylabel("macro-F1"); plt.ylim(0,1.0); plt.title("Evaluation protocol vs. leakage (dataset 1)")
for i,v in enumerate(means): plt.text(i,v+0.03,f"{v:.2f}",ha="center",fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"blocked_cv.png"),dpi=150)
print("saved blocked_cv.png")
