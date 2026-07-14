"""
EXPERIMENT 2 — Head-to-head: Random-Forest fusion vs. a Graph Convolutional Network,
on IDENTICAL splits (dataset 1). Dataset 1 exposes no grid topology, so we build the
standard topology-free graph: a symmetric k-NN graph over standardized fused features
(nodes = time windows). A 2-layer GCN (implemented from scratch, Adam) does transductive
node classification. RF is run on the exact same train/test masks for a fair comparison.

Reports both the random 75/25 split and the temporal hold-out (last 25% in time).
Outputs: results/gnn_headtohead.txt, results/gnn_headtohead.png
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)

OUT="results"
df=pd.read_parquet(os.path.join(OUT,"features_ds1.parquet"))
meta=["label","cls","bin_idx","t_start"]
Xcols=[c for c in df.columns if c not in meta]
X=df[Xcols].values.astype(np.float64); y_lab=df["label"].values
classes=sorted(np.unique(y_lab)); cidx={c:i for i,c in enumerate(classes)}
y=np.array([cidx[c] for c in y_lab]); C=len(classes); N=len(y)
Xs=StandardScaler().fit_transform(X)

def knn_adj(Z,k=10):
    # squared euclidean distances
    sq=(Z*Z).sum(1); D=sq[:,None]+sq[None,:]-2*Z@Z.T; np.fill_diagonal(D,np.inf)
    nn=np.argsort(D,axis=1)[:,:k]
    A=np.zeros((len(Z),len(Z)))
    for i in range(len(Z)):
        A[i,nn[i]]=1.0
    A=np.maximum(A,A.T)                     # symmetric
    A=A+np.eye(len(Z))                      # self loops
    d=A.sum(1); dinv=1.0/np.sqrt(d)
    return (A*dinv[:,None])*dinv[None,:]    # normalized

Ah=knn_adj(Xs,k=10)
AhX=Ah@Xs

def onehot(v):
    o=np.zeros((len(v),C)); o[np.arange(len(v)),v]=1; return o

def train_gcn(train_mask,test_mask,H=64,epochs=300,lr=0.01,wd=5e-4,seed=0):
    rng=np.random.default_rng(seed); F=Xs.shape[1]
    W1=rng.normal(0,np.sqrt(2/F),(F,H)); b1=np.zeros(H)
    W2=rng.normal(0,np.sqrt(2/H),(H,C)); b2=np.zeros(C)
    params=[W1,b1,W2,b2]; m=[np.zeros_like(p) for p in params]; v=[np.zeros_like(p) for p in params]
    Yoh=onehot(y); tr=train_mask; ntr=tr.sum(); b1t,b2t=0.9,0.999
    for ep in range(1,epochs+1):
        Z1=AhX@W1+b1; H1=np.maximum(Z1,0)
        Z2=Ah@H1@W2+b2
        Z2-=Z2.max(1,keepdims=True); P=np.exp(Z2); P/=P.sum(1,keepdims=True)
        dZ2=(P-Yoh); dZ2[~tr]=0; dZ2/=ntr
        dW2=(Ah@H1).T@dZ2 + wd*W2; db2=dZ2.sum(0)
        dH1=Ah@(dZ2@W2.T); dZ1=dH1*(Z1>0)
        dW1=(AhX).T@dZ1 + wd*W1; db1=dZ1.sum(0)
        grads=[dW1,db1,dW2,db2]
        for i,g in enumerate(grads):
            m[i]=b1t*m[i]+(1-b1t)*g; v[i]=b2t*v[i]+(1-b2t)*(g*g)
            mh=m[i]/(1-b1t**ep); vh=v[i]/(1-b2t**ep)
            params[i]-=lr*mh/(np.sqrt(vh)+1e-8)
        W1,b1,W2,b2=params
    Z1=AhX@W1+b1; H1=np.maximum(Z1,0); Z2=Ah@H1@W2+b2
    pred=Z2.argmax(1)
    return f1_score(y[test_mask],pred[test_mask],average="macro")

def rf_split(train_mask,test_mask):
    m=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced")
    m.fit(X[train_mask],y[train_mask])
    return f1_score(y[test_mask],m.predict(X[test_mask]),average="macro")

# --- split A: random 75/25 (paper's headline protocol) ---
idx=np.arange(N)
tr_idx,te_idx=train_test_split(idx,test_size=0.25,random_state=42,stratify=y)
trA=np.zeros(N,bool); trA[tr_idx]=True; teA=~trA
# --- split B: temporal hold-out (last 25% per class) ---
teB=np.zeros(N,bool)
for c in df["cls"].unique():
    ii=df.index[df["cls"]==c]; order=ii[np.argsort(df.loc[ii,"bin_idx"].values)]
    teB[order[int(0.75*len(order)):]]=True
trB=~teB

res={}
res["RF_random"]=rf_split(trA,teA)
res["GCN_random"]=train_gcn(trA,teA)
res["RF_temporal"]=rf_split(trB,teB)
res["GCN_temporal"]=train_gcn(trB,teB)

with open(os.path.join(OUT,"gnn_headtohead.txt"),"w") as f:
    f.write("HEAD-TO-HEAD on identical splits (dataset 1, macro-F1)\n")
    f.write("Graph: symmetric k-NN (k=10) over standardized fused features; GCN 2-layer, H=64.\n\n")
    f.write(f"Random 75/25 :  RF={res['RF_random']:.3f}   GCN={res['GCN_random']:.3f}\n")
    f.write(f"Temporal 25% :  RF={res['RF_temporal']:.3f}   GCN={res['GCN_temporal']:.3f}\n")
print(open(os.path.join(OUT,"gnn_headtohead.txt")).read())

plt.figure(figsize=(5.5,4))
groups=["Random 75/25","Temporal hold-out"]; xpos=np.arange(2); w=0.35
plt.bar(xpos-w/2,[res["RF_random"],res["RF_temporal"]],w,label="RF fusion",color="#2E7D32")
plt.bar(xpos+w/2,[res["GCN_random"],res["GCN_temporal"]],w,label="GCN (k-NN graph)",color="#5B9BD5")
plt.xticks(xpos,groups); plt.ylabel("macro-F1"); plt.ylim(0,1); plt.legend()
plt.title("RF fusion vs. GCN — identical splits")
for i,val in enumerate([res["RF_random"],res["RF_temporal"]]): plt.text(i-w/2,val+0.02,f"{val:.2f}",ha="center",fontsize=8)
for i,val in enumerate([res["GCN_random"],res["GCN_temporal"]]): plt.text(i+w/2,val+0.02,f"{val:.2f}",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"gnn_headtohead.png"),dpi=150)
print("saved gnn_headtohead.png")
