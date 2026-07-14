"""
================================================================================
Advanced experiments on the fused features:
  (A) Model comparison  — Random Forest vs Gradient Boosting vs Neural Net (MLP)
  (B) Stronger attack   — gradient-based PGD (via an MLP surrogate) transferred to
                          the Random Forest, and whether adversarial training holds.

Note on "GNN": a true graph neural network like the dataset authors' needs the
power-grid TOPOLOGY (a graph) plus torch-geometric. Our windowed feature table is
not graph-structured, so here we compare RF against the strongest drop-in models
(MLP neural net, gradient boosting). A topology-based GNN is a separate, larger
effort and a good future step.

Run:
    pip install pandas scikit-learn matplotlib
    python advanced_experiments.py
================================================================================
"""
import os, glob, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR="data/Dataset"; OUTPUT_DIR="results"; WINDOW=0.5
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASSES={"benign":"benign","backdoor":"backdoor","Bruteforce":"bruteforce","FDI":"fdi","ransomware":"ransomware","reverseshell":"reverseshell"}
PROTOS=["TCP","Modbus/TCP","HTTP","TLSv1.2","ARP","ICMPv6"]
PHYS_COLS=["Freq","Theta","V_A","V_B","V_C","I_A","I_B","I_C","ActivePower","ReactivePower","BreakerStatus"]
CANON=["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","_ws.col.Protocol","ip.len","tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"]

def to_num(s): return pd.to_numeric(s.astype(str).str.replace(",","",regex=False).str.replace('"',"",regex=False),errors="coerce")
def find(f,n): h=glob.glob(os.path.join(DATA_DIR,f,n)); return h[0] if h else None
def _ep(s): return pd.to_numeric(s,errors="coerce").between(1.6e9,1.8e9).mean()>0.5
def ncy(cy):
    cy=cy.loc[:,~cy.columns.astype(str).str.startswith("Unnamed")]
    if "frame.time_epoch" in cy.columns and _ep(cy["frame.time_epoch"]): return cy
    cy=cy.iloc[:,:11].copy(); cy.columns=CANON; return cy
def cyf(p):
    cy=pd.read_csv(p,low_memory=False); cy.columns=[c.strip() for c in cy.columns]; cy=ncy(cy)
    t=to_num(cy["frame.time_epoch"]); cy=cy[t.notna()].copy(); cy["t"]=t[t.notna()].values
    cy["bin"]=(cy["t"]//WINDOW).astype(np.int64); cy["iplen"]=to_num(cy["ip.len"]); g=cy.groupby("bin")
    f=pd.DataFrame({"cyb_pkt_count":g.size(),"cyb_iplen_mean":g["iplen"].mean(),"cyb_iplen_sum":g["iplen"].sum(),"cyb_iplen_std":g["iplen"].std(),"cyb_n_src_ip":g["ip.src"].nunique(),"cyb_n_dst_ip":g["ip.dst"].nunique()})
    pr=cy.groupby(["bin","_ws.col.Protocol"]).size().unstack(fill_value=0)
    for p2 in PROTOS: f["cyb_frac_"+p2.replace("/","_")]=pr[p2]/f["cyb_pkt_count"] if p2 in pr.columns else 0.0
    return f
def phf(p):
    ph=pd.read_csv(p,low_memory=False); ph.columns=[c.strip() for c in ph.columns]
    t=to_num(ph["@timestamp"]); ph=ph[t.notna()].copy(); ph["t"]=t[t.notna()].values
    ph["bin"]=(ph["t"]//WINDOW).astype(np.int64)
    for c in PHYS_COLS: ph[c]=to_num(ph[c])
    f=ph.groupby("bin")[PHYS_COLS].agg(["mean","std"]); f.columns=["phys_"+a+"_"+b for a,b in f.columns]; return f
def build():
    rows=[]
    for folder,label in CLASSES.items():
        j=phf(find(folder,"Physical.csv")).join(cyf(find(folder,"[Cc]yber.csv")),how="inner"); j["label"]=label; rows.append(j)
    return pd.concat(rows).reset_index(drop=True).replace([np.inf,-np.inf],np.nan).fillna(0)

print("Building features..."); data=build()
y=data["label"]; X=data.drop(columns=["label"]); cols=list(X.columns)
phys=[cols.index(c) for c in cols if c.startswith("phys_")]

# ---------- (A) MODEL COMPARISON ----------
print("\n=== (A) model comparison (5-fold macro-F1) ===")
cv=StratifiedKFold(5,shuffle=True,random_state=42)
models={"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced"),
        "Grad. Boosting":HistGradientBoostingClassifier(random_state=42),
        "Neural Net (MLP)":make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42))}
res={}
for n,m in models.items():
    s=cross_val_score(m,X.values,y.values,cv=cv,scoring="f1_macro",n_jobs=1); res[n]=s
    print(f"  {n:18s}: {s.mean():.3f} +/- {s.std():.3f}")
plt.figure(figsize=(5.4,4)); ns=list(res); mu=[res[n].mean() for n in ns]; sd=[res[n].std() for n in ns]
plt.bar(ns,mu,yerr=sd,capsize=6,color=["#2E7D32","#ED7D31","#7B4FA3"])
for i,(a,b) in enumerate(zip(mu,sd)): plt.text(i,a+b+0.015,f"{a:.2f}",ha="center")
plt.ylim(0,1); plt.ylabel("macro-F1 (5-fold)"); plt.title("Model comparison on fused features"); plt.xticks(rotation=10)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/model_comparison.png",dpi=150)

# ---------- (B) GRADIENT-BASED PGD ATTACK ----------
print("\n=== (B) gradient-based PGD attack (MLP surrogate -> Random Forest) ===")
Xtr,Xte,ytr,yte=train_test_split(X.values,y.values,test_size=0.25,random_state=42,stratify=y.values)
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)   # std units => eps in std
rf =RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced").fit(Xtr,ytr)
rng=np.random.default_rng(1); mk=ytr!="benign"; Xa=Xtr[mk].copy(); Xa[:,phys]+=rng.uniform(-1,1,(mk.sum(),len(phys)))*0.10
rfd=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced").fit(np.vstack([Xtr,Xa]),np.concatenate([ytr,ytr[mk]]))
mlp=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=600,random_state=42).fit(Xtr,ytr)
W=mlp.coefs_; b=mlp.intercepts_; L=len(W); labels=list(mlp.classes_)
def grad_in(Xb,yidx):
    a=Xb; pres=[]
    for i in range(L):
        z=a@W[i]+b[i]
        if i<L-1: pres.append(z); a=np.maximum(z,0)
    e=np.exp(z-z.max(1,keepdims=True)); sm=e/e.sum(1,keepdims=True)
    oh=np.zeros_like(sm); oh[np.arange(len(Xb)),yidx]=1; delta=sm-oh
    for i in range(L-1,-1,-1):
        dA=delta@W[i].T; delta=dA*(pres[i-1]>0) if i>0 else dA
    return dA
def pgd(Xb,yl,eps,steps=12):
    yi=np.array([labels.index(l) for l in yl]); o=Xb.copy(); Xadv=Xb.copy(); al=eps/4
    for _ in range(steps):
        g=grad_in(Xadv,yi); Xadv[:,phys]+=al*np.sign(g[:,phys])
        d=np.clip(Xadv[:,phys]-o[:,phys],-eps,eps); Xadv[:,phys]=o[:,phys]+d
    return Xadv
def f1(m,Xin): return f1_score(yte,m.predict(Xin),average="macro")
def merge(Xatk):
    ai=np.where(yte!="benign")[0]; out=Xte.copy(); out[ai]=Xatk; return out
clean=f1(rf,Xte); ai=yte!="benign"; EPS=[0.05,0.10,0.20]; pu=[];pd_=[]
for e in EPS:
    pu.append(f1(rf, merge(pgd(Xte[ai],yte[ai],e))))
    pd_.append(f1(rfd,merge(pgd(Xte[ai],yte[ai],e))))
print(f"  clean={clean:.3f}")
for e,a,c in zip(EPS,pu,pd_): print(f"  eps={e:.2f}: PGD undefended={a:.3f}  PGD defended={c:.3f}")
i=EPS.index(0.10); v=[clean,pu[i],pd_[i]]
plt.figure(figsize=(5.4,4))
plt.bar(["Clean","PGD attack","PGD + defense"],v,color=["#2E7D32","#C0392B","#2E75B6"])
for k,val in enumerate(v): plt.text(k,val+0.02,f"{val:.2f}",ha="center")
plt.ylim(0,1); plt.ylabel("macro-F1"); plt.title("Gradient-based (PGD) attack vs defense (eps=0.10)")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/pgd_attack.png",dpi=150)
print(f"\nSaved figures to {OUTPUT_DIR}/")
