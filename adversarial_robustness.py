"""
================================================================================
Adversarial robustness of the cyber-physical FUSION detector  (the novelty step)
================================================================================
What this does, in plain English:
  PART A — Tuning: tries many Random-Forest settings (GridSearch-style) and keeps
           the best, to check if we can push the 0.96 macro-F1 any higher.
  PART B — Robustness: simulates a realistic attacker who can nudge the PHYSICAL
           sensor readings a little (within +/- eps x each feature's std) to try
           to fool the detector. We measure how far the score drops, then DEFEND
           with "adversarial training" (teach the model on perturbed examples too)
           and show the score recovers.

Why it matters: prior fusion work (incl. the dataset authors, Sweeten et al. 2025)
reports CLEAN accuracy only. Showing the attack hurts, and the defense helps, is
the genuinely novel contribution.

Run:
    pip install pandas scikit-learn matplotlib
    python adversarial_robustness.py
Set DATA_DIR to the folder that contains the attack-type subfolders.
================================================================================
"""
import os, glob, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ------------------------------------------------------------------ SETTINGS
DATA_DIR   = "data/Dataset"     # contains benign/, FDI/, ... each with Cyber.csv + Physical.csv
OUTPUT_DIR = "results"
WINDOW     = 0.5                # seconds per time-window
EPS_LEVELS = [0.05, 0.10, 0.20] # attack strengths (fraction of each feature's std)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = {"benign":"benign","backdoor":"backdoor","Bruteforce":"bruteforce",
           "FDI":"fdi","ransomware":"ransomware","reverseshell":"reverseshell"}
PROTOS = ["TCP","Modbus/TCP","HTTP","TLSv1.2","ARP","ICMPv6"]
PHYS_COLS = ["Freq","Theta","V_A","V_B","V_C","I_A","I_B","I_C",
             "ActivePower","ReactivePower","BreakerStatus"]
CANON = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","_ws.col.Protocol",
         "ip.len","tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"]

# ----------------------------------------------------- feature engineering
def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False)
                          .str.replace('"',"",regex=False), errors="coerce")
def find(folder,name):
    h=glob.glob(os.path.join(DATA_DIR,folder,name)); return h[0] if h else None
def _epoch(s):
    return pd.to_numeric(s,errors="coerce").between(1.6e9,1.8e9).mean()>0.5
def normalize_cyber(cy):
    cy=cy.loc[:,~cy.columns.astype(str).str.startswith("Unnamed")]
    if "frame.time_epoch" in cy.columns and _epoch(cy["frame.time_epoch"]): return cy
    cy=cy.iloc[:,:11].copy(); cy.columns=CANON; return cy   # fix the shifted FDI file
def cyber_features(path):
    cy=pd.read_csv(path,low_memory=False); cy.columns=[c.strip() for c in cy.columns]
    cy=normalize_cyber(cy); t=to_num(cy["frame.time_epoch"])
    cy=cy[t.notna()].copy(); cy["t"]=t[t.notna()].values
    cy["bin"]=(cy["t"]//WINDOW).astype(np.int64); cy["iplen"]=to_num(cy["ip.len"])
    g=cy.groupby("bin")
    f=pd.DataFrame({"cyb_pkt_count":g.size(),"cyb_iplen_mean":g["iplen"].mean(),
        "cyb_iplen_sum":g["iplen"].sum(),"cyb_iplen_std":g["iplen"].std(),
        "cyb_n_src_ip":g["ip.src"].nunique(),"cyb_n_dst_ip":g["ip.dst"].nunique()})
    pr=cy.groupby(["bin","_ws.col.Protocol"]).size().unstack(fill_value=0)
    for p in PROTOS:
        f["cyb_frac_"+p.replace("/","_")] = pr[p]/f["cyb_pkt_count"] if p in pr.columns else 0.0
    return f
def phys_features(path):
    ph=pd.read_csv(path,low_memory=False); ph.columns=[c.strip() for c in ph.columns]
    t=to_num(ph["@timestamp"]); ph=ph[t.notna()].copy(); ph["t"]=t[t.notna()].values
    ph["bin"]=(ph["t"]//WINDOW).astype(np.int64)
    for c in PHYS_COLS: ph[c]=to_num(ph[c])
    f=ph.groupby("bin")[PHYS_COLS].agg(["mean","std"])
    f.columns=["phys_"+a+"_"+b for a,b in f.columns]; return f

def build():
    rows=[]
    for folder,label in CLASSES.items():
        j=phys_features(find(folder,"Physical.csv")).join(
            cyber_features(find(folder,"[Cc]yber.csv")), how="inner")
        j["label"]=label; rows.append(j)
    d=pd.concat(rows).reset_index(drop=True).replace([np.inf,-np.inf],np.nan).fillna(0)
    return d

# --------------------------------------------------------------- main
print("Building fused features ...")
data=build(); y=data["label"]; X=data.drop(columns=["label"])
cols=list(X.columns); phys_idx=[cols.index(c) for c in cols if c.startswith("phys_")]
print(f"  {len(X)} windows, {len(cols)} features ({len(phys_idx)} physical)")

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
Xtr,Xte,ytr,yte=Xtr.values,Xte.values,ytr.values,yte.values

# ---- PART A: tuning ----
print("\n=== PART A: hyperparameter tuning ===")
grid={"n_estimators":[200,300,500],"max_depth":[None,10,20],
      "min_samples_leaf":[1,2,4],"max_features":["sqrt","log2",0.5]}
rs=RandomizedSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced"),
    grid,n_iter=10,scoring="f1_macro",cv=3,random_state=42,n_jobs=-1).fit(Xtr,ytr)
default=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1,class_weight="balanced").fit(Xtr,ytr)
tuned=RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced",**rs.best_params_).fit(Xtr,ytr)
print("  best params:",rs.best_params_)
print(f"  test macro-F1: default={f1_score(yte,default.predict(Xte),average='macro'):.3f} "
      f"tuned={f1_score(yte,tuned.predict(Xte),average='macro'):.3f}")

# ---- PART B: adversarial robustness ----
print("\n=== PART B: adversarial robustness ===")
model=tuned; classes=list(model.classes_)
std_phys=np.asarray(pd.DataFrame(Xtr).iloc[:,phys_idx].std()).astype(float); std_phys[std_phys==0]=1e-6

def attack(m,Xv,yv,eps,K=80,seed=0):
    """Black-box evasion: perturb ONLY physical features within +/-eps*std;
       keep the perturbation that least supports the true class."""
    rng=np.random.default_rng(seed); Xadv=Xv.copy()
    ai=np.where(yv!="benign")[0]; Na=len(ai)
    P=np.repeat(Xv[ai],K,axis=0)
    P[:,phys_idx]+=rng.uniform(-1,1,(Na*K,len(phys_idx)))*eps*std_phys
    pr=m.predict_proba(P); tci=np.repeat([classes.index(l) for l in yv[ai]],K)
    best=pr[np.arange(len(pr)),tci].reshape(Na,K).argmin(1)
    Xadv[ai]=P.reshape(Na,K,-1)[np.arange(Na),best]; return Xadv

def adv_train(Xv,yv,eps,seed=1):
    rng=np.random.default_rng(seed); m=yv!="benign"; Xa=Xv[m].copy()
    Xa[:,phys_idx]+=rng.uniform(-1,1,(m.sum(),len(phys_idx)))*eps*std_phys
    return np.vstack([Xv,Xa]),np.concatenate([yv,yv[m]])

clean=f1_score(yte,model.predict(Xte),average="macro")
undef=[f1_score(yte,model.predict(attack(model,Xte,yte,e)),average="macro") for e in EPS_LEVELS]
Xa,ya=adv_train(Xtr,ytr,0.10)
defended=RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced",**rs.best_params_).fit(Xa,ya)
deful=[f1_score(yte,defended.predict(attack(defended,Xte,yte,e)),average="macro") for e in EPS_LEVELS]
print(f"  clean macro-F1 = {clean:.3f}")
for e,u,d in zip(EPS_LEVELS,undef,deful): print(f"  eps={e:.2f}: undefended={u:.3f}  defended={d:.3f}")

# ---- figures ----
i=EPS_LEVELS.index(0.10); v=[clean,undef[i],deful[i]]
plt.figure(figsize=(5.2,4))
plt.bar(["Clean","Attacked\n(no defense)","Attacked\n(adv. trained)"],v,color=["#2E7D32","#C0392B","#2E75B6"])
for k,val in enumerate(v): plt.text(k,val+0.02,f"{val:.2f}",ha="center")
plt.ylim(0,1); plt.ylabel("macro-F1"); plt.title("Adversarial robustness (eps=0.10)")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/adv_robustness_bars.png",dpi=150)
plt.figure(figsize=(5.2,4))
plt.plot(EPS_LEVELS,undef,"o-",color="#C0392B",label="undefended")
plt.plot(EPS_LEVELS,deful,"s-",color="#2E75B6",label="adversarially trained")
plt.axhline(clean,ls="--",color="#2E7D32",label="clean")
plt.xlabel("attack strength eps (x feature std)"); plt.ylabel("macro-F1"); plt.ylim(0,1); plt.legend()
plt.title("Robustness vs attack strength"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/adv_robustness_curve.png",dpi=150)
print(f"\nSaved figures to {OUTPUT_DIR}/")
