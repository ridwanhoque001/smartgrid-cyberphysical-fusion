"""
================================================================================
Cyber-side cross-validation on the real IEC-104 ICS dataset (Matousek et al.).
================================================================================
Windows IEC 60870-5-104 packet flows into network features and detects attacks
with a Random Forest (5-fold out-of-fold). Shows the honest result used in the
paper: signature-based attacks (switching, scanning, injection, rogue-device)
are caught at high recall, while behavioral attacks (DoS flood, connection loss)
that mimic legitimate traffic evade the cyber features -- motivating fusion.

Point DATA_DIR at .../ics-dataset-for-smart-grids/but-iec104-i
Run:  pip install pandas scikit-learn matplotlib ; python iec104_crossval.py
================================================================================
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR = "data/ics-dataset-for-smart-grids/but-iec104-i"
W = 30                                   # packets per window
NOVEL = np.array(["45","46","100","120","121","122","123","124","125"])
ROGUE = "192.168.11.246"
FILES = {"normal":"normal-traffic.csv","switching":"switching-attack.csv",
         "scanning":"scanning-attack.csv","dos":"dos-attack.csv",
         "injection":"injection-attack.csv","rogue":"rogue-device.csv",
         "connloss":"connection-loss.csv"}

nd = pd.read_csv(os.path.join(DATA_DIR,"normal-traffic.csv"),sep=";",dtype=str,keep_default_na=False)
KNOWN = set(nd["srcIP"]) | set(nd["dstIP"])          # IPs seen in normal traffic

def windows(path, atype):
    d = pd.read_csv(path,sep=";",dtype=str,keep_default_na=False); N=(len(d)//W)*W
    rt = pd.to_numeric(d["Relative Time"][:N],errors="coerce").fillna(0).values.reshape(-1,W)
    ipl= pd.to_numeric(d["ipLen"][:N],errors="coerce").fillna(0).values.reshape(-1,W)
    fmt=d["fmt"][:N].values; ut=d["uType"][:N].values; a=d["asduType"][:N].values
    src=d["srcIP"][:N].values; dst=d["dstIP"][:N].values
    unk = (~np.isin(src,list(KNOWN)) | ~np.isin(dst,list(KNOWN))).reshape(-1,W)
    is_u=(fmt=="0x00000003").reshape(-1,W); is_i=(fmt=="0x00000000").reshape(-1,W)
    is_test=np.isin(ut,["0x00000010","0x00000020"]).reshape(-1,W)
    is36=(a=="36").reshape(-1,W); nov=np.isin(a,NOVEL).reshape(-1,W)
    rgp=((src==ROGUE)|(dst==ROGUE)).reshape(-1,W)
    span=rt[:,-1]-rt[:,0]; rate=W/(span+1e-6)
    F=np.column_stack([span,rate,ipl.mean(1),ipl.std(1),is_u.mean(1),is_i.mean(1),
                       is_test.mean(1),is36.mean(1),nov.mean(1),unk.mean(1)])
    return F, nov.any(1), rgp.any(1), span, rate, np.array([atype]*len(F))

FS=[];NV=[];RG=[];SP=[];RT=[];AT=[]
for at,fn in FILES.items():
    f,nv,rg,sp,rt,a=windows(os.path.join(DATA_DIR,fn),at)
    FS.append(f);NV.append(nv);RG.append(rg);SP.append(sp);RT.append(rt);AT.append(a)
F=np.nan_to_num(np.vstack(FS)); NV=np.concatenate(NV); RG=np.concatenate(RG)
SP=np.concatenate(SP); RT=np.concatenate(RT); AT=np.concatenate(AT)

flood=np.percentile(RT[AT=="normal"],99); gap=np.percentile(SP[AT=="normal"],99.5)
y=np.where(AT=="normal",0,
    ((NV)|(RG)|((AT=="dos")&(RT>flood))|((AT=="connloss")&(SP>gap))).astype(int))

pred=cross_val_predict(
    RandomForestClassifier(200,random_state=42,n_jobs=-1,class_weight="balanced"),
    F,y,cv=StratifiedKFold(5,shuffle=True,random_state=42))

print(f"windows={len(F)}  attack={y.sum()}  benign={(y==0).sum()}")
print(f"overall macro-F1      = {f1_score(y,pred,average='macro'):.3f}")
print(f"attack-class F1       = {f1_score(y,pred,pos_label=1):.3f}")
print(f"benign false-positive = {(pred[y==0]==1).mean():.3f}")
print("per-attack recall:")
rec={}
for at in ["injection","rogue","scanning","switching","connloss","dos"]:
    m=(AT==at)&(y==1)
    if m.sum(): rec[at]=(pred[m]==1).mean(); print(f"  {at:11s}: {rec[at]:.2f} ({m.sum()} win)")

# figure
plt.figure(figsize=(6.4,4))
labels=list(rec); vals=[rec[k] for k in labels]
col=["#2E7D32" if v>=0.5 else "#C0392B" for v in vals]
b=plt.bar(labels,vals,color=col)
for r,v in zip(b,vals): plt.text(r.get_x()+r.get_width()/2,v+0.02,f"{v:.2f}",ha="center")
plt.ylim(0,1.08); plt.ylabel("detection recall")
plt.title("Cyber-side cross-validation on real IEC-104 traffic")
plt.tight_layout(); plt.savefig("iec104_crossval.png",dpi=150)
print("saved iec104_crossval.png")
