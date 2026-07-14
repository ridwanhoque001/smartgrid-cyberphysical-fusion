"""EXPERIMENT 10 - Is the temporal drop RF-specific? One model per invocation.
Usage: python exp10_temporal_baselines.py <idx>   (idx 0..4)  ; then: ... plot
"""
import os, sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
OUT="results"; RNG=42; K=4; DB=os.path.join(OUT,"temporal_baselines.json")
df=pd.read_parquet(os.path.join(OUT,"features_ds1.parquet"))
meta=["label","cls","bin_idx","t_start"]; Xcols=[c for c in df.columns if c not in meta]
y=df["label"].values; base=df[Xcols].values
frac=np.empty(len(df)); Xstack=np.zeros((len(df),len(Xcols)*(K+1)))
for c in df["cls"].unique():
    idx=df.index[df["cls"]==c].values; o=idx[np.argsort(df.loc[idx,"bin_idx"].values)]
    frac[o]=np.arange(len(o))/len(o)
    for j,ri in enumerate(o):
        Xstack[ri]=np.concatenate([base[o[max(0,j-d)]] for d in range(K+1)])
X=base
MODELS=[
 ("Random Forest", RandomForestClassifier(250,random_state=RNG,n_jobs=-1,class_weight="balanced"), "X"),
 ("Hist Gradient Boosting", HistGradientBoostingClassifier(max_iter=200,random_state=RNG), "X"),
 ("MLP (per-window)", make_pipeline(StandardScaler(),MLPClassifier((128,64),max_iter=300,random_state=RNG)), "X"),
 ("TDNN (temporal)", make_pipeline(StandardScaler(),MLPClassifier((256,128),max_iter=300,random_state=RNG)), "S"),
 ("Logistic (linear)", make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")), "X"),
]
def rolling(mk,data):
    sc=[]
    for o in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
        tr=frac<o; te=(frac>=o)&(frac<min(o+0.15,1.0))
        if te.sum()<10: continue
        m=clone(mk); m.fit(data[tr],y[tr]); sc.append(f1_score(y[te],m.predict(data[te]),average="macro"))
    return np.array(sc)
if sys.argv[1]=="plot":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif","font.size":15,
      "axes.titlesize":14.5,"axes.labelsize":15,"xtick.labelsize":12,"ytick.labelsize":13,"legend.fontsize":12.5})
    d=json.load(open(DB)); names=[m[0] for m in MODELS]
    disp=[n.replace(" (","\n(").replace(" Gradient","\nGradient").replace("Random ","Random\n") for n in names]
    shuf=[d[n]["shuf"] for n in names]; roll=[d[n]["roll"] for n in names]; sd=[d[n]["sd"] for n in names]
    xp=np.arange(len(names)); w=0.38
    fig,ax=plt.subplots(figsize=(8.4,4.7))
    ax.bar(xp-w/2,shuf,w,label="shuffled CV (optimistic)",color="#B0B0B0")
    ax.bar(xp+w/2,roll,w,yerr=sd,capsize=4,label="rolling-origin (temporal)",color="#1B5E20")
    for i,v in enumerate(shuf): ax.text(xp[i]-w/2,v+0.015,f"{v:.2f}",ha="center",fontsize=11)
    for i,v in enumerate(roll): ax.text(xp[i]+w/2,v+sd[i]+0.015,f"{v:.2f}",ha="center",fontsize=11)
    ax.set_xticks(xp); ax.set_xticklabels(disp); ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.1)
    ax.set_title("Temporal drift is not model-specific: every family drops under rolling-origin")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=2,frameon=False)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"temporal_baselines.png"),dpi=320)
    with open(os.path.join(OUT,"temporal_baselines.txt"),"w") as f:
        f.write("IS THE TEMPORAL DROP RF-SPECIFIC? (dataset 1)\n")
        for n in names: f.write(f"{n:24s} shuffled={d[n]['shuf']:.3f}  rolling-origin={d[n]['roll']:.3f}±{d[n]['sd']:.3f}  drop={d[n]['shuf']-d[n]['roll']:.3f}\n")
    print(open(os.path.join(OUT,"temporal_baselines.txt")).read()); print("saved temporal_baselines.png")
else:
    i=int(sys.argv[1]); nm,mk,tag=MODELS[i]; data=X if tag=="X" else Xstack
    cv=StratifiedKFold(5,shuffle=True,random_state=RNG)
    s=cross_val_score(mk,data,y,cv=cv,scoring="f1_macro",n_jobs=1).mean()
    r=rolling(mk,data)
    d=json.load(open(DB)) if os.path.exists(DB) else {}
    d[nm]={"shuf":float(s),"roll":float(r.mean()),"sd":float(r.std())}
    json.dump(d,open(DB,"w"))
    print(f"{nm}: shuffled={s:.3f} rolling={r.mean():.3f}±{r.std():.3f} drop={s-r.mean():.3f}")
