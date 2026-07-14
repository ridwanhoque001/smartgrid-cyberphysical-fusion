"""Publication-quality regen of fusion_shap_importance.png, matching the exact
computation in cyber_physical_fusion.py (train/test split, aggregation axis=(0,2))."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif","font.size":9,
  "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8})

df = pd.read_parquet("results/features_ds1.parquet")
meta=["label","cls","bin_idx","t_start"]
Xcols=[c for c in df.columns if c not in meta]
X = df[Xcols]; y = df["label"].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
m = RandomForestClassifier(300,random_state=42,n_jobs=-1,class_weight="balanced").fit(Xtr,ytr)
samp = Xte.sample(min(300,len(Xte)),random_state=42)
arr = np.asarray(shap.TreeExplainer(m).shap_values(samp))
imp = np.abs(arr).mean(axis=(0,2)) if arr.ndim==3 else np.abs(arr).mean(axis=0)
imp = np.asarray(imp).ravel()
order = np.argsort(imp)[::-1][:15]
names = [Xcols[i] for i in order]
colors = ["#2E7D32" if n.startswith("cyb_") else "#5B9BD5" for n in names]
fig,ax=plt.subplots(figsize=(3.45,3.15))
ax.barh(range(len(order))[::-1], imp[order], color=colors, edgecolor="white", linewidth=0.4)
ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels(names)
ax.set_xlabel("mean |SHAP| (importance)")
ax.set_title("Top features (green = cyber, blue = physical)")
ax.grid(axis="x", alpha=0.25)
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("results/fusion_shap_importance.png",dpi=320)
print("top5:", names[:5]); print("saved")
