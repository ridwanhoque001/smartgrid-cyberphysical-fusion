"""
Uncertainty-aware selective detection on the cyber-physical fusion detector.
Computes per-window confidence kappa(x)=max_c p_c(x) and predictive entropy,
then a risk-coverage curve: abstaining on the least-confident windows (which are
escalated to physical/operator review) sharply raises accuracy on the retained
set and captures most errors.  Builds features via bf.py (feats.pkl).
Run:  python bf.py  (with DATA_DIR set)  ;  python selective_detection.py
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = pd.read_pickle("feats.pkl"); y = d["label"].values; X = d.drop(columns=["label"]).values
cv = StratifiedKFold(5, shuffle=True, random_state=42)
proba = cross_val_predict(
    RandomForestClassifier(300, random_state=42, n_jobs=-1, class_weight="balanced"),
    X, y, cv=cv, method="predict_proba")
classes = np.unique(y); pred = classes[proba.argmax(1)]
kappa = proba.max(1)                                   # confidence
entropy = -(proba*np.log(proba+1e-12)).sum(1)          # predictive entropy
order = np.argsort(-kappa)                              # most confident first

print(f"full: acc={accuracy_score(y,pred):.3f} macroF1={f1_score(y,pred,average='macro'):.3f}")
covs=[1.0,0.95,0.9,0.85,0.8,0.7]; A=[];F=[]
for c in covs:
    idx=order[:int(c*len(y))]; A.append(accuracy_score(y[idx],pred[idx])); F.append(f1_score(y[idx],pred[idx],average='macro'))
    print(f"  coverage={c*100:3.0f}%  acc={A[-1]:.3f}  macroF1={F[-1]:.3f}")
err=(pred!=y)
print(f"mean confidence: correct={kappa[~err].mean():.3f} errors={kappa[err].mean():.3f}")
print(f"errors captured in abstained 20%: {100*err[order[int(0.8*len(y)):]].sum()/err.sum():.0f}%")

plt.figure(figsize=(5.6,4))
plt.plot([c*100 for c in covs],A,'o-',label='accuracy',color='#2E75B6')
plt.plot([c*100 for c in covs],F,'s-',label='macro-F1',color='#2E7D32')
plt.gca().invert_xaxis(); plt.xlabel('coverage (%)'); plt.ylabel('score on retained windows')
plt.title('Selective detection (abstain -> escalate to physical review)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig('selective.png',dpi=150)
print("saved selective.png")
