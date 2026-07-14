"""
SmartGrid Cyber-Physical Attack Dataset -- FUSION pipeline
Compares: physical-only vs cyber-only vs FUSED detection (the contribution).
"""
import os, glob, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT = os.environ.get("OUT", "outputs"); os.makedirs(OUT, exist_ok=True)
WINDOW = 0.5  # seconds per time window

# folder name on disk -> clean class label
CLASSES = {"benign":"benign","backdoor":"backdoor","Bruteforce":"bruteforce",
           "FDI":"fdi","ransomware":"ransomware","reverseshell":"reverseshell"}
PROTOS = ["TCP","Modbus/TCP","HTTP","TLSv1.2","ARP","ICMPv6"]
PHYS_COLS = ["Freq","Theta","V_A","V_B","V_C","I_A","I_B","I_C",
             "ActivePower","ReactivePower","BreakerStatus"]

def to_num(s):
    """Turn '85,732,060' or '62.0' (text) into real numbers."""
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False)
                          .str.replace('"',"",regex=False), errors="coerce")

def find(folder, name):
    hits = glob.glob(os.path.join(DATA_DIR,"Dataset",folder,name))
    return hits[0] if hits else None

CANON = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst",
         "_ws.col.Protocol","ip.len","tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"]

def _looks_epoch(series):
    v = pd.to_numeric(series, errors="coerce")
    return v.between(1.6e9, 1.8e9).mean() > 0.5

def normalize_cyber(cy):
    """Some files are missing the leading 'node' column (everything shifted)."""
    cy = cy.loc[:, ~cy.columns.astype(str).str.startswith("Unnamed")]
    if "frame.time_epoch" in cy.columns and _looks_epoch(cy["frame.time_epoch"]):
        return cy                                  # already correct
    # shifted: first real column is the timestamp -> drop assumed 'node', re-map
    cy = cy.iloc[:, :11].copy()
    cy.columns = CANON
    return cy

def cyber_features(path):
    cy = pd.read_csv(path, low_memory=False)
    cy.columns = [c.strip() for c in cy.columns]
    cy = normalize_cyber(cy)
    t = to_num(cy["frame.time_epoch"]); cy = cy[t.notna()].copy(); cy["t"]=t[t.notna()].values
    cy["bin"] = (cy["t"]//WINDOW).astype(np.int64)
    cy["iplen"] = to_num(cy["ip.len"])
    g = cy.groupby("bin")
    feat = pd.DataFrame({
        "cyb_pkt_count": g.size(),
        "cyb_iplen_mean": g["iplen"].mean(),
        "cyb_iplen_sum":  g["iplen"].sum(),
        "cyb_iplen_std":  g["iplen"].std(),
        "cyb_n_src_ip":   g["ip.src"].nunique(),
        "cyb_n_dst_ip":   g["ip.dst"].nunique(),
    })
    proto = cy.groupby(["bin","_ws.col.Protocol"]).size().unstack(fill_value=0)
    for p in PROTOS:
        if p in proto.columns:
            feat["cyb_frac_"+p.replace("/","_")] = proto[p]/feat["cyb_pkt_count"]
        else:
            feat["cyb_frac_"+p.replace("/","_")] = 0.0
    return feat

def phys_features(path):
    ph = pd.read_csv(path, low_memory=False)
    ph.columns = [c.strip() for c in ph.columns]
    t = to_num(ph["@timestamp"]); ph = ph[t.notna()].copy(); ph["t"]=t[t.notna()].values
    ph["bin"] = (ph["t"]//WINDOW).astype(np.int64)
    for c in PHYS_COLS: ph[c] = to_num(ph[c])
    g = ph.groupby("bin")[PHYS_COLS]
    feat = g.agg(["mean","std"])
    feat.columns = ["phys_"+a+"_"+b for a,b in feat.columns]
    return feat

rows = []
for folder, label in CLASSES.items():
    cpath = find(folder,"[Cc]yber.csv"); ppath = find(folder,"Physical.csv")
    cf = cyber_features(cpath); pf = phys_features(ppath)
    joined = pf.join(cf, how="inner")          # FUSION: align cyber+physical by time window
    joined["label"] = label
    rows.append(joined)
    print(f"{label:12s}: {len(joined)} fused time-windows")

data = pd.concat(rows).reset_index(drop=True)
data = data.replace([np.inf,-np.inf], np.nan).fillna(0)
y = data["label"]; X = data.drop(columns=["label"])
phys_cols = [c for c in X.columns if c.startswith("phys_")]
cyb_cols  = [c for c in X.columns if c.startswith("cyb_")]
print(f"\nTotal windows: {len(X)} | physical features: {len(phys_cols)} | cyber features: {len(cyb_cols)}")

# ---------------- CROSS-VALIDATION (more trustworthy than one split) ----------
print("\n=== 5-FOLD CROSS-VALIDATION (macro-F1) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for cols, name in [(phys_cols,"Physical"),(cyb_cols,"Cyber"),(list(X.columns),"Fusion")]:
    est = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                  class_weight="balanced")
    sc = cross_val_score(est, X[cols], y, cv=cv, scoring="f1_macro", n_jobs=1)
    cv_results[name] = sc
    print(f"{name:9s}: macro-F1 {sc.mean():.3f} +/- {sc.std():.3f}   folds=[{', '.join(f'{s:.2f}' for s in sc)}]")

plt.figure(figsize=(5,4))
names=list(cv_results); means=[cv_results[n].mean() for n in names]; stds=[cv_results[n].std() for n in names]
plt.bar(names, means, yerr=stds, capsize=6, color=["#5B9BD5","#ED7D31","#2E7D32"])
plt.ylabel("macro-F1 (5-fold mean +/- std)"); plt.ylim(0,1)
plt.title("Cross-validated detection performance")
for i,(mm,ss) in enumerate(zip(means,stds)): plt.text(i, mm+ss+0.02, f"{mm:.2f}", ha="center")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fusion_cv_comparison.png"), dpi=150)
with open(os.path.join(OUT,"cv_results.txt"),"w") as f:
    f.write("5-fold cross-validated macro-F1\n")
    for n in names: f.write(f"{n:9s}: {cv_results[n].mean():.3f} +/- {cv_results[n].std():.3f}\n")
print("Saved fusion_cv_comparison.png + cv_results.txt")


Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

def run(cols, name):
    m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    m.fit(Xtr[cols], ytr); pred = m.predict(Xte[cols])
    f1 = f1_score(yte,pred,average="macro"); acc = accuracy_score(yte,pred)
    print(f"{name:16s} -> accuracy {acc:.3f} | macro-F1 {f1:.3f}")
    return m, pred, f1, acc

print("\n=== RESULTS ===")
_, _, f1_p, acc_p = run(phys_cols, "Physical-only")
_, _, f1_c, acc_c = run(cyb_cols,  "Cyber-only")
m_f, pred_f, f1_f, acc_f = run(list(X.columns), "FUSION")

with open(os.path.join(OUT,"results_summary.txt"),"w") as f:
    f.write("SmartGrid Cyber-Physical Fusion -- results\n")
    f.write(f"windows={len(X)}, window={WINDOW}s\n")
    f.write(f"Physical-only : acc={acc_p:.3f} macroF1={f1_p:.3f}\n")
    f.write(f"Cyber-only    : acc={acc_c:.3f} macroF1={f1_c:.3f}\n")
    f.write(f"FUSION        : acc={acc_f:.3f} macroF1={f1_f:.3f}\n\n")
    f.write(classification_report(yte,pred_f))

# comparison bar
plt.figure(figsize=(5,4))
plt.bar(["Physical","Cyber","Fusion"],[f1_p,f1_c,f1_f],
        color=["#5B9BD5","#ED7D31","#2E7D32"])
plt.ylabel("macro-F1"); plt.ylim(0,1); plt.title("Detection performance by data source")
for i,v in enumerate([f1_p,f1_c,f1_f]): plt.text(i,v+0.02,f"{v:.2f}",ha="center")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fusion_comparison.png"),dpi=150)

# confusion matrix (fusion)
labels_sorted = sorted(y.unique())
cm = confusion_matrix(yte,pred_f,labels=labels_sorted)
plt.figure(figsize=(6,5)); plt.imshow(cm,cmap="Blues")
plt.xticks(range(len(labels_sorted)),labels_sorted,rotation=45,ha="right")
plt.yticks(range(len(labels_sorted)),labels_sorted)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Fusion model confusion matrix")
for i in range(len(labels_sorted)):
    for j in range(len(labels_sorted)):
        plt.text(j,i,cm[i,j],ha="center",va="center",
                 color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fusion_confusion_matrix.png"),dpi=150)

# SHAP feature importance on fusion model
try:
    import shap
    samp = Xte.sample(min(300,len(Xte)),random_state=42)
    sv = shap.TreeExplainer(m_f).shap_values(samp)
    arr = np.asarray(sv)
    # importance per FEATURE: average |SHAP| over samples (and over classes if 3-D)
    if arr.ndim == 3:        # (samples, features, classes)
        imp = np.abs(arr).mean(axis=(0, 2))
    else:                    # (samples, features)
        imp = np.abs(arr).mean(axis=0)
    imp = np.asarray(imp).ravel()
    order = np.argsort(imp)[::-1][:15]
    names = [X.columns[i] for i in order]
    colors = ["#2E7D32" if names[k].startswith("cyb_") else "#5B9BD5" for k in range(len(names))]
    plt.figure(figsize=(7,5))
    plt.barh(range(len(order))[::-1], imp[order], color=colors)
    plt.yticks(range(len(order))[::-1], names, fontsize=8)
    plt.xlabel("mean |SHAP| (importance)")
    plt.title("Top features (green=cyber, blue=physical)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"fusion_shap_importance.png"),dpi=150)
    print("SHAP figure saved.")
except Exception as e:
    print("SHAP step skipped:", e)

print("\nFiles written to", OUT)
