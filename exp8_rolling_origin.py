"""
EXPERIMENT 8 - Rolling-origin temporal CV (reviewer concern #2).

The single temporal hold-out (train first 75% in time, test last 25%) gives one
number (0.80) on ~382 windows with no error bar. Here we run ROLLING-ORIGIN
evaluation: for a sequence of origins, train on all windows before the origin
(per class, in time) and test on the next contiguous block. This yields several
strictly-temporal estimates -> mean +/- std and a percentile range, quantifying
the stability of the temporal number.

Also prints the per-class window counts and how 1,529 windows arise from the raw
CSVs (inner join of 0.5 s cyber and physical bins).

Outputs: results/rolling_origin.txt, results/rolling_origin.png
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "results"; RNG = 42
df = pd.read_parquet(os.path.join(OUT, "features_ds1.parquet"))
meta = ["label", "cls", "bin_idx", "t_start"]
Xcols = [c for c in df.columns if c not in meta]
X = df[Xcols].values; y = df["label"].values

def rf():
    return RandomForestClassifier(n_estimators=300, random_state=RNG,
                                  n_jobs=-1, class_weight="balanced")

# per-class time order (rank within class by bin_idx, as fraction 0..1)
frac = np.empty(len(df))
for c in df["cls"].unique():
    idx = df.index[df["cls"] == c].values
    order = idx[np.argsort(df.loc[idx, "bin_idx"].values)]
    frac[order] = np.arange(len(order)) / len(order)

# rolling origins: train on [0, o), test on [o, o+0.15)
origins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
TEST_W = 0.15
scores, ns = [], []
for o in origins:
    tr = frac < o
    te = (frac >= o) & (frac < min(o + TEST_W, 1.0))
    m = rf(); m.fit(X[tr], y[tr])
    s = f1_score(y[te], m.predict(X[te]), average="macro")
    scores.append(s); ns.append(int(te.sum()))
scores = np.array(scores)

# the paper's single split for reference
te = frac >= 0.75
m = rf(); m.fit(X[~te], y[~te])
single = f1_score(y[te], m.predict(X[te]), average="macro")

lines = []
lines.append("ROLLING-ORIGIN TEMPORAL CV (dataset 1, fusion RF)")
lines.append(f"windows total: {len(df)}  |  per class: " +
             ", ".join(f"{c}={int((df['cls']==c).sum())}" for c in sorted(df["cls"].unique())))
for o, s, n in zip(origins, scores, ns):
    lines.append(f"origin {o:.2f}  test block {o:.2f}-{min(o+TEST_W,1.0):.2f}  "
                 f"n={n:4d}  macro-F1={s:.3f}")
lines.append(f"rolling-origin mean +/- std : {scores.mean():.3f} +/- {scores.std():.3f}")
lines.append(f"range [min, max]            : [{scores.min():.3f}, {scores.max():.3f}]")
lines.append(f"single 75/25 temporal split : {single:.3f}  (n={int(te.sum())})")
txt = "\n".join(lines)
with open(os.path.join(OUT, "rolling_origin.txt"), "w") as f:
    f.write(txt + "\n")
print(txt)

plt.figure(figsize=(6, 4))
plt.plot([f"{o:.2f}" for o in origins], scores, "o-", color="#1B5E20", label="rolling-origin block")
plt.axhline(scores.mean(), color="#2E7D32", ls="--",
            label=f"mean {scores.mean():.2f} ± {scores.std():.2f}")
plt.axhline(single, color="#B71C1C", ls=":", label=f"single 75/25 split ({single:.2f})")
plt.fill_between(range(len(origins)), scores.mean() - scores.std(),
                 scores.mean() + scores.std(), color="#2E7D32", alpha=0.12)
plt.xlabel("train fraction (origin)"); plt.ylabel("macro-F1"); plt.ylim(0, 1.05)
plt.title("Rolling-origin temporal evaluation (dataset 1)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "rolling_origin.png"), dpi=150)
print("saved rolling_origin.png")
