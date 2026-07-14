"""
EXPERIMENT 4 — Semantic value manipulation on IEEE-14: an observability taxonomy of
what each detection layer can and cannot catch, and where fusion closes the gap.

Motivation (ties Testbed 2 <-> Dataset 3):
  On real IEC-104 traffic we showed semantic value-manipulation attacks are invisible to
  the CYBER view (packet headers/timing). This experiment reproduces that boundary on the
  physics-controlled IEEE-14 testbed and shows exactly WHEN the value manipulation becomes
  detectable once we add a physics (state-estimation residual) view.

Three attack classes on the SAME grid (DC model, paper Eq. 1):
  (A) Structural/cyber  : anomalous message pattern (simulated cyber layer); measurements
                          left physically consistent.            -> cyber-visible, physics-blind
  (B) Naive value       : an operator/MITM changes a reported value with NO topology
                          knowledge => perturbation NOT in col(H) => residual spikes.
                          -> cyber-invisible, physics-visible
  (C) Stealthy FDI      : a = H c with full topology (Liu et al. 2011) => residual ~ 0.
                          -> cyber-invisible, physics-invisible  (the fundamental limit)

Three detection views, ONE mixed training set (detector is NOT told the class):
  * Cyber view  : RF on simulated cyber/message features.
  * Physics view: RF on the state-estimation residual vector e = z - H x_hat.
  * Fusion      : RF on [cyber (+) residual].

Also: a GRADED topology-knowledge sweep (0%..100%) for the value attack, mirroring the
graded cyber adversary elsewhere in the paper: as attacker knowledge -> 100%, the attack
slides from naive (physics catches it) to stealthy (nobody does), and the residual falls
to the metering-noise floor.

HONEST SCOPE: the cyber layer on IEEE-14 is simulated (as stated in the paper); the residual
result for naive vs stealthy false data is classical bad-data detection (Liu et al. 2011) —
the contribution here is the UNIFIED taxonomy across one detector/testbed and the graded
knowledge characterization, not a new detector.

Outputs: results/semantic_fusion.txt, results/semantic_fusion.png
"""
import os, numpy as np
import pandapower.networks as nw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

np.random.seed(11); OUT="results"; os.makedirs(OUT, exist_ok=True)

# ---------- IEEE-14 DC measurement model (identical construction to exp3) ----------
net=nw.case14(); nb=len(net.bus); slack=int(net.ext_grid.bus.iloc[0])
non_slack=[b for b in range(nb) if b!=slack]; pos={b:i for i,b in enumerate(non_slack)}
br=[]
for _,l in net.line.iterrows():
    x=l.x_ohm_per_km*l.length_km; br.append((int(l.from_bus),int(l.to_bus),x if x>1e-6 else 0.05))
for _,t in net.trafo.iterrows():
    z=t.vk_percent/100.0; br.append((int(t.hv_bus),int(t.lv_bus),max(z,0.05)))
m=len(br); n=len(non_slack)
H=np.zeros((m,n))
for k,(i,j,x) in enumerate(br):
    if i in pos: H[k,pos[i]]+=1.0/x
    if j in pos: H[k,pos[j]]-=1.0/x
Hpinv=np.linalg.pinv(H)
NOISE=0.0015                                  # metering-noise floor

def make_states(ns, rng):
    Z=[]
    for _ in range(ns):
        theta=rng.normal(0,0.04,n); Z.append(H@theta + rng.normal(0,NOISE,m))
    return np.array(Z)

def residual_vec(z):                          # e = z - H x_hat  (per-branch residual)
    return z - H@(Hpinv@z)
def residual(z):                              # scalar LS residual (BDD statistic)
    return np.linalg.norm(residual_vec(z))

# ---------- simulated cyber/message features (IEEE-14 has no real cyber layer) ----------
# [msg_rate, iat_mean, iat_std, dup_frac, type_entropy, cmd_frac]
CY_MU=np.array([100.,0.040,0.010,0.020,1.80,0.050])
CY_SD=np.array([6.,0.004,0.002,0.005,0.08,0.010])
def cyber_feats(nrows, rng, structural=False):
    f=rng.normal(CY_MU, CY_SD, size=(nrows,len(CY_MU)))
    if structural:                            # replay/report-block signature
        f[:,0]*=0.60                          # fewer distinct updates (report block)
        f[:,3]+=0.25                          # duplicate fraction up (replay)
        f[:,4]-=0.60                          # lower message-type entropy
    return f

# ---------- attacks ----------
TARGET=0; DELTA=0.14                           # meaningful bias on target state coord
def stealthy_fdi(z, rng):                      # a = H c, full topology  -> residual ~ 0
    c=rng.uniform(-0.05,0.05,n); c[TARGET]=DELTA; return z+H@c
def naive_value(z, rng, k=2):                  # change k reported branch values, no topology
    za=z.copy(); idx=rng.choice(m,size=k,replace=False)
    za[idx]+=rng.uniform(0.05,0.12,k)*rng.choice([-1,1],k); return za
def graded_fdi(z, rng, frac):
    """attacker knows fraction 'frac' of branches: applies the consistent injection a=Hc
    only on known branches, leaves unknown branches untouched -> residual from the mismatch.
    frac=0 -> naive-ish (bias only on 1 branch); frac=1 -> fully stealthy."""
    c=np.zeros(n); c[TARGET]=DELTA; c[np.arange(n)!=TARGET]=rng.uniform(-0.05,0.05,n-1)
    a_full=H@c
    nk=max(1,int(round(frac*m))); known=rng.choice(m,size=nk,replace=False)
    a=np.zeros(m); a[known]=a_full[known]      # can only set the branches it knows
    return z+a

# ---------- build a mixed dataset: benign + equal parts A/B/C ----------
rng=np.random.default_rng(11); K=1500
Zben=make_states(K,rng)
Zstruct=make_states(K,rng)                                   # measurements consistent
Znaive=np.array([naive_value(z,rng) for z in make_states(K,rng)])
Zstealth=np.array([stealthy_fdi(z,rng) for z in make_states(K,rng)])

# feature blocks for every sample
def cy_block(Z, structural): return cyber_feats(len(Z),rng,structural=structural)
CY = np.vstack([cy_block(Zben,False), cy_block(Zstruct,True),
                cy_block(Znaive,False), cy_block(Zstealth,False)])
PH = np.vstack([np.array([residual_vec(z) for z in Z]) for Z in (Zben,Zstruct,Znaive,Zstealth)])
y  = np.r_[np.zeros(K), np.ones(3*K)].astype(int)                     # attack vs benign
cls= np.r_[np.full(K,"benign"),np.full(K,"structural"),np.full(K,"naive"),np.full(K,"stealthy")]

# train / calibrate / test split. Detector decision thresholds are calibrated on a
# held-out benign split to a FIXED benign false-positive rate (standard detection practice),
# then per-class recall is read at that operating point.
idx=rng.permutation(len(y)); CY,PH,y,cls=CY[idx],PH[idx],y[idx],cls[idx]
ntr=int(0.55*len(y)); ncal=int(0.70*len(y))
tr=slice(0,ntr); cal=slice(ntr,ncal); te=slice(ncal,None)
TARGET_FPR=0.05

FU=np.hstack([CY,PH])
views={"Cyber":CY,"Physics":PH,"Fusion":FU}
classes=["structural","naive","stealthy"]
clfs={}; scores_te={}; taus={}
for name,X in views.items():
    clf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=1,
                               class_weight="balanced").fit(X[tr],y[tr])
    clfs[name]=clf
    s_cal=clf.predict_proba(X[cal])[:,1]; s_te=clf.predict_proba(X[te])[:,1]
    # threshold = (1-FPR) quantile of benign calibration scores
    ben_cal=s_cal[(y[cal]==0)]
    taus[name]=float(np.quantile(ben_cal, 1-TARGET_FPR))
    scores_te[name]=s_te

cls_te=cls[te]; y_te=y[te]
def per_class(name):
    s=scores_te[name]; tau=taus[name]; pred_te=(s>=tau).astype(int); row={}
    for c in classes:
        mask=cls_te==c
        row[c]=float(pred_te[mask].mean()) if mask.sum() else float("nan")
    bmask=cls_te=="benign"; row["benign_FPR"]=float(pred_te[bmask].mean()) if bmask.sum() else float("nan")
    return row
table={name:per_class(name) for name in views}

# ---------- graded topology-knowledge sweep (value attack -> stealthy) ----------
# physics detection = classic BDD: flag if scalar LS residual exceeds a benign-calibrated
# threshold (5% FPR). As attacker knowledge -> 100%, residual -> noise floor and recall -> FPR.
fracs=[0.0,0.2,0.4,0.6,0.8,1.0]; rng2=np.random.default_rng(23); NG=300
tau_res=float(np.quantile([residual(z) for z in make_states(600,rng2)], 1-TARGET_FPR))
graded_res=[]; graded_rec=[]
for fr in fracs:
    Zg=np.array([graded_fdi(z,rng2,fr) for z in make_states(NG,rng2)])
    rr=[residual(z) for z in Zg]
    graded_res.append(float(np.mean(rr)))
    graded_rec.append(float(np.mean([r>=tau_res for r in rr])))

# ---------- write results ----------
res_stealth=float(np.mean([residual(z) for z in Zstealth[:100]]))
res_naive  =float(np.mean([residual(z) for z in Znaive[:100]]))
res_benign =float(np.mean([residual(z) for z in Zben[:100]]))
with open(os.path.join(OUT,"semantic_fusion.txt"),"w") as f:
    f.write("IEEE-14 SEMANTIC-MANIPULATION OBSERVABILITY TAXONOMY\n")
    f.write(f"branches(m)={m} states(n)={n} noise floor={NOISE}\n")
    f.write(f"LS residual: benign={res_benign:.2e}  naive-value={res_naive:.2e}  stealthy-FDI={res_stealth:.2e}\n\n")
    f.write(f"{'View':8s} | {'structural':>10s} {'naive-val':>10s} {'stealthy':>10s} | {'benign FPR':>10s}\n")
    f.write("-"*60+"\n")
    for name in views:
        r=table[name]
        f.write(f"{name:8s} | {r['structural']:10.2f} {r['naive']:10.2f} {r['stealthy']:10.2f} | {r['benign_FPR']:10.2f}\n")
    f.write("\nReading: cyber catches structural only; physics catches naive value only;\n")
    f.write("FUSION catches structural + naive value; stealthy FDI defeats all three\n")
    f.write("(the fundamental limit -> escalate via the selective policy).\n\n")
    f.write("Graded topology-knowledge sweep (value attack -> stealthy):\n")
    f.write(f"{'knowledge':>10s} {'mean residual':>14s} {'physics recall':>15s}\n")
    for fr,rr,rc in zip(fracs,graded_res,graded_rec):
        f.write(f"{fr*100:9.0f}% {rr:14.2e} {rc:15.2f}\n")
print(open(os.path.join(OUT,"semantic_fusion.txt")).read())

# ---------- figure: taxonomy bars + graded curve ----------
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
w=0.25; xpos=np.arange(len(classes)); colors={"Cyber":"#1f77b4","Physics":"#2E7D32","Fusion":"#8E44AD"}
for i,name in enumerate(views):
    vals=[table[name][c] for c in classes]
    ax[0].bar(xpos+(i-1)*w,vals,w,label=name,color=colors[name])
    for j,v in enumerate(vals): ax[0].text(xpos[j]+(i-1)*w,v+0.02,f"{v:.2f}",ha="center",fontsize=7)
ax[0].set_xticks(xpos); ax[0].set_xticklabels(["structural","naive value","stealthy FDI"])
ax[0].set_ylabel("detection recall"); ax[0].set_ylim(0,1.08); ax[0].legend(fontsize=8,loc="upper right")
ax[0].set_title("Which view catches which attack")

ax2=ax[1]; ax2b=ax2.twinx()
ax2.plot([f*100 for f in fracs],graded_rec,"o-",color="#2E7D32",label="physics recall")
ax2b.plot([f*100 for f in fracs],graded_res,"s--",color="#C0392B",label="mean residual")
ax2.set_xlabel("attacker topology knowledge (%)"); ax2.set_ylabel("physics recall",color="#2E7D32")
ax2b.set_ylabel("mean LS residual",color="#C0392B"); ax2.set_ylim(-0.05,1.05)
ax2.set_title("Naive value -> stealthy FDI as knowledge grows")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"semantic_fusion.png"),dpi=150)
print("saved results/semantic_fusion.png")
