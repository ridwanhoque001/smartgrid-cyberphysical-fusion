"""
Phase 2a — Observability taxonomy with statistical rigor (Reza comments 171,177,197,198).
Resumable: each run processes a few more seeds and checkpoints pooled hit/total counts to
results/exp4b_ckpt.json. Re-run until n_done >= N_TARGET, then it writes the final txt/json.
Wilson 95% CIs are computed on the pooled test-window counts.
"""
import os, json, sys, numpy as np
import pandapower.networks as nw
from sklearn.ensemble import RandomForestClassifier

OUT="results"; os.makedirs(OUT,exist_ok=True)
CKPT=f"{OUT}/exp4b_ckpt.json"
N_TARGET=20; SEEDS_PER_RUN=int(sys.argv[1]) if len(sys.argv)>1 else 5
TARGET_FPR=0.05
classes=["structural","naive","stealthy"]; views=["Cyber","Physics","Fusion"]

def wilson(k,nn,z=1.96):
    if nn==0: return (float("nan"),float("nan"))
    p=k/nn; d=1+z*z/nn; c=p+z*z/(2*nn); h=z*np.sqrt(p*(1-p)/nn+z*z/(4*nn*nn))
    return ((c-h)/d,(c+h)/d)

net=nw.case14(); nb=len(net.bus); slack=int(net.ext_grid.bus.iloc[0])
non_slack=[b for b in range(nb) if b!=slack]; posn={b:i for i,b in enumerate(non_slack)}
br=[]
for _,l in net.line.iterrows():
    x=l.x_ohm_per_km*l.length_km; br.append((int(l.from_bus),int(l.to_bus),x if x>1e-6 else 0.05))
for _,t in net.trafo.iterrows():
    z=t.vk_percent/100.0; br.append((int(t.hv_bus),int(t.lv_bus),max(z,0.05)))
m=len(br); n=len(non_slack); H=np.zeros((m,n))
for k,(i,j,x) in enumerate(br):
    if i in posn: H[k,posn[i]]+=1.0/x
    if j in posn: H[k,posn[j]]-=1.0/x
Hpinv=np.linalg.pinv(H); NOISE=0.0015
def make_states(ns,rng): return np.array([H@rng.normal(0,0.04,n)+rng.normal(0,NOISE,m) for _ in range(ns)])
def residual_vec(z): return z-H@(Hpinv@z)
CY_MU=np.array([100.,0.040,0.010,0.020,1.80,0.050]); CY_SD=np.array([6.,0.004,0.002,0.005,0.08,0.010])
def cyber_feats(nr,rng,structural=False):
    f=rng.normal(CY_MU,CY_SD,size=(nr,len(CY_MU)))
    if structural: f[:,0]*=0.60; f[:,3]+=0.25; f[:,4]-=0.60
    return f
TARGET=0; DELTA=0.14
def stealthy_fdi(z,rng): c=rng.uniform(-0.05,0.05,n); c[TARGET]=DELTA; return z+H@c
def naive_value(z,rng,k=2):
    za=z.copy(); idx=rng.choice(m,size=k,replace=False)
    za[idx]+=rng.uniform(0.05,0.12,k)*rng.choice([-1,1],k); return za

# load ckpt
if os.path.exists(CKPT):
    ck=json.load(open(CKPT))
else:
    ck={"done":[], "pool":{v:{c:[0,0] for c in classes+["benign_FPR"]} for v in views}}

def run_seed(seed):
    rng=np.random.default_rng(1000+seed); K=1500
    Zben=make_states(K,rng); Zstruct=make_states(K,rng)
    Znaive=np.array([naive_value(z,rng) for z in make_states(K,rng)])
    Zstealth=np.array([stealthy_fdi(z,rng) for z in make_states(K,rng)])
    CY=np.vstack([cyber_feats(K,rng,False),cyber_feats(K,rng,True),cyber_feats(K,rng,False),cyber_feats(K,rng,False)])
    PH=np.vstack([np.array([residual_vec(z) for z in Z]) for Z in (Zben,Zstruct,Znaive,Zstealth)])
    y=np.r_[np.zeros(K),np.ones(3*K)].astype(int)
    cls=np.r_[np.full(K,"benign"),np.full(K,"structural"),np.full(K,"naive"),np.full(K,"stealthy")]
    idx=rng.permutation(len(y)); CY,PH,y,cls=CY[idx],PH[idx],y[idx],cls[idx]
    ntr=int(0.55*len(y)); ncal=int(0.70*len(y)); tr=slice(0,ntr); cal=slice(ntr,ncal); te=slice(ncal,None)
    Xv={"Cyber":CY,"Physics":PH,"Fusion":np.hstack([CY,PH])}; cls_te=cls[te]
    for name in views:
        X=Xv[name]
        clf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1,class_weight="balanced").fit(X[tr],y[tr])
        s_cal=clf.predict_proba(X[cal])[:,1]; s_te=clf.predict_proba(X[te])[:,1]
        tau=float(np.quantile(s_cal[y[cal]==0],1-TARGET_FPR)); pred=(s_te>=tau).astype(int)
        for c in classes:
            mask=cls_te==c; ck["pool"][name][c][0]+=int(pred[mask].sum()); ck["pool"][name][c][1]+=int(mask.sum())
        bmask=cls_te=="benign"; ck["pool"][name]["benign_FPR"][0]+=int(pred[bmask].sum()); ck["pool"][name]["benign_FPR"][1]+=int(bmask.sum())

todo=[s for s in range(N_TARGET) if s not in ck["done"]][:SEEDS_PER_RUN]
for s in todo:
    run_seed(s); ck["done"].append(s); json.dump(ck,open(CKPT,"w")); print("done seed",s,"total",len(ck["done"]),flush=True)

if len(ck["done"])>=N_TARGET:
    pool=ck["pool"]; lines=[]
    lines.append(f"IEEE 14-Bus OBSERVABILITY TAXONOMY — {len(ck['done'])} seeds, Wilson 95% CIs on pooled test windows")
    lines.append(f"branches(m)={m} states(n)={n}  target benign FPR={TARGET_FPR}"); lines.append("")
    hdr=f"{'View':8s} | "+" ".join(f"{c:>22s}" for c in classes)+f" | {'benign FPR (measured)':>24s}"
    lines.append(hdr); lines.append("-"*len(hdr))
    out={"n_seeds":len(ck["done"]),"target_fpr":TARGET_FPR,"cells":{}}
    for name in views:
        cs=[]
        for c in classes:
            hit,tot=pool[name][c]; lo,hi=wilson(hit,tot); cs.append(f"{hit/tot:.2f} [{lo:.2f},{hi:.2f}]")
            out["cells"][f"{name}/{c}"]={"mean":hit/tot,"ci":[lo,hi],"hit":hit,"tot":tot}
        hitb,totb=pool[name]["benign_FPR"]; lob,hib=wilson(hitb,totb)
        out["cells"][f"{name}/benign_FPR"]={"mean":hitb/totb,"ci":[lob,hib],"hit":hitb,"tot":totb}
        lines.append(f"{name:8s} | "+" ".join(f"{s:>22s}" for s in cs)+f" | {hitb/totb:.3f} [{lob:.3f},{hib:.3f}]")
    hitb,totb=pool["Cyber"]["benign_FPR"]; lob,hib=wilson(hitb,totb); inside=lob<=TARGET_FPR<=hib
    lines.append(""); lines.append(f"CALIBRATION CHECK (comment 197): Cyber benign FPR={hitb/totb:.3f} 95% CI [{lob:.3f},{hib:.3f}] over {totb} windows.")
    lines.append(f"  Target 5% {'IS' if inside else 'is NOT'} inside CI -> "+("finite-sample calibration error, as stated." if inside else "recalibrate."))
    out["cyber_fpr_target_inside_ci"]=bool(inside)
    txt="\n".join(lines); open(f"{OUT}/semantic_fusion_ci.txt","w").write(txt); json.dump(out,open(f"{OUT}/semantic_fusion_ci.json","w"),indent=2)
    print(txt)
else:
    print(f"progress {len(ck['done'])}/{N_TARGET} — run again")
