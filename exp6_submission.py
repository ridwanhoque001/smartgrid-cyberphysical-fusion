"""
Submission-quality re-run of Experiment 6 (Results V) at 50 trials/point.
Reuses exp6's fast WLS, manifold, and pool. Calibrates all three detectors to a
common 5% false-alarm rate on independent benign streams, then benchmarks with
2-process parallelism. Staged + checkpointed; deterministic seeds.
Usage: python exp6_submission.py {calib|ramp|steproc|finish}
"""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp6_manifold_detector as x6
from sklearn.metrics import roc_auc_score

OUT="results"; CKPT=f"{OUT}/exp6_ckpt.json"
se_stream=x6.se_stream; detector_stats=x6.detector_stats; rho=x6.rho
ac_exact_attack=x6.ac_exact_attack; ref=x6.ref; tgt=x6.tgt; K=x6.K
T=x6.T; T0=x6.T0; DELTA=np.deg2rad(4.0)
NT=50; ROC_N=40; FAR_TGT=0.05
ramp_rates=[2.0,1.0,0.5,0.25,0.1,0.05,0.02]; steps=[1.0,0.5,0.25,0.1]

def load(): return json.load(open(CKPT)) if os.path.exists(CKPT) else {}
def save(d): json.dump(d,open(CKPT,"w"),indent=2)

# thresholds: from checkpoint if calibrated, else exp6 defaults
_ck=load(); Ht,Hm,Hf=(_ck["thr"] if "thr" in _ck else (x6.Ht,x6.Hm,x6.Hf))

def first_cross(series,H,start):
    idx=np.where(series[start:]>H)[0]; return (idx[0]+start) if len(idx) else -1
def maxstats(X):
    cu_t,cu_m=detector_stats(X,T0,ref); return cu_t[T0:].max(),cu_m[T0:].max()
def eval_stream(X):
    cu_t,cu_m=detector_stats(X,T0,ref); fu=np.maximum(cu_t/Ht,cu_m/Hm)
    return first_cross(cu_t,Ht,T0),first_cross(cu_m,Hm,T0),first_cross(fu,Hf,T0)

def w_benign(seed): return maxstats(se_stream(T,np.random.default_rng(seed)))
def w_ramp(a):
    rr,seed=a; rng=np.random.default_rng(seed)
    def atk(x,t,t0): return ac_exact_attack(x,np.deg2rad(min(rr*(t-t0),np.rad2deg(DELTA))))
    return eval_stream(se_stream(T,rng,attack=atk,t0=T0))
def w_step(a):
    ss,seed=a; rng=np.random.default_rng(seed)
    def atk(x,t,t0): return ac_exact_attack(x,np.deg2rad(ss))
    return eval_stream(se_stream(T,rng,attack=atk,t0=T0))
def w_roc(a):
    seed,label=a; rng=np.random.default_rng(seed)
    if label==0: X=se_stream(T,rng)
    else:
        def atk(x,t,t0): return ac_exact_attack(x,np.deg2rad(min(0.03*(t-t0),4.0)))
        X=se_stream(T,rng,attack=atk,t0=T0)
    a2,b2=maxstats(X); fu=max(a2/Ht,b2/Hm); return (a2,b2,fu,label)

def stage_calib():
    pool=mp.Pool(2)
    cal=np.array(pool.map(w_benign,[20000+t for t in range(140)]))   # calibrate
    far=np.array(pool.map(w_benign,[30000+t for t in range(90)]))    # held-out
    pool.close()
    ht=float(np.quantile(cal[:,0],1-FAR_TGT)); hm=float(np.quantile(cal[:,1],1-FAR_TGT))
    fu_cal=np.maximum(cal[:,0]/ht,cal[:,1]/hm); hf=float(np.quantile(fu_cal,1-FAR_TGT))
    fart=float(np.mean(far[:,0]>ht)); farm=float(np.mean(far[:,1]>hm))
    farf=float(np.mean(np.maximum(far[:,0]/ht,far[:,1]/hm)>hf))
    d=load(); d.update(thr=[ht,hm,hf],far=[fart,farm,farf]); save(d)
    print("calib done | thr=%.2f/%.2f/%.2f | held-out FAR=%.0f%%/%.0f%%/%.0f%%"%(ht,hm,hf,fart*100,farm*100,farf*100))

def stage_ramp(lo,hi):
    pool=mp.Pool(2); keys=["temporal","manifold","fused"]
    idxs=list(range(lo,hi))
    jobs=[(ramp_rates[i],1000+i*NT+t) for i in idxs for t in range(NT)]
    R=np.array(pool.map(w_ramp,jobs)).reshape(len(idxs),NT,3); pool.close()
    d=load(); pp=d.get("pdet_by_i",{}); dl_=d.get("delay_by_i",{})
    for a,i in enumerate(idxs):
        pp[str(i)]=[float(np.mean(R[a,:,j]>0)) for j in range(3)]
        dl_[str(i)]=[ (lambda v: float(np.median(v)) if len(v) else None)(R[a,:,j][R[a,:,j]>0]-T0) for j in range(3)]
    d.update(pdet_by_i=pp,delay_by_i=dl_,ramp_rates=ramp_rates); save(d); print("ramp %d-%d done"%(lo,hi))

def stage_steproc():
    pool=mp.Pool(2)
    jobs=[(ss,5000+i*NT+t) for i,ss in enumerate(steps) for t in range(NT)]
    S=np.array(pool.map(w_step,jobs)).reshape(len(steps),NT,3)
    Q=pool.map(w_roc,[(9000+t,0) for t in range(ROC_N)]+[(9500+t,1) for t in range(ROC_N)]); pool.close()
    keys=["temporal","manifold","fused"]; floor={k:[] for k in keys}
    for i in range(len(steps)):
        for j,k in enumerate(keys): floor[k].append(float(np.mean(S[i,:,j]>0)))
    st={k:[] for k in keys}; lab=[]
    for a,b,c,y in Q: st["temporal"].append(a);st["manifold"].append(b);st["fused"].append(c);lab.append(y)
    lab=np.array(lab); auc={k:round(float(roc_auc_score(lab,st[k])),3) for k in keys}
    d=load(); d.update(floor=floor,step_sizes=steps,auc_slowramp=auc); save(d); print("steproc done",auc)

def stage_finish():
    d=load(); keys=["temporal","manifold","fused"]
    biases=np.linspace(0.1,3.0,12); base=x6.Xpool[:60]; e=np.eye(x6.NX)[tgt]
    zres=np.array([np.mean([(rho(x+e*np.deg2rad(b))-ref["mu_rho"])/ref["sig_rho"] for x in base]) for b in biases])
    gate=3.0; idx=np.where(zres>gate)[0]; bias_limit=float(biases[idx[0]]) if len(idx) else float("nan")
    keys=["temporal","manifold","fused"]
    pbi=d["pdet_by_i"]; dbi=d["delay_by_i"]
    pdet={k:[pbi[str(i)][j] for i in range(len(ramp_rates))] for j,k in enumerate(keys)}
    delay={k:[dbi[str(i)][j] for i in range(len(ramp_rates))] for j,k in enumerate(keys)}
    floor=d["floor"]; auc=d["auc_slowramp"]; far=d["far"]
    print("SUBMISSION exp6 | NT=%d/pt ROC %d/class horizon=%d K=%d | FAR t/m/f=%.0f/%.0f/%.0f%%"%(
        NT,ROC_N,T-T0,K,far[0]*100,far[1]*100,far[2]*100))
    print("P(detect) vs ramp:  rate  temporal manifold fused")
    for i,rr in enumerate(ramp_rates): print("  %5.2f    %.2f    %.2f   %.2f"%(rr,pdet["temporal"][i],pdet["manifold"][i],pdet["fused"][i]))
    print("step floor:");  [print("  %4.2f  %.2f %.2f %.2f"%(ss,floor["temporal"][i],floor["manifold"][i],floor["fused"][i])) for i,ss in enumerate(steps)]
    print("AUC@0.03:",auc,"| bias limit %.2f deg"%bias_limit)
    cols={"temporal":"#C0392B","manifold":"#2E7D32","fused":"#1F4E79"}; mk={"temporal":"o","manifold":"s","fused":"^"}
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    for k in keys: ax[0].plot(ramp_rates,pdet[k],mk[k]+"-",color=cols[k],label=k)
    ax[0].set_xscale("log");ax[0].set_xlabel("attack ramp rate (deg/step)");ax[0].set_ylabel("P(detect)");ax[0].set_ylim(-.05,1.05)
    ax[0].set_title("(a) Ramp-escape floor:\nmanifold/fused catch slow ramps temporal misses");ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
    ax[1].plot(biases,zres,"o-",color="#1F4E79");ax[1].axhline(gate,ls="--",color="#C0392B",label="3σ spatial gate")
    if np.isfinite(bias_limit): ax[1].axvline(bias_limit,ls=":",color="#2E7D32",label=f"bias limit ≈ {bias_limit:.2f}°")
    ax[1].set_xlabel("targeted bias (deg)");ax[1].set_ylabel("off-manifold residual (σ)")
    ax[1].set_title("(b) New (tighter) limit:\ntargeted bias budget under manifold gate");ax[1].legend(fontsize=8);ax[1].grid(alpha=.3)
    xk=np.arange(len(steps))
    for j,k in enumerate(keys): ax[2].bar(xk+(j-1)*0.26,floor[k],0.26,color=cols[k],label=k)
    ax[2].set_xticks(xk);ax[2].set_xticklabels([f"{s:g}°" for s in steps]);ax[2].set_xlabel("step-bias size");ax[2].set_ylabel("P(detect)");ax[2].set_ylim(0,1.05)
    ax[2].set_title("(c) Step floor:\ntiny persistent steps: fusion marginally best");ax[2].legend(fontsize=8);ax[2].grid(alpha=.3,axis="y")
    plt.tight_layout(); plt.savefig(f"{OUT}/exp6_manifold_detector.png",dpi=150)
    summary=dict(trials_per_point=NT,roc_n_per_class=ROC_N,horizon=int(T-T0),manifold_dim=int(K),
        far_temporal=round(far[0],3),far_manifold=round(far[1],3),far_fused=round(far[2],3),
        ramp_rates=ramp_rates,pdet_temporal=pdet["temporal"],pdet_manifold=pdet["manifold"],pdet_fused=pdet["fused"],
        step_sizes=steps,floor_temporal=floor["temporal"],floor_manifold=floor["manifold"],floor_fused=floor["fused"],
        auc_slowramp=auc,bias_limit_deg=None if not np.isfinite(bias_limit) else round(bias_limit,3))
    json.dump(summary,open(f"{OUT}/exp6_summary.json","w"),indent=2); print("saved summary+figure")

if __name__=="__main__":
    cmd=sys.argv[1]
    if cmd=="ramp": stage_ramp(int(sys.argv[2]),int(sys.argv[3]))
    else: {"calib":stage_calib,"steproc":stage_steproc,"finish":stage_finish}[cmd]()
