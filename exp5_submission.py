"""
Submission-quality re-run of the Experiment-5 limit panels (Fig. exp5_gradient_limit)
at 50 trials/point, plus the static-vs-temporal AUCs. Fast fixed-Jacobian WLS +
2-process parallelism, staged + checkpointed. Deterministic seeds.
Usage: python exp5_submission.py {calib|ramp|step|finish}
"""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import multiprocessing as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp5_ac_stealth_temporal as e
from sklearn.metrics import roc_auc_score

OUT="results"; CKPT=f"{OUT}/exp5_ckpt.json"
h=e.h; jac=e.jac; W=e.W; NX=e.NX; sig=e.sig; NB=e.NB; NL=e.NL; M=e.M; bdd=e.bdd
CHI2_THR=e.CHI2_THR
T=64; T0=32; WARM=12; K_SLACK=0.5; FAR=0.05
tgt=e.NONSLACK.index(3); DELTA=np.deg2rad(4.0)
ramp_rates=[2.0,1.0,0.5,0.25,0.1,0.05]; steps=[6.0,4.0,3.0,2.0,1.0]

Xtrue,Zclean,prof=e.run_series(T)   # one benign day (AC power flow), reused

def wls_fast(z,x0,iters=5):
    x=x0.copy(); H=jac(x); HtW=H.T*W; Ginv=np.linalg.inv(HtW@H+1e-9*np.eye(NX))
    for _ in range(iters): x=x+Ginv@(HtW@(z-h(x)))
    return x,H
def ac_attack(x,bias): xa=x.copy(); xa[tgt]+=bias; return h(xa)-h(x)
def se_traj(seed,attack=None,t0=None):
    rng=np.random.default_rng(seed); traj=np.empty(T); xprev=Xtrue[0]
    for k in range(T):
        z=Zclean[k]+rng.normal(0,sig)
        if attack is not None and k>=t0: z=z+attack(Xtrue[k],k,t0)
        xh,_=wls_fast(z,xprev); xprev=xh; traj[k]=np.rad2deg(xh[tgt])
    return traj
def holt_innov(s,al=0.4,be=0.15):
    lv=s[0];tr=0.0;inn=np.zeros(len(s))
    for k in range(1,len(s)):
        inn[k]=s[k]-(lv+tr); p=lv; lv=al*s[k]+(1-al)*(lv+tr); tr=be*(lv-p)+(1-be)*tr
    return inn
def isigma(inn):
    seg=inn[WARM:T0]; return max(1.4826*np.median(np.abs(seg-np.median(seg))),1e-6)
def cusum(inn,sg,H,start=T0):
    Sp=Sn=0.0;det=-1;mx=0.0
    for k in range(start,len(inn)):
        z=inn[k]/sg; Sp=max(0.0,Sp+z-K_SLACK); Sn=min(0.0,Sn+z+K_SLACK); mx=max(mx,Sp,-Sn)
        if (Sp>H or -Sn>H) and det<0: det=k
    return det,mx

def load(): return json.load(open(CKPT)) if os.path.exists(CKPT) else {}
def save(d): json.dump(d,open(CKPT,"w"),indent=2)
_ck=load(); Hthr=_ck.get("Hthr",None)

def w_ben(seed):
    inn=holt_innov(se_traj(seed)); return cusum(inn,isigma(inn),np.inf)[1]
def w_ramp(a):
    rr,seed=a
    def atk(x,k,t0): return ac_attack(x,np.deg2rad(min(rr*(k-t0),np.rad2deg(DELTA))))
    inn=holt_innov(se_traj(seed,atk,T0)); d,_=cusum(inn,isigma(inn),Hthr)
    return d
def w_step(a):
    ss,seed=a
    def atk(x,k,t0): return ac_attack(x,np.deg2rad(ss))
    inn=holt_innov(se_traj(seed,atk,T0)); d,_=cusum(inn,isigma(inn),Hthr)
    return 1 if d>0 else 0
def w_atkstep(seed):
    def atk(x,k,t0): return ac_attack(x,DELTA)
    inn=holt_innov(se_traj(seed,atk,T0)); return cusum(inn,isigma(inn),np.inf)[1]

def stage_calib():
    pool=mp.Pool(2)
    cal=np.array(pool.map(w_ben,[20000+t for t in range(120)]))
    far_streams=np.array(pool.map(w_ben,[30000+t for t in range(80)])); pool.close()
    H=float(np.quantile(cal,1-FAR)); far=float(np.mean(far_streams>H))
    d=load(); d.update(Hthr=H,far=far); save(d); print("calib H=%.2f held-out FAR=%.0f%%"%(H,far*100))

def stage_ramp():
    NT=50; pool=mp.Pool(2)
    R=np.array(pool.map(w_ramp,[(rr,1000+i*NT+t) for i,rr in enumerate(ramp_rates) for t in range(NT)])).reshape(len(ramp_rates),NT)
    pool.close()
    pdet=[float(np.mean(R[i]>0)) for i in range(len(ramp_rates))]
    delay=[ (lambda v: float(np.median(v)) if len(v) else None)(R[i][R[i]>0]-T0) for i in range(len(ramp_rates))]
    d=load(); d.update(ramp_pdet=pdet,ramp_delay=delay,ramp_rates=ramp_rates); save(d); print("ramp done",pdet)

def stage_step():
    NT=50; pool=mp.Pool(2)
    S=np.array(pool.map(w_step,[(ss,5000+i*NT+t) for i,ss in enumerate(steps) for t in range(NT)])).reshape(len(steps),NT)
    pool.close()
    floor=[float(np.mean(S[i])) for i in range(len(steps))]
    d=load(); d.update(step_sizes=steps,step_floor=floor); save(d); print("step done",floor)

def stage_finish():
    d=load(); H=d["Hthr"]
    # knowledge gradient (serial; DC-designed -> AC-exact), 30 snapshots/alpha
    mask=np.zeros(M,bool); mask[NB:2*NB]=True; mask[NB+2*NB:NB+2*NB+NL]=True
    def dc_designed(x,alpha):
        a_ac=ac_attack(x,DELTA); a_dc=np.zeros(M); a_dc[mask]=a_ac[mask]
        return alpha*a_ac+(1-alpha)*a_dc
    alphas=np.linspace(0,1,11); rng=np.random.default_rng(3); Jby=[]
    for al in alphas:
        Js=[]
        for k in range(34,64):
            z=Zclean[k]+rng.normal(0,sig); za=z+dc_designed(Xtrue[k],al)
            xh,Hh=wls_fast(za,Xtrue[k]); J,_=bdd(za,xh,Hh); Js.append(J)
        Jby.append(float(np.mean(Js)))
    Jby=np.array(Jby)
    # static vs temporal AUC on stealthy step (parallel)
    pool=mp.Pool(2)
    ben=np.array(pool.map(w_ben,[40000+t for t in range(50)]))
    atk=np.array(pool.map(w_atkstep,[41000+t for t in range(50)])); pool.close()
    lab=np.r_[np.zeros(len(ben)),np.ones(len(atk))]
    auc_temporal=float(roc_auc_score(lab,np.r_[ben,atk]))
    # static chi-square AUC (benign vs attacked snapshots)
    Jb=[];Ja=[]
    for k in range(20,45):
        z=Zclean[k]+rng.normal(0,sig); xh,Hh=wls_fast(z,Xtrue[k]); Jb.append(bdd(z,xh,Hh)[0])
        za=z+ac_attack(Xtrue[k],DELTA); xa,Ha=wls_fast(za,Xtrue[k]); Ja.append(bdd(za,xa,Ha)[0])
    auc_static=float(roc_auc_score(np.r_[np.zeros(len(Jb)),np.ones(len(Ja))],np.r_[Jb,Ja]))

    rr=d["ramp_rates"]; delay=d["ramp_delay"]; floor=d["step_floor"]; ss=d["step_sizes"]; far=d["far"]
    print("SUBMISSION exp5 | 50 trials/pt | temporal FAR=%.0f%% | AUC temporal=%.3f static=%.3f"%(far*100,auc_temporal,auc_static))
    print("ramp P(detect):",d["ramp_pdet"]); print("ramp median delay:",delay); print("step floor:",floor)

    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(alphas*100,Jby,"o-",color="#1f77b4"); ax[0].axhline(CHI2_THR,ls="--",color="#C0392B",label=f"chi2 thr ({CHI2_THR:.0f})")
    ax[0].set_xlabel("attacker AC topology knowledge (%)"); ax[0].set_ylabel("AC chi-square J after attack")
    ax[0].set_title("(a) Knowledge gradient:\nDC-designed FDI is not AC-stealthy"); ax[0].legend(); ax[0].grid(alpha=.3)
    dl=[np.nan if x is None else x for x in delay]
    ax[1].plot(rr,dl,"s-",color="#2E7D32"); ax[1].set_xscale("log")
    ax[1].set_xlabel("attack ramp rate (deg/step)"); ax[1].set_ylabel("median detection delay (steps)")
    ax[1].set_title("(b) Honest limit #1:\nundetected window grows as ramp slows"); ax[1].grid(alpha=.3)
    ax[2].plot(ss,floor,"^-",color="#8E44AD"); ax[2].set_xscale("log")
    ax[2].set_xlabel("step-bias size (deg)"); ax[2].set_ylabel("P(temporal detect)"); ax[2].set_ylim(-.05,1.05)
    ax[2].set_title("(c) Honest limit #2:\ndetectability floor at SE-noise level"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/exp5_gradient_limit.png",dpi=150)
    js=dict(trials_per_point=50,temporal_far=round(far,3),auc_temporal=round(auc_temporal,3),
        auc_static=round(auc_static,3),ramp_rates=rr,ramp_pdet=d["ramp_pdet"],ramp_delay=delay,
        step_sizes=ss,step_floor=floor,knowledge_alpha=list(alphas),knowledge_J=list(map(float,Jby)))
    json.dump(js,open(f"{OUT}/exp5_limit_summary.json","w"),indent=2)
    print("saved exp5_gradient_limit.png + exp5_limit_summary.json")

if __name__=="__main__":
    {"calib":stage_calib,"ramp":stage_ramp,"step":stage_step,"finish":stage_finish}[sys.argv[1]]()
