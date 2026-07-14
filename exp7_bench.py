"""
IEEE-118 detector benchmark (reduced-order, exploiting x_hat = x_a + N(0,Sigma)).
Temporal (Holt-CUSUM) vs manifold (off-manifold residual) vs fused, at 50 trials/pt,
5% FAR. Regenerates results/exp7_ieee118.png + results/exp7_summary.json.
"""
import os, json, time, numpy as np, warnings; warnings.filterwarnings("ignore")
import pandapower as pp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import exp7_ieee118 as e

OUT="results"; rng=np.random.default_rng(7)
h=e.h; NX=e.NX; NB=e.NB; NTH=e.NTH; NONSLACK=e.NONSLACK
bp=e.base_p.copy(); bq=e.base_q.copy(); nL=e.nLoad
tgt=20; DELTA=np.deg2rad(4.0)
Sigma=np.load(f"{OUT}/_ieee118_Sigma.npy"); L=np.linalg.cholesky(Sigma+1e-12*np.eye(NX))
sig_tgt=np.sqrt(Sigma[tgt,tgt])
T=48; T0=24; WARM=14; K_SLACK=0.5; FAR=0.05

# ---- benign true-state day (independent per-load variation) ----
def load_scales(Tn,rng):
    s=np.ones((Tn,nL)); a=np.zeros(nL); ph=rng.uniform(0,2*np.pi,nL)
    for t in range(Tn):
        a=0.9*a+0.1*rng.normal(0,0.06,nL)
        s[t]=np.clip(1.0+0.12*np.sin(2*np.pi*t/Tn+ph)+a,0.6,1.4)
    return s
def build_day(Tn,rng):
    S=load_scales(Tn,rng); X=np.empty((Tn,NX))
    for t in range(Tn):
        e.net.load.p_mw=bp*S[t]; e.net.load.q_mvar=bq*S[t]
        pp.runpp(e.net,init="results" if t else "auto"); X[t]=e.x_from_pf(e.net)
    return X
t0=time.time(); POOL=build_day(200,np.random.default_rng(101)); print("pool build %.1fs"%(time.time()-t0))

# ---- learn manifold on benign ESTIMATED states (true + SE noise) ----
def noise(): return L@rng.standard_normal(NX)
Xb=np.array([POOL[i%len(POOL)]+L@rng.standard_normal(NX) for i in range(600)])
mu=Xb.mean(0); sd=np.maximum(Xb.std(0),5e-3); Zmean=((Xb-mu)/sd).mean(0)
_,Sv,Vt=np.linalg.svd((Xb-mu)/sd-Zmean,full_matrices=False)
K=int(np.argmax(np.cumsum(Sv**2/np.sum(Sv**2))>=0.999)+1); Vk=Vt[:K]
def rho(x):
    z=(x-mu)/sd-Zmean; return np.linalg.norm(z-z@Vk.T@Vk)
rb=np.array([rho(POOL[i%len(POOL)]+L@rng.standard_normal(NX)) for i in range(300)])
mu_rho=np.median(rb); sig_rho=max(1.4826*np.median(np.abs(rb-mu_rho)),1e-6)
print("manifold dim K=%d of %d states | benign rho mu=%.2f sd=%.2f"%(K,NX,mu_rho,sig_rho))

# ---- detectors on a synthesized estimated-state stream ----
def holt(s,al=0.4,be=0.15):
    lv=s[0];tr=0.0;inn=np.zeros(len(s))
    for k in range(1,len(s)):
        inn[k]=s[k]-(lv+tr);p=lv;lv=al*s[k]+(1-al)*(lv+tr);tr=be*(lv-p)+(1-be)*tr
    return inn
def isigma(inn):
    seg=inn[WARM:T0]; return max(1.4826*np.median(np.abs(seg-np.median(seg))),1e-6)
def cusum(z,start):
    Sp=Sn=0.0;det=-1;mx=0.0
    for k in range(start,len(z)):
        Sp=max(0.0,Sp+z[k]-K_SLACK);Sn=min(0.0,Sn+z[k]+K_SLACK);mx=max(mx,Sp,-Sn)
        if (Sp>1e9 or -Sn>1e9) and det<0: det=k
    return mx
def stream(seed,bias_fn=None):
    r=np.random.default_rng(seed); w0=int(r.integers(0,len(POOL)-T))
    win=POOL[w0:w0+T]; ang=np.empty(T); rr=np.empty(T)
    for k in range(T):
        nz=L@r.standard_normal(NX); v=win[k]+nz
        b=0.0 if bias_fn is None else bias_fn(k)
        v=v.copy(); v[tgt]+=b; 
        ang[k]=np.rad2deg(win[k][tgt]+b)+np.rad2deg(nz[tgt])
        rr[k]=rho(v)
    inn=holt(ang); zt=inn/isigma(inn)
    zm=(rr-mu_rho)/sig_rho
    def maxc(z):
        Sp=Sn=0.0;mx=0.0
        for k in range(T0,T):
            Sp=max(0.0,Sp+z[k]-K_SLACK);Sn=min(0.0,Sn+z[k]+K_SLACK);mx=max(mx,Sp,-Sn)
        return mx
    return maxc(zt),maxc(zm)

# ---- calibrate thresholds to 5% FAR on independent benign streams ----
cal=np.array([stream(20000+i) for i in range(300)])
far_s=np.array([stream(30000+i) for i in range(200)])
Ht=np.quantile(cal[:,0],1-FAR); Hm=np.quantile(cal[:,1],1-FAR)
Hf=np.quantile(np.maximum(cal[:,0]/Ht,cal[:,1]/Hm),1-FAR)
far=[float(np.mean(far_s[:,0]>Ht)),float(np.mean(far_s[:,1]>Hm)),
     float(np.mean(np.maximum(far_s[:,0]/Ht,far_s[:,1]/Hm)>Hf))]
def detect(mt,mm): return int(mt>Ht),int(mm>Hm),int(max(mt/Ht,mm/Hm)>Hf)

# ---- benchmark: P(detect) vs ramp rate ----
ramp_rates=[2.0,1.0,0.5,0.25,0.1,0.05,0.02]; NT=50
def ramp_fn(rr):
    def f(k): return np.deg2rad(min(rr*max(k-T0,0),np.rad2deg(DELTA))) if k>=T0 else 0.0
    return f
pdet={"temporal":[],"manifold":[],"fused":[]}
for rr in ramp_rates:
    D=np.array([detect(*stream(40000+int(rr*1e4)+i,ramp_fn(rr))) for i in range(NT)])
    for j,k in enumerate(["temporal","manifold","fused"]): pdet[k].append(float(np.mean(D[:,j])))
# AUC @ slow 0.03
def slow(k): return np.deg2rad(min(0.03*max(k-T0,0),4.0)) if k>=T0 else 0.0
ben=np.array([stream(50000+i) for i in range(40)]); atk=np.array([stream(51000+i,slow) for i in range(40)])
lab=np.r_[np.zeros(40),np.ones(40)]
auc={k:round(float(roc_auc_score(lab,np.r_[ben[:,j],atk[:,j]])),3) for j,k in enumerate(["temporal","manifold"])}
# bias budget (single snapshot)
biases=np.linspace(0.1,3.0,12)
zres=np.array([np.mean([(rho(POOL[i]+L@rng.standard_normal(NX)+np.eye(NX)[tgt]*np.deg2rad(b))-mu_rho)/sig_rho for i in range(40)]) for b in biases])
gate=3.0; idx=np.where(zres>gate)[0]; bias_lim=float(biases[idx[0]]) if len(idx) else float("nan")

print("SUBMISSION IEEE-118 | NT=%d | FAR t/m/f=%.0f/%.0f/%.0f%% | AUC@0.03 t=%.2f m=%.2f | bias lim %.2f deg"%(
    NT,far[0]*100,far[1]*100,far[2]*100,auc["temporal"],auc["manifold"],bias_lim))
print("P(detect) vs ramp:  rate temporal manifold fused")
for i,rr in enumerate(ramp_rates): print("  %5.2f   %.2f    %.2f   %.2f"%(rr,pdet["temporal"][i],pdet["manifold"][i],pdet["fused"][i]))

# ---- figure ----
cols={"temporal":"#C0392B","manifold":"#2E7D32","fused":"#1F4E79"}; mk={"temporal":"o","manifold":"s","fused":"^"}
fig,ax=plt.subplots(1,2,figsize=(10,4))
for k in pdet: ax[0].plot(ramp_rates,pdet[k],mk[k]+"-",color=cols[k],label=k)
ax[0].set_xscale("log");ax[0].set_xlabel("attack ramp rate (deg/step)");ax[0].set_ylabel("P(detect)");ax[0].set_ylim(-.05,1.05)
ax[0].set_title("(a) IEEE-118 ramp-escape floor:\nmanifold/fused rate-invariant, temporal escapes");ax[0].legend(fontsize=8);ax[0].grid(alpha=.3)
ax[1].plot(biases,zres,"o-",color="#1F4E79");ax[1].axhline(gate,ls="--",color="#C0392B",label="3σ gate")
if np.isfinite(bias_lim): ax[1].axvline(bias_lim,ls=":",color="#2E7D32",label=f"bias limit ≈ {bias_lim:.2f}°")
ax[1].set_xlabel("targeted bias (deg)");ax[1].set_ylabel("off-manifold residual (σ)")
ax[1].set_title("(b) IEEE-118 targeted-bias budget\n(manifold dim K=%d of %d states)"%(K,NX));ax[1].legend(fontsize=8);ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/exp7_ieee118.png",dpi=150)
json.dump(dict(system="IEEE-118",n_states=NX,m_meas=M if (M:=e.M) else 0,manifold_dim=int(K),trials_per_point=NT,
    far_temporal=round(far[0],3),far_manifold=round(far[1],3),far_fused=round(far[2],3),
    ramp_rates=ramp_rates,pdet_temporal=pdet["temporal"],pdet_manifold=pdet["manifold"],pdet_fused=pdet["fused"],
    auc_slowramp=auc,bias_limit_deg=None if not np.isfinite(bias_lim) else round(bias_lim,3)),
    open(f"{OUT}/exp7_summary.json","w"),indent=2)
print("saved exp7_ieee118.png + exp7_summary.json")
