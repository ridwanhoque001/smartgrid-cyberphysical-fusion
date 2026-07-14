"""
EXPERIMENT 7 — Generality check on IEEE-118: does the Results-V story hold at scale?
Full AC WLS SE + BDD on case118. Verifies (i) stealthy-FDI blindness with a real
solve, then evaluates the temporal vs manifold detectors using the exact identity
that a consistent FDI yields x_hat = x_a, so the estimated-state stream equals the
benign stream + injection + SE noise ~ N(0,(H^T W H)^-1). This makes 50-trial Monte
Carlo tractable at scale without a per-timestep solve.
"""
import os, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as nw
from scipy.stats import chi2
np.random.seed(0)

net=nw.case118(); pp.runpp(net)
Ybus=net["_ppc"]["internal"]["Ybus"].toarray(); NB=len(net.bus)
SLACK=int(net.ext_grid.bus.iloc[0]); NONSLACK=[b for b in range(NB) if b!=SLACK]
NTH=len(NONSLACK); NX=NTH+NB
LINES=[(int(l.from_bus),int(l.to_bus)) for _,l in net.line.iterrows()]; NL=len(LINES)
base_p=net.load.p_mw.values.copy(); base_q=net.load.q_mvar.values.copy(); nLoad=len(base_p)
M=NB+2*NB+2*NL
sig=np.concatenate([np.full(NB,0.004),np.full(NB,0.008),np.full(NB,0.008),np.full(NL,0.008),np.full(NL,0.008)])
W=1.0/sig**2; DOF=M-NX; CHI2_THR=chi2.ppf(0.99,DOF); LNR_THR=3.0

def unpack(x):
    th=np.zeros(NB); th[NONSLACK]=x[:NTH]; return x[NTH:]*np.exp(1j*th)
def h(x):
    V=unpack(x); S=V*np.conj(Ybus@V)
    Pf=np.empty(NL); Qf=np.empty(NL)
    for k,(i,j) in enumerate(LINES):
        Sij=V[i]*np.conj(Ybus[i,j]*(V[i]-V[j])); Pf[k]=Sij.real; Qf[k]=Sij.imag
    return np.concatenate([np.abs(V),S.real,S.imag,Pf,Qf])
def x_from_pf(net):
    return np.concatenate([np.deg2rad(net.res_bus.va_degree.values)[NONSLACK],net.res_bus.vm_pu.values])
def jac(x,eps=1e-6):
    h0=h(x); H=np.empty((M,NX))
    for k in range(NX):
        xp=x.copy(); xp[k]+=eps; H[:,k]=(h(xp)-h0)/eps
    return H
def wls(z,x0,iters=6,tol=1e-6):
    x=x0.copy()
    for _ in range(iters):
        H=jac(x); HtW=H.T*W
        dx=np.linalg.solve(HtW@H+1e-9*np.eye(NX),HtW@(z-h(x))); x=x+dx
        if np.linalg.norm(dx)<tol: break
    return x,H
def bdd(z,xh,H):
    r=z-h(xh); J=float(np.sum(W*r**2))
    G=(H.T*W)@H; Ginv=np.linalg.inv(G+1e-9*np.eye(NX))
    Omega=np.diag(sig**2)-H@Ginv@H.T
    rN=np.abs(r)/np.sqrt(np.clip(np.diag(Omega),1e-12,None))
    return J,float(rN.max())

if __name__=="__main__":
    import time
    print("IEEE-118 | states n=%d (theta %d, V %d) | meas m=%d | redundancy %.2f | chi2 dof=%d thr=%.0f"%(
        NX,NTH,NB,M,M/NX,DOF,CHI2_THR))
    xt=x_from_pf(net); rng=np.random.default_rng(1)
    t=time.time(); H0=jac(xt); print("jac: %.0f ms"%((time.time()-t)*1000))
    t=time.time(); xh,H=wls(h(xt)+rng.normal(0,sig),xt,iters=5); print("wls(5): %.0f ms"%((time.time()-t)*1000))
    J,lnr=bdd(h(xt)+rng.normal(0,sig),xh,H)
    print("clean chi2 J=%.1f (dof=%d thr=%.0f) LNR=%.2f | state err %.3f deg"%(J,DOF,CHI2_THR,lnr,
        np.rad2deg(np.abs(xh[:NTH]-xt[:NTH]).max())))
    # blindness: AC-exact stealthy FDI on a load bus angle
    TB=20; tgt=NONSLACK.index(TB) if TB in NONSLACK else 20; DELTA=np.deg2rad(4.0)
    xa=xt.copy(); xa[tgt]+=DELTA; a=h(xa)-h(xt)
    z=h(xt)+rng.normal(0,sig)
    xhc,Hc=wls(z,xt); Jc,lc=bdd(z,xhc,Hc)
    xha,Ha=wls(z+a,xt); Ja,la=bdd(z+a,xha,Ha)
    print("[blindness] chi2 clean=%.1f attacked=%.1f (thr %.0f) | LNR %.2f/%.2f | est moved %.2f deg (target %.1f)"%(
        Jc,Ja,CHI2_THR,lc,la,np.rad2deg(xha[tgt]-xt[tgt]),np.rad2deg(DELTA)))
    # SE covariance at nominal
    G=(H0.T*W)@H0; Sigma=np.linalg.inv(G+1e-9*np.eye(NX))
    print("SE noise std at target angle = %.4f deg"%np.rad2deg(np.sqrt(Sigma[tgt,tgt])))
    np.save("results/_ieee118_Sigma.npy",Sigma)
    print("saved Sigma; tgt idx=%d"%tgt)
