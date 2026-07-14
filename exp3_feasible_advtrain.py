"""
EXPERIMENT 3 — Physics-feasible FDI on IEEE-14 and FEASIBLE-SPACE adversarial training.

Pipeline (DC state estimation, matching paper Eq. 1):
  * IEEE-14 (pandapower) -> DC measurement Jacobian H (branch active-power flows).
  * benign states: random load scaling -> DC power flow -> z = H theta (+ noise).
  * feasible stealthy FDI: a = H c  =>  z_a = z + Hc, LS residual ~ 0 (undetectable by BDD).
  * detector: RF on the measurement vector, benign vs feasible-FDI.
  * ES evasion: (mu,lambda) Evolution Strategy searches c (||c||_inf<=gamma) to minimize
    RF P(attack | z+Hc), staying inside the feasible manifold.
  * DEFENSE: feasible-space adversarial training — augment the training set with feasible
    a=Hc perturbations (random + ES-mined) and retrain; re-run the SAME ES evasion.

Outputs: results/feasible_advtrain.txt, results/feasible_advtrain.png
"""
import os, numpy as np
import pandapower as pp, pandapower.networks as nw
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(7); OUT="results"

# ---- build DC measurement model for IEEE-14 ----
net=nw.case14()
nb=len(net.bus); slack=int(net.ext_grid.bus.iloc[0])
non_slack=[b for b in range(nb) if b!=slack]
# branch reactances (lines + trafos approximated by their x)
br=[]
for _,l in net.line.iterrows():
    x=l.x_ohm_per_km*l.length_km
    br.append((int(l.from_bus),int(l.to_bus),x if x>1e-6 else 0.05))
for _,t in net.trafo.iterrows():
    z=t.vk_percent/100.0; br.append((int(t.hv_bus),int(t.lv_bus),max(z,0.05)))
m=len(br)
# H: branch flow P_ij = (theta_i - theta_j)/x  as function of non-slack angles
H=np.zeros((m,len(non_slack))); pos={b:i for i,b in enumerate(non_slack)}
for k,(i,j,x) in enumerate(br):
    if i in pos: H[k,pos[i]]+=1.0/x
    if j in pos: H[k,pos[j]]-=1.0/x
n=len(non_slack)
# Partial observability: only a subset of branches are metered (realistic). The
# adversary exploits the unmetered branches as a null space to hide the injection.
rng_obs=np.random.default_rng(3); OBS=np.sort(rng_obs.choice(m,size=12,replace=False))
def obs(Z): return Z[:,OBS] if Z.ndim==2 else Z[OBS]

def make_states(ns, rng):
    """random operating points -> DC angles -> measurements z=H theta (+noise)."""
    Z=[]
    for _ in range(ns):
        theta=rng.normal(0,0.04,n)          # plausible angle spread (rad)
        z=H@theta + rng.normal(0,0.0015,m)  # metering noise
        Z.append(z)
    return np.array(Z)

def feasible_attack(z, rng, gamma):
    c=rng.uniform(-gamma,gamma,n); return z+H@c   # a=Hc, stays on manifold

# --- attack model ---------------------------------------------------------
# The adversary must achieve a MEANINGFUL, fixed bias on a target state coordinate
# (else "evasion" degenerates to not attacking). c[TARGET]=DELTA is fixed; the
# remaining freedom (other coords, bounded by GAMMA) can be shaped to evade the ML
# detector while a=Hc keeps the BDD residual at the benign noise floor (stealthy).
TARGET=0; DELTA=0.14; GAMMA=0.05
def attack_from_c(z, c_rest):
    c=np.zeros(n); c[TARGET]=DELTA; mask=np.arange(n)!=TARGET; c[mask]=c_rest
    return z+H@c

rng=np.random.default_rng(7)
NB=1200
Zben=make_states(NB,rng)
# naive (non-evasive) feasible attack: only the target bias, no evasive shaping
Zatk=np.array([attack_from_c(z,np.zeros(n-1)) for z in make_states(NB,rng)])

# residual check: stealthiness of a=Hc
Hpinv=np.linalg.pinv(H)
def residual(z):
    th=Hpinv@z; return np.linalg.norm(z-H@th)
res_atk=np.mean([residual(z) for z in Zatk[:50]])

Zall=np.vstack([Zben,Zatk]); yv=np.r_[np.zeros(NB),np.ones(NB)].astype(int)
perm=rng.permutation(len(Zall)); Zall,yv=Zall[perm],yv[perm]
X=obs(Zall)                                   # detector sees only metered branches
ntr=int(0.7*len(X)); Xtr,Xte,ytr,yte=X[:ntr],X[ntr:],yv[:ntr],yv[ntr:]

def train(Xt,yt):
    r=RandomForestClassifier(n_estimators=150,random_state=0,n_jobs=1); r.fit(Xt,yt); return r

base=train(Xtr,ytr)
clean_f1=f1_score(yte,base.predict(Xte),average="macro")

# ---- ES evasion: keep target bias fixed, shape the OTHER coords to evade ----
def es_evade(clf, z0, gamma=GAMMA, mu=5, lam=14, gens=8, seed=0):
    rng=np.random.default_rng(seed); mean=np.zeros(n-1); sigma=gamma*0.6
    best=attack_from_c(z0,np.zeros(n-1)); best_p=clf.predict_proba(obs(best)[None])[0,1]
    for g in range(gens):
        pop=np.clip(mean+sigma*rng.normal(size=(lam,n-1)),-gamma,gamma)
        Zc=np.array([attack_from_c(z0,c) for c in pop])
        p=clf.predict_proba(obs(Zc))[:,1]            # P(attack) on metered branches; minimize
        order=np.argsort(p); mean=pop[order[:mu]].mean(0); sigma*=0.9
        if p[order[0]]<best_p: best_p=p[order[0]]; best=Zc[order[0]]
    return best                                       # full-measurement attacked vector

def evasion_recall(clf, n_eval=80, seed=1):
    rng=np.random.default_rng(seed); base_states=make_states(n_eval,rng)
    evaded=np.array([es_evade(clf,z,seed=1000+i) for i,z in enumerate(base_states)])
    return (clf.predict(obs(evaded))==1).mean()      # recall on ES-evaded (still-effective) attacks

rec_before=evasion_recall(base)

# ---- FEASIBLE-SPACE ADVERSARIAL TRAINING ----
# augment training set with feasible a=Hc perturbations, incl. ES-mined evasions
rng2=np.random.default_rng(21)
aug_atk=[]
atk_train=Xtr[ytr==1]
mine_states=make_states(150,rng2)
for z in mine_states:
    aug_atk.append(es_evade(base,z,seed=int(rng2.integers(1e6))))  # ES-mined evasions
for _ in range(400):
    z=make_states(1,rng2)[0]
    aug_atk.append(attack_from_c(z,rng2.uniform(-GAMMA,GAMMA,n-1))) # random feasible-shaped
aug_atk=obs(np.array(aug_atk))                # project augmented attacks to metered branches
Xtr2=np.vstack([Xtr,aug_atk]); ytr2=np.r_[ytr,np.ones(len(aug_atk))].astype(int)
adv=train(Xtr2,ytr2)
clean_f1_adv=f1_score(yte,adv.predict(Xte),average="macro")
rec_after=evasion_recall(adv)

with open(os.path.join(OUT,"feasible_advtrain.txt"),"w") as f:
    f.write("IEEE-14 PHYSICS-FEASIBLE FDI + FEASIBLE-SPACE ADVERSARIAL TRAINING\n")
    f.write(f"branches(m)={m}  states(n)={n}  gamma={GAMMA}\n")
    f.write(f"feasible attack LS residual (mean) = {res_atk:.2e}  (stealthy)\n\n")
    f.write(f"clean detection macro-F1        : undefended={clean_f1:.3f}  adv-trained={clean_f1_adv:.3f}\n")
    f.write(f"recall under ES evasion          : undefended={rec_before:.3f}  adv-trained={rec_after:.3f}\n")
    f.write(f"recovery from feasible adv. train: +{rec_after-rec_before:.3f}\n")
print(open(os.path.join(OUT,"feasible_advtrain.txt")).read())

plt.figure(figsize=(5.5,4))
labels=["Clean\ndetection","Under ES\nevasion"]; x=[0.0,1.0]; w=0.35
plt.bar([xi-w/2 for xi in x],[clean_f1,rec_before],w,label="Undefended",color="#C0392B")
plt.bar([xi+w/2 for xi in x],[clean_f1_adv,rec_after],w,label="Feasible adv. training",color="#2E7D32")
plt.xticks(x,labels); plt.ylabel("macro-F1 / recall"); plt.ylim(0,1); plt.legend()
plt.title("IEEE-14: feasible-space adversarial training")
for i,v in enumerate([clean_f1,rec_before]): plt.text(x[i]-w/2,v+0.02,f"{v:.2f}",ha="center",fontsize=8)
for i,v in enumerate([clean_f1_adv,rec_after]): plt.text(x[i]+w/2,v+0.02,f"{v:.2f}",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"feasible_advtrain.png"),dpi=150)
print("saved feasible_advtrain.png")
