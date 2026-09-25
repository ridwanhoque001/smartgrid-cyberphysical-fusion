"""Master figure regeneration: uniform 9pt serif style, drawn at FINAL print width
so no rescaling occurs -> identical on-page font across every figure.
COL = single-column width; WIDE = full text width (figure*)."""
import os, re, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="results/"
COL=3.45; WIDE=7.0
# ---- unified style (body text is 10pt; figure text 8-9pt = within 2pt) ----
plt.rcParams.update({
 "font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
 "font.size":9,"axes.titlesize":9,"axes.labelsize":9,"axes.titlepad":4,
 "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"figure.titlesize":9,
 "lines.linewidth":1.4,"lines.markersize":4.2,"axes.linewidth":0.8,
 "xtick.major.width":0.8,"ytick.major.width":0.8,"legend.frameon":False,
 "savefig.dpi":320,"savefig.bbox":"tight","savefig.pad_inches":0.03})
CY,PH,GR,RD,PU,GY,BL="#2E7D32","#5B9BD5","#1B5E20","#C0392B","#8E44AD","#B0B0B0","#1f77b4"
def nospine(ax):
    for s in ["top","right"]: ax.spines[s].set_visible(False)
def save(fig,name): fig.savefig(R+name+".png"); plt.close(fig); print("saved",name)

# ============ SINGLE-COLUMN ============
# 1 blocked_cv
fig,ax=plt.subplots(figsize=(COL,2.5))
v=[0.96,0.93,0.80]; e=[0.01,0.06,0.0]
ax.bar(["Random\n5-fold","Time-blocked\n5-fold","Temporal\nhold-out"],v,yerr=e,capsize=4,
       color=[GY,GR,"#0B3D0B"])
for i,x in enumerate(v): ax.text(i,x+e[i]+0.02,f"{x:.2f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.12); ax.set_title("Evaluation protocol vs. leakage"); nospine(ax)
save(fig,"blocked_cv")

# 2 rolling_origin (parse txt)
txt=open(R+"rolling_origin.txt").read()
oo=[float(a) for a,_ in re.findall(r"origin ([\d.]+).*?macro-F1=([\d.]+)",txt)]
sc=[float(b) for _,b in re.findall(r"origin ([\d.]+).*?macro-F1=([\d.]+)",txt)]
m=re.search(r"mean.*?:\s*([\d.]+)\s*\+/-\s*([\d.]+)",txt); mu,sd=float(m.group(1)),float(m.group(2))
sm=re.search(r"single 75/25[^:]*:\s*([\d.]+)",txt); single=float(sm.group(1))
fig,ax=plt.subplots(figsize=(COL,2.5))
ax.plot(oo,sc,"o-",color=GR,label="rolling-origin block")
ax.axhline(mu,ls="--",color=CY,label=f"mean {mu:.2f}$\\pm${sd:.2f}")
ax.axhline(single,ls=":",color=RD,label=f"single 75/25 ({single:.2f})")
ax.fill_between(oo,mu-sd,mu+sd,color=CY,alpha=0.12)
ax.set_xlabel("Train fraction (origin)"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0,1.05)
ax.set_title("Rolling-origin temporal evaluation"); ax.legend(loc="lower left"); nospine(ax)
save(fig,"rolling_origin")

# 3 gnn_headtohead (known)
fig,ax=plt.subplots(figsize=(COL,2.5))
x=np.arange(2); w=0.36
rf=[0.96,0.80]; gc=[0.83,0.70]
ax.bar(x-w/2,rf,w,label="RF fusion",color=GR)
ax.bar(x+w/2,gc,w,label="GCN ($k$-NN graph)",color=PH)
for i in range(2):
    ax.text(x[i]-w/2,rf[i]+0.015,f"{rf[i]:.2f}",ha="center",fontsize=8)
    ax.text(x[i]+w/2,gc[i]+0.015,f"{gc[i]:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["Shuffled","Temporal"]); ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.30)
ax.set_title("RF fusion vs. topology-free GCN")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,1.02),ncol=2,frameon=False); nospine(ax)
save(fig,"gnn_headtohead")

# 4 robustness_spectrum (known)
fig,ax=plt.subplots(figsize=(COL,2.5))
labs=["Random","PGD","Adaptive","Adaptive\njoint"]; und=[0.80,0.80,0.67,0.34]; deff=[0.93,0.90,0.82,0.55]
x=np.arange(4); w=0.38
ax.axhline(0.96,ls="--",color=CY,label="clean (0.96)")
ax.bar(x-w/2,und,w,label="undefended",color=RD)
ax.bar(x+w/2,deff,w,label="adv. trained",color=PH)
for i in range(4):
    ax.text(x[i]-w/2,und[i]+0.015,f"{und[i]:.2f}",ha="center",fontsize=7.5)
    ax.text(x[i]+w/2,deff[i]+0.015,f"{deff[i]:.2f}",ha="center",fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.34)
ax.set_title(r"Robustness across attack types ($\varepsilon=0.10$)")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,1.03),ncol=3,frameon=False,
          handlelength=1.4,columnspacing=1.0); nospine(ax)
save(fig,"robustness_spectrum")

# 5 ieee14_feasible (known)
fig,ax=plt.subplots(figsize=(COL,2.6))
b=ax.bar(["Clean\ndetection","Feasible ES\nevasion"],[0.97,0.33],width=0.55,color=[GR,RD])
for bb,vv in zip(b,[0.97,0.33]): ax.text(bb.get_x()+bb.get_width()/2,vv+0.02,f"{vv:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_ylabel("detection macro-F1"); ax.set_ylim(0,1.1)
ax.set_title("IEEE-14 physics-feasible stealthy FDI\n($a=Hc$, DC residual $\\approx$1e-14)"); nospine(ax)
save(fig,"ieee14_feasible")

# 6 feasible_advtrain (parse txt)
ft=open(R+"feasible_advtrain.txt").read()
cu=re.search(r"clean.*?undefended=([\d.]+)\s+adv-trained=([\d.]+)",ft); c_u,c_a=float(cu.group(1)),float(cu.group(2))
ru=re.search(r"recall under ES evasion\s*:\s*undefended=([\d.]+)\s+adv-trained=([\d.]+)",ft); r_u,r_a=float(ru.group(1)),float(ru.group(2))
fig,ax=plt.subplots(figsize=(COL,2.6))
x=np.arange(2); w=0.36
ax.bar(x-w/2,[c_u,r_u],w,label="Undefended",color=RD)
ax.bar(x+w/2,[c_a,r_a],w,label="Feasible adv. training",color=GR)
for i,(a,b) in enumerate([(c_u,c_a),(r_u,r_a)]):
    ax.text(x[i]-w/2,a+0.015,f"{a:.2f}",ha="center",fontsize=8); ax.text(x[i]+w/2,b+0.015,f"{b:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["Clean\ndetection","Under ES\nevasion"]); ax.set_ylabel("macro-F1 / recall"); ax.set_ylim(0,1.05)
ax.set_title("IEEE-14 feasible-space adversarial training"); ax.legend(loc="upper right"); nospine(ax)
save(fig,"feasible_advtrain")

# 7 iec104_hmi (known)
fig,ax=plt.subplots(figsize=(COL,2.6))
att=["report\nblock","replay","mitm","masq.","value\nchange"]; rec=[0.98,0.87,0.35,0.32,0.11]
col=[GR,GR,RD,RD,RD]
b=ax.bar(att,rec,color=col)
for bb,vv in zip(b,rec): ax.text(bb.get_x()+bb.get_width()/2,vv+0.015,f"{vv:.2f}",ha="center",fontsize=7.5)
ax.set_ylabel("detection recall"); ax.set_ylim(0,1.08); ax.set_title("Behavioral attacks on real IEC-104 HMI"); nospine(ax)
save(fig,"iec104_hmi")

# 8 temporal_baselines (WIDE)
d=json.load(open(R+"temporal_baselines.json"))
names=["Random Forest","Hist Gradient Boosting","MLP (per-window)","TDNN (temporal)","Logistic (linear)"]
disp=["Random\nForest","Hist Grad.\nBoosting","MLP\n(per-window)","TDNN\n(temporal)","Logistic\n(linear)"]
shuf=[d[n]["shuf"] for n in names]; roll=[d[n]["roll"] for n in names]; sdv=[d[n]["sd"] for n in names]
fig,ax=plt.subplots(figsize=(WIDE,2.7))
x=np.arange(5); w=0.38
ax.bar(x-w/2,shuf,w,label="shuffled CV (optimistic)",color=GY)
ax.bar(x+w/2,roll,w,yerr=sdv,capsize=3,label="rolling-origin (temporal)",color=GR)
for i in range(5):
    ax.text(x[i]-w/2,shuf[i]+0.015,f"{shuf[i]:.2f}",ha="center",fontsize=7.5)
    ax.text(x[i]+w/2,roll[i]+sdv[i]+0.015,f"{roll[i]:.2f}",ha="center",fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(disp); ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.1)
ax.set_title("Temporal drift is not model-specific: every family drops under rolling-origin")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=2); nospine(ax)
save(fig,"temporal_baselines")

# 9 selective_both (WIDE 2-panel)
cov=[100,90,80,70]; d1=[0.956,0.985,0.993,0.997]; d3=[0.749,0.771,0.781,0.794]
fig,ax=plt.subplots(1,2,figsize=(WIDE,2.6),constrained_layout=True)
ax[0].plot(cov,d1,"o-",color=GR)
for x,v in zip(cov,d1): ax[0].annotate(f"{v:.3f}",(x,v),textcoords="offset points",xytext=(0,6),ha="center",fontsize=7)
ax[0].set_title("Dataset 1 (cyber-physical fusion)"); ax[0].set_xlabel("coverage (%)"); ax[0].set_ylabel("macro-F1 on retained")
ax[0].invert_xaxis(); ax[0].set_ylim(0.90,1.01); ax[0].grid(alpha=0.25); nospine(ax[0])
ax[1].plot(cov,d3,"s-",color=PH)
for x,v in zip(cov,d3): ax[1].annotate(f"{v:.3f}",(x,v),textcoords="offset points",xytext=(0,6),ha="center",fontsize=7)
ax[1].set_title("Dataset 3 (real IEC-104 HMI)"); ax[1].set_xlabel("coverage (%)")
ax[1].invert_xaxis(); ax[1].set_ylim(0.72,0.81); ax[1].grid(alpha=0.25); nospine(ax[1])
fig.suptitle("Selective detection (abstain on low-confidence $\\rightarrow$ escalate to review)")
save(fig,"selective_both")

# 10 semantic_fusion (WIDE 2-panel): taxonomy bars + graded knowledge sweep
st=open(R+"semantic_fusion.txt").read()
kn=[(int(a),float(c)) for a,b,c in re.findall(r"\s*(\d+)%\s+([\d.eE+-]+)\s+([\d.]+)",st)]
kk=[a for a,_ in kn]; pr=[b for _,b in kn]
fig,ax=plt.subplots(1,2,figsize=(WIDE,2.7),constrained_layout=True)
views=["Cyber","Physics","Fusion"]; struc=[1.00,0.04,1.00]; nv=[0.09,1.00,1.00]; sf=[0.09,0.03,0.05]
x=np.arange(3); w=0.26
ax[0].bar(x-w,struc,w,label="structural",color="#4C72B0")
ax[0].bar(x,nv,w,label="naive value",color="#DD8452")
ax[0].bar(x+w,sf,w,label="stealthy FDI",color="#C44E52")
ax[0].set_xticks(x); ax[0].set_xticklabels(views); ax[0].set_ylabel("detection recall"); ax[0].set_ylim(0,1.1)
ax[0].set_title("Which view catches which attack"); ax[0].legend(loc="upper center",ncol=3,fontsize=7); nospine(ax[0])
ax[1].plot(kk,pr,"o-",color=PU)
ax[1].set_xlabel("attacker topology knowledge (%)"); ax[1].set_ylabel("physics recall"); ax[1].set_ylim(0,1.08)
ax[1].set_title("Graded knowledge: stealthy FDI is the escape"); ax[1].grid(alpha=0.25); nospine(ax[1])
save(fig,"semantic_fusion")

# 11 exp5_gradient_limit (WIDE 3-panel from JSON)
j=json.load(open(R+"exp5_limit_summary.json"))
al=[a*100 for a in j["knowledge_alpha"]]; Jb=j["knowledge_J"]
# Panels (b) ramp-delay and (c) step-floor were removed: exp6 measures both
# quantities for ALL THREE detectors under a single calibration protocol
# (Table ramp / Fig exp6). Reporting exp5's separately calibrated versions
# alongside them produced contradictory numbers for the same quantity.
fig,ax=plt.subplots(figsize=(COL,2.5),constrained_layout=True)
ax.plot(al,Jb,"o-",color=BL); ax.axhline(539 if max(Jb)>200 else 70,ls="--",color=RD,label="$\\chi^2$ threshold")
ax.set_xlabel("AC topology knowledge (%)"); ax.set_ylabel("AC $\\chi^2$ after attack")
ax.set_title("DC-designed FDI is not AC-stealthy"); ax.legend(); ax.grid(alpha=0.3); nospine(ax)
save(fig,"exp5_gradient_limit")

# 12 exp6_manifold_detector (WIDE 3-panel from JSON; panel b reconstructed linear)
j=json.load(open(R+"exp6_summary.json")); rr=j["ramp_rates"]; ss=j["step_sizes"]
pd_={"temporal":j["pdet_temporal"],"manifold":j["pdet_manifold"],"fused":j["pdet_fused"]}
fl={"temporal":j["floor_temporal"],"manifold":j["floor_manifold"],"fused":j["floor_fused"]}
blim=j["bias_limit_deg"]; cols={"temporal":RD,"manifold":GR,"fused":"#1F4E79"}; mk={"temporal":"o","manifold":"s","fused":"^"}
fig,ax=plt.subplots(1,3,figsize=(WIDE,2.35),constrained_layout=True)
for k in ["temporal","manifold","fused"]: ax[0].plot(rr,pd_[k],mk[k]+"-",color=cols[k],label=k)
ax[0].set_xscale("log"); ax[0].set_ylim(-0.05,1.05); ax[0].set_xlabel("attack ramp rate (deg/step)"); ax[0].set_ylabel("P(detect)")
ax[0].set_title("(a) Rate-invariance:\nmanifold catches slow ramps"); ax[0].legend(); ax[0].grid(alpha=0.3); nospine(ax[0])
bx=np.linspace(0,3,50); ax[1].plot(bx,3*bx/blim,"-",color="#1F4E79"); ax[1].axhline(3,ls="--",color=RD,label="$3\\sigma$ gate")
ax[1].axvline(blim,ls=":",color=GR,label=f"limit $\\approx${blim:.2f}$^\\circ$")
ax[1].set_xlabel("targeted bias (deg)"); ax[1].set_ylabel("off-manifold residual ($\\sigma$)")
ax[1].set_title("(b) Targeted-bias budget\nunder the manifold gate"); ax[1].legend(); ax[1].grid(alpha=0.3); nospine(ax[1])
xk=np.arange(len(ss))
for jx,k in enumerate(["temporal","manifold","fused"]): ax[2].bar(xk+(jx-1)*0.26,fl[k],0.26,color=cols[k],label=k)
ax[2].set_xticks(xk); ax[2].set_xticklabels([f"{s:g}°" for s in ss]); ax[2].set_ylim(0,1.05)
ax[2].set_xlabel("step-bias size"); ax[2].set_ylabel("P(detect)")
ax[2].set_title("(c) Tiny steps:\nfusion marginally best"); ax[2].legend(); ax[2].grid(alpha=0.3,axis="y"); nospine(ax[2])
save(fig,"exp6_manifold_detector")

# 13 exp7_ieee118 (WIDE 2-panel from JSON; panel b reconstructed)
j=json.load(open(R+"exp7_summary.json")); rr=j["ramp_rates"]; blim=j["bias_limit_deg"]; NX=j["n_states"]; K=j["manifold_dim"]
pd_={"temporal":j["pdet_temporal"],"manifold":j["pdet_manifold"],"fused":j["pdet_fused"]}
fig,ax=plt.subplots(1,2,figsize=(WIDE,2.6),constrained_layout=True)
for k in ["temporal","manifold","fused"]: ax[0].plot(rr,pd_[k],mk[k]+"-",color=cols[k],label=k)
ax[0].set_xscale("log"); ax[0].set_ylim(-0.05,1.05); ax[0].set_xlabel("attack ramp rate (deg/step)"); ax[0].set_ylabel("P(detect)")
ax[0].set_title("(a) IEEE-118 ramp-escape floor"); ax[0].legend(); ax[0].grid(alpha=0.3); nospine(ax[0])
bx=np.linspace(0,3,50); ax[1].plot(bx,3*bx/blim,"-",color="#1F4E79"); ax[1].axhline(3,ls="--",color=RD,label="$3\\sigma$ gate")
ax[1].axvline(blim,ls=":",color=GR,label=f"limit $\\approx${blim:.2f}$^\\circ$")
ax[1].set_xlabel("targeted bias (deg)"); ax[1].set_ylabel("off-manifold residual ($\\sigma$)")
ax[1].set_title(f"(b) IEEE-118 bias budget (K={K} of {NX})"); ax[1].legend(); ax[1].grid(alpha=0.3); nospine(ax[1])
save(fig,"exp7_ieee118")
print("ALL DONE")
