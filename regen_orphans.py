import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif","font.size":15,
  "axes.titlesize":15,"axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12.5})

# ---- ieee14_feasible.png ----
fig,ax=plt.subplots(figsize=(5.6,4.6))
bars=ax.bar(["Clean\ndetection","Feasible ES\nevasion"],[0.97,0.33],width=0.55,
            color=["#2E7D32","#C0392B"],edgecolor="white")
for b,v in zip(bars,[0.97,0.33]): ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=15,fontweight="bold")
ax.set_ylabel("detection macro-F1"); ax.set_ylim(0,1.08)
ax.set_title("IEEE-14: physics-feasible stealthy FDI\n(residual $\\approx$ 1e-14; DC $a=Hc$)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("results/ieee14_feasible.png",dpi=320); print("saved ieee14_feasible.png")

# ---- selective_both.png ----
cov=[100,90,80,70]
d1=[0.956,0.985,0.993,0.997]; d3=[0.749,0.771,0.781,0.794]
fig,ax=plt.subplots(1,2,figsize=(11.0,4.3))
ax[0].plot(cov,d1,"o-",color="#2E7D32",markersize=8)
for x,v in zip(cov,d1): ax[0].annotate(f"{v:.3f}",(x,v),textcoords="offset points",xytext=(0,9),ha="center",fontsize=12)
ax[0].set_title("Dataset 1 (cyber-physical fusion)"); ax[0].set_xlabel("coverage (%)"); ax[0].set_ylabel("macro-F1 on retained")
ax[0].invert_xaxis(); ax[0].set_ylim(0.90,1.01); ax[0].grid(alpha=0.25)
ax[1].plot(cov,d3,"s-",color="#2E75B6",markersize=8)
for x,v in zip(cov,d3): ax[1].annotate(f"{v:.3f}",(x,v),textcoords="offset points",xytext=(0,9),ha="center",fontsize=12)
ax[1].set_title("Dataset 3 (real IEC-104 HMI)"); ax[1].set_xlabel("coverage (%)")
ax[1].invert_xaxis(); ax[1].set_ylim(0.72,0.81); ax[1].grid(alpha=0.25)
fig.suptitle("Selective detection (abstain on low-confidence $\\rightarrow$ escalate to physical review)",fontsize=15)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("results/selective_both.png",dpi=320); print("saved selective_both.png")
