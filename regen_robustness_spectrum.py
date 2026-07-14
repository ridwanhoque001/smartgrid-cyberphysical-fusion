import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif","font.size":15,
  "axes.titlesize":15,"axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12.5})
labels=["Random\n(phys)","PGD\n(phys)","Adaptive\n(phys)","Adaptive\nJOINT"]
undef=[0.80,0.80,0.67,0.34]; deff=[0.93,0.90,0.82,0.55]; clean=0.96
x=np.arange(len(labels)); w=0.38
fig,ax=plt.subplots(figsize=(6.2,4.4))
ax.axhline(clean,ls="--",color="#2E7D32",label=f"clean ({clean:.2f})")
ax.bar(x-w/2,undef,w,label="undefended",color="#C0392B")
ax.bar(x+w/2,deff,w,label="adversarially trained",color="#2E75B6")
for xi,v in zip(x-w/2,undef): ax.text(xi,v+0.015,f"{v:.2f}",ha="center",fontsize=13)
for xi,v in zip(x+w/2,deff): ax.text(xi,v+0.015,f"{v:.2f}",ha="center",fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("macro-F1"); ax.set_ylim(0,1.08)
ax.set_title(r"Robustness across attack types ($\varepsilon=0.10$)")
ax.legend(loc="lower left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("results/robustness_spectrum.png",dpi=320); print("saved robustness_spectrum.png")
