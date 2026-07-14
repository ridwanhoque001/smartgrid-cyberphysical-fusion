"""
Build the fused dataset-1 feature table ONCE and cache it (features_ds1.parquet).
Also emits a per-window time key so downstream scripts can do time-blocked CV.
Reuses the exact feature engineering from cyber_physical_fusion.py.
"""
import os, glob, numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT = os.environ.get("OUT", "results"); os.makedirs(OUT, exist_ok=True)
WINDOW = 0.5

CLASSES = {"benign":"benign","backdoor":"backdoor","Bruteforce":"bruteforce",
           "FDI":"fdi","ransomware":"ransomware","reverseshell":"reverseshell"}
PROTOS = ["TCP","Modbus/TCP","HTTP","TLSv1.2","ARP","ICMPv6"]
PHYS_COLS = ["Freq","Theta","V_A","V_B","V_C","I_A","I_B","I_C",
             "ActivePower","ReactivePower","BreakerStatus"]
CANON = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst",
         "_ws.col.Protocol","ip.len","tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"]

def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",","",regex=False)
                          .str.replace('"',"",regex=False), errors="coerce")
def find(folder, name):
    hits = glob.glob(os.path.join(DATA_DIR,"Dataset",folder,name))
    return hits[0] if hits else None
def _looks_epoch(series):
    v = pd.to_numeric(series, errors="coerce")
    return v.between(1.6e9, 1.8e9).mean() > 0.5
def normalize_cyber(cy):
    cy = cy.loc[:, ~cy.columns.astype(str).str.startswith("Unnamed")]
    if "frame.time_epoch" in cy.columns and _looks_epoch(cy["frame.time_epoch"]):
        return cy
    cy = cy.iloc[:, :11].copy(); cy.columns = CANON
    return cy
def cyber_features(path):
    cy = pd.read_csv(path, low_memory=False); cy.columns=[c.strip() for c in cy.columns]
    cy = normalize_cyber(cy)
    t = to_num(cy["frame.time_epoch"]); cy=cy[t.notna()].copy(); cy["t"]=t[t.notna()].values
    cy["bin"]=(cy["t"]//WINDOW).astype(np.int64); cy["iplen"]=to_num(cy["ip.len"])
    g=cy.groupby("bin")
    feat=pd.DataFrame({"cyb_pkt_count":g.size(),"cyb_iplen_mean":g["iplen"].mean(),
        "cyb_iplen_sum":g["iplen"].sum(),"cyb_iplen_std":g["iplen"].std(),
        "cyb_n_src_ip":g["ip.src"].nunique(),"cyb_n_dst_ip":g["ip.dst"].nunique()})
    proto=cy.groupby(["bin","_ws.col.Protocol"]).size().unstack(fill_value=0)
    for p in PROTOS:
        feat["cyb_frac_"+p.replace("/","_")] = (proto[p]/feat["cyb_pkt_count"]) if p in proto.columns else 0.0
    feat["t_start"] = feat.index.values * WINDOW
    return feat
def phys_features(path):
    ph=pd.read_csv(path, low_memory=False); ph.columns=[c.strip() for c in ph.columns]
    t=to_num(ph["@timestamp"]); ph=ph[t.notna()].copy(); ph["t"]=t[t.notna()].values
    ph["bin"]=(ph["t"]//WINDOW).astype(np.int64)
    for c in PHYS_COLS: ph[c]=to_num(ph[c])
    g=ph.groupby("bin")[PHYS_COLS]; feat=g.agg(["mean","std"])
    feat.columns=["phys_"+a+"_"+b for a,b in feat.columns]
    return feat

rows=[]
for folder,label in CLASSES.items():
    cf=cyber_features(find(folder,"[Cc]yber.csv")); pf=phys_features(find(folder,"Physical.csv"))
    joined=pf.join(cf,how="inner"); joined["label"]=label
    joined["bin_idx"]=joined.index.values          # time-ordered window index within class
    joined["cls"]=label
    rows.append(joined); print(f"{label:12s}: {len(joined)} windows")
data=pd.concat(rows).reset_index(drop=True)
data=data.replace([np.inf,-np.inf],np.nan).fillna(0)
data.to_parquet(os.path.join(OUT,"features_ds1.parquet"))
print("\nSaved", os.path.join(OUT,"features_ds1.parquet"), "shape", data.shape)
