"""
================================================================================
Well-powered cyber-side cross-validation on REAL IEC-104 HMI traffic
(vrt-iec104 subset of Matousek et al.), with time-based attack labels.
================================================================================
Five behavioral attacks (MITM, replay, value-change, report-block, masquerade)
captured over 1-3 days each, with attack intervals given in the dataset Readme.
We window flows by time (120 s), extract network features, and detect attacks
(5-fold out-of-fold). Result: structurally-visible attacks (report-block, replay)
are caught (recall 0.87-0.98), but semantic value-manipulation attacks
(value-change, MITM, masquerade) largely evade the cyber layer (0.11-0.35) --
their effect would instead appear in the PHYSICAL measurements, motivating fusion.

Point DATA_DIR at .../ics-dataset-for-smart-grids/vrt-iec104
Run:  pip install pandas scikit-learn matplotlib ; python iec104_hmi_crossval.py
================================================================================
"""
import os, numpy as np, pandas as pd
from datetime import datetime as D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA_DIR = "data/ics-dataset-for-smart-grids/vrt-iec104"
W = 120                                   # seconds per window
NOVEL = ['45','46','120','121','122','123','124','125']
# capture start + attack intervals (absolute times from the dataset Readme)
FILES = {
 'HMI_Standard.csv':     (D(2021,6,29,7,8,31),  [], 'normal'),
 'HMI_MITM.csv':         (D(2021,6,16,10,29,12),
    [(D(2021,6,16,20,23,4),D(2021,6,16,21,31,34)),(D(2021,6,17,5,29,45),D(2021,6,17,6,52,15)),
     (D(2021,6,18,22,15,19),D(2021,6,19,2,38,9))], 'mitm'),
 'report_block_HMI.csv': (D(2021,11,26,13,20,48),[(D(2021,11,26,23,35,52),D(2021,11,27,1,47,32))],'report_block'),
 'replay_HMI.csv':       (D(2021,11,25,7,59,50),
    [(D(2021,11,25,20,16,31),D(2021,11,25,22,35,52)),(D(2021,11,26,7,38,53),D(2021,11,26,8,49,53))],'replay'),
 'value_change_HMI.csv': (D(2021,11,23,21,1,30),
    [(D(2021,11,24,7,22,21),D(2021,11,24,9,3,1)),(D(2021,11,24,20,56,23),D(2021,11,24,22,57,13))],'value_change'),
 'masquerating_HMI.csv': (D(2021,11,27,14,0,44),[(D(2021,11,27,14,15,14),D(2021,11,27,19,11,15))],'masquerade'),
}
COLS=['Relative Time','srcIP','dstIP','ipLen','len','fmt','uType','asduType','cot']

def features(fn, start, atk):
    d=pd.read_csv(os.path.join(DATA_DIR,fn),sep=';',dtype=str,keep_default_na=False,usecols=COLS)
    rt=pd.to_numeric(d['Relative Time'],errors='coerce').fillna(0).values
    df=pd.DataFrame({'b':(rt//W).astype(int),
        'ipl':pd.to_numeric(d['ipLen'],errors='coerce').fillna(0).values,
        'ln': pd.to_numeric(d['len'],errors='coerce').fillna(0).values,
        'fu':(d['fmt']=='0x00000003').values,'fi':(d['fmt']=='0x00000000').values,
        'test':d['uType'].isin(['0x00000010','0x00000020']).values,
        'a100':(d['asduType']=='100').values,'a46':(d['asduType']=='46').values,
        'a1':(d['asduType']=='1').values,'a3':(d['asduType']=='3').values,'a36':(d['asduType']=='36').values,
        'anov':d['asduType'].isin(NOVEL).values,
        'c3':(d['cot']=='3').values,'c6':(d['cot']=='6').values,'c7':(d['cot']=='7').values,
        'c20':(d['cot']=='20').values,'c45':(d['cot']=='45').values,
        'src':d['srcIP'].values,'dst':d['dstIP'].values})
    g=df.groupby('b')
    F=pd.DataFrame({'pkt':g.size(),'iplen':g['ipl'].mean(),'lenm':g['ln'].mean(),'fu':g['fu'].mean(),
        'fi':g['fi'].mean(),'test':g['test'].mean(),'a100':g['a100'].mean(),'a46':g['a46'].mean(),
        'a1':g['a1'].mean(),'a3':g['a3'].mean(),'a36':g['a36'].mean(),'anov':g['anov'].mean(),
        'c3':g['c3'].mean(),'c6':g['c6'].mean(),'c7':g['c7'].mean(),'c20':g['c20'].mean(),'c45':g['c45'].mean(),
        'usrc':g['src'].nunique(),'udst':g['dst'].nunique()})
    binc=F.index.values*W+W/2; lab=np.zeros(len(F),int)
    for a,b in atk:
        lab |= ((binc>=(a-start).total_seconds())&(binc<=(b-start).total_seconds())).astype(int)
    return F.values, lab

X=[];y=[];AT=[]
for fn,(start,atk,at) in FILES.items():
    f,l=features(fn,start,atk); X.append(f);y.append(l);AT.append(np.array([at]*len(f)))
    print(f"{fn:22s} windows={len(f):4d} attack={l.sum():3d}")
X=np.nan_to_num(np.vstack(X)); y=np.concatenate(y); AT=np.concatenate(AT)
pred=cross_val_predict(RandomForestClassifier(300,random_state=42,n_jobs=-1,class_weight="balanced"),
                       X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42))
print(f"\noverall macro-F1={f1_score(y,pred,average='macro'):.3f}  attack-F1={f1_score(y,pred,pos_label=1):.3f}  FPR={(pred[y==0]==1).mean():.3f}")
rec={}
for at in ['report_block','replay','mitm','masquerade','value_change']:
    m=(AT==at)&(y==1)
    if m.sum(): rec[at]=(pred[m]==1).mean(); print(f"  {at:13s}: recall {rec[at]:.2f} ({m.sum()} win)")
plt.figure(figsize=(6.6,4)); ks=list(rec); vs=[rec[k] for k in ks]
plt.bar(ks,vs,color=["#2E7D32" if v>=0.5 else "#C0392B" for v in vs])
plt.ylim(0,1.08); plt.ylabel("detection recall"); plt.xticks(rotation=15)
plt.title("Behavioral attacks on real IEC-104 HMI traffic"); plt.tight_layout()
plt.savefig("iec104_hmi.png",dpi=140); print("saved iec104_hmi.png")
