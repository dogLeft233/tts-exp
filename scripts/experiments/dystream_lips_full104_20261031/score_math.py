"""Independent source-level paired analysis; reads raw scorer artifacts only."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

METRICS=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']
FOLDERS=dict(zip(METRICS,['syncnet','cogenav','peavs','syncreward','avs_u','mtd','vocalist','synchformer','sparsesync']))

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def statistics(n,t,seed=20261030):
    n=np.asarray(n,dtype=float);t=np.asarray(t,dtype=float)
    assert n.shape==t.shape and n.ndim==1 and len(n) and np.isfinite(n).all() and np.isfinite(t).all()
    d=t-n;rng=np.random.default_rng(seed)
    ci=np.quantile(d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1),[.025,.975])
    exceed=0
    for _ in range(50):
        signs=rng.integers(0,2,size=(2000,len(d)))*2-1
        exceed+=int(np.sum(abs((signs*d).mean(1))>=abs(d.mean())-1e-15))
    sd=float(d.std(ddof=1)) if len(d)>1 else 0
    return dict(n_sources=len(d),N_mean=float(n.mean()),T_mean=float(t.mean()),delta=float(d.mean()),
        ci95=ci.tolist(),paired_dz=float(d.mean()/sd) if sd>0 else None,
        T_higher=int(np.sum(d>0)),N_higher=int(np.sum(d<0)),ties=int(np.sum(d==0)),signflip_p=(exceed+1)/100001)

def holm(ps):
    a=np.asarray(ps,dtype=float);order=np.argsort(a,kind='stable');out=np.empty_like(a)
    out[order]=np.minimum(1,np.maximum.accumulate(a[order]*np.arange(len(a),0,-1)))
    return out.tolist()

def load_values(root,metric):
    folder=FOLDERS[metric]
    if metric in ['SyncNet','CoGenAV']:
        return {r['id']:float(r['C' if metric=='SyncNet' else 'eq1']) for r in read(root/folder/'scores.json') if r.get('status','PASS')=='PASS'}
    if metric=='PEAVS':
        return {Path(r['filename']).stem:float(r['score']) for r in read(root/folder/'scores.json')['scores']}
    if metric=='SyncReward':
        return {Path(r['video_path']).stem:float(r['reward']) for r in map(json.loads,(root/folder/'scores.jsonl').read_text().splitlines())}
    result={}
    for row in read(root/folder/'batch_scores.json'):
        if row['status']!='PASS':continue
        # Resolve from the delivery root rather than trusting original machine paths.
        r=read(root/folder/row['id']/'deployment_validation.json')['results']['0']
        result[row['id']]=float(r['frame_mean'] if metric=='AVS_u' else r['author_metrics']['K5'] if metric in ['MTDVocaLiST','VocaLiST'] else r['p_zero'])
    return result

