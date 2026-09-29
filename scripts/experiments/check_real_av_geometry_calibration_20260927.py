"""Independent raw-coordinate reconstruction of the real-input calibration gate."""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/real_av_geometry_calibration_20260927'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def euler(matrix):
    a=matrix[:3,:3];u,s,vt=np.linalg.svd(a);r=u@vt
    assert np.linalg.det(r)>0 and max(s)/min(s)<=1.02
    return np.array([np.arctan2(-r[2,0],np.hypot(r[0,0],r[1,0])),np.arctan2(r[2,1],r[2,2]),np.arctan2(r[1,0],r[0,0])])*180/np.pi
def reconstruct(z):
    ref=z['raw'][0,:,:2].astype(float)*224;eye=np.linalg.norm(ref[0]-ref[3]);base=euler(z['matrices'][0]);valid=z['valid'][1:].copy();aps=[];res=[];poses=[]
    for k,(raw,m) in enumerate(zip(z['raw'][1:],z['matrices'][1:])):
        if not valid[k]:continue
        pts=raw[:,:2].astype(float)*224;a=pts[:7];b=ref[:7];ac=a-a.mean(0);bc=b-b.mean(0)
        # Closed-form2D least squares, independent of worker's SVD similarity.
        aa=np.sum(ac*bc);bb=np.sum(ac[:,0]*bc[:,1]-ac[:,1]*bc[:,0]);theta=np.arctan2(bb,aa)
        rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        scale=np.sqrt(aa*aa+bb*bb)/np.sum(ac*ac);p=scale*(pts-a.mean(0))@rot+b.mean(0)
        aps.append(np.linalg.norm(p[7]-p[8])/eye);res.append(np.sqrt(np.mean(np.sum((p[:7]-b)**2,1)))/eye);poses.append(euler(m))
    bad=np.flatnonzero(~valid);longest=0
    for _,vals in __import__('itertools').groupby(enumerate(bad),lambda x:x[1]-x[0]):longest=max(longest,len(list(vals)))
    pose95=np.percentile(abs((np.array(poses)[:,:2]-base[:2]+180)%360-180),95,axis=0)
    result={'valid_fraction':float(valid.mean()),'missing_run':longest,'reference_angles_deg':base.tolist(),'pose_change_p95_deg':pose95.tolist(),'stable_residual_median':float(np.median(res)),'stable_residual_p95':float(np.percentile(res,95)),'aperture_sd':float(np.std(aps))}
    result['passed']=bool(valid.mean()>=.95 and longest<=2 and max(abs(base[:2]))<=20 and max(pose95)<=10 and np.median(res)<=.02 and np.percentile(res,95)<=.04 and np.std(aps)>=.005)
    return result
def main():
    p=json.load(open(OUT/'protocol.json'));checks=[];maxerr=0
    for row in p['calibration']:
        dest=OUT/'input'/row['sample_id'];meta=json.load(open(dest.with_suffix('.json')));assert sha(dest.with_suffix('.npz'))==meta['feature_sha256']
        z=np.load(dest.with_suffix('.npz'));actual=meta['qc']
        if z['valid'][0]:
            result=reconstruct(z);assert result['passed']==actual['passed']
            for name,v in result.items():
                if name=='passed':continue
                err=float(np.max(np.abs(np.asarray(v)-np.asarray(actual[name]))));maxerr=max(maxerr,err);assert err<1e-7,(name,err)
        else:assert not actual['passed'];result={'passed':False,'reference_invalid':True}
        assert np.max(abs(np.diff(z['pts'][1:])-.04))<1e-6
        checks.append({'id':row['sample_id'],'recomputed':result})
    gate=json.load(open(OUT/'input_gate.json'));assert sum(r['recomputed']['passed'] for r in checks)==gate['passed_sources']
    output={'status':'PASS','checked_sources':8,'maximum_numeric_error':maxerr,'source_input_pass_count':gate['passed_sources'],'meaning':'independent numerical engineering check, not scientific gate pass','records':checks,'checker_sha256':sha(__file__)}
    (OUT/'independent_input_check.json').write_text(json.dumps(output,indent=2)+'\n');print(output['status'],maxerr,gate['passed_sources'])
if __name__=='__main__':main()
