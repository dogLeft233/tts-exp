"""Independent feature covariance and cohort completeness audit."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_native_spectrum_generation_cross_20260926'
def read(p):return json.loads(Path(p).read_text())
support=read(OUT/'support.json');effects=read(OUT/'effects.json');expected={s:{r['id'] for r in support if r['split']=='evaluation' and (s=='all74' or r['guard20_eligible'])} for s in ['common71','all74']}
assert len(expected['common71'])==71 and len(expected['all74'])==74
buckets={}
for r in effects:buckets.setdefault((r['geometry'],r['policy'],r['support'],r['metric'],r['condition'],r['arm']),set()).add(r['id'])
assert all(ids==expected[key[2]] for key,ids in buckets.items())
err=0.;ident=0.;count=0
for row in support:
    score=read(OUT/'scores'/(row['id']+'.json'))
    for arm in ['N','T']:
        L=row['arms'][arm]['joint_L']
        for geom in ['raw','unit']:
            for key,cell in score['cells'][arm][geom].items():
                if key=='historical' or cell['bridge'] is None:continue
                vc,ac=key.split('__');v=np.load(OUT/'features'/row['id']/arm/(vc+'.npz'))['visual'][:L];a=np.load(OUT/'features'/row['id']/arm/(ac+'.npz'))['audio'][:L]
                if geom=='unit':v=(v.astype(float)/np.linalg.norm(v.astype(float),axis=1)[:,None]).astype(np.float32);a=(a.astype(float)/np.linalg.norm(a.astype(float),axis=1)[:,None]).astype(np.float32)
                v=v[20:L-20].astype(float);a=a[23:L-17].astype(float);vv=v-v.mean(0);aa=a-a.mean(0);mm=(vv+aa)/2
                values={'sigmaA':np.sqrt(np.einsum('ij,ij->',aa,aa)/len(aa)),'sigmaV':np.sqrt(np.einsum('ij,ij->',vv,vv)/len(vv)),'sigmaM':np.sqrt(np.einsum('ij,ij->',mm,mm)/len(mm)),'covAV':np.einsum('ij,ij->',aa,vv)/len(aa)}
                err=max(err,max(abs(values[k]-cell['bridge'][k]) for k in values));ident=max(ident,abs(values['sigmaM']**2-(values['sigmaA']**2+values['sigmaV']**2+2*values['covAV'])/4));count+=1
assert err<1e-10 and ident<1e-10
result={'passed':True,'bridge_cells':count,'max_description_error':float(err),'max_covariance_identity_error':float(ident),'common71_ids':sorted(expected['common71']),'all74_ids':sorted(expected['all74']),'effect_cohorts_exact':True,'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'independent_bridge_support_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
