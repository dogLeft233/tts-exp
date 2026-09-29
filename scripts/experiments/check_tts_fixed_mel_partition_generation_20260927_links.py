"""Independent effect-row/arm/source linkage and exact historical endpoint replay."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_mel_partition_generation_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 seal=read(OUT/'postprocessing_seal.json');assert sha(__file__)==seal['code'][str(Path(__file__).resolve())]
 for f,h in seal['old_score_hashes'].items():assert sha(f)==h
 protocol=read(OUT/'protocol.json');data={r['id']:read(OUT/'scores'/r['id']/'cells.json') for r in read(OUT/'support.json')};count=0;maxerr=0.;qcount=0
 for sid,new in data.items():
  old=read(OLD/'scores'/(sid+'.json'))
  for arm,geoms in new['cells'].items():
   for geom,cs in geoms.items():
    for c,oc in [('RAW','raw__raw'),('FIXED','FIXED__identity')]:
     for pol,m in cs[c]['policies'].items():
      om=old['cells'][arm][geom][oc]['policies'][pol]
      assert np.array_equal(m['curve'],om['curve'])
      for k in ['C','B','D','C_anchor','D_anchor','best_lag']:maxerr=max(maxerr,abs(m[k]-om[k]))
      count+=1
 for r in read(OUT/'effects.json'):
  geom,pol,pop=r['view'].split('/');sid=r['id'];assert sid in protocol['main_ids' if pop=='common71' else 'all_ids'];d=data[sid];assert r['speaker']==d['speaker'];qq={a:np.array([d['cells'][a][geom][c]['policies'][pol][r['metric']] for c in ['RAW','U','S','FIXED']],dtype=float) for a in ['N','T']};qq['T_minus_N']=qq['T']-qq['N'];assert np.array_equal(qq[r['arm']],r['q']);qcount+=1
 assert maxerr==0
 result={'status':'PASS','old_RAW_FIXED_curves_exact':count,'old_endpoint_metric_max_error':maxerr,'effect_row_q_arm_support_links_exact':qcount,'checker_sha256':sha(__file__)};(OUT/'independent_links.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
