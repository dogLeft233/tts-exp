"""Independent scalar FFT, explicit distance/curve, speaker bootstrap and retention audit."""
from pathlib import Path
import argparse,hashlib,json,types,wave
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_natural_holdout80_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def validate_protocol():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in {**p['dependencies'],**p['new_code_hashes']}.items():assert sha(f)==h
 return p

def acoustic():
 import soundfile as sf
 p=validate_protocol();path=ROOT/'runs/tts_native_level_match_calibration_20260927/code_snapshot/check_tts_native_level_match_calibration.py';m=types.ModuleType('independent80');m.__file__=str(ROOT/'scripts/experiments/check_tts_native_level_match_calibration.py');exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
 maximum={}
 def check(k,v,limit):maximum[k]=max(maximum.get(k,0.),float(v));assert v<=limit,(k,v,limit)
 rows=read(OUT/'support.json');assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 for row in rows:
  src=row['source'];assert sha(src['path'])==src['sha256']
  with wave.open(src['path'],'rb') as w:
   assert w.getnchannels()==1 and w.getsampwidth()==2 and w.getframerate()==16000;pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.int64)
  x=pcm.astype(float)/32768;rms=np.sqrt(float(np.sum(pcm*pcm))/len(pcm))/32768;assert rms>1e-8;gain=p['target_RMS']/rms;y=x*gain;info=row['FIXED'];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256'];z,sr=sf.read(info['waveform'],dtype='float64');assert sr==16000 and len(z)==len(x) and np.isfinite(z).all() and abs(z).max()<=.98 and np.array_equal(z,y.astype(np.float32).astype(float)) and np.all(z[x==0]==0)
  check('gain',abs(gain-row['gain']),1e-12);check('target_relative',abs(m.energy(z)/p['target_RMS']-1),1e-5);check('recover',abs(z/gain-x).max(),1e-7)
  a,b,c=m.features(x),m.features(y),m.features(z);check('float64_shape',abs(b[0]-a[0]).max(),1e-8);check('float64_R',abs(b[1]-a[1]),1e-8);check('saved_shape',abs(c[0]-a[0]).max(),1e-4);check('saved_R',abs(c[1]-a[1]),1e-5)
  for label,v in [('float64',y),('saved',z)]:check(label+'_cosine',.5*np.sum((v/np.linalg.norm(v)-x/np.linalg.norm(x))**2),1e-10)
  assert row['L']>40;print('scalar checked',row['id'],flush=True)
 write(OUT/'independent_acoustic.json',{'status':'PASS','rows':80,'max_errors':maximum,'method':'independent int16 squaredsum and explicit FFT definitions, no primary transform','checker_sha256':sha(__file__)})

def matrix(v,a):
 v=np.asarray(v,dtype=np.float32);a=np.asarray(a,dtype=np.float32);n=len(v);out=np.empty((n,31));ix=np.arange(n)
 for j,lag in enumerate(range(-15,16)):
  partner=np.zeros_like(a);valid=(ix+lag>=0)&(ix+lag<n);partner[valid]=a[ix[valid]+lag];delta=(v-partner+np.float32(1e-6)).astype(float);out[:,j]=np.sqrt(np.einsum('ij,ij->i',delta,delta))
 return out

def curve(m,pol):
 n=len(m)
 if pol=='valid':return np.array([m[max(0,-k):min(n,n-k),k+15].mean(0) for k in range(-15,16)])
 g=int(pol[5:]);return m[g:n-g].mean(0) if g else m.mean(0)

def stats(values,groups):
 names=sorted(set(groups));means=np.array([sum(v for v,g in zip(values,groups) if g==s)/sum(g==s for g in groups) for s in names]);rng=np.random.Generator(np.random.PCG64(20260926));draws=rng.integers(0,len(names),(20000,len(names)));draw=means[draws].sum(1)/len(names);return float(sum(means)/len(names)),np.percentile(draw,[.5,99.5])

def scores():
 p=validate_protocol();rows=read(OUT/'support.json');maxdist=0.;maxmetric=0.;maxstat=0.;closure=0.;count=0;curves=0;bridgecount=0;bridgeerr=0.
 for r in rows:
  sid=r['id'];L=r['L'];d=read(OUT/'scores'/(sid+'.json'));mp=OUT/'scores'/(sid+'.npz');assert sha(mp)==d['matrices_sha256'];mm=np.load(mp);ff={c:np.load(OUT/'features'/sid/(c+'.npz')) for c in ['RAW','FIXED']}
  for geom,cs in d['cells'].items():
   for q,c in cs.items():
    vc,ac=p['cells'][q];v=ff[vc]['visual'][:L];a=ff[ac]['audio'][:L]
    if geom=='unit':v=(v.astype(float)/np.sqrt(np.sum(v.astype(float)**2,axis=1))[:,None]).astype(np.float32);a=(a.astype(float)/np.sqrt(np.sum(a.astype(float)**2,axis=1))[:,None]).astype(np.float32)
    expected=matrix(v,a);stored=mm[geom+'__'+q];maxdist=max(maxdist,float(abs(expected-stored).max()));count+=expected.size
    for pol,metrics in c['policies'].items():
     ec=curve(expected,pol);sc=curve(stored,pol);assert np.array_equal(sc,metrics['curve']) and int(np.argmin(sc))-15==metrics['best_lag'];B=float(np.median(ec));D=float(ec.min());vals={'C':B-D,'B':B,'D':D,'C_anchor':B-float(ec[18]),'D_anchor':float(ec[18])};maxmetric=max(maxmetric,max(abs(vv-metrics[k]) for k,vv in vals.items()));curves+=1
    b=c['bridge'];bridgeerr=max(bridgeerr,abs(4*b['sigmaM']**2-b['sigmaA']**2-b['sigmaV']**2-2*b['covAV']));bridgecount+=1
  print('distance checked',sid,flush=True)
 summary=read(OUT/'summary.json');effects=read(OUT/'effects.json');buckets={}
 for r in effects:
  expected={'G':r['q10']-r['q00'],'E':r['q01']-r['q00'],'I':r['q11']-r['q10']-r['q01']+r['q00'],'total':r['q11']-r['q00']}
  for k,v in expected.items():assert abs(v-r[k])<1e-12
  closure=max(closure,abs(r['G']+r['E']+r['I']-r['total']));buckets.setdefault((r['geometry']+'/'+r['policy'],r['metric']),[]).append(r)
 for view,block in summary['results'].items():
  for metric,d in block.items():
   r=buckets[view,metric];groups=[x['speaker'] for x in r]
   for k,stored in [('q00',d['baseline']),('q11',d['FIXED_native']),*d['effects'].items()]:
    mean,ci=stats([x[k] for x in r],groups);maxstat=max(maxstat,abs(mean-stored['mean']),float(abs(ci-stored['ci99']).max()))
 assert maxdist<1e-5 and maxmetric<1e-5 and maxstat<1e-10 and closure<1e-12 and bridgeerr<1e-10
 result={'status':'PASS','distance_entries':count,'max_distance_error':maxdist,'max_metric_error':maxmetric,'max_statistics_error':maxstat,'max_fourcell_closure':closure,'curves_and_argmin_exact':curves,'bridge_identities':bridgecount,'max_bridge_identity':bridgeerr,'checker_sha256':sha(__file__)};write(OUT/'independent_scores.json',result);print(result)

def hashes():
 p=validate_protocol();count=len(p['dependencies'])+len(p['new_code_hashes']);streamed=0;retained=0
 for f,h in read(OUT/'feature_seal.json').items():assert sha(f)==h;count+=1
 for r in read(OUT/'support.json'):
  assert sha(r['source']['path'])==r['source']['sha256'];info=r['FIXED'];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256'];count+=3
  for c in ['RAW','FIXED']:
   mp=OUT/'features'/r['id']/(c+'.json');m=read(mp);assert sha(m['features'])==m['sha256'];count+=1
   if m['retained_video']:assert sha(m['video']['path'])==m['video']['sha256'];count+=1;retained+=1
   else:
    assert c=='FIXED' and not Path(m['video']['path']).exists();rec=read(OUT/'retention_commits'/'FIXED'/(r['id']+'.json'));assert rec['metadata_sha256']==sha(mp) and rec['features_sha256']==m['sha256'] and rec['video_sha256_at_creation']==m['video']['sha256'];streamed+=1
 for r in read(OUT/'reuse_manifest.json'):assert sha(r['source_metadata'])==r['source_metadata_sha256'];count+=1
 for sid in p['controls']['ids']:
  for name in ['RAW_repeat','FIXED_stream']:
   mp=OUT/'controls'/sid/(name+'.json');m=read(mp);assert m['passed'] and all(m['tests'].values()) and sha(m['features'])==m['sha256'];assert not Path(m['video']['path']).exists();rec=read(OUT/'retention_commits'/'controls'/sid/(name+'.json'));assert rec['metadata_sha256']==sha(mp) and rec['features_sha256']==m['sha256']
   original=np.load(OUT/'features'/sid/('RAW.npz' if name=='RAW_repeat' else 'FIXED.npz'));repeated=np.load(m['features']);assert all(np.array_equal(original[k],repeated[k]) for k in ['audio','visual'])
  delay=read(OUT/'controls'/sid/'delay5.json');assert delay['passed'] and sha(OUT/'controls'/sid/'delay5.npz')==delay['array_sha256']
 assert streamed==78 and retained==82 and not list((OUT/'temporary').rglob('*.avi'))
 result={'status':'PASS','existing_hashes_verified':count,'retained_old_RAW_videos':80,'retained_new_FIXED_videos':2,'streamed_new_FIXED_creation_records':78,'streamed_final_video_file_hash_verified':False,'controls_reopened_exact':True};write(OUT/'independent_hashes.json',result);print(result)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['acoustic','scores','hashes']);args=ap.parse_args();globals()[args.stage]()
