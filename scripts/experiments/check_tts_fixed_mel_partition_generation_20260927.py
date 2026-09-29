"""Independent endpoint/input partition, explicit distance and speaker-bootstrap checks."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_mel_partition_generation_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def validate():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in {**p['dependencies'],**p['new_code_hashes']}.items():assert sha(f)==h
 return p
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def chunk(x):
 result=[];i=0
 while True:
  first=int(i*3.2)
  if first+16>x.shape[1]:result.append(x[:,-16:]);break
  result.append(x[:,first:first+16]);i+=1
 return np.asarray(result,dtype=np.float32)
def inputs():
 p=validate();rows=read(OUT/'support.json');assert len(rows)==75 and sum(r['main'] for r in rows)==71;bindings=read(OUT/'bindings.json')
 for f,h in bindings.items():assert sha(f)==h
 n=0;cells=0
 for r in rows:
  for arm,z in r['arms'].items():
   a=np.load(z['RAW']['frontend'])['mel'];b=np.load(z['FIXED']['frontend'])['mel'];assert a.dtype==b.dtype==np.float32 and a.shape==b.shape
   u=~((a==-4)|(a==4)|(b==-4)|(b==4));stored=np.unpackbits(np.load(z['mask']['path']))[:a.size].reshape(a.shape).astype(bool);assert sha(z['mask']['path'])==z['mask']['sha256'] and np.array_equal(u,stored)
   uu=a.copy();ss=a.copy();uu[u]=b[u];ss[~u]=b[~u];variants={'RAW':a,'U':uu,'S':ss,'FIXED':b}
   assert np.array_equal((uu.astype(float)-a)+(ss.astype(float)-a),b.astype(float)-a)
   for c,x in variants.items():
    assert np.isfinite(x).all() and np.all(abs(x)<=4);cc=chunk(x);assert ah(x)==z['inputs'][c]['mel_sha256'] and ah(cc)==z['inputs'][c]['chunks_sha256'] and len(cc)==z['inputs'][c]['frames'];assert min(len(cc),z['RAW']['samples']//640)-5==z['L']
   assert int(u.sum())==z['mask']['U_count'];n+=1;cells+=a.size
 write(OUT/'independent_inputs.json',{'status':'PASS','arms':n,'cells':cells,'bound_files':len(bindings),'where_partition_range_chunk_exact':True,'checker_sha256':sha(__file__)})
def matrix(v,a):
 v=np.asarray(v,dtype=np.float32);a=np.asarray(a,dtype=np.float32);n=len(v);ix=np.arange(n);out=np.empty((n,31))
 for j,k in enumerate(range(-15,16)):
  b=np.zeros_like(a);valid=(ix+k>=0)&(ix+k<n);b[valid]=a[ix[valid]+k];d=(v-b+np.float32(1e-6)).astype(float);out[:,j]=np.sqrt(np.einsum('ij,ij->i',d,d))
 return out
def curve(m,p):
 if p=='valid':return np.array([m[max(0,-k):min(len(m),len(m)-k),k+15].mean(0) for k in range(-15,16)])
 g=int(p[5:]);return m[g:len(m)-g].mean(0) if g else m.mean(0)
def stats(values,groups):
 names=sorted(set(groups));m=np.array([sum(v for v,g in zip(values,groups) if g==s)/sum(g==s for g in groups) for s in names]);rng=np.random.Generator(np.random.PCG64(20260926));draws=rng.integers(0,len(names),(20000,len(names)));return m.mean(),np.percentile(m[draws].sum(1)/len(names),[.5,99.5])
def scores():
 p=validate();mx=0.;mmx=0.;cnt=0;curves=0
 for r in read(OUT/'support.json'):
  d=read(OUT/'scores'/r['id']/'cells.json')
  for arm,z in r['arms'].items():
   raw=np.load(z['RAW']['feature']);fixed=np.load(z['FIXED']['feature']);a=raw['audio'][:z['L']];v={'RAW':raw['visual'],'FIXED':fixed['visual']}
   for c in ['U','S']:v[c]=np.load(OUT/'features'/r['id']/arm/(c+'.npy'))
   for geom,cs in d['cells'][arm].items():
    aa=(a.astype(float)/np.sqrt((a.astype(float)**2).sum(1))[:,None]).astype(np.float32) if geom=='unit' else a
    for c,meta in cs.items():
     vv=v[c][:z['L']];vv=(vv.astype(float)/np.sqrt((vv.astype(float)**2).sum(1))[:,None]).astype(np.float32) if geom=='unit' else vv;expected=matrix(vv,aa);assert sha(meta['matrix'])==meta['matrix_sha256'];stored=np.load(meta['matrix']);mx=max(mx,float(abs(expected-stored).max()));cnt+=expected.size
     for pol,metrics in meta['policies'].items():
      cc=curve(stored,pol);ee=curve(expected,pol);assert np.array_equal(cc,metrics['curve']) and np.argmin(cc)-15==metrics['best_lag'];B=np.median(ee);D=ee.min();vvv={'C':B-D,'B':B,'D':D,'C_anchor':B-ee[18],'D_anchor':ee[18]};mmx=max(mmx,max(abs(v-metrics[k]) for k,v in vvv.items()));curves+=1
  print('checked',r['id'],flush=True)
 summary=read(OUT/'summary.json');rows=read(OUT/'effects.json');buckets={};closure=0.;staterr=0.;checks=0
 for r in rows:
  q=np.array(r['q']);computed={'U':q[1]-q[0],'S':q[2]-q[0],'interaction':q[3]-q[1]-q[2]+q[0],'FIXED':q[3]-q[0],'U_after_S':q[3]-q[2],'S_after_U':q[3]-q[1],'U_minus_S':q[1]-q[2]}
  for k,v in computed.items():assert abs(v-r[k])<1e-12
  closure=max(closure,abs(r['U']+r['S']+r['interaction']-r['FIXED']));buckets.setdefault((r['view'],r['metric'],r['arm']),[]).append(r)
 for view,fields in summary['results'].items():
  for field,arms in fields.items():
   for arm,block in arms.items():
    rr=buckets[view,field,arm];groups=[r['speaker'] for r in rr]
    for family,entries in block.items():
     for k,s in entries.items():
      vals=[r['q'][['RAW','U','S','FIXED'].index(k)] if family=='cells' else r[k] for r in rr];m,ci=stats(vals,groups);staterr=max(staterr,abs(m-s['mean']),float(abs(ci-s['ci99']).max()));checks+=1
 assert mx<1e-5 and mmx<1e-5 and closure<1e-12 and staterr<1e-10
 write(OUT/'independent_scores.json',{'status':'PASS','distance_entries':cnt,'max_distance_error':mx,'max_metric_error':mmx,'curves_argmin_exact':curves,'statistics_checked':checks,'max_statistics_error':staterr,'max_fourcell_closure':closure,'checker_sha256':sha(__file__)})
def hashes():
 p=validate();n=0
 for f,h in {**read(OUT/'bindings.json'),**read(OUT/'feature_seal.json')}.items():assert sha(f)==h;n+=1
 commits=list((OUT/'retention_commits').rglob('*.json'));assert len(commits)==304 and not list((OUT/'temporary').rglob('*.avi'))
 for cp in commits:
  rel=cp.relative_to(OUT/'retention_commits');mp=OUT/'features'/rel;rec=read(cp);m=read(mp);assert sha(mp)==rec['metadata_sha256'] and sha(m['feature'])==rec['feature_sha256']==m['feature_sha256'];assert rec['video_sha256_at_creation']==m['video']['sha256'] and rec['pixel_sha256_at_creation']==m['video']['pixel_sha256'];assert not Path(m['video']['path']).exists() and m['video']['decoded_pixel_equals_render'];n+=2
 assert read(OUT/'engineering_gate.json')['passed'] and read(OUT/'cal_U_S_gate.json')['passed']
 write(OUT/'independent_hashes.json',{'status':'PASS','existing_hashes':n,'creation_records':304,'new_final_videos':0,'deleted_video_final_file_hash_verified':False,'no_old_deletion':True})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inputs','scores','hashes']);globals()[ap.parse_args().stage]()
