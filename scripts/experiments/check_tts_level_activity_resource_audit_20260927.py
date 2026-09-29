"""Independent PCM integer energy, TextGrid masks and group bootstrap audit."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_activity_resource_audit_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in {**p['dependencies'],**p['code']}.items():assert sha(f)==h
 lookup={(r['id'],r['arm'],r['method']):r for r in read(OUT/'metrics.json')};pairrows=read(OUT/'pairs.json');summary=read(OUT/'summary.json');maxima={};checked=0;cal=[];old=read(ROOT/'runs/tts_native_level_generation_cross_20260927/pairs.json');old={r['id']:r for r in old}
 def close(a,b,k):
  e=float(np.max(abs(np.asarray(a)-b)));maxima[k]=max(maxima.get(k,0),e);assert e<1e-10,(k,e)
 expected={}
 for row in p['rows']:
  for arm in ['N','T']:
   source=row['audio'][arm];assert sha(source['path'])==source['sha256'];pcm,sr=sf.read(source['path'],dtype='int16');assert sr==16000
   n=len(pcm);squares=pcm.astype(np.int64)**2;total=int(squares.sum());globalr=np.sqrt(total/n)/32768
   if row['split']=='calibration':cal.append(float(np.log(globalr)))
   masks={};binding=next(b for b in p['mfa_bindings'] if b['id']==row['id'] and b['arm']==arm)
   if binding['available']:
    assert sha(binding['textgrid'])==binding['textgrid_sha256'];text=Path(binding['textgrid']).read_text();phones=text.split('name = "phones"',1)[1]
    intervals=re.findall(r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"([^"]*)"',phones)
    center=(np.arange(n)+.5)/16000;active=np.zeros(n,dtype=bool);unk=np.zeros(n,dtype=bool)
    for start,end,label in intervals:
     span=(center>=float(start))&(center<float(end))
     if label.strip() not in ['','sil','sp','spn','<eps>','h#']:active|=span
     if label.strip()=='spn':unk|=span
    masks['MFA']=active;masks['MFA_unknown_active']=active|unk
   for threshold in [-20,-30,-40]:
    values=[np.sqrt(int(squares[i:i+320].sum())/len(squares[i:i+320])) for i in range(0,n,320)];peak=max(values);mask=np.zeros(n,dtype=bool)
    for block,val in enumerate(values):mask[block*320:min(n,(block+1)*320)]=val>peak*10**(threshold/20) if peak else False
    masks[f'block_{threshold}dB']=mask
   for method,mask in masks.items():
    r=lookup[row['id'],arm,method];assert sha(r['mask_path'])==r['mask_sha256'];saved=np.load(r['mask_path'])[method];assert np.array_equal(mask,saved),(row['id'],arm,method)
    na=int(mask.sum());ni=n-na;ea=int(squares[mask].sum());ei=total-ea;ar=np.sqrt(ea/na)/32768 if na else None;proportion=na/n
    close(globalr,r['global_RMS'],'global_RMS');close(proportion,r['p'],'occupancy')
    if ar is not None:close(ar,r['active_RMS'],'active_RMS')
    if ea and na:
     logs=[np.log(ar),np.log(proportion)/2,np.log((ea+ei)/ea)/2];db=np.array(logs)*20/np.log(10)
     for k,v in zip(['activity_level','occupancy','inactive_correction'],db):close(v,r['dB_terms'][k],'dB_terms')
     close(sum(logs),np.log(globalr),'log_closure')
     expected[row['id'],arm,method]={'db':dict(zip(['activity_level','occupancy','inactive_correction'],db)),'global':20*np.log10(globalr),'active_RMS':ar}
    checked+=1
 target=float(np.exp(np.median(cal)));close(target,read(OUT/'fixed_target.json')['target_RMS'],'target')
 fixed=read(OUT/'fixed_target_feasibility.json')
 for r in fixed:
  gain=target/r['global_RMS'];close(gain,r['fixed_gain'],'fixed_gain');close(gain*r['original_peak'],r['hypothetical_peak'],'peak')
 for r in pairrows:
  if not r['defined']:continue
  en,et=[expected[r['id'],a,r['method']] for a in ['N','T']]
  for key in ['activity_level','occupancy','inactive_correction']:close(et['db'][key]-en['db'][key],r['delta_dB'][key],'pair_terms')
  close(et['global']-en['global'],r['delta_dB']['global'],'pair_global')
  geometric=np.exp((np.log(en['active_RMS'])+np.log(et['active_RMS']))/2)
  for a,e in [('N',en),('T',et)]:close(20*np.log10(geometric/e['active_RMS']/old[r['id']]['gain'][a]),r['active_gain'][a]['difference_from_LEVEL_dB'],'active_gain_difference')
 def check_stats(vals,groups,saved):
  names=sorted(set(groups));means=np.array([sum(v for v,g in zip(vals,groups) if g==name)/sum(g==name for g in groups) for name in names]);ix=np.random.Generator(np.random.PCG64(20260926)).integers(0,len(names),(20000,len(names)))
  multiplicity=np.stack([(ix==j).sum(1) for j in range(len(names))],axis=1)/len(names);draw=multiplicity@means
  close(float(sum(means)/len(means)),saved['mean'],'statistics');close(np.percentile(draw,[.5,99.5]),saved['ci99'],'statistics')
 for split,methods in summary['results'].items():
  for method,detail in methods.items():
   pp=[r for r in pairrows if r['split']==split and r['method']==method];groups=[r['speaker'] for r in pp]
   for field,result in detail['delta_dB'].items():check_stats([r['delta_dB'][field] for r in pp],groups,result)
   for a,result in detail['gain_difference_from_LEVEL_dB'].items():check_stats([r['active_gain'][a]['difference_from_LEVEL_dB'] for r in pp],groups,result)
   mm=[r for r in lookup.values() if r['split']==split and r['method']==method]
   for a,fields in detail['arms'].items():
    rr=[r for r in mm if r['arm']==a]
    for k,result in fields.items():check_stats([r[k] for r in rr],[r['speaker'] for r in rr],result)
 for split,detail in summary['fixed_target'].items():
  for a,result in detail['gain_diff'].items():
   rr=[r for r in fixed if r['split']==split and r['arm']==a];check_stats([r['gain_dB_difference'] for r in rr],[r['speaker'] for r in rr],result)
 result={'status':'PASS','arm_method_records':checked,'max_errors':maxima,'method':'No primary import; original PCM integer square sums, sample-center TextGrid parsing, explicit blocks, independent bootstrap multiplicities; inputs/masks/source hashes verified'}
 (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
