"""Pre-execution protocol, dependency, pooled-template and synthetic clock audit."""
from pathlib import Path
import hashlib,json,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=ROOT/'runs/tts_fixed_generator_shift_holdout80_20260927';O=ROOT/'runs/tts_fixed_generator_shift_holdout80_independent_audit_20260927'
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as core
from scripts.experiments import tts_fixed_generator_shift_holdout80_scores_20260927 as scoring
j=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=True);p=j(R/'protocol.json');assert sha(R/'protocol.json')==j(R/'seal.json')['protocol_sha256'];assert not (R/'mfa_cal_gate.json').exists() and not (R/'score_lock.json').exists()
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 assert core.runtime()==p['gpu_config']['runtime'];rows=j(R/'rows.json');cal=j(R/'cal_rows.json');assert len(rows)==80 and len(cal)==26;names=[r['speaker'] for r in rows];assert len(set(names))==40 and all(names.count(s)==2 for s in names)
 old=j(core.OLD/'support.json');assert [r['id'] for r in rows]==[r['id'] for r in old];fitrows=j(core.FIT/'rows.json');assert set(names).isdisjoint({r['speaker'] for r in fitrows});assert j(core.SHIFT/'final.json')['status']=='concluded' and j(core.SHIFT/'resource_closure.json')['passed']
 fm=j(core.FIT/'fit.json');f=np.load(core.FIT/'fit.npz');indices=[fm['folds'].index(x) for x in ['S0912','S0913','S0915']];assert len(fm['training_ids']['S0912'])==20
 for ix,label in zip(indices,['S0912','S0913','S0915']):assert np.array_equal(f['raw_mu'][ix],f['raw_mu'][indices[0]]) and np.array_equal(f['speaker_counts'][ix],f['speaker_counts'][indices[0]]) and fm['training_ids'][label]==fm['training_ids']['S0912']
 asset=j(R/'asset_binding.json');assert set(fm['labels'])<=set(asset['MFA_inventory']) and len(fm['labels'])==82
 vb=zb=0
 for r,oldrow in zip(rows,old):
  a=r['arms']['N'];assert r['L']==oldrow['L']==min(a['frames'],a['samples']//640)-5 and r['L']>40
  with np.load(a['frontend_path']) as ff:
   M=ff['mel'].shape[1];starts=[];i=0
   while int(i*80/25)+16<=M:starts.append(int(i*80/25));i+=1
   starts.append(M-16);assert starts==a['mel_starts'] and len(starts)==a['frames']
  vb+=4*math.ceil(((a['frames']-4)*1024*4+128)/4096)*4096;zb+=math.ceil((a['frames']*512*4+128)/4096)*4096
 assert vb<=p['resources']['V_allocated_bound'] and zb<=p['resources']['z_allocated_bound'];assert 4*math.ceil(((rows[0]['arms']['N']['frames']-4)*1024*4+128)/4096)*4096<=p['resources']['calV_allocated_bound']
 # Synthetic Decimal half-open boundary: final padding center outside duration stays inactive.
 fixture=O/'synthetic.TextGrid';fixture.write_text('name = "phones"\nintervals [1]:\nxmin = 0\nxmax = 0.1\ntext = "a˥"\nintervals [2]:\nxmin = 0.1\nxmax = 0.2\ntext = "sil"\nintervals [3]:\nxmin = 0.2\nxmax = 0.3\ntext = "pʲ"\n')
 z=core.parse_grid(fixture,4800,[0,1,8,9,16,17],['a˥','pʲ'],['a˥']);assert z['phone_labels']==['a˥',None,None,'pʲ','pʲ',None] and z['active']==3 and z['matched']==1
 # Source sign arithmetic and projection include inactive frame identity.
 from scripts.experiments.tts_fixed_generator_shift_ops_20260927 import shift_table
 x=np.array([[.0,.1],[.2,.0]],np.float32);a=np.array([.4,.0],np.float32);b=np.array([.0,.6],np.float32)
 # Actual op expects512; tile preserves the counterexample and shape contract.
 x=np.tile(x,(1,256));a=np.tile(a,256);b=np.tile(b,256)
 for own,other in [(a,b),(b,a)]:
  out,_=shift_table(x,['a','a'],[True,False],{},own,{},other,'global');expected=x.copy();expected[0]=np.maximum(np.add(x[0],np.multiply(np.subtract(other,own,dtype=np.float32),np.float32(.5),dtype=np.float32),dtype=np.float32),np.float32(0));assert np.array_equal(out,expected)
 terms=scoring.contrasts(np.zeros((80,5)));assert len(terms)==14 and all(np.array_equal(v,np.zeros(80)) for v in terms.values());assert p['primary']=='global_plus-baseline and global_plus-global_minus each C,C_anchor rawguard20 allfour99CI lower>0'
 assert p['resources']['main']==176*2**20 and p['resources']['audit']==16*2**20 and p['resources']['floor']==4*2**30
 out=O/'static_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'code_sha256':sha(__file__),'dependencies':len(p['dependencies']),'all80_40speaker_26cal_preserved':True,'pooled20cal_three_identical_folds':True,'exact82_inventory':True,'V_bound':vb,'z_bound':zb,'synthetic_clock_projection_contrasts':'PASS','runtime':core.runtime(),'resources':core.limits(),'scope':'static and synthetic only; authorizes cal MFA then80 input plus oldbaseline CPU bridge; GPU awaits independent input seal and subsequent live gates','no_GPU_model_forward_or_new_effects':True},indent=2)+'\n');rv=R/'reviewer_pass.json';assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','protocol_sha256':sha(R/'protocol.json'),'receipt':str(out),'receipt_sha256':sha(out)},indent=2)+'\n');print(sha(out))
if __name__=='__main__':main()
