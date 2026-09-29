"""Independent Decimal/Fraction TextGrid masks and complete input-seal audit."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
import hashlib,json,re,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_holdout80_20260927';O=ROOT/'runs/tts_fixed_generator_shift_holdout80_independent_audit_20260927';EXCLUDED={'','sil','sp','spn','<unk>','<eps>','silence'}
j=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def grid(path):
 name=None;active=False;part=None;out=[]
 for line in Path(path).read_text().splitlines():
  s=line.strip()
  if re.fullmatch(r'item \[\d+\]:',s):name=None;active=False
  elif s.startswith('name = '):name=s.split('=',1)[1].strip().strip('"');active=name=='phones'
  elif active and re.fullmatch(r'intervals \[\d+\]:',s):part={}
  elif active and part is not None and s.startswith(('xmin = ','xmax = ','text = ')):
   key,value=s.split('=',1);key=key.strip();value=value.strip();part[key]=value.strip('"') if key=='text' else value
   if key=='text':assert set(part)=={'xmin','xmax','text'};out.append(part);part=None
 assert out;return out

def main():
 p=j(R/'protocol.json');assert sha(R/'protocol.json')==j(R/'seal.json')['protocol_sha256'];assert j(O/'static_receipt.json')['protocol_sha256']==sha(R/'protocol.json');assert not (R/'gpu_runtime/timer_start_identity.json').exists();seal=j(R/'input_seal.json');assert seal['status']=='PASS' and seal['protocol_sha256']==sha(R/'protocol.json')
 for f,h in {**p['dependencies'],**seal['files']}.items():assert sha(f)==h,f
 assets=j(R/'asset_binding.json');labels=set(assets['template_labels']);inventory=set(assets['MFA_inventory']);inp=j(R/'input_rows.json');rows=j(R/'rows.json');assert [r['id'] for r in inp]==[r['id'] for r in rows];reports={};allfiles={}
 for split,src,n in [('cal',j(R/'cal_rows.json'),26),('evaluation',rows,80)]:
  gate=j(R/('mfa_'+split+'_gate.json'));assert gate['status']=='PASS' and gate['source_count']==n and len(gate['rows'])==n;by={r['id']:r for r in gate['rows']};assert set(by)=={r['id'] for r in src};total=matched=0
  for r in src:
   z=by[r['id']];assert z['speaker']==r['speaker'] and sha(z['TextGrid'])==z['TextGrid_sha256'];allfiles[z['TextGrid']]=z['TextGrid_sha256'];a=r if split=='cal' else r['arms']['N'];tokens=grid(z['TextGrid']);last=Fraction(0);duration=Fraction(a['samples'],16000);tt=[]
   for t in tokens:
    lo,hi=Fraction(Decimal(t['xmin'])),Fraction(Decimal(t['xmax']));assert lo==last and hi>=lo and hi<=duration+Fraction(1,16000);usable=hi>lo and t['text'].strip().casefold() not in EXCLUDED
    if usable:assert t['text'] in inventory
    tt.append((lo,hi,t['text'],usable));last=hi
   assert abs(last-duration)<=Fraction(1,16000)
   phones=[]
   for start in a['mel_starts']:
    center=Fraction(2*start+15,160);hits=[t for t in tt if t[0]<=center<t[1]];assert len(hits)<=1;phones.append(hits[0][2] if hits and hits[0][3] else None)
   mask=[x is not None for x in phones];active=sum(mask);found=sum(x in labels for x in phones if x is not None);unknown=sum(t[2].casefold() in {'spn','<unk>'} for t in tt);assert active>=1
   assert phones==z['phone_labels'] and mask==z['speech_mask'] and active==z['active'] and found==z['matched'] and unknown==z['unknown_intervals'] and z['coverage']==found/active
   assert z['tokens']==[{'lo_decimal':t['xmin'],'hi_decimal':t['xmax'],'label':t['text'],'usable':tt[i][3]} for i,t in enumerate(tokens)]
   if split=='evaluation':
    rr=next(x for x in inp if x['id']==r['id']);expected=json.loads(json.dumps(r));expected['arms']['N'].update(phone_labels=phones,speech_mask=mask,TextGrid=z['TextGrid'],TextGrid_sha256=z['TextGrid_sha256']);assert rr==expected
   total+=active;matched+=found
  assert gate['active']==total and gate['matched']==matched and gate['coverage']==matched/total>=.8;reports[split]={'clips':n,'active':total,'matched':matched,'coverage':matched/total}
 # Complete source hashes and original support remain unchanged; no treatment-dependent input.
 assert len(inp)==80 and len({r['speaker'] for r in inp})==40 and j(R/'mfa_cal_gate.json')['created_epoch']<j(R/'mfa_evaluation_gate.json')['created_epoch']<seal['created_epoch']
 out=O/'input_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'input_seal_sha256':sha(R/'input_seal.json'),'audit_code_sha256':sha(__file__),'all106_TextGrids_independently_parsed':True,'TextGrid_hashes':allfiles,'gates':reports,'all80_and_all26_preserved':True,'no_GPU_or_model_forward_or_effect_scores':True},indent=2)+'\n');rv=R/'input_reviewer_pass.json';assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','input_seal_sha256':sha(R/'input_seal.json'),'receipt':str(out),'receipt_sha256':sha(out)},indent=2)+'\n');print(sha(out))
if __name__=='__main__':main()
