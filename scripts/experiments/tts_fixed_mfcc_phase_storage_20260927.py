"""Lossless score JSON storage adapter; scientific producer remains byte-identical."""
import argparse,gzip,hashlib,importlib.util,json,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927'
WORKER=ROOT/'scripts/experiments/tts_fixed_mfcc_phase_reversal_20260927.py'
SPEC=importlib.util.spec_from_file_location('sealed_phase_science',WORKER)
s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
ORIGINAL_READ=s.read;ORIGINAL_WRITE=s.write

def compressible(p):
 p=Path(p)
 try:r=p.relative_to(OUT)
 except ValueError:return False
 return r.suffix=='.json' and (r.parts[0] in ('scores','event_scores') or str(r) in ('effects.json','summary.json'))

def read(p):
 p=Path(p)
 if compressible(p) and not p.exists():
  with gzip.open(str(p)+'.gz','rt',encoding='utf-8') as f:return json.load(f)
 return ORIGINAL_READ(p)

def write(p,z):
 p=Path(p)
 if not compressible(p):return ORIGINAL_WRITE(p,z)
 raw=(json.dumps(z,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
 payload=gzip.compress(raw,compresslevel=9,mtime=0)
 assert gzip.decompress(payload)==raw
 target=Path(str(p)+'.gz');assert not p.exists() and not target.exists()
 s.limits(math.ceil(len(payload)/4096)*4096)
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(payload)
 restored=gzip.decompress(target.read_bytes());assert restored==raw and json.loads(restored)==z
 s.limits()

def verify():
 seal=ORIGINAL_READ(OUT/'storage_seal.json')
 for p,h in seal['dependencies'].items():assert s.sha(p)==h,p
 receipt=ORIGINAL_READ(OUT/'storage_reviewer_pass.json')
 assert receipt['status']=='PASS' and receipt['storage_seal_sha256']==s.sha(OUT/'storage_seal.json')
 assert s.sha(receipt['receipt'])==receipt['receipt_sha256']
 assert not (OUT/'scores').exists() or all(p.suffix=='.gz' for p in (OUT/'scores').iterdir())
 return seal

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['score','analyze']);args=a.parse_args()
 verify();s.read=read;s.write=write;getattr(s,args.stage)()
