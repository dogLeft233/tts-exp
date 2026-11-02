"""Stream exact large intermediate caches into verified F gzip files, then free SHM."""
import argparse,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017');REMOTE='/dev/shm/synctalk-multispeaker-data-20261017';HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST]
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
p=argparse.ArgumentParser()
p.add_argument('--identities',nargs='+',choices=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers'],default=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers'])
p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test'])
p.add_argument('--worker-receipt',default='stage2_worker_receipt.json')
options=p.parse_args();assert Path(options.worker_receipt).name==options.worker_receipt
old=ROOT/'flow_cache_persistence_Cathy_independent_receipt.json'
prior=json.loads(old.read_text()) if old.exists() else {}
retained=[x for x in prior.get('verified',[]) if x['identity'] not in options.identities or x['split'] not in options.splits]
state=dict(state='running',verified=retained,passed=False,pid=os.getpid())
def save():(ROOT/'flow_cache_persistence_Cathy_independent_receipt.json').write_text(json.dumps(state,indent=2))
def read_remote(path):
    p=subprocess.run(SSH+['cat '+path],capture_output=True,encoding='utf-8',errors='replace',timeout=60)
    if p.returncode:return None
    try:return json.loads(p.stdout)
    except json.JSONDecodeError:return None
save()
try:
    for identity in options.identities:
        for split in options.splits:
            state.update(identity=identity,split=split);save();remote=f'{REMOTE}/prepared/{identity}/{split}';deadline=time.time()+14400
            while True:
                receipt=read_remote(remote+'/flow_receipt.json')
                if receipt:break
                worker=read_remote(REMOTE+'/'+options.worker_receipt)
                assert not worker or worker.get('passed') is not False,worker
                assert time.time()<deadline,'flow timeout';time.sleep(20)
            assert receipt['passed']
            folder=ROOT/'flow_caches'/identity/split;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'rigid_flow.npy.gz';assert not target.exists();partial=target.with_suffix('.partial')
            with partial.open('xb') as output:
                result=subprocess.run(SSH+['gzip -c -- '+remote+'/rigid_flow.npy'],stdout=output,stderr=subprocess.PIPE,timeout=1800)
                assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
                output.flush();os.fsync(output.fileno())
            raw_hash=hashlib.sha256();raw_bytes=0
            with gzip.open(partial,'rb') as data:
                for b in iter(lambda:data.read(8*1024**2),b''):raw_hash.update(b);raw_bytes+=len(b)
            assert raw_hash.hexdigest()==receipt['rigid_flow_sha256']
            partial.rename(target);packed_hash=hashlib.sha256()
            with target.open('rb') as data:
                for b in iter(lambda:data.read(8*1024**2),b''):packed_hash.update(b)
            verified=dict(passed=True,identity=identity,split=split,path=str(target),bytes=target.stat().st_size,
                gzip_sha256=packed_hash.hexdigest(),raw_bytes=raw_bytes,raw_sha256=raw_hash.hexdigest(),decompressed_source_verified=True)
            metadata=folder/'persistence.json';metadata.write_text(json.dumps(verified,indent=2))
            subprocess.run(SCP+[str(metadata),HOST+':'+remote+'/rigid_flow_persistence.json'],check=True,timeout=120)
            result=subprocess.run(SSH+[f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/reclaim_flow_cache_remote.py --identity {identity} --split {split}'],capture_output=True,encoding='utf-8',errors='replace',timeout=180)
            assert result.returncode==0,result.stderr
            verified['reclamation']=json.loads(result.stdout);assert verified['reclamation']['passed']
            metadata.write_text(json.dumps(verified,indent=2));state['verified'].append(verified);save();print(json.dumps(verified),flush=True)
    state['state']='completed';state['passed']=True;save()
except Exception as error:
    state['state']='failed';state['error']=repr(error);save();raise
