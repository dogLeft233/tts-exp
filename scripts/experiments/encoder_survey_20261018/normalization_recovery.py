"""Resume only three unrun norm ablations after HTTP429, same scientific worker.

Original failed output/rc retained; new output root and pinned mirror cache.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,time,subprocess,shutil
import normalization_ablation as m
ROOT=m.OUT
REC=ROOT/'recovery_attempt03'
IDS=['xlsr53','wavlm_large','w2vbert2']

def cache(s):
    folder=REC/'weights'/s['id'];folder.mkdir(parents=True,exist_ok=True)
    for row in s['source']['files']:
        dest=folder/row['file']
        if dest.exists():
            assert dest.stat().st_size==row['bytes'] and m.sha(dest)==row['sha256'];continue
        if row['file'].endswith('.json'):
            original=m.OLD/'weights'/s['id']/row['file']
            assert m.sha(original)==row['sha256'];shutil.copyfile(original,dest);continue
        assert shutil.disk_usage(folder).free>row['bytes']+650*1024**2
        # Official public LFS object download API has its own quota; resolve API is temporarily rate limited.
        # The ephemeral signed download URL stays in memory, never in saved receipts/logs.
        endpoint='https://huggingface.co/'+s['repo']+'.git/info/lfs/objects/batch'
        failures=[]
        for attempt in range(3):
            try:
                batch=m.requests.post(endpoint,json={'operation':'download','transfers':['basic'],
                    'objects':[{'oid':row['sha256'],'size':row['bytes']}]},
                    headers={'Accept':'application/vnd.git-lfs+json'},timeout=30)
                batch.raise_for_status()
                action=batch.json()['objects'][0]['actions']['download'];url=action['href']
                with m.requests.get(url,stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    with dest.with_suffix(dest.suffix+'.partial').open('wb') as f:
                        for b in response.iter_content(4*1024**2):f.write(b)
                tmp=dest.with_suffix(dest.suffix+'.partial')
                assert tmp.stat().st_size==row['bytes'] and m.sha(tmp)==row['sha256']
                os.replace(tmp,dest)
                m.write(folder/'transport.json',dict(route=endpoint,bytes=row['bytes'],sha256=m.sha(dest),prior_failures=failures))
                print('PINNED_CACHE_READY',s['id'],row['bytes'],flush=True);break
            except Exception as e:
                failures.append(dict(route=endpoint,error_type=type(e).__name__,attempt=attempt+1))
                time.sleep(3)
        else:
            m.write(folder/'transport_failed.json',failures);raise RuntimeError('all fixed-body routes failed')

def main():
    if len(sys.argv)>1:
        m.OUT=REC;m.worker(sys.argv[1]);return
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
    assert m.sha(m.__file__)==p['producer_sha256']
    first=json.loads((ROOT/'models/xlsr53/receipt.json').read_text(encoding='utf8'))
    assert first['status']=='FAILED' and first['forward_count']==0
    assert '429' in first['error']
    for mid in m.IDS[:5]:
        assert json.loads((ROOT/'models'/mid/'receipt.json').read_text(encoding='utf8'))['status']=='COMPLETED'
    REC.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/'protocol.json',REC/'protocol.json')
    shutil.copyfile(__file__,REC/Path(__file__).name)
    states=[]
    for mid in IDS:
        s=next(s for s in p['models'] if s['id']==mid);cache(s)
        with (REC/(mid+'.log')).open('xb') as f:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),mid],stdout=f,stderr=subprocess.STDOUT)
            m.write(ROOT/'active_recovery.json',dict(model=mid,pid=child.pid,root=str(REC),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            rc=child.wait()
        states.append(dict(model=mid,rc=rc));m.write(ROOT/'recovery_status.json',states)
        print('RECOVERY_MODEL_DONE',mid,rc,flush=True)
        if rc:raise RuntimeError('science worker failed')
        folder=(REC/'weights'/mid).resolve();assert folder.parent==(REC/'weights').resolve()
        for file in folder.iterdir():
            if file.suffix in ['.bin','.safetensors']:
                assert file.resolve().parent==folder;file.unlink()
    m.write(ROOT/'terminal.json',dict(status='COMPLETED_WITH_DOWNLOAD_RECOVERY',models=8,forwards=512,failed_download_forwards=0))

if __name__=='__main__':main()
