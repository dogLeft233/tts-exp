"""Resume immutable streaming workers only with sufficient host commit headroom."""
from pathlib import Path
import argparse,json,subprocess,sys,ctypes,time,hashlib,datetime

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
def memory():
    s=MemoryStatus();s.length=ctypes.sizeof(s);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s));return dict(available_commit=s.avail_page,available_physical=s.avail_phys)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def main(out):
    root=out.resolve();worker=Path(__file__).with_name('recover_streaming.py');assert sha(worker)==read(out/'recovery_protocol.json')['recovery_source_sha256']
    entry=out/'resume_driver_protocol.json'
    if not entry.exists():
        write(entry,dict(status='FROZEN_BEFORE_RESUMPTION',source_sha256=sha(__file__),recovery_protocol_sha256=sha(out/'recovery_protocol.json'),reason='initial streaming job0 passed; subsequent job1 failed before first bridge output on1MiB allocation during host commit pressure; no scientific conditions changed',minimum_available_commit_fp32=3<<30,minimum_available_commit_fp64=4<<30,training_updates=0))
    assert read(entry)['source_sha256']==sha(__file__)
    for job in read(out/'jobs.json'):
        index=job['index'];receipt=out/'worker_receipts'/f'{index:02d}.json'
        if receipt.exists():assert read(receipt)['passed'];continue
        partial_paths=[out/job['mode']/job['split']/f"{job['kind']}_{job['seed']}",out/'recovery_controls'/f'job{index:02d}']
        if job['scientific_anchors']:partial_paths+=[out/job['mode']/job['split']/('anchor'+arm) for arm in ['A','B']]
        present=[p for p in partial_paths if p.exists()]
        if present:
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');bundle=out/'recovery_failures'/f'job{index:02d}_{stamp}';bundle.mkdir(parents=True,exist_ok=False);snapshot=[]
            for path in present:
                assert path.resolve().is_relative_to(root) and path.resolve()!=root
                dest=bundle/path.relative_to(out);assert dest.resolve().is_relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
                for file in path.rglob('*'):
                    if file.is_file():snapshot.append(dict(path=file.relative_to(out).as_posix(),bytes=file.stat().st_size,sha256=sha(file)))
                path.rename(dest)
            write(bundle/'preserved_partial_manifest.json',dict(excluded=True,job=index,members=snapshot))
        minimum=(3 if job['mode']=='fp32' else 4)<<30;last_notice=0
        while memory()['available_commit']<minimum:
            if time.monotonic()-last_notice>45:print(json.dumps(dict(waiting_for_host_memory=True,job=index,minimum=minimum,**memory())),flush=True);last_notice=time.monotonic()
            time.sleep(5)
        log=out/'execution_logs'/f'job{index:02d}.log';log.parent.mkdir(exist_ok=True)
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,str(worker),'worker','--output',str(out),'--index',str(index)],stdout=handle,stderr=subprocess.STDOUT)
        write(out/'execution_logs'/f'job{index:02d}_exit.json',dict(job=index,exit_code=result.returncode,seconds=time.monotonic()-start,post_memory=memory()))
        assert result.returncode==0,f'job{index} failed; exact log preserved at {log}'
        print(json.dumps(dict(job=index,passed=True,completed_jobs=len(list((out/'worker_receipts').glob('*.json'))),**memory())),flush=True)
    subprocess.run([sys.executable,str(worker),'finalize','--output',str(out)],check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
