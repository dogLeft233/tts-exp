"""Independent cloud range reads into task-owned SHM, then full byte-order SHA."""
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);args=parser.parse_args()
    manifest=json.loads(args.input.read_text());bundle=manifest['parent_bundle'];parts=manifest['parts']
    source=Path(manifest['cloud_file']);root=Path('/dev/shm/synctalk-may-cloud-readback-attempt04')
    assert not root.exists() and source.stat().st_size==bundle['bytes']
    assert shutil.disk_usage('/dev/shm').free>bundle['bytes']+512*1024**2
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>bundle['bytes']+1024**3
    root.mkdir();before=shutil.disk_usage('/').free;start=time.monotonic();offset=0
    jobs=[]
    for index,part in enumerate(parts):jobs.append((index,offset,part));offset+=part['bytes']
    assert offset==bundle['bytes']
    def read(job):
        index,offset,row=job;path=root/f'range_{index:03d}.bin';temp=path.with_suffix('.partial')
        h=hashlib.sha256();remaining=row['bytes']
        with source.open('rb') as f,temp.open('xb') as out:
            f.seek(offset)
            while remaining:
                assert shutil.disk_usage('/').free>128*1024**2
                block=f.read(min(4*1024**2,remaining));assert block,'unexpected EOF'
                out.write(block);h.update(block);remaining-=len(block)
        assert h.hexdigest()==row['sha256'],index
        temp.replace(path);return index
    with ThreadPoolExecutor(max_workers=9) as pool:
        pending={pool.submit(read,job) for job in jobs}
        verified=[]
        while pending:
            done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:verified.append(future.result())
            total=sum(p.stat().st_size for p in root.glob('range_*'))
            print(json.dumps({'staged_bytes':total,'expected_bytes':bundle['bytes'],'verified_ranges':sorted(verified)}),flush=True)
    whole=hashlib.sha256();total=0
    for index,_,row in jobs:
        path=root/f'range_{index:03d}.bin'
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''):whole.update(block);total+=len(block)
    assert total==bundle['bytes'] and whole.hexdigest()==bundle['sha256']
    receipt={'passed':True,'path':str(source),'bytes':total,'sha256':whole.hexdigest(),
        'independent_linux_readback':True,'verification_attempt':4,'parallel_ranges_verified':len(parts),
        'method':'same cloud archive read in disjoint ranges, verify each range and concatenate byte-order SHA256',
        'root_free_before':before,'root_free_after':shutil.disk_usage('/').free,
        'elapsed_seconds':time.monotonic()-start,'staging_is_volatile':True,
        'transport_note':'Attempt01 visibility timeout; attempt02 parts staged but not counted as verified cloud backup; attempt03 slow sequential read replaced by attempt04. Original archive bytes and primary experiment unchanged.'}
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
