"""Independent Linux-mounted Baidu readback, with root/SHM guards and ordered full SHA."""
import argparse,hashlib,json,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def staging_path(target):return target.with_name(target.name+'.partial')

def run(manifest_path,readback_attempt=None):
    manifest=json.loads(manifest_path.read_text());source=Path(manifest['linux_cloud_directory'])
    assert source.is_relative_to(Path('/home/wjj/百度网盘/tts-exp-research-models/synctalk-multispeaker-full-20261017'))
    readback_attempt=manifest['attempt'] if readback_attempt is None else readback_attempt
    assert readback_attempt>0
    directory=ROOT/'cloud_readback'/manifest['name']/f"attempt{readback_attempt:02d}"
    directory.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+3600
    while not all((source/row['file']).exists() and (source/row['file']).stat().st_size==row['bytes'] for row in manifest['parts']):
        assert time.monotonic()<deadline,'cloud upload not visible in independent mount; preserve originals';time.sleep(20)
    assert shutil.disk_usage(ROOT).free>manifest['bytes']+512*1024**2,'reserve full independent readback staging first'
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>manifest['bytes']+1024**3
    before=shutil.disk_usage('/').free;start=time.monotonic();verified=[]
    def read(row):
        target=directory/row['file'];partial=staging_path(target);h=hashlib.sha256();count=0
        with (source/row['file']).open('rb') as src,partial.open('xb') as dst:
            while True:
                assert shutil.disk_usage('/').free>128*1024**2,'root free-space guard'
                assert shutil.disk_usage(ROOT).free>256*1024**2,'SHM free-space guard'
                block=src.read(4*1024**2)
                if not block:break
                dst.write(block);h.update(block);count+=len(block)
        assert count==row['bytes'] and h.hexdigest()==row['sha256'];partial.rename(target);return row['file']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(read,row) for row in manifest['parts']]):
            verified.append(future.result());print(json.dumps(dict(verified_parts=len(verified),total_parts=len(manifest['parts']))),flush=True)
    h=hashlib.sha256();count=0
    for row in manifest['parts']:
        with (directory/row['file']).open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);count+=len(block)
    assert count==manifest['bytes'] and h.hexdigest()==manifest['sha256']
    result=dict(passed=True,name=manifest['name'],attempt=manifest['attempt'],readback_attempt=readback_attempt,bytes=count,sha256=h.hexdigest(),
        independent_linux_cloud_readback=True,parts_verified=len(verified),cloud_directory=str(source),
        root_free_before=before,root_free_after=shutil.disk_usage('/').free,elapsed_seconds=time.monotonic()-start,
        manifest_sha256=sha(manifest_path),source_sha256=sha(Path(__file__)))
    (directory/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def retire(manifest_path,proof_path):
    manifest=json.loads(manifest_path.read_text());proof=json.loads(proof_path.read_text())
    readback_attempt=proof.get('readback_attempt',manifest['attempt'])
    directory=ROOT/'cloud_readback'/manifest['name']/f"attempt{readback_attempt:02d}"
    assert directory.resolve().is_relative_to((ROOT/'cloud_readback').resolve())
    receipt=json.loads((directory/'receipt.json').read_text());assert receipt['passed']
    assert sha(directory/'receipt.json')==proof['F_verified_receipt_sha256'] and proof['passed']
    for row in manifest['parts']:
        file=directory/row['file'];assert file.resolve().is_relative_to(directory.resolve()) and sha(file)==row['sha256']
    for row in manifest['parts']:(directory/row['file']).unlink()
    (directory/'staging_retirement.json').write_text(json.dumps(dict(passed=True,retired_bytes=manifest['bytes'],cloud_and_F_source_unchanged=True),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--retire-proof',type=Path);p.add_argument('--readback-attempt',type=int);a=p.parse_args()
    if a.retire_proof:retire(a.manifest,a.retire_proof)
    else:run(a.manifest,a.readback_attempt)
