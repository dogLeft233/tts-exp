"""Independent ordered cloud-byte verification with bounded memory and no payload staging."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
CLOUD_PREFIX=Path('/home/wjj/百度网盘/tts-exp-research-models/synctalk-multispeaker-full-20261017')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(manifest_path,readback_attempt=None):
    manifest=json.loads(manifest_path.read_text());source=Path(manifest['linux_cloud_directory']);assert source.is_relative_to(CLOUD_PREFIX)
    attempt=manifest['attempt'] if readback_attempt is None else readback_attempt;assert attempt>0
    directory=ROOT/'cloud_readback'/manifest['name']/f'attempt{attempt:02d}';directory.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+3600
    while not all((source/r['file']).is_file() and (source/r['file']).stat().st_size==r['bytes'] for r in manifest['parts']):
        assert time.monotonic()<deadline,'cloud parts not visible; originals retained';time.sleep(20)
    before=shutil.disk_usage(ROOT).free;start=time.monotonic();whole=hashlib.sha256();total=0
    for index,row in enumerate(manifest['parts']):
        assert Path(row['file']).name==row['file']
        h=hashlib.sha256();count=0
        with (source/row['file']).open('rb') as f:
            while True:
                assert shutil.disk_usage('/').free>128*1024**2,'root free-space guard'
                assert shutil.disk_usage(ROOT).free>128*1024**2,'metadata workspace guard; payload never staged'
                block=f.read(4*1024**2)
                if not block:break
                h.update(block);whole.update(block);count+=len(block);total+=len(block)
        assert count==row['bytes'] and h.hexdigest()==row['sha256']
        print(json.dumps(dict(verified_parts=index+1,total_parts=len(manifest['parts']))),flush=True)
    assert total==manifest['bytes'] and whole.hexdigest()==manifest['sha256']
    result=dict(passed=True,name=manifest['name'],attempt=manifest['attempt'],readback_attempt=attempt,bytes=total,
        sha256=whole.hexdigest(),independent_linux_cloud_readback=True,parts_verified=len(manifest['parts']),cloud_directory=str(source),
        verification_mode='ordered_streaming_zero_payload_staging',payload_staged_bytes=0,maximum_read_block_bytes=4*1024**2,
        workspace_free_before=before,workspace_free_after=shutil.disk_usage(ROOT).free,elapsed_seconds=time.monotonic()-start,
        manifest_sha256=sha(manifest_path),source_sha256=sha(Path(__file__)))
    (directory/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result
def retire(manifest_path,proof_path):
    manifest=json.loads(manifest_path.read_text());proof=json.loads(proof_path.read_text())
    attempt=proof.get('readback_attempt',manifest['attempt']);directory=ROOT/'cloud_readback'/manifest['name']/f'attempt{attempt:02d}'
    assert directory.resolve().is_relative_to((ROOT/'cloud_readback').resolve())
    receipt=json.loads((directory/'receipt.json').read_text());assert receipt['passed'] and receipt['verification_mode']=='ordered_streaming_zero_payload_staging'
    assert proof['passed'] and sha(directory/'receipt.json')==proof['F_verified_receipt_sha256'] and receipt['sha256']==proof['archive_sha256']
    assert not any(p.name!='receipt.json' for p in directory.iterdir()),'no payload cache may be silently discarded'
    (directory/'staging_retirement.json').write_text(json.dumps(dict(passed=True,retired_bytes=0,
        payload_bytes_independently_stream_verified=manifest['bytes'],zero_payload_staging=True,cloud_and_F_source_unchanged=True),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--retire-proof',type=Path);p.add_argument('--readback-attempt',type=int);a=p.parse_args()
    if a.retire_proof:retire(a.manifest,a.retire_proof)
    else:run(a.manifest,a.readback_attempt)
