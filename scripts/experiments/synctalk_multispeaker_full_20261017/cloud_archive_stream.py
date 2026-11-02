"""Windows Baidu staging plus independent Linux full-archive readback; no config changes."""
import argparse,hashlib,json,re,shlex,shutil,subprocess
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
from execution_journal import atomic_json

def run(archive,expected,name,attempt):
    archive=Path(archive);assert archive.resolve().is_relative_to(ROOT.resolve()) and sha(archive)==expected
    assert re.fullmatch('[a-zA-Z0-9_-]+',name) and attempt>0
    out=ROOT/'cloud_delivery'/name/f'transport_attempt{attempt:02d}';out.mkdir(parents=True,exist_ok=False)
    cloud_root=Path('B:/tts-exp-research-models');assert cloud_root.is_dir(),'existing Windows Baidu mount required'
    assert shutil.disk_usage('C:/').free>archive.stat().st_size+512*1024**2,'Windows upload cache capacity guard'
    target=cloud_root/'synctalk-multispeaker-full-20261017'/name/f'transport_attempt{attempt:02d}'
    target.mkdir(parents=True,exist_ok=False)
    parts=[];whole=hashlib.sha256();total=0;size=256*1024**2
    with archive.open('rb') as source:
        index=0
        while True:
            block=source.read(4*1024**2)
            if not block:break
            file=f'bundle.part{index:03d}';h=hashlib.sha256();count=0
            with (target/file).open('xb') as output:
                while block:
                    assert shutil.disk_usage('C:/').free>128*1024**2
                    output.write(block);h.update(block);whole.update(block);count+=len(block);total+=len(block)
                    if count==size:break
                    block=source.read(min(4*1024**2,size-count))
            parts.append(dict(file=file,bytes=count,sha256=h.hexdigest()));index+=1
            atomic_json(out/'staging_receipt.json',dict(passed=False,staged_parts=parts,cloud_independently_verified=False))
    assert total==archive.stat().st_size and whole.hexdigest()==expected
    relative=f'synctalk-multispeaker-full-20261017/{name}/transport_attempt{attempt:02d}'
    manifest=dict(name=name,attempt=attempt,archive=str(archive),bytes=total,sha256=expected,parts=parts,
        windows_cloud_directory=str(target),linux_cloud_directory='/home/wjj/百度网盘/tts-exp-research-models/'+relative,
        transport_only=True,source_sha256=sha(Path(__file__)))
    atomic_json(out/'manifest.json',manifest);atomic_json(target/'manifest.json',manifest)
    remote_manifest=f'{REMOTE}/cloud_manifests/{name}_attempt{attempt:02d}.json'
    code="from pathlib import Path;import sys;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())"%remote_manifest
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=(out/'manifest.json').read_bytes(),check=True)
    command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/cloud_archive_readback_stream.py','--manifest',remote_manifest]
    with (out/'independent_readback_transport.log').open('xb') as log:
        subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt_path=f'{REMOTE}/cloud_readback/{name}/attempt{attempt:02d}/receipt.json'
    raw=subprocess.check_output(SSH+['cat '+shlex.quote(receipt_path)]);receipt=json.loads(raw)
    assert receipt['passed'] and receipt['independent_linux_cloud_readback'] and receipt['sha256']==expected and receipt['bytes']==total
    target_receipt=out/'cloud_readback_receipt.json';target_receipt.write_bytes(raw)
    remote_sha=subprocess.check_output(SSH+['sha256sum '+shlex.quote(receipt_path)]).decode().split()[0]
    assert sha(target_receipt)==remote_sha
    proof=dict(passed=True,F_verified_receipt_sha256=remote_sha,archive_sha256=expected)
    atomic_json(out/'readback_persistence_verification.json',proof)
    remote_proof=f'{REMOTE}/cloud_manifests/{name}_attempt{attempt:02d}_F_proof.json'
    code="from pathlib import Path;import sys;p=Path(%r);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())"%remote_proof
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=json.dumps(proof).encode(),check=True)
    subprocess.run(SSH+[shlex.join(command+['--retire-proof',remote_proof])],check=True)
    print(json.dumps(receipt),flush=True);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--name',required=True);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();run(a.archive,a.expected_sha,a.name,a.attempt)
