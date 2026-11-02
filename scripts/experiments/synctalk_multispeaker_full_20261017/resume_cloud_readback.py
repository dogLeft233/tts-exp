"""Versioned independent re-read of existing uploaded parts; no re-upload or overwrite."""
import argparse,json,shlex,subprocess
from pathlib import Path
from execution_journal import atomic_json
from persist_prepared import ROOT,REMOTE,SSH,sha
def run(name,transport_attempt,readback_attempt):
    assert readback_attempt>transport_attempt>0 and Path(name).name==name
    base=ROOT/'cloud_delivery'/name/f'transport_attempt{transport_attempt:02d}'
    manifest=json.loads((base/'manifest.json').read_text());expected=manifest['sha256']
    assert sha(Path(manifest['archive']))==expected,'retain and freshly verify original archive'
    out=base/f'readback_attempt{readback_attempt:02d}';out.mkdir(exist_ok=False)
    remote_manifest=f'{REMOTE}/cloud_manifests/{name}_attempt{transport_attempt:02d}.json'
    command=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/cloud_archive_readback.py','--manifest',remote_manifest,'--readback-attempt',str(readback_attempt)]
    with (out/'transport.log').open('xb') as log:subprocess.run(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
    path=f'{REMOTE}/cloud_readback/{name}/attempt{readback_attempt:02d}/receipt.json'
    raw=subprocess.check_output(SSH+['cat '+shlex.quote(path)]);receipt=json.loads(raw)
    assert receipt['passed'] and receipt['sha256']==expected and receipt['bytes']==manifest['bytes'] and receipt['readback_attempt']==readback_attempt
    target=out/'receipt.json';target.write_bytes(raw)
    digest=subprocess.check_output(SSH+['sha256sum '+shlex.quote(path)]).decode().split()[0];assert sha(target)==digest
    proof=dict(passed=True,F_verified_receipt_sha256=digest,archive_sha256=expected,readback_attempt=readback_attempt)
    atomic_json(out/'persistence_verification.json',proof)
    canonical=base/'cloud_readback_receipt.json';assert not canonical.exists();canonical.write_bytes(raw)
    remote_proof=f'{REMOTE}/cloud_manifests/{name}_attempt{transport_attempt:02d}_readback{readback_attempt:02d}_F_proof.json'
    code="from pathlib import Path;import sys;p=Path(%r);assert not p.exists();p.write_bytes(sys.stdin.buffer.read())"%remote_proof
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=json.dumps(proof).encode(),check=True)
    subprocess.run(SSH+[shlex.join(command+['--retire-proof',remote_proof])],check=True)
    atomic_json(base/'readback_persistence_verification.json',proof)
    print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--transport-attempt',type=int,default=1);p.add_argument('--readback-attempt',type=int,required=True);a=p.parse_args();run(a.name,a.transport_attempt,a.readback_attempt)
