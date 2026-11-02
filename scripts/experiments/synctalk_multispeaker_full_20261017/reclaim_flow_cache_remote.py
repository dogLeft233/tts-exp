import argparse,hashlib,json
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--split',required=True);a=p.parse_args()
assert a.identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers'] and a.split in ['train','val','test']
root=Path('/dev/shm/synctalk-multispeaker-data-20261017');folder=root/'prepared'/a.identity/a.split
receipt=json.loads((folder/'rigid_flow_persistence.json').read_text());flow=json.loads((folder/'flow_receipt.json').read_text())
assert receipt['passed'] and receipt['decompressed_source_verified'] and receipt['raw_sha256']==flow['rigid_flow_sha256']
target=folder/'rigid_flow.npy';assert not target.is_symlink() and target.resolve()==target and target.is_relative_to(root/'prepared')
before=target.stat();h=hashlib.sha256()
with target.open('rb') as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
assert h.hexdigest()==receipt['raw_sha256'] and target.stat().st_mtime_ns==before.st_mtime_ns
for process in psutil.process_iter():
    try:assert not any(x.path==str(target) for x in process.open_files()),f'cache busy: {process.pid}'
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
target.unlink()
r=dict(passed=True,identity=a.identity,split=a.split,reclaimed_bytes=before.st_size,raw_sha256=h.hexdigest(),persistent_gzip=receipt['path'],tracks_and_pose_inputs_retained=True)
(folder/'rigid_flow_reclamation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
