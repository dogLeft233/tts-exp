"""Retire two canonical SHM video caches after geometry, scoring and F verification."""
import argparse,json
from pathlib import Path
import psutil
from flow_stream import sha
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args()
assert a.identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
folder=ROOT/'material'/a.identity
proof=json.loads((folder/'F_canonical_verification.json').read_text())
assert proof['passed'] and proof['identity']==a.identity
for path in [folder/'independent_material_qa.json',ROOT/'prepared'/a.identity/'geometry_v2/geometry_qualification.json',ROOT/'score_reference'/a.identity/'score_reference_receipt.json',ROOT/'snapshots'/a.identity/'supervision_reclamation.json']:
    assert json.loads(path.read_text())['passed']
source=json.loads((folder/'material_receipt.json').read_text())
targets=[folder/'main.mkv',folder/'test.mkv']
for target in targets:assert target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(folder.resolve())
for process in psutil.process_iter():
    try:
        assert not any(Path(x.path) in targets for x in process.open_files()),'active canonical video reader'
    except (psutil.NoSuchProcess,psutil.AccessDenied):continue
for target in targets:
    role=target.stem;original=next(x for x in source['recordings'] if x['role']==role)
    verified=next(x for x in proof['files'] if x['role']==role)
    assert target.stat().st_size==original['bytes']==verified['bytes']
    assert sha(target)==original['sha256']==verified['sha256']
output=folder/'canonical_reclamation.json';assert not output.exists()
size=sum(t.stat().st_size for t in targets)
for target in targets:target.unlink()
result=dict(passed=True,identity=a.identity,reclaimed_bytes=size,removed=[str(x) for x in targets],
    persistent_files=proof['files'],whole_audio_ave_images_geometry_retained=True,
    requires_restore_for_future_original_video_redecode=True,training_admitted=False)
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
