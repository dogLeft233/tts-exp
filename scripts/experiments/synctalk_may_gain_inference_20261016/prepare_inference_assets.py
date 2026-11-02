"""Freeze the official May validation prefix before any generated scores."""
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

out=Path('runs/synctalk_may_gain_inference_20261016')
root=out/'prepared'
assert not root.exists()
root.mkdir(parents=True)
archive=Path('runs/synctalk_amplitude_audit_20261016/resources/May.zip')
with zipfile.ZipFile(archive) as z:
    config=json.loads(z.read('May/transforms_val.json'))
    frames=config['frames'][:300]
    assert len(frames)==300 and [x['img_id'] for x in frames]==list(range(5520,5820))
    names=['May/aud.wav','May/bs.npy','May/bc.jpg','May/transforms_val.json']
    for f in frames:
        i=f['img_id']
        names.extend([f'May/gt_imgs/{i}.jpg',f'May/ori_imgs/{i}.lms',f'May/torso_imgs/{i}.png'])
    members=[]
    for name in names:
        raw=z.read(name)
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        members.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
protocol=dict(status='FROZEN_BEFORE_RENDER',official_commit='43079fb3f349675f6cb33031bb07262f7566cdd5',
              checkpoint_sha256='79e5c0056702357e176e9895b237bfcab1a517b85f4e468586282954e7379e79',
              source='Official May validation prefix',img_ids=list(range(5520,5820)),
              audio_ids=[f['aud_id'] for f in frames],frames=300,fps=25,duration_seconds=12,
              score_audio_start_sample=5520*640,score_audio_samples=300*640,sample_rate=16000,
              arms=[dict(id='gain_m6',db=-6),dict(id='gain_0',db=0),
                    dict(id='gain_0_repeat',db=0),dict(id='gain_p6',db=6)],
              preprocessing='Official full-utterance AVE recomputed from DOUBLE WAV, no normalization or limiter',
              scoring='Identical unmodified original mono PCM segment in every arm',
              renderer='Official non-portrait head with frame-specific original torso/background; no training',
              shared_controls=['checkpoint','pose','eye','background','frame ids','render settings','codec','score PCM'],
              gain_safety='Every full-utterance arm peak < 1; verify scalar relationship exactly',
              scope='One identity and one 12-second clip; not a population or quality conclusion')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
(root/'members.json').write_text(json.dumps(members,indent=2))
with tarfile.open(out/'resources/may_inference_subset.tar.gz','w:gz') as t:
    t.add(root,arcname='data')
print('Extracted bytes',sum(m['bytes'] for m in members))
