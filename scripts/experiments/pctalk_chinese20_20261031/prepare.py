"""Freeze prior Chinese inputs; never change old runs or shared model files."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path('/home/data2/wangjiajun/av-evaluation')
D = Path('/home/data2/wangjiajun/PCTalkDeploy')
SOURCE = Path('/home/data2/wangjiajun/PC-Talk')
PARENT = B/'jobs/flashhead_chinese20_rms24_20261030_attempt01'
RUN = B/'jobs/pctalk_chinese20_rms24_20261031_attempt01'
GPU = 'GPU-7272933b-1219-b5f8-a9dd-ceade9bbde37'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024**2):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)

def native_frames(samples):
    # Match upstream operation order, including float round-off at integer edges.
    return max(1, int(max(1, 1+samples//200-16)/80*25))

def main():
    import numpy as np
    import soundfile as sf
    assert not RUN.exists(), 'Fresh run required'
    old = json.loads((PARENT/'protocol.json').read_text())
    rows = json.loads((PARENT/'inputs.json').read_text())['rows']
    assert len(rows) == 80 and len({r['source'] for r in rows}) == 20
    face = Path(old['generation']['face_path'])
    assert sha(face) == old['generation']['face_sha256']
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=SOURCE, text=True).strip()
    assert commit == 'a0392744cbac857d96db1b99ceab70cf7b5f6c83'
    assert not subprocess.check_output(['git','diff','HEAD','--'], cwd=SOURCE)
    tracked = subprocess.check_output(['git','ls-files'], cwd=SOURCE, text=True).splitlines()
    pins = {p:sha(SOURCE/p) for p in tracked if (SOURCE/p).is_file()}
    weights = SOURCE/'pctalk/checkpoints'
    assetpins = {}
    for line in (weights/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('* ')
        name = name.removeprefix('pctalk/checkpoints/')
        assert sha(weights/name) == expected, name
        assetpins[name] = expected
    assert len(assetpins) == 12
    RUN.mkdir(parents=True)
    for row in rows:
        original = Path(row['audio_path'])
        assert sha(original) == row['audio_sha256']
        target = RUN/'audio'/original.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        pcm, sr = sf.read(target, dtype='int16')
        assert sr == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
        assert sha(target) == row['audio_sha256']
        x = pcm.astype(np.float64)/32768
        assert abs(20*np.log10(np.sqrt(np.mean(x*x)))+24) < .002
        assert abs(x).max() <= 10**(-1/20)
        row.update(parent_normalized_audio_path=str(original), audio_path=str(target),
                   expected_frames=native_frames(len(pcm)),
                   unsupported_tail_samples=len(pcm)-native_frames(len(pcm))*640)
    groups = {}
    for row in rows:
        groups.setdefault(row['audio_sha256'], []).append(row['id'])
    protocol = dict(old)
    protocol.update(run_id=RUN.name,parent_run=str(PARENT),parent_protocol_sha256=sha(PARENT/'protocol.json'),
        scope='SyncNet only; stop before other eight scorers to compare official results',
        generation=dict(model='PC-Talk',source_commit=commit,
            weights_revision='e6220816118bd8714ff57dc83e5906122b4e9168',
            face_path=str(face),face_sha256=sha(face),fps=25,seed=42,
            person_id=192,lip_scale=1.,emotion=[],emc_enabled=False,face_index=1,
            crop_scale=2.3,motion_reference='source image static pose',
            fresh_process_per_unique_audio=True,
            frame_policy='Official complete mel windows; preserve full input PCM and zero origin. No padding/retime/offset correction.',
            audio_preprocess='Unmodified official mel/preemphasis/encoder; no extra external normalization',
            choice='Recommended preset192, CLI default lip scale1, no emotional control. Fixed before scoring, no parameter search.'),
        resources=dict(gpu_uuid=GPU,torch_memory_fraction=.45,cpu_threads=4,execution='nohup; sequential GPU phases; private directories only'),
        duplicate_audio_groups=[g for g in groups.values() if len(g)>1],unique_audio_files=len(groups))
    write(RUN/'protocol.json',protocol)
    write(RUN/'inputs.json',dict(rows=rows))
    write(RUN/'source_pins.json',dict(source_root=str(SOURCE),source=pins,weights=assetpins))
    write(RUN/'preparation.json',dict(status='PASS',audio_rows=80,sources=20,unique_audio=len(groups),
        parent_inputs_sha256=sha(PARENT/'inputs.json'),official_asset_files_verified=len(assetpins),source_files_verified=len(pins)))
    print((RUN/'preparation.json').read_text())

if __name__ == '__main__':
    main()
