"""Freeze exact prior Chinese20 inputs and the official TalkLikeYou configuration."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path('/home/data2/wangjiajun/av-evaluation')
D = Path('/home/data2/wangjiajun/TalkLikeYouDeploy')
PARENT = B/'jobs/flashhead_chinese20_rms24_20261030_attempt01'
RUN = B/'jobs/talklikeyou_chinese20_rms24_20261030_attempt01'
GPU = 'GPU-3464c5ee-e2ed-81e4-e275-b612a929bfe1'


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
    # Official centered librosa.stft, hop200: 1+samples//200 mel frames.
    # AudioFeatureExtractor: int((mel_frames-16)/80*25).
    value = (samples//200-15)*25//80
    if value <= 0:
        raise ValueError('No complete native mel windows')
    return value


def main():
    import numpy as np
    import soundfile as sf
    assert not RUN.exists(), 'Fresh run required'
    old = json.loads((PARENT/'protocol.json').read_text())
    rows = json.loads((PARENT/'inputs.json').read_text())['rows']
    assert len(rows) == 80 and len({r['source'] for r in rows}) == 20
    face = Path(old['generation']['face_path'])
    assert sha(face) == old['generation']['face_sha256']
    assets = json.loads((D/'download_receipt.json').read_text())
    source = json.loads((D/'source_verification.json').read_text())
    assert assets['status'] == source['status'] == 'PASS'
    for entry in assets['files']:
        asset = D/'TalkLikeYou'/entry['path']
        assert asset.stat().st_size == entry['bytes'] and sha(asset) == entry['sha256']
    for entry in source['files']:
        assert sha(D/'TalkLikeYou'/entry['path']) == entry['sha256']
    RUN.mkdir(parents=True)
    for row in rows:
        original = Path(row['audio_path'])
        assert sha(original) == row['audio_sha256']
        target = RUN/'audio'/original.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        assert sha(target) == row['audio_sha256']
        pcm, sr = sf.read(target, dtype='int16')
        assert sr == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
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
    protocol.update(run_id=RUN.name, parent_run=str(PARENT), parent_protocol_sha256=sha(PARENT/'protocol.json'),
        input_binding='Byte-identical copies of all80 prior RMS24 WAVs; no synthesis, gain or time changes',
        generation=dict(model='TalkLikeYou', source_commit='d1088c49071148853c76d8a8f0c1fb97d895758b',
            weights_revision='652b744bbf86510c370ff6f302d046bdd63632e1',
            face_path=str(face), face_sha256=sha(face), fps=25, seed=42,
            person_id=192, stage0=True, neutral=True, motion_scale=1., guidance_scale=1.3,
            sampling_steps=1, pose_mode='audio', pose_sampling_steps=1,
            overlap=5, smooth=True, crop_scale=2.3, face_index=0,
            fresh_process_per_unique_audio=True, size='native; no resize beyond upstream renderer',
            audio_preprocess='All official lip mel and Ditto pose preprocessing retained; no external waveform normalization beyond frozen RMS24 inputs',
            frame_policy='Official centered-mel complete16-window count; min(lip,pose). Full audio preserved with original zero origin. No padding, time stretch, time shift or offset correction.',
            reuse='Exact audio+face+seed/settings aliases explicitly reference one generated artifact'),
        resources=dict(gpu_uuid=GPU, torch_memory_fraction=.60, cpu_threads=4, execution='nohup; sequential GPU phases; private directories only'),
        duplicate_audio_groups=[g for g in groups.values() if len(g)>1], unique_audio_files=len(groups))
    write(RUN/'protocol.json', protocol)
    write(RUN/'inputs.json', dict(rows=rows))
    write(RUN/'preparation.json', dict(status='PASS', audio_rows=80, sources=20,
        unique_audio=len(groups), parent_inputs_sha256=sha(PARENT/'inputs.json'),
        official_asset_files_verified=len(assets['files']), source_files_verified=len(source['files']),
        expected_frame_range=[min(r['expected_frames'] for r in rows),max(r['expected_frames'] for r in rows)],
        unsupported_tail_samples_range=[min(r['unsupported_tail_samples'] for r in rows),max(r['unsupported_tail_samples'] for r in rows)]))
    print((RUN/'preparation.json').read_text())


if __name__ == '__main__':
    main()
