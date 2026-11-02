"""Bind the exact frozen RMS24 inputs to a fresh EchoMimicV3-Flash run."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path('/home/data2/wangjiajun/av-evaluation')
D = Path('/home/data2/wangjiajun/EchoMimicV3-Flash')
PARENT = B/'jobs/flashhead_chinese20_rms24_20261030_attempt01'
RUN = B/'jobs/echomimic_v3_flash_chinese20_rms24_20261030_attempt01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def native_frames(samples):
    """Official infer_flash: floor to 25fps, then round down to 4k+1."""
    frames = samples//640
    if frames < 1:
        raise ValueError('Audio has no complete video frame')
    return 4*((frames-1)//4)+1


def main():
    import numpy as np
    import soundfile as sf
    assert not RUN.exists(), 'Fresh run required; do not overwrite previous attempts'
    old = json.loads((PARENT/'protocol.json').read_text())
    rows = json.loads((PARENT/'inputs.json').read_text())['rows']
    assert len(rows) == 80 and len({r['source'] for r in rows}) == 20
    face = Path(old['generation']['face_path'])
    assert sha(face) == old['generation']['face_sha256']
    RUN.mkdir(parents=True)
    for row in rows:
        source = Path(row['audio_path'])
        assert sha(source) == row['audio_sha256']
        dest = RUN/'audio'/source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        assert sha(dest) == row['audio_sha256']
        pcm, sr = sf.read(dest, dtype='int16')
        assert sr == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
        x = pcm.astype(np.float64)/32768
        assert abs(20*np.log10(np.sqrt(np.mean(x*x)))+24) < .002
        assert abs(x).max() <= 10**(-1/20)
        row['parent_normalized_audio_path'] = str(source)
        row['audio_path'] = str(dest)
        row['expected_frames'] = native_frames(len(pcm))
        row['unsupported_tail_samples'] = len(pcm)-row['expected_frames']*640
    groups = {}
    for row in rows:
        groups.setdefault(row['audio_sha256'], []).append(row['id'])
    protocol = dict(old)
    protocol.update(run_id=RUN.name, parent_run=str(PARENT), parent_protocol_sha256=sha(PARENT/'protocol.json'),
        input_binding='Exact byte copies of all80 previous RMS24 WAVs; no second gain or resynthesis',
        generation=dict(model='EchoMimicV3-Flash', released_weights='echomimicv3-flash-pro',
            upstream_revision='7e89489ca51c0d008fc1963ec6c03fc5bd0b9397',
            face_path=str(face), face_sha256=sha(face), fps=25, size=[512,512], seed=42,
            prompt='A person is speaking.', negative_prompt='', num_inference_steps=8,
            sampler='Flow_Unipc', guidance_scale=6., audio_guidance_scale=3., audio_scale=1.,
            neg_scale=1., neg_steps=0, shift=5., weight_dtype='bfloat16',
            teacache=dict(threshold=.1, num_skip_start_steps=5, reset_per_clip=True),
            riflex=False, use_un_ip_mask=False, cpu_offload='model',
            peak_torch_memory_cap_gib=22., threads=4,
            frame_policy='Official floor(samples/640), then 4k+1 round down, with no81-frame cap. No input/audio padding or time shift. Last <0.16s may lack video. Full unchanged audio remains in lossless scoring media; scorers use actual common support.',
            audio_preprocess='Bypass official loudness_norm(-23LUFS) only, to preserve user RMS-24dBFS. Official W2V feature-extractor normalization remains. Official features use native supported prefix.',
            reuse='Exact normalized WAV+face+seed/settings permits explicit artifact aliases, never independent replicas'),
        support='Native actual video/audio common support; fixed5s classifiers require >=125 actual full frames and audio frames. SyncReward native loader internal padding retained. No external padding.',
        resources='Personal directories only; nohup; one GPU UUID; CPU4; torch<=22GiB for generator; sequential GPU phases',
        duplicate_audio_groups=[v for v in groups.values() if len(v)>1], unique_audio_files=len(groups))
    write(RUN/'protocol.json', protocol)
    write(RUN/'inputs.json', dict(rows=rows))
    write(RUN/'preparation.json', dict(status='PASS', rows=80, sources=20, unique_audio_files=len(groups),
        protocol_sha256=sha(RUN/'protocol.json'), parent_inputs_sha256=sha(PARENT/'inputs.json'),
        exact_parent_normalized_wav_copies=80, expected_frame_range=[min(r['expected_frames'] for r in rows),max(r['expected_frames'] for r in rows)],
        max_unsupported_tail_samples=max(r['unsupported_tail_samples'] for r in rows)))
    print((RUN/'preparation.json').read_text())


if __name__ == '__main__':
    main()
