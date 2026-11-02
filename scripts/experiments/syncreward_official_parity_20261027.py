"""Execute official score.main with explicit V100 FP32/loader adaptations.

No upstream source edits. Stage2 strictly overwrites the config-only encoder,
avoiding the redundant9GB initial backbone download. Both adaptations are
recorded, so this must not be described as H20/BF16 bit-exact replication.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

DEPLOY = Path('/home/wjj/av-eval-20261026')
sys.path.insert(0, str(DEPLOY / 'SyncReward'))
import syncreward.model as official
from syncreward.data import load_av
from transformers import PeAudioVideoConfig, PeAudioVideoModel


def main(out):
    torch.set_num_threads(4)
    backbone = DEPLOY / 'weights/pe-av-large'
    checkpoint = DEPLOY / 'weights/syncreward/syncreward-stage2'
    observed = {}

    def initialize_encoder(cls, encoder_path):
        cfg = PeAudioVideoConfig.from_pretrained(encoder_path, local_files_only=True)
        raw = json.loads((Path(encoder_path) / 'config.json').read_text())
        cfg.video_config.video_config.vision_config.num_labels = int(raw['audio_video_config']['video_config']['vision_config']['num_classes'])
        base = PeAudioVideoModel(cfg)
        encoder = cls(base.audio_model.audio_encoder, base.video_model.video_encoder,
                      base.audio_model.audio_head, base.video_model.video_head)
        observed['encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
        return encoder

    official.PeAVEncoder.from_pretrained = classmethod(initialize_encoder)
    original_build = official.build_reward_model

    def record_build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        observed['model'] = model
        return model

    official.build_reward_model = record_build
    actual_autocast = torch.autocast
    autocast_requests = []

    def fp32_autocast(device_type, *args, **kwargs):
        autocast_requests.append(dict(device=device_type, dtype=str(kwargs.get('dtype')), enabled=kwargs.get('enabled')))
        return contextlib.nullcontext()

    torch.autocast = fp32_autocast
    path = DEPLOY / 'SyncReward/score.py'
    spec = importlib.util.spec_from_file_location('unmodified_official_score', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    root = Path(__file__).resolve().parents[2]
    old = root / 'runs/av_metric_preference_20261026_attempt02'
    names = ['native__BAC009S0002W0122_N_0', 'shift__BAC009S0002W0122__-6400',
             'shift__BAC009S0002W0122__6400']
    videos = [old / 'inputs' / (name + '.mp4') for name in names]
    output = out / 'official_score_fp32.jsonl'
    sys.argv = [str(path), *map(str, videos), '--checkpoint', str(checkpoint),
                '--encoder_path', str(backbone), '--encode_batch_size', '1', '--device', 'cuda', '--output', str(output)]
    started = time.monotonic()
    reference.main()
    torch.autocast = actual_autocast
    official_rows = [json.loads(line) for line in output.read_text().splitlines()]
    cached = json.loads((out / 'cached_control_reproduction.json').read_text())['syncreward']['rows']
    expected = {names[0]: cached[0]['native']}
    expected.update({r['id']: r['shifted'] for r in cached})
    differences = []
    for row in official_rows:
        key = Path(row['video_path']).stem
        error = abs(row['reward'] - expected[key])
        assert error < 5e-6, (key, error)
        differences.append(dict(id=key, official_fp32=row['reward'], deployed=expected[key], absolute_error=error))
    processor = official.load_processor(str(backbone))
    sr = official.processor_sample_rate(processor)
    args = SimpleNamespace(num_segments=18, segment_duration=.5, segment_stride=.25)
    batch_results = []
    model = observed.pop('model')
    for video, target in zip(videos[:2], official_rows[:2]):
        frames, fps, audio = load_av(video, sr)
        inputs = official.prepare_inputs(processor, {'video': [frames], 'audio': [audio], 'fps': [fps]}, args, 'center', torch.device('cuda'))
        with torch.inference_mode():
            score = float(model(inputs, encode_batch_size=2).float()[0])
        error = abs(score - target['reward'])
        assert error < 1e-4
        batch_results.append(dict(id=video.stem, batch1=target['reward'], batch2=score, error=error))
    # The official window/crop and decoder are also exercised on a long clip,
    # proving that the induced audio shift survives resampling and shared crop.
    nv, nf, na = load_av(videos[0], sr)
    sv, sf, sa = load_av(videos[1], sr)
    shift = int(round(.4 * sr))
    interior_error = float(np.max(np.abs(sa[64:-shift-64] - na[shift+64:-64])))
    assert np.array_equal(nv, sv) and nf == sf and interior_error < 1e-6
    # Apply the same official model to independent diagnostic inputs.
    probes = []
    for video in sorted((out / 'probe_inputs').glob('*.mp4')):
        frames, fps, audio = load_av(video, sr)
        inputs = official.prepare_inputs(processor, {'video': [frames], 'audio': [audio], 'fps': [fps]}, args, 'center', torch.device('cuda'))
        with torch.inference_mode():
            score = float(model(inputs, encode_batch_size=1).float()[0])
        probes.append(dict(id=video.stem, reward=score))
        print('PROBE', video.stem, score, flush=True)
    report = dict(status='PASS', official_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        caveat='Official score.main, with recorded FP32 instead of BF16 on V100 and config-only initial backbone before official strict Stage2 load',
        requested_autocast=autocast_requests, cuda_bf16_supported=torch.cuda.is_bf16_supported(),
        parameters=observed, parity=differences, encode_batch1_vs2=batch_results,
        decoder_resampled_shift_max_error=interior_error, processor_sample_rate=sr,
        shared_center_start_seconds=max(0, min(len(nv)/nf, len(na)/sr)-4.75)/2,
        probes=probes, elapsed_seconds=time.monotonic()-started)
    (out / 'syncreward_official_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
