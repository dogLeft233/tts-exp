"""Test the official CUDA BF16 arithmetic on all existing timing controls."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

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

    def initialize_encoder(cls, encoder_path):
        cfg = PeAudioVideoConfig.from_pretrained(encoder_path, local_files_only=True)
        raw = json.loads((Path(encoder_path) / 'config.json').read_text())
        cfg.video_config.video_config.vision_config.num_labels = int(raw['audio_video_config']['video_config']['vision_config']['num_classes'])
        base = PeAudioVideoModel(cfg)
        return cls(base.audio_model.audio_encoder, base.video_model.video_encoder,
                   base.audio_model.audio_head, base.video_model.video_head)

    official.PeAVEncoder.from_pretrained = classmethod(initialize_encoder)
    model = official.build_reward_model(str(backbone), checkpoint=str(checkpoint), num_segments=18).cuda().eval()
    processor = official.load_processor(str(backbone))
    sr = official.processor_sample_rate(processor)
    args = SimpleNamespace(num_segments=18, segment_duration=.5, segment_stride=.25)
    root = Path(__file__).resolve().parents[2]
    old = root / 'runs/av_metric_preference_20261026_attempt02'
    controls = json.loads((out / 'cached_control_reproduction.json').read_text())['syncreward']['rows']
    names = sorted({r['id'] for r in controls} | {'native__' + r['source'] + '_N_0' for r in controls})
    videos = [old / 'inputs' / (name + '.mp4') for name in names]
    videos += sorted((out / 'probe_inputs').glob('official*.mp4'))
    saved = out / 'syncreward_bf16_scores.jsonl'
    existing = [json.loads(x) for x in saved.read_text().splitlines()] if saved.exists() else []
    scores = {r['id']: r['reward'] for r in existing}
    for video in videos:
        if video.stem in scores:
            continue
        frames, fps, audio = load_av(video, sr)
        inputs = official.prepare_inputs(processor, {'video': [frames], 'audio': [audio], 'fps': [fps]}, args, 'center', torch.device('cuda'))
        started = time.monotonic()
        # Exactly the CUDA inference precision selected by upstream score.py.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            reward = float(model(inputs, encode_batch_size=1).float()[0])
        row = dict(id=video.stem, reward=reward, elapsed_seconds=time.monotonic()-started)
        with saved.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        scores[video.stem] = reward
        print('BF16', len(scores), '/', len(videos), json.dumps(row), flush=True)
    fp32 = {'native__' + r['source'] + '_N_0': r['native'] for r in controls}
    fp32.update({r['id']: r['shifted'] for r in controls})
    rows = []
    for r in controls:
        native = 'native__' + r['source'] + '_N_0'
        rows.append(dict(id=r['id'], native=scores[native], shifted=scores[r['id']],
            delta=scores[r['id']]-scores[native], fp32_delta=r['delta']))
    report = dict(status='COMPLETE', controls=40, lower=sum(r['delta']<0 for r in rows),
        fp32_lower=sum(r['delta']<0 for r in controls), threshold=32,
        fp32_vs_bf16_max_score_difference=max(abs(fp32[k]-scores[k]) for k in fp32),
        direction_flips=sum((r['delta']>0)!=(r['fp32_delta']>0) for r in rows),
        device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        bf16_native=torch.cuda.is_bf16_supported(including_emulation=False),
        bf16_including_emulation=torch.cuda.is_bf16_supported(),
        precision='Official CUDA BF16 autocast; encode_batch_size1 is documented memory option',
        loader_adaptation='Config-only backbone init then original official builder strict Stage2 load',
        rows=rows, official_probes={k:v for k,v in scores.items() if k.startswith('official_')})
    (out / 'syncreward_bf16_controls.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    try:
        main(out)
    except BaseException as exc:
        (out / 'syncreward_bf16_failure.json').write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()), indent=2))
        raise
