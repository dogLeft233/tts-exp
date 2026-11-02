"""Official SyncReward inference precision by default; explicit FP32 replay option.

Config-only backbone construction is overwritten by the complete strict Stage2.
Encode batch1 is the documented low-memory option; official CUDA BF16 is kept.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/home/wjj/av-eval-20261026')
sys.path.insert(0, str(ROOT / 'SyncReward'))
import torch
from syncreward.data import load_av, MIN_AUDIO_SAMPLES, MIN_VIDEO_FRAMES
from syncreward.model import PeAVEncoder, SyncRewardModel, load_config, load_state_dict, remap_legacy_keys, load_processor, prepare_inputs, processor_sample_rate, ARCH_KEYS
from transformers import PeAudioVideoConfig, PeAudioVideoModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument('videos', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--encode_batch_size', type=int, default=1)
    p.add_argument('--precision', choices=['bf16','float32'], default='bf16')
    args = p.parse_args()
    torch.set_num_threads(4)
    checkpoint = ROOT / 'weights/syncreward/syncreward-stage2'
    encoder_path = ROOT / 'weights/pe-av-large'
    cfg = PeAudioVideoConfig.from_pretrained(encoder_path, local_files_only=True)
    # Timm config serializes num_classes=1024, while transformers defaults num_labels to 2.
    # Recover the declared class count; strict Stage-2 loading verifies the resulting tensor shapes.
    backbone = cfg.video_config.video_config.vision_config
    raw_config = json.loads((encoder_path / 'config.json').read_text())
    backbone.num_labels = int(raw_config['audio_video_config']['video_config']['vision_config']['num_classes'])
    # Only modules retained in Stage 2 are used; every retained tensor is overwritten by strict load.
    base = PeAudioVideoModel(cfg)
    encoder = PeAVEncoder(base.audio_model.audio_encoder, base.video_model.video_encoder,
                          base.audio_model.audio_head, base.video_model.video_head)
    del base
    arch = {k: v for k, v in load_config(checkpoint).items() if k in ARCH_KEYS}
    model = SyncRewardModel(encoder, **arch)
    match = model.load_state_dict(remap_legacy_keys(load_state_dict(checkpoint)), strict=True)
    assert not match.missing_keys and not match.unexpected_keys
    model.cuda().eval()
    processor = load_processor(str(encoder_path))
    sr = processor_sample_rate(processor)
    args.num_segments = model.num_segments
    args.segment_duration = 0.5
    args.segment_stride = 0.25
    results = []
    for video_path in args.videos:
        video, fps, audio = load_av(video_path, sr)
        if len(video) < MIN_VIDEO_FRAMES or len(audio) < MIN_AUDIO_SAMPLES:
            raise ValueError('Input clip too short: ' + video_path)
        inputs = prepare_inputs(processor, {'video': [video], 'audio': [audio], 'fps': [fps]}, args, 'center', torch.device('cuda'))
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            reward = float(model(inputs, encode_batch_size=args.encode_batch_size).float()[0])
        if not math.isfinite(reward) or not 0 <= reward <= 2:
            raise RuntimeError(f'Invalid SyncReward: {reward}')
        row = {'video_path': str(Path(video_path).resolve()), 'reward': reward, 'precision': args.precision, 'encode_batch_size': args.encode_batch_size, 'elapsed_seconds': time.monotonic()-started}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row) + '\n' for row in results))
    print('peak_cuda_memory_bytes', torch.cuda.max_memory_allocated())

if __name__ == '__main__':
    main()
