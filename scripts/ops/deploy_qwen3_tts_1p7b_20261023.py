"""V100 deployment check; no training and no research scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['download', 'smoke'])
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    repo = args.root.resolve()
    out = repo / 'runs/qwen3_tts_1p7b_deployment_20261023_attempt01'
    model_dir = repo / 'models/Qwen3-TTS-12Hz-1.7B-Base'
    out.mkdir(parents=True, exist_ok=True)

    def save(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

    if args.stage == 'download':
        import requests
        listing = json.loads((out / 'modelscope_listing.json').read_text())
        records = []
        for item in listing['Data']['Files']:
            if item['Type'] != 'blob':
                continue
            target = model_dir / item['Path']
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = item['Sha256']
            old = repo / 'models/Qwen3-TTS-12Hz-0.6B-Base' / item['Path']
            if not target.exists() and old.is_file() and sha(old) == expected:
                import shutil
                shutil.copy2(old, target)
            url = ('https://modelscope.cn/api/v1/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base/repo'
                   '?Revision=' + item['Revision'] + '&FilePath=' + item['Path'])
            if not target.exists():
                temporary = target.with_name(target.name + '.part')
                print('DOWNLOAD', item['Path'], item['Size'], flush=True)
                with requests.get(url, stream=True, timeout=(15, 90)) as r:
                    r.raise_for_status()
                    with temporary.open('wb') as f:
                        for block in r.iter_content(4 << 20):
                            f.write(block)
                if temporary.stat().st_size != item['Size'] or sha(temporary) != expected:
                    raise RuntimeError('download hash mismatch: ' + item['Path'])
                temporary.rename(target)
            if target.stat().st_size != item['Size'] or sha(target) != expected:
                raise RuntimeError('installed file mismatch: ' + item['Path'])
            records.append(dict(path=item['Path'], sha256=expected, size=item['Size'],
                                revision=item['Revision'], url=url))
            print('VERIFIED', item['Path'], flush=True)
        save('weights_manifest.json', dict(model_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                                         source='official Qwen ModelScope repository', files=records))
        return

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
    if (out / 'smoke_receipt.json').exists() or (out / 'smoke_full_icl_float.wav').exists():
        raise FileExistsError('deployment smoke already exists; inspect it instead of overwriting')
    import gc
    import sys
    import importlib.metadata as md
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    manifest = json.loads((out / 'weights_manifest.json').read_text())
    for record in manifest['files']:
        if sha(model_dir / record['path']) != record['sha256']:
            raise RuntimeError('weight changed')
    sample = json.loads((repo / 'runs/qwen_audio31_cloud_subset10_20261023_attempt01/protocol.json').read_text())['samples'][0]
    reference = Path(sample['input24_path'])
    if sha(reference) != sample['input24_sha256']:
        raise RuntimeError('reference mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    print('LOAD FP32 eager', flush=True)
    model = Qwen3TTSModel.from_pretrained(str(model_dir), local_files_only=True,
        device_map='cuda:0', dtype=torch.float32, attn_implementation='eager')
    model.model.eval().requires_grad_(False)
    # Explicit codec load matches the verified 0.6B FP32 implementation.
    old = model.model.speech_tokenizer
    model.model.speech_tokenizer = None
    del old
    gc.collect()
    torch.cuda.empty_cache()
    codec = Qwen3TTSTokenizer.from_pretrained(str(model_dir / 'speech_tokenizer'),
        local_files_only=True, device_map='cuda:0', dtype=torch.float32, attn_implementation='eager')
    codec.model.eval().requires_grad_(False)
    model.model.speech_tokenizer = codec
    if model.model.dtype != torch.float32 or codec.model.dtype != torch.float32:
        raise RuntimeError('FP32 model and codec required')
    loaded = time.monotonic()
    text = ''.join(sample['transcript'].split())
    ref, rate = sf.read(reference, dtype='float32')
    with torch.inference_mode():
        prompt = model.create_voice_clone_prompt(ref_audio=(ref, rate), ref_text=text,
                                                x_vector_only_mode=False)
        if not prompt[0].icl_mode or prompt[0].x_vector_only_mode:
            raise RuntimeError('full ICL prompt required')
        print('GENERATE full ICL', flush=True)
        waves, sr = model.generate_voice_clone(text=text, language='Chinese', voice_clone_prompt=prompt,
            non_streaming_mode=False, do_sample=True, temperature=0.9, top_k=50, top_p=1., max_new_tokens=512)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - loaded
    wave = np.asarray(waves[0], dtype=np.float32).reshape(-1)
    if not np.isfinite(wave).all() or not 0.5 < len(wave) / sr < 45 or np.max(np.abs(wave)) < 1e-5:
        raise RuntimeError('invalid waveform')
    dest = out / 'smoke_full_icl_float.wav'
    sf.write(dest, wave, sr, subtype='FLOAT')
    save('smoke_receipt.json', dict(status='PASS_WAVEFORM_CHECKS', no_training=True,
        scope='one deployment smoke, not a scientific comparison or perceptual quality validation',
        executable=sys.executable, model=str(model_dir), gpu=torch.cuda.get_device_name(),
        dtype='float32', attention='eager', seed=20261023, full_icl=True, text=text,
        parameter_count=sum(p.numel() for p in model.model.parameters()),
        model_size=model.model.config.tts_model_size,
        generation_kwargs=dict(non_streaming_mode=False, do_sample=True,
            temperature=0.9, top_k=50, top_p=1., max_new_tokens=512),
        reference_path=str(reference), reference_sha256=sha(reference),
        weights_manifest_sha256=sha(out / 'weights_manifest.json'), script_sha256=sha(__file__),
        packages={n: md.version(n) for n in ['torch', 'qwen-tts', 'transformers', 'numpy', 'soundfile']},
        load_seconds=loaded-start, inference_seconds=elapsed, sample_rate=sr,
        duration_seconds=len(wave)/sr, real_time_factor=elapsed/(len(wave)/sr),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        output_path=str(dest), output_sha256=sha(dest), peak_abs=float(np.max(np.abs(wave))),
        rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2)))))
    print((out / 'smoke_receipt.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
