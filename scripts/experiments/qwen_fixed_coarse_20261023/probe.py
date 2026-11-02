"""Single-source inference-only coarse-code intervention, with strict replay gates.

Only talker logits are constrained. Sampling still runs at its original shape,
so own-source replay tests the otherwise easy-to-miss RNG-consumption bridge.
No installed model code or historical artifact is modified.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/qwen_fixed_coarse_20261023_attempt01'
SID = 'BAC009S0002W0122'
sys.path.insert(0, str(ROOT))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


class ForceSequence:
    """One forced coarse token per ordinary sampling step, including final EOS."""
    def __init__(self, sequence):
        self.sequence = [int(v) for v in sequence]
        self.calls = 0

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1 or self.calls >= len(self.sequence):
            raise ValueError('forced sequence exhausted or batch not one')
        token = self.sequence[self.calls]
        if not 0 <= token < scores.shape[1]:
            raise ValueError('forced token out of vocabulary')
        self.calls += 1
        out = scores.new_full(scores.shape, -float('inf'))
        out[:, token] = 0
        return out


@contextmanager
def constrain(talker, sequence):
    from transformers import LogitsProcessorList
    original = talker.generate
    force = ForceSequence(sequence)
    def generate(*args, **kwargs):
        if kwargs.get('logits_processor'):
            raise ValueError('unexpected existing custom logits processors')
        kwargs['logits_processor'] = LogitsProcessorList([force])
        return original(*args, **kwargs)
    talker.generate = generate
    try:
        yield force
    finally:
        talker.generate = original


def prepare():
    RUN.mkdir(exist_ok=False)
    parent = ROOT / 'runs/qwen17_level_comparison10_20261023_attempt01'
    p = read(parent / 'protocol.json')
    sample = sorted(p['samples'], key=lambda r:r['sample_id'])[0]
    assert sample['sample_id'] == SID
    paths = {
        'L06': ROOT / f'runs/qwen_full_tts_extension_20261022_attempt01/tts/{SID}/seed1/record.json',
        'L17': parent / f'generation/{SID}/draw1.json'}
    records = {k:read(v) for k,v in paths.items()}
    pins = {str(v):sha(v) for v in paths.values()}
    for r in records.values():
        assert r['sample_id'] == SID and r['seed'] == 2026102401
        for rec in [r, r['new_codes'], r['reference_codes'], r['talker_sequences']]:
            assert sha(rec['path']) == rec['sha256']
            pins[rec['path']] = rec['sha256']
    assert sha(sample['input24_path']) == sample['input24_sha256']
    pins[sample['input24_path']] = sample['input24_sha256']
    model_dirs = {k:str(ROOT / ('models/Qwen3-TTS-12Hz-' + n + '-Base'))
                  for k,n in [('L06','0.6B'),('L17','1.7B')]}
    for d in model_dirs.values():
        for rel in ['model.safetensors','speech_tokenizer/model.safetensors','config.json']:
            pins[str(Path(d)/rel)] = sha(Path(d)/rel)
    save(RUN/'protocol.json', dict(sample=sample, records=records, pins=pins,
        model_dirs=model_dirs, generation_kwargs=p['generation_kwargs'], seed=2026102401,
        selection='lexicographically first old cloud10 sample, draw1; no score selection',
        controls='old full-code decoder replay, fresh unforced exact replay, forced own-coarse exact replay; both models before cross',
        content_gate='ASR exact normalized transcript, without initial prompt; forced alignment is not independent content evidence',
        timing_gate='same samples; matching phone identities; center median <=20ms, p95 <=40ms, max <=80ms; envelope best lag <=10ms',
        interpretation='single-source technical gate, not equivalence proof or pure semantic/acoustic factorial',
        no_training=True, no_tfg=True, no_syncnet=True))
    (RUN/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED', flush=True)


def run(model_name, stage):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2')
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('probe 900s')))
    signal.alarm(900)
    p = read(RUN/'protocol.json')
    if stage == 'cross':
        for name in ['L06','L17']:
            assert read(RUN/f'{name}_control/result.json')['status'] == 'PASS'
    out = RUN/f'{model_name}_{stage}'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    for path, digest in p['pins'].items():
        assert sha(path) == digest, path
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    def seed(n):
        random.seed(n); np.random.seed(n % 2**32)
        torch.manual_seed(n); torch.cuda.manual_seed_all(n)
    seed(20261022)
    d = p['model_dirs'][model_name]
    w = Qwen3TTSModel.from_pretrained(d, local_files_only=True, device_map='cuda:0',
                                    dtype=torch.float32, attn_implementation='eager')
    w.model.eval().requires_grad_(False)
    old = w.model.speech_tokenizer
    w.model.speech_tokenizer = None
    del old
    gc.collect(); torch.cuda.empty_cache()
    codec = Qwen3TTSTokenizer.from_pretrained(d+'/speech_tokenizer', local_files_only=True,
                  device_map='cuda:0', dtype=torch.float32, attn_implementation='eager')
    codec.model.eval().requires_grad_(False)
    w.model.speech_tokenizer = codec
    r = p['records'][model_name]
    x, sr = sf.read(p['sample']['input24_path'], dtype='float32')
    text = r['target_text']
    with torch.inference_mode():
        prompt = w.create_voice_clone_prompt(ref_audio=(x,sr),ref_text=text,x_vector_only_mode=False)
    ref = h.codes(prompt[0].ref_code)
    assert np.array_equal(ref,np.load(r['reference_codes']['path']))
    source = 'L17' if model_name == 'L06' else 'L06'
    donor = p['records'][model_name if stage == 'control' else source]
    codes = np.load(donor['new_codes']['path'])
    sequence = np.load(donor['talker_sequences']['path'])[0]
    assert len(sequence) == len(codes)+1 and np.array_equal(sequence[:-1], codes[:,0])
    assert sequence[-1] == w.model.config.talker_config.codec_eos_token_id
    runtime = dict(model=model_name, stage=stage, source=source if stage=='cross' else model_name,
        torch=torch.__version__, executable=sys.executable, gpu=torch.cuda.get_device_name(),
        source_files={str(Path(inspect.getfile(c))):sha(inspect.getfile(c)) for c in [Qwen3TTSModel,Qwen3TTSTokenizer,type(w.model)]},
        protocol_sha256=sha(RUN/'protocol.json'), source_sha256=sha(__file__))
    save(out/'runtime.json',runtime)
    results = {}
    if stage == 'control':
        joined = np.concatenate([ref,codes])
        with torch.inference_mode():
            waves,rate = codec.decode([{'audio_codes':torch.as_tensor(joined,device=codec.device)}])
        cut = h.official_cut(len(ref),len(joined),len(waves[0]))
        oldwave,_ = sf.read(r['path'],dtype='float32')
        results['decoder_replay'] = h.compare_waveforms(waves[0][cut:],oldwave)
        h.save_wave(out/'decoded_replay.wav',waves[0][cut:])
        save(out/'decoder_replay.json',results['decoder_replay'])
        assert results['decoder_replay']['array_exact'], 'decoder replay failed'
    for mode in (['unforced','forced'] if stage == 'control' else ['forced']):
        seed(p['seed'])
        with torch.inference_mode(), h.capture_official(w) as cap:
            if mode == 'forced':
                with constrain(w.model.talker,sequence) as force:
                    waves,rate = w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,
                                non_streaming_mode=False,**p['generation_kwargs'])
                assert force.calls == len(sequence)
            else:
                waves,rate = w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,
                                non_streaming_mode=False,**p['generation_kwargs'])
        got = cap['new_codes'][0]
        result = dict(wave=h.save_wave(out/f'{mode}.wav',waves[0]),
             codes=h.save_array(out/f'{mode}_codes.npy',got),
             sequences=h.save_array(out/f'{mode}_sequences.npy',cap['talker_sequences']),
             coarse_exact=bool(np.array_equal(got[:,0],codes[:,0])),
             full_codes_exact=bool(np.array_equal(got,codes)),
             waveform_comparison=h.compare_waveforms(waves[0],sf.read(donor['path'],dtype='float32')[0]))
        results[mode] = result
        save(out/f'{mode}.json',result)
        if stage == 'control':
            assert result['full_codes_exact'] and result['waveform_comparison']['array_exact'], mode+' self replay failed'
        else:
            assert result['coarse_exact'] and result['waveform_comparison']['length_delta_samples']==0
        print(model_name,stage,mode,'PASS',flush=True)
    save(out/'result.json',dict(status='PASS',results=results,elapsed_s=time.monotonic()-start,
        peak_reserved=torch.cuda.max_memory_reserved(),no_training=True,no_tfg=True,no_syncnet=True))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','control','cross'])
    ap.add_argument('--model',choices=['L06','L17'])
    args=ap.parse_args()
    if args.stage=='prepare': prepare()
    else: run(args.model,args.stage)
