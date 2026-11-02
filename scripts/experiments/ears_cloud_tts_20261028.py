"""EARS40 English cloned cloud TTS. API credential is read from stdin, never saved."""
from __future__ import annotations
import argparse, base64, concurrent.futures, hashlib, json, sys, time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

MODEL = 'qwen-audio-3.1-tts-plus'
BASE = 'https://dashscope.aliyuncs.com/api/v1'

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    tmp.replace(path)

def english_text(text):
    return ' '.join(text.split())

def mono_resample(x, sr, target):
    import math
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2: x = x.mean(axis=1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid audio')
    gcd = math.gcd(sr, target)
    return resample_poly(x, target // gcd, sr // gcd) if sr != target else x

def prepare(dataset, run):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    lookup = {row['sample_id']: row for row in manifest}
    speakers = sorted({row['speaker_id'] for row in manifest})
    if len(speakers) != 40: raise ValueError('Expected EARS40')
    samples = []
    for speaker in speakers:
        ref = lookup[speaker + '_sentences_15_regular']
        target = lookup[speaker + '_sentences_16_regular']
        reference = run / 'references24' / (speaker + '.wav')
        natural = run / 'natural16' / (speaker + '.wav')
        reference.parent.mkdir(parents=True, exist_ok=True)
        natural.parent.mkdir(parents=True, exist_ok=True)
        for row, path, rate in [(ref, reference, 24000), (target, natural, 16000)]:
            if not path.exists():
                x, sr = sf.read(dataset / row['path'], dtype='float64')
                sf.write(path, mono_resample(x, sr, rate), rate, subtype='PCM_16')
        info = sf.info(reference)
        if not 5 <= info.duration <= 30: raise ValueError('Reference outside frozen duration limits')
        samples.append(dict(speaker_id=speaker, gender=ref['gender'], reference=str(reference),
                            reference_original=str(dataset / ref['path']), natural=str(natural),
                            natural_original=str(dataset / target['path']), text=english_text(target['text']),
                            reference_text=english_text(ref['text']), reference_duration_s=info.duration,
                            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
    protocol = dict(model=MODEL, dataset=str(dataset), samples=samples, speakers=40, draws_per_speaker=1,
                    reference_key='sentences_15_regular', target_key='sentences_16_regular',
                    language='en', enable_preprocess=False, enable_volume_normalization='false',
                    max_prompt_audio_length=30.0, amplitude_normalization=False, time_editing=False,
                    provider_original_bytes_retained=True, natural_canonical_sr=16000, reference_sr=24000,
                    output_sr_requested=24000, selection='All 40 acquired speakers, one fixed prompt pair; no score selection')
    path = run / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Existing protocol differs')
    save(path, protocol)
    return protocol

def generate(protocol, run, key):
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    def clean(value): return str(value).replace(key, '[REDACTED]')
    def post(endpoint, payload, name):
        response = requests.post(BASE + endpoint, json=payload, headers=headers, timeout=(15, 300))
        try: body = response.json()
        except ValueError: body = {'message': clean(response.text[:500])}
        receipt = dict(endpoint=BASE + endpoint, model=payload['model'], http_status=response.status_code,
                       request_id=body.get('request_id'), usage=body.get('usage'), code=body.get('code'),
                       message=clean(body.get('message', '')), requested_once=True)
        save(run / 'api_receipts' / (name + '.json'), receipt)
        if response.status_code != 200 or body.get('code'): raise RuntimeError(json.dumps(receipt))
        return body
    def worker(row):
        speaker = row['speaker_id']
        record_path = run / 'records' / (speaker + '.json')
        if record_path.exists(): return json.loads(record_path.read_text())
        voice_path = run / 'voices' / (speaker + '.json')
        if voice_path.exists():
            voice_record = json.loads(voice_path.read_text())
            if voice_record['reference_sha256'] != row['reference_sha256']: raise ValueError('Voice reference changed')
            voice = voice_record['voice_id']
        else:
            ref_bytes = Path(row['reference']).read_bytes()
            body = post('/services/audio/tts/customization', dict(model='voice-enrollment', input=dict(
                action='create_voice', target_model=MODEL, prefix='ears' + speaker,
                audio='data:audio/wav;base64,' + base64.b64encode(ref_bytes).decode('ascii'),
                language_hints=['en'], enable_preprocess=False, enable_volume_normalization='false',
                max_prompt_audio_length=30.0)), 'enroll_' + speaker)
            voice = body['output']['voice_id']
            save(voice_path, dict(voice_id=voice, reference_sha256=row['reference_sha256'], request_id=body.get('request_id')))
        request_name = 'synthesis_' + speaker
        prior_receipt = run / 'api_receipts' / (request_name + '.json')
        if prior_receipt.exists():
            previous = json.loads(prior_receipt.read_text())
            if previous.get('http_status') == 429 and previous.get('code') == 'Throttling.RateQuota':
                prior_receipt.rename(prior_receipt.with_name(request_name + '_rejected429_' + str(time.time_ns()) + '.json'))
                time.sleep(10)
            else:
                raise RuntimeError('Synthesis was already issued but record missing; inspect before another paid request: ' + speaker)
        body = post('/services/audio/tts/SpeechSynthesizer', dict(model=MODEL, input=dict(
                    text=row['text'], voice=voice, format='wav', sample_rate=24000)), request_name)
        audio = body['output']['audio']
        url = audio['url'] if isinstance(audio, dict) else audio
        provider = run / 'provider_audio' / (speaker + '.wav')
        provider.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                downloaded = requests.get(url, timeout=(15, 180)); downloaded.raise_for_status()
                provider.write_bytes(downloaded.content)
                x, sr = sf.read(provider, dtype='float64')
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(2)
        canonical = run / 'tts16' / (speaker + '.wav')
        canonical.parent.mkdir(parents=True, exist_ok=True)
        sf.write(canonical, mono_resample(x, sr, 16000), 16000, subtype='PCM_16')
        u = urlsplit(url)
        result = dict(speaker_id=speaker, gender=row['gender'], model=MODEL, voice_id=voice,
                      reference=row['reference'], reference_sha256=row['reference_sha256'], natural=row['natural'],
                      text=row['text'], provider_audio=str(provider), tts16=str(canonical),
                      provider_info=dict(sr=sr, frames=len(x), duration_s=len(x)/sr),
                      provider_sha256=hashlib.sha256(downloaded.content).hexdigest(),
                      request_id=body.get('request_id'), usage=body.get('usage'),
                      url_without_query=urlunsplit((u.scheme,u.netloc,u.path,'','')),
                      normalization=False, time_editing=False, draws=1)
        save(record_path, result)
        return result
    records, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(worker, row): row['speaker_id'] for row in protocol['samples']}
        for future in concurrent.futures.as_completed(futures):
            speaker = futures[future]
            try:
                records.append(future.result())
                print(json.dumps(dict(event='complete', speaker_id=speaker, completed=len(records), planned=40)), flush=True)
            except Exception as error:
                failures.append(dict(speaker_id=speaker, error=clean(error)))
                print(json.dumps(dict(event='failed', speaker_id=speaker, error=clean(error))), flush=True)
            save(run / 'status.json', dict(status='RUNNING', completed=len(records), planned=40, failures=failures))
    save(run / 'audio_manifest.json', dict(records=sorted(records, key=lambda r:r['speaker_id']), failures=failures))
    final = dict(status='COMPLETE' if len(records)==40 and not failures else 'INCOMPLETE', completed=len(records),
                 planned=40, failures=failures, key_persisted=False, no_TFG=True, no_scoring=True)
    save(run / 'status.json', final)
    print(json.dumps(final), flush=True)
    return 0 if final['status']=='COMPLETE' else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    protocol = prepare(args.dataset, args.run)
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED', speakers=len(protocol['samples']), run=str(args.run))))
        return 0
    key = sys.stdin.readline().strip()
    if not key: raise RuntimeError('Missing runtime key on stdin')
    try: return generate(protocol, args.run, key)
    finally: key = ''

if __name__ == '__main__': raise SystemExit(main())
