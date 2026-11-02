"""Frozen HDTF ASR -> self-cloned Plus TTS -> ASR QC; secret via getpass only."""
from __future__ import annotations
import argparse
import base64
import getpass
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ASR = 'qwen-audio-3.1-asr-flash'
TTS = 'qwen-audio-3.1-tts-plus'
BASE = 'https://dashscope.aliyuncs.com/api/v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def word_error(reference, hypothesis):
    a, b = [re.findall(r"[a-z0-9']+", x.lower()) for x in (reference, hypothesis)]
    distances = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, distances[j] + 1, distances[j - 1] + (x != y)))
        distances = new
    return distances[-1] / max(1, len(a))


def extract_text(body):
    output = body['output']
    text = output.get('text') or output.get('output', {}).get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('No nonempty ASR text in response')
    return ' '.join(text.split())


def main():
    import imageio_ffmpeg
    import numpy as np
    import requests
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    dataset, run = args.dataset.resolve(), args.run.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE' and len(manifest['records']) == 40
    assert len({r['speaker'] for r in manifest['records']}) == 20
    run.mkdir(parents=True, exist_ok=True)
    protocol = dict(dataset=str(dataset), dataset_manifest_sha256=sha(dataset / 'manifest.json'),
        asr_model=ASR, tts_model=TTS, clips=40, speakers=20, tts_draws_per_clip=1,
        language='en', voice_reference='same ten-second natural clip; no content-based replacement',
        enable_preprocess=False, enable_volume_normalization='false', max_prompt_audio_length=30.,
        target_rms_dbfs=-24, speaker_unit='mean of two clip scores per arm, then paired twenty speakers',
        image='each segment start frame, identical within N/T pair',
        boundary_policy='frozen ten-second windows may contain partial words/sentences; ASR is not human ground truth',
        qc='ASR on TTS with requested ASR model; flag WER>.20 or duration outside [.5,2]x; do not silently exclude/resample',
        paid_retry='no automatic repeat of uncertain/failed POST; verify saved responses before resuming',
        tts_parameters='text, voice, format=wav, sample_rate=24000; no unsupported seed/rate/pitch fields')
    pp = run / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        save(pp, protocol)
    key = getpass.getpass('Runtime DashScope key (not saved): ')
    if not key:
        raise ValueError('Missing runtime credential')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    started = time.monotonic()

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            value = value.replace(key, '[REDACTED]')
            if value.startswith('https://'):
                u = urlsplit(value)
                value = urlunsplit((u.scheme, u.netloc, u.path, '', ''))
        return value

    def event(stage, **kwargs):
        payload = dict(stage=stage, elapsed_s=round(time.monotonic()-started, 2), **kwargs)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        with (run / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def post(route, payload, name, binding):
        path = run / 'responses' / (name + '.json')
        attempt = run / 'requests' / (name + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['binding'] == binding and saved['http_status'] == 200 and not saved['body'].get('code')
            return saved['body']
        if attempt.exists():
            raise RuntimeError('Uncertain prior paid request, inspect before repeating: ' + name)
        save(attempt, dict(model=payload['model'], route=route, binding=binding, started=time.time(),
                           request_metadata_only=True, payload_audio_omitted=True))
        response = requests.post(BASE + route, headers=headers, json=payload, timeout=(20, 300))
        body = response.json()
        save(path, dict(http_status=response.status_code, binding=binding, body=clean(body)))
        if response.status_code != 200 or body.get('code'):
            raise RuntimeError('API failure ' + name + ': ' + str(response.status_code) + ' ' + str(body.get('code')))
        return body

    def asr(path, name):
        import wave
        with wave.open(str(path)) as wav:
            rate = wav.getframerate()
        body = post('/services/aigc/multimodal-generation/generation',
            dict(model=ASR, input=dict(messages=[dict(role='user', content=[dict(type='input_audio',
                input_audio=dict(data='data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode()))])]),
                parameters=dict(format='wav', sample_rate=str(rate), language_hints=['en'])),
            name, dict(audio_sha256=sha(path), model=ASR))
        return extract_text(body)

    try:
        transcripts = []
        for row in manifest['records']:
            sid = row['id']
            natural = dataset / row['files']['audio']['path']
            assert sha(natural) == row['files']['audio']['sha256']
            text = asr(natural, 'natural_' + sid)
            transcripts.append(dict(id=sid, speaker=row['speaker'], text=text, natural=str(natural),
                natural_sha256=sha(natural), image=str(dataset / row['files']['image']['path']),
                image_sha256=row['files']['image']['sha256'], real_video=str(dataset / row['files']['video']['path']),
                real_video_sha256=row['files']['video']['sha256'], source_start_s=row['start_s']))
            save(run / 'transcripts.json', dict(status='COMPLETE' if len(transcripts)==40 else 'RUNNING',
                 model=ASR, human_verified=False, rows=transcripts))
            event('ASR', id=sid, complete=len(transcripts), planned=40)
        records = []
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for number, row in enumerate(transcripts, 1):
            sid, natural = row['id'], Path(row['natural'])
            record_path = run / 'records' / (sid + '.json')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                assert record['natural_sha256'] == row['natural_sha256']
                assert sha(record['tts16']) == record['tts16_sha256']
                records.append(record)
                continue
            reference = run / 'reference24' / (sid + '.wav')
            reference.parent.mkdir(exist_ok=True)
            if not reference.exists():
                subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-n', '-i', str(natural),
                                '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', str(reference)], check=True)
            voice_body = post('/services/audio/tts/customization', dict(model='voice-enrollment', input=dict(
                action='create_voice', target_model=TTS, prefix=f'hdtf31{number:03d}',
                audio='data:audio/wav;base64,' + base64.b64encode(reference.read_bytes()).decode(),
                language_hints=['en'], enable_preprocess=False, enable_volume_normalization='false',
                max_prompt_audio_length=30.)), 'voice_' + sid, dict(reference_sha256=sha(reference), model=TTS))
            voice = voice_body['output']['voice_id']
            provider = run / 'provider_audio' / (sid + '.wav')
            payload = dict(model=TTS, input=dict(text=row['text'], voice=voice, format='wav', sample_rate=24000))
            response_path = run / 'responses' / ('tts_' + sid + '.json')
            if response_path.exists() and not provider.exists():
                raise RuntimeError('Prior successful synthesis has no downloaded media; inspect without another paid POST: ' + sid)
            body = post('/services/audio/tts/SpeechSynthesizer', payload, 'tts_' + sid,
                        dict(text=row['text'], reference_sha256=sha(reference), voice_id=voice, model=TTS))
            if not provider.exists():
                audio = body['output']['audio']
                url = audio['url'] if isinstance(audio, dict) else audio
                provider.parent.mkdir(exist_ok=True)
                for attempt in range(3):
                    try:
                        response = requests.get(url, timeout=(20, 180))
                        response.raise_for_status()
                        temporary = provider.with_suffix('.part')
                        temporary.write_bytes(response.content)
                        sf.info(temporary)
                        temporary.replace(provider)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2)
            raw, rate = sf.read(provider, dtype='float64', always_2d=True)
            assert rate == 24000 and raw.shape[1] == 1 and np.isfinite(raw).all()
            tts16 = run / 'tts16' / (sid + '.wav')
            tts16.parent.mkdir(exist_ok=True)
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-y', '-i', str(provider),
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(tts16)], check=True)
            hypothesis = asr(provider, 'tts_qc_' + sid)
            wer = word_error(row['text'], hypothesis)
            ratio = len(raw)/rate/10
            record = dict(**row, asr_model=ASR, tts_model=TTS, voice_id=voice,
                reference=str(reference), reference_sha256=sha(reference), provider_audio=str(provider),
                provider_sha256=sha(provider), provider_sr=rate, provider_samples=len(raw),
                tts16=str(tts16), tts16_sha256=sha(tts16), tts_asr=hypothesis, word_error_rate=wer,
                duration_ratio=ratio, qc_flag=bool(wer>.2 or not .5<=ratio<=2),
                normalization=False, time_editing=False, human_verified=False)
            save(record_path, record)
            records.append(record)
            save(run / 'audio_manifest.json', dict(status='COMPLETE' if len(records)==40 else 'RUNNING', rows=records))
            event('TTS_AND_QC', id=sid, complete=len(records), planned=40, wer=wer, seconds=len(raw)/rate,
                  flagged=record['qc_flag'])
        assert len(records)==40
        # Verify the provided credential was never persisted anywhere in this run.
        assert all(key.encode() not in path.read_bytes() for path in run.rglob('*') if path.is_file())
        save(run / 'cloud_complete.json', dict(status='PASS', asr=ASR, tts=TTS, pairs=40, speakers=20,
            qc_flagged=[r['id'] for r in records if r['qc_flag']], api_key_persisted=False,
            cloud_requests=len(list((run/'responses').glob('*.json'))), no_TFG=True))
        event('CLOUD_COMPLETE', pairs=40, speakers=20)
    except BaseException as error:
        save(run / 'failure.json', dict(type=type(error).__name__, error=clean(str(error)), time=time.time()))
        print('CLOUD_FAILED', clean(str(error)), flush=True)
        raise SystemExit(1)
    finally:
        headers.clear()
        key = ''


if __name__ == '__main__':
    main()
