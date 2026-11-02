"""Independent ASR family audit; never rewrites frozen text or media."""
import hashlib
from pathlib import Path
import importlib.metadata
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from calibrate import ROOT, read, write, sha

RUN = ROOT / 'runs/phoneme_direction_20261026_repair03'
MODEL = Path('/home/wjj/tts-audio/model_cache/whisper-small-20261026')


def main():
    x = read(ROOT / 'runs/phoneme_direction_20261026_repair02/full_peak95_quality.json')['failed'][0]
    assert x['id'] == 'BAC009S0168W0257_N_0'
    original = ROOT / 'runs/audio_data_inventory200_20261024_attempt04/natural/BAC009S0168W0257.wav'
    assert sha(x['source']) == x['source_sha256']
    files = {p.name: sha(p) for p in MODEL.iterdir() if p.is_file()}
    write(RUN / 'whisper_content_contract.json', {
        'source': x['source'], 'source_sha256': x['source_sha256'],
        'inventory_original': str(original), 'inventory_original_sha256': sha(original),
        'model_files_sha256': files, 'model': 'Systran/faster-whisper-small',
        'faster_whisper': importlib.metadata.version('faster-whisper'),
        'segments_seconds': [[0, 3], [3, 6], [6, 9.024]],
        'language': 'zh', 'beam_size': 5, 'temperature': 0, 'vad_filter': False,
        'condition_on_previous_text': False, 'raw_and_fixed_peak95': True,
        'script_sha256': sha(__file__), 'frozen_text_changed': False,
        'not_manual_ground_truth': True, 'development_allowed': False, 'confirmation_allowed': False})
    model = WhisperModel(str(MODEL), device='cpu', compute_type='int8', cpu_threads=4,
                         num_workers=1, local_files_only=True)
    a = decode_audio(x['source'], sampling_rate=16000)
    source = decode_audio(str(original), sampling_rate=16000)
    # Canonical conversion may have a final pad sample. Preserve both byte hashes.
    n = min(len(a), len(source))
    comparison = {'canonical_samples': len(a), 'inventory_samples': len(source),
                  'common_prefix_max_absolute_difference': float(abs(a[:n] - source[:n]).max()),
                  'common_prefix_correlation': float(np.corrcoef(a[:n], source[:n])[0, 1])}
    outputs = []
    for kind, waveform in [('canonical', a), ('inventory', source)]:
        spans = [(0., len(waveform) / 16000)]
        if kind == 'canonical': spans += [(0., 3.), (3., 6.), (6., len(waveform) / 16000)]
        for lo, hi in spans:
            clip = waveform[int(lo * 16000):int(round(hi * 16000))]
            for level in ['raw', 'peak95']:
                audio = clip if level == 'raw' else (clip * (.95 / max(float(abs(clip).max()), 1e-8))).astype('float32')
                segments, info = model.transcribe(audio, language='zh', beam_size=5,
                    temperature=0., vad_filter=False, condition_on_previous_text=False)
                seg = [dict(start=float(s.start), end=float(s.end), text=s.text,
                    avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob)) for s in segments]
                row = dict(kind=kind, level=level, start=lo, end=hi, segments=seg,
                           text=''.join(s['text'] for s in seg))
                outputs.append(row); print(row, flush=True)
    write(RUN / 'whisper_content_result.json', {
        'rows': outputs, 'original_comparison': comparison,
        'frozen_text': '十一月二十日消息', 'not_manual_ground_truth': True,
        'contract_sha256': sha(RUN / 'whisper_content_contract.json')})


if __name__ == '__main__': main()
