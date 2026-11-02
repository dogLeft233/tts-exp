"""Freeze metadata-selected natural inputs before Qwen codec inference."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import wave
import datetime


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root = Path('/home/wjj/tts-audio/tts-exp')
    run = root / 'runs/qwen_codec_roundtrip_20261022_attempt01'
    run.mkdir(exist_ok=False)
    base = root / 'data/native_mechanism_holdout_20260926'
    model = Path('/home/wjj/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/speech_tokenizer')
    samples = []
    seen = set()
    for p in sorted((base / 'audio').glob('*.wav')):
        sid = re.search(r'S\d{4}', p.stem).group()
        text = base / 'transcripts' / (p.stem + '.txt')
        if sid in seen or not text.is_file():
            continue
        with wave.open(str(p)) as w:
            duration = w.getnframes() / w.getframerate()
            source_meta = dict(samples=w.getnframes(), sample_rate_hz=w.getframerate(), channels=w.getnchannels())
        if not 3 <= duration <= 8:
            continue
        dst = run / 'input24' / (p.stem + '.wav')
        dst.parent.mkdir(exist_ok=True)
        command = ['/usr/bin/ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(p), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_f32le', str(dst)]
        subprocess.run(command, check=True, timeout=30)
        samples.append(dict(sample_id=p.stem, speaker_id=sid, source_path=str(p), source_sha256=sha(p),
                            transcript_path=str(text), transcript_sha256=sha(text), transcript=text.read_text().strip(),
                            duration_s=duration, source_meta=source_meta, input24_path=str(dst), input24_sha256=sha(dst),
                            conversion_command=command))
        seen.add(sid)
        if len(samples) == 10:
            break
    if len(samples) != 10:
        raise RuntimeError('not enough metadata-qualified distinct speakers')
    protocol = dict(run_id=run.name, status='FROZEN_BEFORE_CODEC_FORWARD', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    seed=20261022, model_dir=str(model), model_weights_sha256=sha(model/'model.safetensors'),
                    selection='sorted original utterance IDs; first existing transcript and duration 3..8s per speaker; first 10 speakers; no score-based selection',
                    scientific_scope='historically exposed AISHELL development pilot; no upstream training-overlap exclusion',
                    samples=samples,
                    codec=dict(all_codebooks=16, dtype='float32', attention='eager', load='offline tokenizer only', text_or_LM_generation=False,
                               preserve_raw_decoded_length=True, output_format='24k mono FLOAT WAV', alignment='none; no time stretch, trim, imposed equal length, or SyncNet-derived delay correction'),
                    primary_audio=dict(N='input24 sham conversion only',R='same input24 -> Qwen tokenizer encode/decode',
                                       downstream='each independently /usr/bin/ffmpeg -> mono16k FLOAT -> same scalar RMS -> PCM16', target_rms=0.0376838172675746),
                    timing=dict(methods=['length audit','10ms energy/spectral-flux local lag diagnostics','independent Mandarin MFA original phone and silence boundaries'],
                                local_lag='no waveform warp; acoustic lag is proxy, not physical event ground truth',
                                MFA='on original native clock before and after subtracting first matched-phone shift; exact phone LCS; report all unmatched and unknown labels',
                                human_annotation='not performed by AI; output blind annotation packet, no human timing equivalence claim'),
                    generation=dict(model='Wav2Lip GAN', weights_sha256='ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8',
                                    face=str(root/'data/data/image/3.png'), face_sha256=sha(root/'data/data/image/3.png'), score_box_xyxy=[33,18,462,447],
                                    fixed_face=True, conditions=['N','R'], max_outputs=20, fps=25, options=['--nosmooth','--wav2lip_batch_size 4']),
                    scoring=dict(layout='2x2 for every source; source PCM rather than mux AAC', lags=list(range(-15,16)),
                                 common_support='intersection of all N/R video/audio receptive-field windows and every lag; no embedding padding',
                                 endpoints=['C','B','Dmin','D0','offset','full_distance_curve'],
                                 contrasts=['VR_AR-VR_AN: fixed reconstructed video audio swap', 'VN_AR-VN_AN: fixed natural video audio swap',
                                            'VR_AN-VN_AN: fixed natural evaluation audio generator response','VR_AR-VN_AN: self-paired total response']),
                    statistics=dict(unit='speaker, 10 speakers', bootstrap_draws=10000, seed=20261022, CI='ordinary descriptive 95%; not confirmatory or perception calibrated',
                                    no_invariance_from_nonsignificance=True, no_frames_as_independent_samples=True),
                    controls=dict(input_hashes=True, no_clipping=True, full_raw_decoder_output=True, no_training=True,
                                  timing_shift='report; do not infer codec timing change from SyncNet alone',
                                  failure_policy='stop affected stage on nonfinite, missing input/output, clipping, source/model hash mismatch or insufficient common support; do not replace sample based on outputs'),
                    budgets=dict(codec_timeout_seconds=1800, tfg_timeout_seconds=1800, score_timeout_seconds=900, mfa_timeout_seconds=1800, storage_cap_bytes=1500000000))
    dest = run/'input_manifest.json'
    dest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run':str(run),'input_manifest':str(dest),'manifest_sha256':sha(dest),'speakers':sorted(seen),'source_count':len(samples)},ensure_ascii=False))


if __name__ == '__main__':
    main()
