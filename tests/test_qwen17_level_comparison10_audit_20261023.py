import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import wave

import numpy as np

BASE = Path(__file__).parents[1] / 'scripts/experiments'


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


analyzer = module('qwen17_level_comparison10_analysis_20261023')
auditor = module('qwen17_level_comparison10_audit_20261023')


def test_saved_distance_first_track_parity_and_statistics_audited(tmp_path):
    def save(name, obj):
        path = tmp_path / name
        path.write_text(json.dumps(obj), encoding='utf8')
        return str(path)

    pcm = bytes(64)
    audio = tmp_path / 'audio.wav'
    with wave.open(str(audio), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(pcm)
    video, mux = tmp_path / 'video.mp4', tmp_path / 'input.mkv'
    video.write_bytes(b'media fixture')
    mux.write_bytes(b'mux fixture')
    matrices = np.full((10, 31), 4., dtype=np.float32)
    matrices[:, 15] = 2.
    distance = tmp_path / 'distance.pckl'
    distance.write_bytes(pickle.dumps([matrices]))
    tracks = tmp_path / 'tracks.pckl'
    tracks.write_bytes(pickle.dumps([{'track': {'frame': np.arange(16)}}]))
    log = tmp_path / 'sync.log'
    log.write_text('AV offset: 0\nMin dist: 2.000\nConfidence: 2.000\n')
    checks = {}
    for filename in ['source.framemd5', 'mux.framemd5']:
        p = tmp_path / filename
        p.write_text('# hash fixture\n0,0,0,1,4,abcd\n')
        checks[str(p)] = auditor.sha(p)
    for filename in ['source_frames.json', 'mux_frames.json']:
        p = save(filename, {'frames': [{'best_effort_timestamp_time': '0.000'}]})
        checks[p] = auditor.sha(p)
    p = tmp_path / 'mux_PCM.s16le'
    p.write_bytes(pcm)
    checks[str(p)] = auditor.sha(p)
    receipt = {'command': ['fixture'], 'returncode': 0, 'timed_out': False,
               'log': str(log), 'log_sha256': auditor.sha(log)}
    samples = [{'sample_id': f'utt{i}', 'speaker_id': 'S0016' if i == 1 else f'spk{i}', 'duration_s': 1.} for i in range(10)]
    index = {'records': []}
    for sample in samples:
        for level in analyzer.LEVELS:
            for model in analyzer.MODELS:
                for draw in ([0] if model == 'N' else [1, 2, 3]):
                    if level == 'norm' and model != 'L17':
                        role = 'N' if model == 'N' else ('T' if model == 'L06' else 'C') + str(draw)
                    else:
                        role = model + '_' + level + (str(draw) if draw else '')
                    score = dict(sample, status='COMPLETE', sync_c=2., sync_d=2., av_offset=0,
                                 role=role,
                                 canonical_audio_path=str(audio), canonical_audio_sha256=auditor.sha(audio),
                                 video_path=str(video), source_video_sha256=auditor.sha(video),
                                 input_mkv=str(mux), input_mkv_sha256=auditor.sha(mux),
                                 PCM_sha256=hashlib.sha256(pcm).hexdigest(), audio_samples=32,
                                 decoded_frames_exact=True, frame_PTS_exact=True, mux_PCM_exact=True,
                                 checks=checks, remux_receipt=receipt, pipeline_receipt=receipt,
                                 syncnet_receipt=receipt, syncnet_log=str(log),
                                 activesd_path=str(distance), activesd_sha256=auditor.sha(distance),
                                 tracks_pickle_path=str(tracks), tracks_pickle_sha256=auditor.sha(tracks))
                    path = save(f"{sample['sample_id']}_{level}_{model}_{draw}.json", score)
                    index['records'].append(dict(sample, level=level, model=model, draw_index=draw,
                                                 score_path=path, score_sha256=auditor.sha(path),
                                                 audio_path=str(audio), audio_sha256=auditor.sha(audio)))
    save('protocol.json', {'samples': samples})
    save('comparison_index.json', index)
    result = analyzer.analyze(index, samples)
    result['analyzer_sha256'] = auditor.sha(BASE / 'qwen17_level_comparison10_analysis_20261023.py')
    save('analysis.json', result)
    save('root_input_seal.json', {'files': {'protocol.json': auditor.sha(tmp_path / 'protocol.json')},
                                  'analyzer_sha256': result['analyzer_sha256']})
    sequence = tmp_path / 'talker.npy'
    np.save(sequence, np.asarray([[2150]], dtype=np.int64))
    records = [dict(sample, seed_index=d, draw_index=d, samples=24000, sample_rate_hz=24000,
                    talker_sequences={'path': str(sequence), 'sha256': auditor.sha(sequence)},
                    talker_effective_kwargs={'eos_token_id': 2150}, eos_in_talker_sequences=True)
               for sample in samples for d in [1, 2, 3]]
    meta = save('old_metadata.json', {'records': records})
    (tmp_path / 'generation').mkdir()
    save('generation/tts_manifest.json', {'records': records})
    plan = save('analysis_sensitivity_protocol.json', {'source_sha256': 'fixture'})
    summary = analyzer.summarize
    contrasts = {level: {name: summary([0.] * 10) for name in analyzer.PAIRS} for level in analyzer.LEVELS}
    strict = dict(n=9,
                  groups={level: {m: summary([2.] * 9) for m in analyzer.MODELS} for level in analyzer.LEVELS},
                  contrasts={level: {name: summary([0.] * 9) for name in analyzer.PAIRS} for level in analyzer.LEVELS},
                  norm_minus_raw={m: summary([0.] * 9) for m in analyzer.MODELS},
                  contrast_norm_minus_raw={name: summary([0.] * 9) for name in analyzer.PAIRS})
    save('reliability_sensitivity.json', dict(metadata_bindings={meta: auditor.sha(meta)},
         sensitivity_protocol_sha256=auditor.sha(plan), contrasts=contrasts, strict9=strict,
         qualifications=[dict(sample_id=s['sample_id'], model=m, draw_index=d, EOS=True, duration_ratio=1., qualified=True)
                         for s in samples for m in ['L06', 'L17'] for d in [1, 2, 3]]))
    actual = auditor.audit(tmp_path)
    assert actual['status'] == 'PASS', actual['errors']
    assert actual['score_cells_checked'] == 200
    assert actual['summaries_checked'] == 90
    assert actual['supplementary_summaries_checked'] == 42
    # Altering saved evidence must fail even if score receipts are untouched.
    log.write_text('AV offset: 0\nMin dist: 2.000\nConfidence: 3.000\n')
    altered = auditor.audit(tmp_path)
    assert altered['status'] == 'FAIL'
    assert any('first printed track' in s for s in altered['errors'])
    assert any('raw C/D calculation' in s for s in altered['errors'])
