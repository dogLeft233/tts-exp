"""Independent CPU checks of score evidence and source-speaker arithmetic.

Does not import the runner, generator, or analyzer. No model forwards.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import pickle
import re
import wave

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def frame_hashes(path):
    return [line.split(',')[-1].strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def audit(run):
    index = read(run / 'comparison_index.json')
    protocol = read(run / 'protocol.json')
    result = read(run / 'analysis.json')
    errors, checked, sources, score_values = [], [], {}, {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    seal = read(run / 'root_input_seal.json')
    for name, expected_sha in seal['files'].items():
        require(sha(run / name) == expected_sha, 'pre-score root input seal changed: ' + name)
    require(result['analyzer_sha256'] == seal['analyzer_sha256'], 'pre-score analyzer seal changed')
    for row in index['records']:
        key = (row['sample_id'], row['model'], row['level'], row['draw_index'])
        label = '/'.join(map(str, key))
        start_errors = len(errors)
        try:
            score = read(row['score_path'])
            require(sha(row['score_path']) == row['score_sha256'], label + ': score SHA')
            require(score['status'] == 'COMPLETE', label + ': incomplete score')
            require(score['sample_id'] == row['sample_id'], label + ': sample binding')
            if row['level'] == 'norm' and row['model'] != 'L17':
                expected_role = 'N' if row['model'] == 'N' else ('T' if row['model'] == 'L06' else 'C') + str(row['draw_index'])
            else:
                expected_role = row['model'] + '_' + row['level'] + (str(row['draw_index']) if row['draw_index'] else '')
            require(score['role'] == expected_role, label + ': model/level/draw receipt binding')
            require(score['speaker_id'] == row['speaker_id'], label + ': speaker binding')
            require(sha(row['audio_path']) == row['audio_sha256'], label + ': index audio SHA')
            require(score['canonical_audio_sha256'] == row['audio_sha256'], label + ': score audio binding')
            require(sha(score['canonical_audio_path']) == row['audio_sha256'], label + ': actual score audio SHA')
            require(sha(score['video_path']) == score['source_video_sha256'], label + ': video SHA')
            require(sha(score['input_mkv']) == score['input_mkv_sha256'], label + ': input MKV SHA')
            with wave.open(str(row['audio_path']), 'rb') as f:
                require((f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000), label + ': audio format')
                pcm = f.readframes(f.getnframes())
                require(f.getnframes() == score['audio_samples'], label + ': audio sample count')
            require(hashlib.sha256(pcm).hexdigest() == score['PCM_sha256'], label + ': PCM SHA')
            require(all(score[k] is True for k in ('decoded_frames_exact', 'frame_PTS_exact', 'mux_PCM_exact')),
                    label + ': parity flags')
            checks = score['checks']
            for path, expected in checks.items():
                require(sha(path) == expected, label + ': parity evidence SHA ' + path)
            checkpaths = {Path(p).name: Path(p) for p in checks}
            a, b = frame_hashes(checkpaths['source.framemd5']), frame_hashes(checkpaths['mux.framemd5'])
            require(bool(a) and a == b, label + ': decoded frames differ')
            clocks = []
            for name in ('source_frames.json', 'mux_frames.json'):
                clocks.append([Decimal(f['best_effort_timestamp_time']) for f in read(checkpaths[name])['frames']])
            require(bool(clocks[0]) and clocks[0] == clocks[1], label + ': PTS differ')
            require(checkpaths['mux_PCM.s16le'].read_bytes() == pcm, label + ': remux PCM differs')
            stage_receipts = [v for k, v in score.items() if isinstance(v, dict) and 'returncode' in v and 'command' in v]
            require(len(stage_receipts) >= 3, label + ': stage receipt coverage')
            for receipt in stage_receipts:
                require(receipt['returncode'] == 0 and not receipt.get('timed_out'), label + ': stage exit')
                require(sha(receipt['log']) == receipt['log_sha256'], label + ': stage log SHA')
            log = Path(score['syncnet_log']).read_text(encoding='utf8')
            cs = [float(v) for v in re.findall(r'\bConfidence:\s*([-+\d.eE]+)', log)]
            ds = [float(v) for v in re.findall(r'\bMin dist:\s*([-+\d.eE]+)', log)]
            os = [int(v) for v in re.findall(r'\bAV offset:\s*(-?\d+)', log)]
            require(bool(cs) and len(cs) == len(ds) == len(os), label + ': printed track triples')
            require([score['sync_c'], score['sync_d'], score['av_offset']] == [cs[0], ds[0], os[0]],
                    label + ': first printed track')
            require(sha(score['activesd_path']) == score['activesd_sha256'], label + ': distance SHA')
            require(sha(score['tracks_pickle_path']) == score['tracks_pickle_sha256'], label + ': tracks SHA')
            matrices = pickle.loads(Path(score['activesd_path']).read_bytes())
            tracks = pickle.loads(Path(score['tracks_pickle_path']).read_bytes())
            require(len(matrices) == len(tracks) == len(cs), label + ': track count')
            max_error = 0.
            for i, matrix in enumerate(matrices):
                x = np.asarray(matrix, dtype=np.float64)
                require(x.ndim == 2 and x.shape[1] == 31 and np.isfinite(x).all(), label + ': matrix dimensions')
                curve = x.mean(axis=0)
                minimum = int(curve.argmin())
                max_error = max(max_error, abs(float(np.median(curve) - curve[minimum]) - cs[i]),
                                abs(float(curve[minimum]) - ds[i]))
                require(15 - minimum == os[i], label + ': raw offset')
                require(0 < len(x) <= len(tracks[i]['track']['frame']) - 5, label + ': support')
            require(max_error <= .00051, label + ': raw C/D calculation')
            score_values[key] = [score['sync_c'], score['sync_d'], score['av_offset']]
            sources[label] = row['score_sha256']
            checked.append({'cell': key, 'PASS': len(errors) == start_errors, 'tracks': len(cs),
                            'max_print_error': max_error})
        except Exception as exc:
            errors.append(label + ': ' + type(exc).__name__ + ': ' + str(exc))
    sample_ids = [p['sample_id'] for p in protocol['samples']]
    require(len(sample_ids) == len(set(sample_ids)) == 10, 'cohort sample coverage')
    models = ['N', 'L06', 'L17', 'CLOUD']
    levels = ['norm', 'raw']
    expected = {(sid, model, level, draw) for sid in sample_ids for model in models for level in levels
                for draw in ([0] if model == 'N' else [1, 2, 3])}
    require(set(score_values) == expected and len(index['records']) == 200, 'all200 distinct complete planned cells')
    math_error, summaries, supplementary_summaries = 0., 0, 0
    if set(score_values) == expected:
        # Different algorithm: bootstrap counts times person values, not index/mean.
        picks = np.random.Generator(np.random.PCG64(20261023)).integers(0, 10, (10000, 10))
        frequencies = np.asarray([np.bincount(row, minlength=10) for row in picks], dtype=float) / 10
        means = {}
        for mi, metric in enumerate(['C', 'D', 'offset']):
            for level in levels:
                for model in models:
                    means[metric, level, model] = np.asarray([
                        sum(score_values[sid, model, level, d][mi] for d in ([0] if model == 'N' else [1, 2, 3])) /
                        (1 if model == 'N' else 3) for sid in sample_ids])
        pairs = {'L17-N': ('L17', 'N'), 'L17-L06': ('L17', 'L06'), 'L17-CLOUD': ('L17', 'CLOUD'),
                 'L06-N': ('L06', 'N'), 'CLOUD-N': ('CLOUD', 'N'), 'CLOUD-L06': ('CLOUD', 'L06')}

        def compare(values, published):
            nonlocal math_error, summaries
            boot = frequencies @ values
            calculated = [float(sum(values) / 10), *np.percentile(boot, [2.5, 97.5, 100 * .05 / 6, 100 * (1 - .05 / 6)])]
            saved = [published['mean'], *published['ci95'], *published['ci98_333']]
            math_error = max(math_error, float(np.max(np.abs(np.asarray(calculated) - saved))))
            require(published['n'] == 10 and published['positive'] == int((values > 0).sum()), 'speaker n/positive count')
            require(np.max(np.abs(values - np.asarray(published['values']))) <= 1e-12, 'per-speaker values')
            summaries += 1

        for metric in ['C', 'D', 'offset']:
            for level in levels:
                block = result['by_level'][metric][level]
                for model in models:
                    compare(means[metric, level, model], block['groups'][model])
                for label, (a, b) in pairs.items():
                    compare(means[metric, level, a] - means[metric, level, b], block['contrasts'][label])
            for model in models:
                compare(means[metric, 'norm', model] - means[metric, 'raw', model], result['norm_minus_raw'][metric][model])
            for label, (a, b) in pairs.items():
                values = means[metric, 'norm', a] - means[metric, 'norm', b] - means[metric, 'raw', a] + means[metric, 'raw', b]
                compare(values, result['contrast_norm_minus_raw'][metric][label])
        require(math_error <= 1e-12, 'bootstrap arithmetic mismatch')
        supplementary = read(run / 'reliability_sensitivity.json')
        sensitivity_plan = read(run / 'analysis_sensitivity_protocol.json')
        require(supplementary['sensitivity_protocol_sha256'] == sha(run / 'analysis_sensitivity_protocol.json'),
                'supplement pre-score protocol binding')
        source_metadata = {}
        for path, expected_sha in supplementary['metadata_bindings'].items():
            require(sha(path) == expected_sha, 'old generation metadata SHA')
            for r in read(path)['records']:
                source_metadata[r['sample_id'], 'L06', r['seed_index']] = r
        for r in read(protocol.get('generation_source_manifest_path', run / 'generation/tts_manifest.json'))['records']:
            source_metadata[r['sample_id'], 'L17', r['draw_index']] = r
        allowed = {}
        published_qc = {(r['sample_id'], r['model'], r['draw_index']): r for r in supplementary['qualifications']}
        require(len(published_qc) == 60, 'reliability draw coverage')
        sample_by_id = {p['sample_id']: p for p in protocol['samples']}
        for sid in sample_ids:
            for model in ['L06', 'L17']:
                allowed[sid, model] = []
                for draw in [1, 2, 3]:
                    r = source_metadata[sid, model, draw]
                    sequence = r['talker_sequences']
                    require(sha(sequence['path']) == sequence['sha256'], 'EOS sequence SHA')
                    actual_eos = bool(np.any(np.load(sequence['path']) == r['talker_effective_kwargs']['eos_token_id']))
                    require(actual_eos == r['eos_in_talker_sequences'], 'EOS metadata versus saved sequence')
                    duration_ratio = r['samples'] / r['sample_rate_hz'] / sample_by_id[sid]['duration_s']
                    q = published_qc[sid, model, draw]
                    require(q['EOS'] == actual_eos and abs(q['duration_ratio'] - duration_ratio) <= 1e-12
                            and q['qualified'] == (actual_eos and .5 <= duration_ratio <= 2), 'published reliability qualification')
                    if actual_eos and .5 <= duration_ratio <= 2:
                        allowed[sid, model].append(draw)

        def supplementary_compare(values, published):
            nonlocal math_error, supplementary_summaries
            v = np.asarray(values, dtype=float)
            require(published['n'] == len(v), 'supplement n')
            if len(v):
                picks = np.random.Generator(np.random.PCG64(20261023)).integers(0, len(v), (10000, len(v)))
                freq = np.asarray([np.bincount(row, minlength=len(v)) for row in picks], dtype=float) / len(v)
                boot = freq @ v
                actual = [float(sum(v) / len(v)), *np.percentile(boot, [2.5, 97.5, .05 / 6 * 100, (1 - .05 / 6) * 100])]
                saved = [published['mean'], *published['ci95'], *published['ci98_333']]
                math_error = max(math_error, float(np.max(np.abs(np.asarray(actual) - saved))))
                require(np.max(np.abs(v - published['values'])) < 1e-12, 'supplement per-person values')
                require(int((v > 0).sum()) == published['positive'], 'supplement positive count')
            supplementary_summaries += 1

        qualified = {}
        for sid in sample_ids:
            for level in levels:
                for model in models:
                    ds = [0] if model == 'N' else [1, 2, 3] if model == 'CLOUD' else allowed[sid, model]
                    qualified[sid, level, model] = (sum(score_values[sid, model, level, d][0] for d in ds) / len(ds)) if ds else None
        for level in levels:
            for label, (a, b) in pairs.items():
                values = [qualified[sid, level, a] - qualified[sid, level, b] for sid in sample_ids
                          if qualified[sid, level, a] is not None and qualified[sid, level, b] is not None]
                supplementary_compare(values, supplementary['contrasts'][level][label])
        keep = np.asarray([sample_by_id[sid]['speaker_id'] != 'S0016' for sid in sample_ids])
        require(keep.sum() == 9 and supplementary['strict9']['n'] == 9, 'strict9 speaker support')
        for level in levels:
            for model in models:
                supplementary_compare(means['C', level, model][keep], supplementary['strict9']['groups'][level][model])
            for label, (a, b) in pairs.items():
                supplementary_compare((means['C', level, a] - means['C', level, b])[keep], supplementary['strict9']['contrasts'][level][label])
        for model in models:
            supplementary_compare((means['C', 'norm', model] - means['C', 'raw', model])[keep], supplementary['strict9']['norm_minus_raw'][model])
        for label, (a, b) in pairs.items():
            v = means['C', 'norm', a] - means['C', 'norm', b] - means['C', 'raw', a] + means['C', 'raw', b]
            supplementary_compare(v[keep], supplementary['strict9']['contrast_norm_minus_raw'][label])
        require(math_error <= 1e-12, 'supplementary bootstrap arithmetic mismatch')
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'score_cells_checked': len(checked), 'planned_cells': len(expected),
            'summaries_checked': summaries, 'max_arithmetic_error': math_error,
            'supplementary_summaries_checked': supplementary_summaries,
            'tracks_checked': sum(r['tracks'] for r in checked), 'checks': checked,
            'analysis_sha256': sha(run / 'analysis.json'), 'index_sha256': sha(run / 'comparison_index.json'),
            'protocol_sha256': sha(run / 'protocol.json'), 'audit_source_sha256': sha(__file__),
            'limitations': ['Independent CPU audit of saved media evidence/distances; no second stochastic generation.',
                            'No human synchronization or lip-accuracy evaluation.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.run)
    with (args.run / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
