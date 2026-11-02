"""Pre-score supplementary reliability analysis; no scientific re-generation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    code = Path(__file__).with_name('qwen17_level_comparison10_analysis_20261023.py')
    spec = importlib.util.spec_from_file_location('frozen_summarizer', code)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    protocol = read(run / 'protocol.json')
    addendum = read(run / 'analysis_sensitivity_protocol.json')
    if sha(__file__) != addendum['source_sha256']:
        raise ValueError('Sensitivity source differs from pre-score seal')
    parent = Path(protocol['model_dir']).parents[1]
    local06_paths = [parent / 'runs/qwen_full_tts_extension_20261022_attempt01/tts/tts_manifest.json',
                     parent / 'runs/qwen_nrt_expansion40_20261022_attempt01/generation/tts_manifest.json']
    metadata, bindings = {}, {}
    for path in local06_paths:
        bindings[str(path)] = sha(path)
        for r in read(path)['records']:
            metadata[r['sample_id'], 'L06', r['seed_index']] = r
    for r in read(protocol.get('generation_source_manifest_path', run / 'generation/tts_manifest.json'))['records']:
        metadata[r['sample_id'], 'L17', r['draw_index']] = r
    index = read(run / 'comparison_index.json')['records']
    cells = {(r['sample_id'], r['model'], r['level'], r['draw_index']): r for r in index}
    qc, people = [], []
    for sample in protocol['samples']:
        sid = sample['sample_id']
        person = dict(sample_id=sid, speaker_id=sample['speaker_id'], qualified={}, values={})
        for model in ['L06', 'L17']:
            valid = []
            for draw in [1, 2, 3]:
                r = metadata[sid, model, draw]
                source = cells[sid, model, 'raw', draw]
                if sha(source['source_path']) != source['source_sha256'] or source['source_sha256'] != r['sha256']:
                    raise ValueError('Generation metadata source binding mismatch')
                ratio = r['samples'] / r['sample_rate_hz'] / sample['duration_s']
                qualified = r['eos_in_talker_sequences'] is True and .5 <= ratio <= 2.
                if qualified:
                    valid.append(draw)
                qc.append(dict(sample_id=sid, model=model, draw_index=draw, EOS=r['eos_in_talker_sequences'],
                               duration_ratio=ratio, qualified=qualified, seed=r['seed']))
            person['qualified'][model] = valid
        for level in ['norm', 'raw']:
            person['values'][level] = {}
            for model in ['N', 'L06', 'L17', 'CLOUD']:
                draws = [0] if model == 'N' else [1, 2, 3] if model == 'CLOUD' else person['qualified'][model]
                scores = [read(cells[sid, model, level, d]['score_path']) for d in draws]
                if any(r['status'] != 'COMPLETE' for r in scores):
                    raise ValueError('Missing score in reliability supplement')
                person['values'][level][model] = float(np.mean([r['sync_c'] for r in scores])) if scores else None
        people.append(person)
    comparisons = {}
    for level in ['norm', 'raw']:
        comparisons[level] = {}
        for name, (a, b) in analyzer.PAIRS.items():
            rows = [p for p in people if p['values'][level][a] is not None and p['values'][level][b] is not None]
            comparisons[level][name] = dict(analyzer.summarize([p['values'][level][a] - p['values'][level][b] for p in rows]),
                                             sample_ids=[p['sample_id'] for p in rows])
    main_result = read(run / 'analysis.json')
    strict_people = [p for p in main_result['records'] if p['speaker_id'] != 'S0016']
    strict = {'excluded_speaker': 'S0016', 'reason': 'Exclude whole source speaker containing one raw eight-point PCM16 exception',
              'n': len(strict_people), 'groups': {}, 'contrasts': {}, 'norm_minus_raw': {}, 'contrast_norm_minus_raw': {}}
    for level in ['norm', 'raw']:
        strict['groups'][level] = {m: analyzer.summarize([p['values'][level][m]['C'] for p in strict_people])
                                   for m in ['N', 'L06', 'L17', 'CLOUD']}
        strict['contrasts'][level] = {name: analyzer.summarize([p['values'][level][a]['C'] - p['values'][level][b]['C']
                                                               for p in strict_people]) for name, (a, b) in analyzer.PAIRS.items()}
    strict['norm_minus_raw'] = {m: analyzer.summarize([p['values']['norm'][m]['C'] - p['values']['raw'][m]['C']
                                                     for p in strict_people]) for m in ['N', 'L06', 'L17', 'CLOUD']}
    strict['contrast_norm_minus_raw'] = {name: analyzer.summarize([
        (p['values']['norm'][a]['C'] - p['values']['norm'][b]['C']) -
        (p['values']['raw'][a]['C'] - p['values']['raw'][b]['C']) for p in strict_people]) for name, (a, b) in analyzer.PAIRS.items()}
    result = dict(status='DESCRIPTIVE_POSTTREATMENT_SENSITIVITY', records=people, qualifications=qc,
                  contrasts=comparisons, metadata_bindings=bindings,
                  strict9=strict,
                  sensitivity_protocol_sha256=sha(run / 'analysis_sensitivity_protocol.json'),
                  main_unchanged=True,
                  limitation='EOS/duration are treatment outcomes; conditional results do not replace all-draw main results.')
    with (run / 'reliability_sensitivity.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({model: sum(r['qualified'] for r in qc if r['model'] == model) for model in ['L06', 'L17']}))


if __name__ == '__main__':
    main()
