import importlib.util
import json
from pathlib import Path
import sys


def test_qualified_supplement_reports_missing_person_and_keeps_main(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / 'scripts/experiments/qwen17_level_sensitivity_20261023.py'
    spec = importlib.util.spec_from_file_location('q17_sensitivity', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / 'runs/current'
    run.mkdir(parents=True)
    original = tmp_path / 'original.wav'
    original.write_bytes(b'source provenance fixture')
    digest = mod.sha(original)
    samples = [dict(sample_id=f'utt{i}', speaker_id='S0016' if i == 1 else f'p{i}', duration_s=2.) for i in range(10)]
    old, new, index = [], [], []
    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf8')
    for person, sample in enumerate(samples):
        for model in ['N', 'L06', 'L17', 'CLOUD']:
            for draw in ([0] if model == 'N' else [1, 2, 3]):
                qualified = not (model == 'L17' and person == 0) and not (model == 'L06' and person == 1 and draw == 3)
                if model in ['L06', 'L17']:
                    metadata = dict(sample, seed_index=draw, draw_index=draw, seed=draw,
                                    eos_in_talker_sequences=qualified, samples=48000,
                                    sample_rate_hz=24000, path=str(original), sha256=digest)
                    (old if model == 'L06' else new).append(metadata)
                for level in ['raw', 'norm']:
                    path = run / f"score_{sample['sample_id']}_{model}_{level}_{draw}.json"
                    value = {'N': 7., 'L06': 8. if qualified else 0., 'L17': 8.5, 'CLOUD': 9.}[model]
                    save(path, dict(status='COMPLETE', sync_c=value))
                    index.append(dict(sample, model=model, level=level, draw_index=draw, score_path=str(path),
                                      source_path=str(original), source_sha256=digest))
    save(run / 'protocol.json', dict(samples=samples, model_dir=str(tmp_path / 'models/checkpoint')))
    save(tmp_path / 'runs/qwen_full_tts_extension_20261022_attempt01/tts/tts_manifest.json', {'records': old})
    save(tmp_path / 'runs/qwen_nrt_expansion40_20261022_attempt01/generation/tts_manifest.json', {'records': []})
    save(run / 'generation/tts_manifest.json', {'records': new})
    save(run / 'comparison_index.json', {'records': index})
    save(run / 'analysis_sensitivity_protocol.json', {'source_sha256': mod.sha(source)})
    main_path = run / 'analysis.json'
    main_records = []
    for sample in samples:
        values = {level: {model: {'C': {'N': 7., 'L06': 8., 'L17': 8.5, 'CLOUD': 9.}[model]}
                          for model in ['N', 'L06', 'L17', 'CLOUD']} for level in ['raw', 'norm']}
        main_records.append(dict(sample, values=values))
    save(main_path, {'records': main_records})
    main_before = main_path.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['sensitivity', '--run', str(run)])
    mod.main()
    result = mod.read(run / 'reliability_sensitivity.json')
    assert main_path.read_bytes() == main_before
    assert result['contrasts']['norm']['L17-L06']['n'] == 9
    assert result['contrasts']['norm']['L17-L06']['mean'] == .5
    assert result['records'][0]['values']['norm']['L17'] is None
    assert result['records'][1]['qualified']['L06'] == [1, 2]
    assert result['strict9']['n'] == 9
    assert result['strict9']['contrasts']['norm']['L17-L06']['mean'] == .5
