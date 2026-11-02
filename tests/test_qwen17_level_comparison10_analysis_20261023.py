import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/experiments/qwen17_level_comparison10_analysis_20261023.py'
spec = importlib.util.spec_from_file_location('qwen17_analysis', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tmp_path):
    samples = [{'sample_id': f'utt{i}', 'speaker_id': f'spk{i}'} for i in range(10)]
    records = []
    for sample in samples:
        for level in mod.LEVELS:
            for model in mod.MODELS:
                for draw in ([0] if model == 'N' else [1, 2, 3]):
                    # A large draw-specific spread must be averaged before the bootstrap.
                    value = (0 if model == 'N' else draw) + (1 if level == 'norm' else 0)
                    score = dict(sample, status='COMPLETE', sync_c=value, sync_d=9-value, av_offset=0)
                    dest = tmp_path / f"{sample['sample_id']}_{level}_{model}_{draw}.json"
                    dest.write_text(json.dumps(score), encoding='utf8')
                    records.append(dict(sample, level=level, model=model, draw_index=draw,
                                        score_path=str(dest), score_sha256=mod.sha(dest)))
    return {'records': records}, samples


def test_draws_averaged_before_speaker_resampling_and_interaction(tmp_path):
    index, samples = fixture(tmp_path)
    result = mod.analyze(index, samples)
    assert result['coverage']['valid_cells'] == 200
    assert result['by_level']['C']['norm']['contrasts']['L17-N']['n'] == 10
    assert result['by_level']['C']['norm']['contrasts']['L17-N']['mean'] == 2
    assert result['by_level']['C']['norm']['contrasts']['L17-N']['ci98_333'] == [2, 2]
    assert result['norm_minus_raw']['C']['L17']['mean'] == 1
    assert result['contrast_norm_minus_raw']['C']['L17-N']['mean'] == 0


def test_failure_not_silently_averaged_over_surviving_draws(tmp_path):
    index, samples = fixture(tmp_path)
    row = next(r for r in index['records'] if r['model'] == 'L17' and r['level'] == 'norm')
    Path(row['score_path']).unlink()
    result = mod.analyze(index, samples)
    assert result['status'] == 'ANALYZED_INCOMPLETE'
    assert result['coverage']['valid_cells'] == 199
    assert result['coverage']['complete_speakers'] == 9
    assert result['by_level']['C']['norm']['contrasts']['L17-N']['n'] == 9
    assert result['by_level']['C']['raw']['contrasts']['L17-N']['n'] == 10
    assert result['records'][0]['values']['norm']['L17']['C'] is None


def test_unplanned_duplicate_and_changed_receipt_rejected(tmp_path):
    index, samples = fixture(tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        mod.analyze({'records': index['records'] + index['records'][:1]}, samples)
    row = index['records'][0]
    Path(row['score_path']).write_text('{}')
    # A non-COMPLETE receipt is recorded as failure, not inferred to be valid.
    assert mod.analyze(index, samples)['coverage']['valid_cells'] == 199
    Path(row['score_path']).write_text(json.dumps(dict(samples[0], status='COMPLETE', sync_c=999, sync_d=0, av_offset=0)))
    with pytest.raises(ValueError, match='SHA'):
        mod.analyze(index, samples)
