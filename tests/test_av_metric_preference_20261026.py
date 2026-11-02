import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('avpref', Path(__file__).parents[1] / 'scripts/experiments/av_metric_preference_20261026.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_video_cross_contains_four_cells_and_no_missing_draw():
    rows = [dict(id='n', sample_id='s', arm='N', draw=0),
            dict(id='t1', sample_id='s', arm='L17', draw=1),
            dict(id='bad', sample_id='s', arm='L06', draw=3)]
    items, cells = module.plan(rows, {'n', 't1'})
    assert set(items) == {'n', 't1'}
    cross = [c for c in cells if c['kind'] == 'cross']
    assert {(c['video_id'], c['audio_id']) for c in cross} == {('n', 'n'), ('n', 't1'), ('t1', 'n'), ('t1', 't1')}
    assert all(c['common_ids'] == ['n', 't1'] for c in cross)
    assert len([c for c in cells if c['kind'] == 'shift']) == 2


def test_direction_and_single_source_interval():
    assert module.direction(2, 1) == 'TTS'
    assert module.direction(1, 2) == 'NATURAL'
    assert module.direction(1, 1) == 'TIE'
    assert module.interval([0.25]) == dict(mean=0.25, ci95=[0.25, 0.25], sources=1)


def test_paused_controller_does_not_block_but_running_production_does():
    controller = '/venv/python /repo/scripts/experiments/audio200_nl17_controller_20261025.py --authorization x'
    worker = '/venv/python /repo/scripts/experiments/audio200_production_20261025.py worker-synthesize --run x'
    assert module.active_production_commands(['Ts ' + controller, 'Z ' + worker]) == []
    assert module.active_production_commands(['Ss ' + controller, 'Rl ' + worker]) == [controller, worker]


def test_summary_weights_sources_and_keeps_conflict_directions(tmp_path):
    import json
    selected = []
    for source, draws in [('sourceA', 3), ('sourceB', 1)]:
        selected.append(dict(id=source + '_N', sample_id=source, arm='N', draw=0, score_float32=7, eq1=.3))
        for arm in ['L06', 'L17']:
            for draw in range(draws):
                selected.append(dict(id=f'{source}_{arm}_{draw}', sample_id=source,
                                     arm=arm, draw=draw, score_float32=8, eq1=.31))
    _, cells = module.plan(selected, {r['id'] for r in selected})
    prior = tmp_path / 'runs/cogenav_sync_full20_20261024_attempt01'
    prior.mkdir(parents=True)
    (prior / 'scores.json').write_text(json.dumps(selected))
    out = tmp_path / 'result'
    out.mkdir()
    module.write(out / 'cells.json', cells)
    for metric in ['syncreward', 'peavs']:
        rows = []
        for c in cells:
            value = 3.0
            if c['kind'] == 'native' and c['arm'] != 'N':
                value += 1 if c['source'] == 'sourceA' else -1
            elif c['kind'] == 'shift':
                value -= .2
            elif c['kind'] == 'cross':
                value += .1 if c['video_label'] == c['audio_label'] else -.1
            rows.append(dict(video_path=c['id'] + '.mp4', filename=c['id'], reward=value, score=value))
        module.write(out / metric / 'batchnative000.json', rows)
        module.write(out / metric / 'batchnative000.raw.json', {'scores': rows})
    summary = module.summarize(tmp_path, out)
    for value in summary.values():
        assert value['native_delta']['mean'] == 0
        assert value['native_TTS_higher'] == 6
        assert value['conflicts']['delta_sync']['both_TTS'] == 6
        assert value['conflicts']['delta_sync']['other_TTS_new_N'] == 2
        assert value['shift_lower'] == value['shift_total'] == 4
