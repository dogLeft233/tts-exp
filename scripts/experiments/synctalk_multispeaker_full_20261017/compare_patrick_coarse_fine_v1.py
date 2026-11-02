"""Pair fixed coarse10/fine17 cells and clips; descriptive differences only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

GAINS = (-6, 0, 6)
METRICS = ('sync_c', 'sync_d', 'distance_d0', 'background_median')
SUPPORTS = ('guard20', 'full')
CLIPS = tuple(f'test_{i:02d}' for i in range(6))


def validated_cells(document, phase, epoch):
    assert document['passed'] and document['seed'] == 42
    assert document['phase'] == phase and document['epoch'] == epoch
    assert document['descriptive_only']
    assert document['training_control_status'] == 'CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES'
    cells = {}
    for cell in document['cells']:
        key = tuple(cell[k] for k in ('training_gain_db', 'driver_gain_db', 'evaluation_gain_db'))
        assert key not in cells, 'duplicate axis cell'
        clips = {clip['clip_id']: clip for clip in cell['clips']}
        assert len(cell['clips']) == len(clips) == 6 and set(clips) == set(CLIPS)
        for support in SUPPORTS:
            for metric in METRICS:
                values = [clips[c][support][metric] for c in CLIPS]
                assert all(math.isfinite(x) for x in values)
                actual = cell['mean_' + support][metric]
                assert math.isfinite(actual) and abs(sum(values)/6-actual) < 1e-10
            assert cell['mean_' + support]['offsets'] == [clip[support]['av_offset'] for clip in cell['clips']]
        cells[key] = (cell, clips)
    assert set(cells) == {(a, b, c) for a in GAINS for b in GAINS for c in GAINS}
    return cells


def compare(coarse, fine):
    before = validated_cells(coarse, 'coarse', 10)
    after = validated_cells(fine, 'fine', 17)
    result = []
    for key in sorted(before):
        old, old_clips = before[key]
        new, new_clips = after[key]
        row = dict(zip(('training_gain_db', 'driver_gain_db', 'evaluation_gain_db'), key))
        row['clips'] = []
        for clip_id in CLIPS:
            clip = dict(clip_id=clip_id)
            for support in SUPPORTS:
                clip[support] = {
                    metric: dict(coarse=old_clips[clip_id][support][metric],
                                 fine=new_clips[clip_id][support][metric],
                                 fine_minus_coarse=new_clips[clip_id][support][metric]-old_clips[clip_id][support][metric])
                    for metric in METRICS}
                clip[support]['offset_frames'] = dict(coarse=old_clips[clip_id][support]['av_offset'],
                                                     fine=new_clips[clip_id][support]['av_offset'])
            row['clips'].append(clip)
        for support in SUPPORTS:
            row['mean_' + support] = {}
            for metric in METRICS:
                delta = sum(c[support][metric]['fine_minus_coarse'] for c in row['clips'])/6
                assert abs(delta-(new['mean_'+support][metric]-old['mean_'+support][metric])) < 1e-10
                row['mean_'+support][metric] = dict(coarse=old['mean_'+support][metric],
                                                    fine=new['mean_'+support][metric],
                                                    fine_minus_coarse=delta)
        result.append(row)
    return dict(passed=True, identity='PatrickLeahy', seed=42, coarse_epoch=10, fine_epoch=17,
                cells=result, paired_cells=27, paired_clips_per_cell=6,
                primary_support='guard20', secondary_support='full', delta_definition='fine minus coarse',
                training_control_status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES',
                parent_control_failed_retained=True, descriptive_only=True,
                human_perception_not_measured=True, no_independent_identity_or_seed_replication=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--fine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Bind the sole already delivered Patrick coarse input, never a selected rerun.
    raw = args.coarse.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '0c727888b1fbeb09e573ab26bc7129d453aa03b45de345b460d66973ac6fac1a'
    final_raw = args.fine.read_bytes()
    result = compare(json.loads(raw), json.loads(final_raw))
    result['input_sha256'] = dict(coarse=hashlib.sha256(raw).hexdigest(), fine=hashlib.sha256(final_raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()
