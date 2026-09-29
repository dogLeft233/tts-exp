"""Report-only feature dose descriptions requested after parent effects; no new scores."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments import tts_fixed_generator_shift_cross_20260927 as core
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as stream
from scripts.experiments.tts_fixed_generator_shift_ops_20260927 import shift_table


def describe():
    protocol, rows = core.locked()
    assert core.read(core.OUT / 'bridge_validation.json')['status'] == 'PASS'
    fit = np.load(core.PARENT / 'fit.npz')
    metadata = core.read(core.PARENT / 'fit.json')
    clips = []
    for row in rows:
        if row['split'] != 'evaluation':
            continue
        fold = metadata['folds'].index(row['speaker'])
        for ai, arm in enumerate('NT'):
            q = row['arms'][arm]
            with np.load(q['features_path']) as f:
                z = f['z']
            own, own_global = stream.template_table(fit['raw_mu'], fit['speaker_counts'], metadata['labels'], fold, ai)
            other, other_global = stream.template_table(fit['raw_mu'], fit['speaker_counts'], metadata['labels'], fold, 1-ai)
            active = np.asarray(q['speech_mask'], bool)
            for kind in core.KINDS:
                own_z, _ = stream.replacement_table(z, q['phone_labels'], q['speech_mask'], own, own_global, kind)
                shift_z, desc = shift_table(z, q['phone_labels'], q['speech_mask'], own, own_global, other, other_global, kind)
                change = own_z[active].astype(np.float64) - z[active].astype(np.float64)
                assert len(change) > 0
                B = {'RMS': float(np.sqrt(np.mean(change**2))), 'L2': float(np.linalg.norm(change)),
                     'max_abs': float(np.abs(change).max()), 'frame_L2_mean': float(np.linalg.norm(change, axis=1).mean())}
                clips.append({'id': row['id'], 'speaker': row['speaker'], 'arm': arm, 'kind': kind,
                              'primary71': row['guard20_eligible'], 'speech_frames': int(active.sum()),
                              'B_own_prototype_change': B, 'S_applied_projected_change': desc['applied'],
                              'S_preprojection_half_delta': desc['half_delta'], 'projection': desc['projection'],
                              'projection_fraction': desc['pre_relu_negative_fraction']})
    summaries = {}
    for support in ['all74', 'common71']:
        for arm in 'NT':
            for kind in core.KINDS:
                selected = [r for r in clips if r['arm'] == arm and r['kind'] == kind and (support == 'all74' or r['primary71'])]
                cell = {}
                for key in ['B_own_prototype_change', 'S_applied_projected_change', 'S_preprojection_half_delta', 'projection']:
                    values = np.array([r[key]['RMS'] for r in selected])
                    speaker_means = [np.mean([r[key]['RMS'] for r in selected if r['speaker'] == sp]) for sp in sorted({r['speaker'] for r in selected})]
                    cell[key] = {'speaker_equal_mean_RMS': float(np.mean(speaker_means)), 'clip_median_RMS': float(np.median(values)),
                                 'clip_min_RMS': float(values.min()), 'clip_max_RMS': float(values.max())}
                fractions = [r['projection_fraction'] for r in selected]
                cell['projection_fraction'] = {'clip_median': float(np.median(fractions)), 'min': min(fractions), 'max': max(fractions)}
                cell['clips'] = len(selected)
                summaries[support + '/' + arm + '/' + kind] = cell
    core.write(core.OUT / 'dose_descriptions.json.gz', {
        'purpose': 'post-parent-result report-only descriptive supplement authorized by root; no endpoint or selection change',
        'limitations': 'same lambda is not same dose; B includes speaker/utterance offsets and within-phone dynamics; differences do not identify a root cause',
        'protocol_sha256': core.sha(core.OUT / 'protocol.json'), 'fit_sha256': core.sha(core.PARENT / 'fit.npz'),
        'script_sha256': core.sha(__file__), 'all74_retained': True, 'no_new_forward_or_effect_score': True,
        'clips': clips, 'summaries': summaries,
    })


if __name__ == '__main__':
    describe()
