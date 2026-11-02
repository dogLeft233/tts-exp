"""One scientific QA of the actual output-owned candidate arrays."""
from pathlib import Path
from fractions import Fraction
import datetime
import hashlib
import importlib.util
import json
import math
import numpy as np

A = Path(__file__).resolve().parent
ROOT = A.parents[2]
OUT = A / 'output_owned_candidates_attempt01'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rawsha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))


def main():
    spec = importlib.util.spec_from_file_location('independent_rotation_math', A / 'root_actual_rotation_qa.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    receipt_path = OUT / 'receipt.json'
    assert sha(receipt_path) == '37ff2a054bb1e4f1dcca1d37de4fe6f292bee433832371886109b879841be705'
    receipt = read(receipt_path)
    assert receipt['consumer_source_sha256'] == sha(A / 'output_owned_rotation_candidate_source_attempt01.py')
    admission_path = A / 'root_output_owned_cpu_admission_attempt01.json'
    admission = read(admission_path)
    for p, expected in admission['pins'].items():
        assert sha(ROOT / p) == expected, p
    inventory_path = ROOT / '.codex/scratch/mandarin_viseme_phone_support_20261015_attempt03/receipt.json'
    assert sha(inventory_path) == receipt['fit_inventory_sha256']
    assert sha(old.FITS['musetalk']) == receipt['fit_musetalk_array_sha256']
    anchors, supporters = old.fit_anchors('musetalk', read(inventory_path))
    axis = old.unit(anchors['high'] - anchors['low'])
    frames_by_sample = {}
    summaries = []
    count = 0
    for row in receipt['rows']:
        sample = row['sample_id']
        source_path = A / 'readback_attempt04' / f'{sample}.musetalk.native.npz'
        assert sha(source_path) == row['cpu6_source_file_sha256']
        with np.load(source_path, allow_pickle=False) as z:
            baseline, old_clock = z['features'], z['native_time_s']
        assert rawsha(baseline) == row['cpu6_feature_array_sha256']
        assert baseline.dtype == np.float32 and baseline.shape[1:] == (50, 384)
        cp = Path(row['native_time_s']['path'])
        assert sha(cp) == row['native_time_s']['file_sha256']
        clock = np.load(cp, allow_pickle=False)
        assert clock.dtype == np.float64 and clock.tobytes() == old_clock.tobytes()
        assert np.array_equal(clock, np.asarray([float(Fraction(j, 25)) for j in range(len(clock))]))
        grid = Path(row['textgrid_path'])
        assert sha(grid) == row['textgrid_sha256']
        expected_tokens = {}
        for i, phone, lo, hi in old.phones(grid):
            core = old.indices(len(baseline), 'musetalk', lo, hi)
            if phone in old.HEIGHT and len(core) >= 2:
                expected_tokens[i] = (phone, core, core + 4)
        tokens = [t for t in row['tokens'] if t['status'] == 'ROTATED_OUTPUT_OWNED']
        assert {t['phone_index'] for t in tokens} == set(expected_tokens)
        effective = []
        for t in tokens:
            phone, core, owned = expected_tokens[t['phone_index']]
            assert core.tolist() == t['original_audio_core_I']
            assert owned.tolist() == t['output_owned_prompt_rows_I_plus_p']
            assert t['height_class'] == old.HEIGHT[phone] and t['anchor_supporters'] == supporters
            assert all(row['speaker_id'] not in s and len(s) >= 3 for s in supporters.values())
            assert np.all((owned >= 0) & (owned < len(baseline)))
            assert np.all(owned * 640 < {'BAC009S0022W0123':51376, 'BAC009S0024W0122':56561}[sample])
            effective.extend(owned.tolist())
        assert len(effective) == len(set(effective))
        effective = sorted(effective)
        assert len(effective) == 7
        frames_by_sample[sample] = effective
        mask = np.zeros(len(baseline), dtype=bool)
        mask[effective] = True
        x = baseline.reshape(len(baseline), -1).astype(np.float64)
        radii = np.linalg.norm(x, axis=1)
        u = x / radii[:, None]
        result = {'sample_id':sample, 'effective_owned_frame_indices':effective,
                  'proposal_only_identity_rows':sorted(set(row['mapped_prompt_row_indices']) - set(effective)),
                  'rotated_tokens':len(tokens), 'arms':{}}
        for arm, sign in [('zero',0), ('plus',1), ('minus',-1)]:
            meta = row['arrays']['output_owned_height_' + arm]
            p = Path(meta['path'])
            assert sha(p) == meta['file_sha256']
            candidate = np.load(p, allow_pickle=False)
            assert rawsha(candidate) == meta['array_sha256']
            assert candidate.shape == baseline.shape and candidate.dtype == baseline.dtype and np.isfinite(candidate).all()
            changed = np.flatnonzero(np.any(candidate.reshape(len(candidate), -1) != baseline.reshape(len(baseline), -1), axis=1)).tolist()
            assert changed == ([] if sign == 0 else effective)
            assert candidate[~mask].tobytes() == baseline[~mask].tobytes()
            if sign == 0:
                assert candidate.tobytes() == baseline.tobytes()
                result['arms'][arm] = {'bitwise_identity':True}
                continue
            y = candidate.reshape(len(candidate), -1).astype(np.float64)
            ry = np.linalg.norm(y, axis=1)
            uy = y / ry[:, None]
            norm_error = float(np.max(np.abs(ry - radii) / radii))
            gram_error = formula_error = 0.0
            deltas = []
            token_results = []
            for t in tokens:
                phone, core, owned = expected_tokens[t['phone_index']]
                klass = old.HEIGHT[phone]
                m = old.unit(old.direction(x[owned]).astype(np.float64))
                assert rawsha(m) == t['baseline_output_prompt_direction_sha256']
                target = axis if klass == 'high' else -axis
                c = float(np.clip(m @ target, -1, 1))
                w = old.unit(target - c * m)
                theta = min(0.05, 0.5 * math.acos(c))
                assert theta == t['theta_eff_rad']
                rotated = (u[owned] + (math.cos(theta)-1)*((u[owned]@m)[:,None]*m + (u[owned]@w)[:,None]*w)
                           + sign*math.sin(theta)*((u[owned]@m)[:,None]*w - (u[owned]@w)[:,None]*m))
                expected = (rotated * radii[owned, None]).astype(np.float32)
                formula_error = max(formula_error, float(np.max(np.abs(candidate.reshape(len(candidate), -1)[owned] - expected))))
                gram_error = max(gram_error, float(np.max(np.abs(u[owned]@u[owned].T - uy[owned]@uy[owned].T))))
                margin_axis = anchors[klass] - anchors['low' if klass == 'high' else 'high']
                after = old.unit(old.direction(y[owned]).astype(np.float64))
                delta = float((after - m) @ margin_axis)
                assert sign * delta > 1e-7
                assert abs(float(after @ margin_axis) - t['quantized_diagnostics']['signed_class_margin_' + arm]) < 1e-12
                deltas.append(delta)
                token_results.append({'phone_index':t['phone_index'], 'phone':phone, 'owned':owned.tolist(), 'signed_margin_delta':delta})
                count += 1
            assert norm_error <= 2e-6 and gram_error <= 2e-6 and formula_error <= 2e-6
            result['arms'][arm] = {'changed_rows':changed, 'non_effective_bitwise_identity':True,
                'max_relative_row_norm_error':norm_error, 'max_unit_intracore_gram_error':gram_error,
                'max_independent_rotation_formula_error':formula_error, 'signed_margin_delta_min':min(deltas),
                'signed_margin_delta_max':max(deltas), 'tokens':token_results}
        summaries.append(result)
    assert set(frames_by_sample) == {'BAC009S0022W0123','BAC009S0024W0122'} and count == 12
    qa = {'schema':'root_output_owned_musetalk_actual_array_qa_v1', 'status':'PASS_ACTUAL_ARRAYS_ONLY_NOT_VIDEO_EFFECT',
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256':sha(Path(__file__)),
          'cpu_receipt_sha256':sha(receipt_path), 'admission_sha256':sha(admission_path),
          'rotated_token_arm_checks':count, 'directly_exposed_cp_frames':14, 'rows':summaries,
          'scope':'Two clips, six native f32 arrays and two exact clocks; no model/encoder/MFA/scorer run; no efficacy claim.'}
    qa_path = A / 'root_actual_output_owned_qa.json'
    with qa_path.open('x', encoding='utf-8') as f:
        json.dump(qa, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    owner = {'schema':'output_owned_musetalk_edit_frame_map_v1', 'status':'PASS_ACTUAL_CHANGED_ROWS',
             'cpu_receipt_sha256':sha(receipt_path), 'root_array_qa_sha256':sha(qa_path),
             'mapping':'original eligible MFA core I + fixed p4, unchanged clock; union only of actually rotated eligible tokens',
             'owned_output_frame_indices_by_sample':frames_by_sample}
    owner_path = A / 'output_owned_actual_frame_map_attempt01.json'
    with owner_path.open('x', encoding='utf-8') as f:
        json.dump(owner, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status':qa['status'],'qa_sha256':sha(qa_path),'frame_map_sha256':sha(owner_path),'checks':count,
                      'max_norm_error':max(r['arms'][a]['max_relative_row_norm_error'] for r in summaries for a in ('plus','minus')),
                      'max_gram_error':max(r['arms'][a]['max_unit_intracore_gram_error'] for r in summaries for a in ('plus','minus')),
                      'max_formula_error':max(r['arms'][a]['max_independent_rotation_formula_error'] for r in summaries for a in ('plus','minus'))}))


if __name__ == '__main__':
    main()
