"""Integration checks against actual frozen-weight diagnostic receipts, without GPU replay."""
import json
import math
import os
from pathlib import Path
import unittest

ROOT = Path(os.environ.get('PATRICK_DENSITY_DIAGNOSTIC',
    'F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1/diagnostics/density_refresh_v1'))

def receipt(attempt, case):
    return json.loads((ROOT/'remote_results'/attempt/(case+'.json')).read_text())

def test_first_nonfinite_is_float32_exponential_range_not_mlp_nan():
    c=receipt('attempt05','rms_amp_a')
    assert all(x['output']['finite'] == math.prod(x['output']['shape']) for x in c['modules'])
    expected=sum(math.exp(x)>3.4028234663852886e38 for x in c['overflow_logit_values'])
    assert expected == c['density'][0]['sigma']['posinf'] == c['pre_dilation']['posinf'] == 9
    assert c['sigma_logit']['max'] == 124.25
    assert c['post_dilation']['posinf'] == len(c['post_dilation_bad_indices']) == 52
    assert c['new_bad_mask_matches_terminal']

def test_repeat_same_coordinates_and_raw_and_dilated_bad_positions():
    a,b=(receipt('attempt05',x) for x in ('rms_amp_a','rms_amp_b'))
    for field in ('xyz_sha256','encoded_audio_sha256','selected_eye','overflow_logit_values',
                  'pre_dilation_bad_indices','post_dilation_bad_indices'):
        assert a[field] == b[field]

def test_counterfactuals_change_one_current_input_without_claiming_training_cause():
    a=receipt('attempt05','rms_amp_a')
    for case in ('old_amp_a','rms_fp32'):
        c=receipt('attempt05',case)
        assert c['xyz_sha256'] == a['xyz_sha256'] and c['selected_eye'] == a['selected_eye']
        assert c['post_dilation']['posinf'] == 52 and c['new_bad_mask_matches_terminal']
    for attempt in ('attempt06_old_coarse10','attempt07_rms_coarse3_zero'):
        for case in ('rms_amp_a','old_amp_a'):
            c=receipt(attempt,case)
            assert c['xyz_sha256'] == a['xyz_sha256']
            assert c['pre_dilation']['posinf'] == c['post_dilation']['posinf'] == 0
    review=json.loads((ROOT/'independent_review.json').read_text())
    assert review['learned_trajectory_causal_effect_of_normalization'].startswith('NOT_IDENTIFIED')

def test_all_actual_cases_have_unchanged_parameters_and_zero_training_updates():
    count=0
    for p in (ROOT/'remote_results').glob('*/result.json'):
        r=json.loads(p.read_text())
        assert r['optimizer_calls'] == r['backward_calls'] == r['trainer_instantiations'] == 0
        assert r['terminal_file_unchanged'] and r['original_source_pins_unchanged']
        for c in r['cases']:
            assert c['parameters_unchanged'] and c['gradients_absent']
            count+=1
    assert count == 16

def load_tests(loader, tests, pattern):
    suite=unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            checked=unittest.skipUnless((ROOT/'independent_review.json').exists(), 'actual diagnostic evidence not available')(function)
            suite.addTest(unittest.FunctionTestCase(checked))
    return suite

if __name__ == '__main__':
    unittest.main()
