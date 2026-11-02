"""Artifact checks for units, missing legacy inference, and the new Holm family."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/cross_tfg/nine_scorer_comparison_20261031/comparison.json'


def test_frozen_statistics_and_percentage_units_are_preserved():
    data=json.loads(OUTPUT.read_text())
    folders={'TalkLikeYou':'talklikeyou','FlashHead':'soulx_flashhead_pro','Echo':'echomimic_v3_flash'}
    assert len(data['rows'])==81
    for model,folder in folders.items():
        frozen=json.loads((ROOT/'results'/folder/'chinese20_rms24_20261030_attempt01/summary.json').read_text())
        for row in [r for r in data['rows'] if r['model']==model]:
            old=frozen['subsets'][row['subset']][row['metric']]
            assert row['delta']==old['delta'] and row['holm9_p']==old['holm9_p']
            assert row['n_sources']==old['n_sources']
            multiplier=100 if row['unit']=='percentage_points' else 1
            assert row['display_delta']==row['delta']*multiplier


def test_missing_legacy_q_is_not_classified_as_nonsignificance():
    rows=json.loads(OUTPUT.read_text())['rows']
    old=[r for r in rows if r['model']=='Wav2Lip-ZH']
    assert len(old)==18
    assert all(r['holm9_p'] is None and r['significance']=='NOT_COMPUTED_NOT_NONSIGNIFICANT' for r in old)
    assert all(r['n_sources']==6 and 'mismatch' in r['caveat'] for r in old if r['metric'] in ['Synchformer','SparseSync'])


def test_new_contrast_family_and_existing_raw_contrasts_match():
    data=json.loads(OUTPUT.read_text())
    old=json.loads((ROOT/'results/talklikeyou/chinese20_rms24_20261030_attempt01/cross_model_comparison.json').read_text())
    for subset in ['same_reference12','all20']:
        family=[r for r in data['pairwise_contrasts'] if r['subset']==subset]
        assert len(family)==27
        running=0.
        for rank,row in enumerate(sorted(family,key=lambda x:x['p'])):
            running=min(1.,max(running,(27-rank)*row['p']))
            assert math.isclose(running,row['holm27_p'],abs_tol=1e-12)
            if row['model_a']=='TalkLikeYou':
                name='EchoMimicV3-Flash' if row['model_b']=='Echo' else 'FlashHead'
                reference=old['contrasts'][name]['subsets'][subset][row['metric']]
                assert math.isclose(row['delta'],reference['effect_difference'],abs_tol=1e-12)
                assert row['p']==reference['signflip_p']
