"""Isolation and deterministic selection checks for the fit-only calibration."""
import importlib.util
from pathlib import Path
import unittest
import sys

PATH = Path(__file__).parents[1]/'scripts/experiments/phoneme_direction_20261026/calibrate.py'
spec = importlib.util.spec_from_file_location('phone_calibration', PATH)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
sys.path.insert(0,str(PATH.parent))
from audit_alignment_readback import matched_words
sys.path.pop(0)


def rows():
    return [dict(id=f'{s}_{q}_{a}_{d}',sample_id=f'{s}_{q}',speaker_id=f'S{s:04}',
                 text='测试',split='fit',arm=a,draw=d)
            for s in range(100) for q in range(3)
            for a,d in [('N',0),('L17',1),('L17',2),('L17',3)]]


class CalibrationTests(unittest.TestCase):
    def test_deterministic_selection_ignores_input_order(self):
        a=cal.select_calibration(rows())
        b=cal.select_calibration(list(reversed(rows())))
        self.assertEqual(a,b)
        self.assertEqual(len(a),80)
        self.assertEqual({x['speaker_id'] for x in a},{f'S{s:04}' for s in range(0,100,5)})

    def test_nonfit_and_duplicate_rejected(self):
        r=rows();r[0]['split']='confirm'
        with self.assertRaises(ValueError):cal.select_calibration(r)
        r=rows();r[0]=r[1]
        with self.assertRaises(ValueError):cal.select_calibration(r)

    def test_draw_and_text_identity_rejected(self):
        r=rows();r[0]['draw']=1
        with self.assertRaises(ValueError):cal.select_calibration(r)
        r=rows();r[0]['text']='别的句子'
        with self.assertRaises(ValueError):cal.select_calibration(r)

    def test_normalization_never_invents_number_readings(self):
        self.assertEqual(cal.normalize_text('你，好。'),'你 好')
        for t in ['第12次','A股','']:
            with self.assertRaises(ValueError):cal.normalize_text(t)

    def test_unknown_phone_never_counts_as_match(self):
        tokens=[dict(text_position=0,label='spn'),dict(text_position=1,label='a˥˩')]
        matched,unsupported=matched_words(tokens,tokens)
        self.assertEqual(len(matched),1)
        self.assertEqual(unsupported,[0])

    def test_within_word_mismatch_rejects_whole_position(self):
        a=[dict(text_position=0,label='t'),dict(text_position=0,label='a˥˩')]
        b=[dict(text_position=0,label='t'),dict(text_position=0,label='a˧˥')]
        matched,unsupported=matched_words(a,b)
        self.assertEqual(matched,[])
        self.assertEqual(unsupported,[0])


if __name__=='__main__':unittest.main()
