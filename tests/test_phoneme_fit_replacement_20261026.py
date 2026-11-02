import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/phoneme_direction_20261026'))
from replace_fit_source import replace_rows, OLD_ID, NEW_ID


class ReplacementContractTests(unittest.TestCase):
    def setUp(self):
        self.original = []
        for speaker in range(100):
            for sentence in range(3):
                sid = OLD_ID if speaker == 0 and sentence == 0 else f'S{speaker:04d}W{sentence:04d}'
                spk = 'S0168' if speaker == 0 else f'F{speaker:04d}'
                for arm, draw in [('N', 0), ('L17', 1), ('L17', 2), ('L17', 3)]:
                    self.original.append(dict(id=f'{sid}_{arm}_{draw}', sample_id=sid,
                        speaker_id=spk, split='fit', arm=arm, draw=draw, text='原句'))
        self.replacement = [dict(x, id=x['id'].replace(OLD_ID, NEW_ID), sample_id=NEW_ID, text='替代句')
                            for x in self.original[:4]]

    def test_exact_replacement_preserves_original_and_other_1196_units(self):
        before = copy.deepcopy(self.original)
        result = replace_rows(self.original, self.replacement)
        self.assertEqual(before, self.original)
        self.assertEqual([x for x in result if x['sample_id'] != NEW_ID],
                         sorted(self.original[4:], key=lambda x: x['id']))
        self.assertEqual(len(result), 1200)

    def test_reject_split_leak_and_wrong_speaker(self):
        for key, value in [('split', 'development'), ('speaker_id', 'S0002')]:
            bad = copy.deepcopy(self.replacement); bad[0][key] = value
            with self.assertRaises(ValueError): replace_rows(self.original, bad)

    def test_reject_incomplete_draws_and_reference_mismatch(self):
        with self.assertRaises(ValueError): replace_rows(self.original, self.replacement[:-1])
        bad = copy.deepcopy(self.replacement); bad[0]['text'] = '不同文本'
        with self.assertRaises(ValueError): replace_rows(self.original, bad)


if __name__ == '__main__': unittest.main()
