"""Lexicon repairs must preserve phone/tone provenance and existing entries."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experiments/phoneme_direction_20261026'))
from repair_dictionary import parse_dictionary,additions
sys.path.pop(0)


class DictionaryRepairTests(unittest.TestCase):
    def setUp(self):
        self.text='生涯\t0.99\t0.32\t0.62\t1.08\tʂ o˥ ŋ j a˧˥\n撰写\t0.99\t0.09\t1.14\t0.86\tʈʂ w a˥˩ n ɕ e˨˩˦\n炉膛\t0.99\t0.22\t1.0\t1.0\tl u˧˥ tʰ a˧˥ ŋ\n'

    def test_missing_characters_added_with_original_phone_and_tone(self):
        before=parse_dictionary(self.text)
        self.assertTrue(all(c not in before for c in '涯撰膛'))
        rows,evidence=additions(before,set('涯撰膛'))
        after=parse_dictionary(self.text+'\n'.join(rows))
        self.assertTrue(all(c in after for c in '涯撰膛'))
        self.assertEqual(after['涯'][0][1],['j','a˧˥'])
        self.assertEqual(after['撰'][0][1],['ʈʂ','w','a˥˩','n'])
        self.assertEqual(after['膛'][0][1],['tʰ','a˧˥','ŋ'])
        self.assertEqual(before['生涯'],after['生涯'])
        self.assertEqual(len(evidence),3)

    def test_existing_entry_cannot_be_overwritten(self):
        before=parse_dictionary(self.text+'涯\t1\t1\t1\t1\tj a˧˥\n')
        with self.assertRaises(ValueError):additions(before,{'涯'})

    def test_unproven_tone_cannot_be_invented(self):
        before=parse_dictionary(self.text.replace('j a˧˥','j a˥˩'))
        with self.assertRaises(ValueError):additions(before,{'涯'})

    def test_old_format_not_silently_parsed(self):
        with self.assertRaises(ValueError):parse_dictionary('涯 j a˧˥')


if __name__=='__main__':unittest.main()
