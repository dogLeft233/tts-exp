import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experiments/phoneme_direction_20261026'))
from canonical_alignment import canonical_words
sys.path.pop(0)


class CanonicalAlignmentTests(unittest.TestCase):
    def test_alias_changes_no_times(self):
        w,c=canonical_words([(0.,1.,'綠'),(1.,2.,'為')],[],['绿','为'])
        self.assertEqual(w,[(0.,1.,'绿'),(1.,2.,'为')]);self.assertEqual(len(c),2)

    def test_proven_silent_duplicate_fixed_without_moving_phone(self):
        phones=[(0.,1.,'i˥'),(1.,40.,'')]
        words,changes=canonical_words([(0.,1.,'息'),(1.,40.,'息')],phones,['息'],[(1.,40.)])
        self.assertEqual(words,[(0.,1.,'息')]);self.assertEqual(len(changes),1)
        self.assertEqual(phones,[(0.,1.,'i˥'),(1.,40.,'')])

    def test_unknown_or_speech_is_never_discarded(self):
        for label in ['spn','i˥']:
            with self.assertRaises(ValueError):
                canonical_words([(0.,1.,'息'),(1.,40.,'息')],[(0.,1.,'i˥'),(1.,40.,label)],['息'],[(1.,40.)])

    def test_database_evidence_and_full_transcript_are_required(self):
        words=[(0.,1.,'息'),(1.,40.,'息')];phones=[(0.,1.,'i˥'),(1.,40.,'')]
        with self.assertRaises(ValueError):canonical_words(words,phones,['息'])
        with self.assertRaises(ValueError):canonical_words(words,phones,['消'],[(1.,40.)])


if __name__=='__main__':unittest.main()
