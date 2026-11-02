import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/compare_patrick_coarse_fine_v1.py'
SPEC = importlib.util.spec_from_file_location('comparison', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(phase, epoch, increment):
    cells = []
    for a in MODULE.GAINS:
        for b in MODULE.GAINS:
            for c in MODULE.GAINS:
                clips = []
                for i, name in enumerate(MODULE.CLIPS):
                    row = dict(clip_id=name)
                    for support in MODULE.SUPPORTS:
                        row[support] = {metric: 5+i/10+a/100+b/1000+c/10000+increment for metric in MODULE.METRICS}
                        row[support]['av_offset'] = i-3
                    clips.append(row)
                cell = dict(training_gain_db=a, driver_gain_db=b, evaluation_gain_db=c, clips=clips)
                for support in MODULE.SUPPORTS:
                    cell['mean_'+support] = {m: sum(x[support][m] for x in clips)/6 for m in MODULE.METRICS}
                    cell['mean_'+support]['offsets'] = [x[support]['av_offset'] for x in clips]
                cells.append(cell)
    return dict(passed=True, phase=phase, epoch=epoch, seed=42, descriptive_only=True,
                training_control_status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES', cells=cells)


class ComparisonTests(unittest.TestCase):
    def test_pairing_survives_cell_and_clip_permutation(self):
        coarse = fixture('coarse', 10, 0)
        fine = fixture('fine', 17, .25)
        fine['cells'].reverse()
        for cell in fine['cells']:
            cell['clips'].reverse()
            for support in MODULE.SUPPORTS:
                cell['mean_'+support]['offsets'].reverse()
        result = MODULE.compare(coarse, fine)
        self.assertEqual(len(result['cells']), 27)
        for cell in result['cells']:
            self.assertAlmostEqual(cell['mean_guard20']['sync_c']['fine_minus_coarse'], .25)
            self.assertTrue(result['parent_control_failed_retained'])

    def test_missing_duplicate_or_wrong_phase_fails(self):
        for change in ('missing', 'duplicate', 'coarse_as_fine'):
            with self.subTest(change=change):
                fine = fixture('fine', 17, .25)
                if change == 'missing':fine['cells'].pop()
                elif change == 'duplicate':fine['cells'][-1] = copy.deepcopy(fine['cells'][0])
                else:fine['epoch'] = 10
                with self.assertRaises(AssertionError):MODULE.compare(fixture('coarse', 10, 0), fine)

    def test_bad_aggregation_or_nonfinite_score_fails(self):
        for bad_value in (.5, float('nan')):
            fine = fixture('fine', 17, .25)
            fine['cells'][0]['mean_guard20']['sync_c'] = bad_value
            with self.assertRaises(AssertionError):MODULE.compare(fixture('coarse', 10, 0), fine)


if __name__ == '__main__':unittest.main()
