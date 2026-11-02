import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_newbatch_parent_audit_20261025.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ParentAudit(unittest.TestCase):
    def test_fixed_cross_model_draws_exclude_mismatched_draw(self):
        rows = [dict(arm='N', draw=0, id='N')]+[dict(arm=a, draw=d, id=a+str(d)) for a in ('L06', 'L17') for d in (1, 2, 3)]
        pairs = m.expected_pairs(rows)
        self.assertEqual(len(pairs), 15)
        self.assertNotIn(('L17-L06', 'L171', 'L062'), pairs)
        with self.assertRaises(AssertionError):
            m.expected_pairs(rows[:-1])

    def test_clock_center_mapping_and_unpadded_boundary(self):
        centers = [(max(0, 640*i-1)+640*i+3439)/32000 for i in range(30)]
        _, i, good = m.nearest_window([400, 440], .069, 0., centers)
        self.assertEqual(i, 10)
        self.assertTrue(good)
        self.assertFalse(m.nearest_window([0, 10], 0., 0., centers)[2])
        self.assertEqual(m.nearest_window([400, 440], .069, .04, centers)[1], 11)


if __name__ == '__main__':
    unittest.main()
