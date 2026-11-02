import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).resolve().parents[1] / "scripts/experiments/wimhf_audio_smoke_20261025.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeContracts(unittest.TestCase):
    def test_signed_global_batchtopk_and_threshold(self):
        h = torch.arange(1., 129.).reshape(2, 64)
        a = smoke.sparse_signed(h)
        self.assertEqual(int((a != 0).sum()), 8)
        self.assertTrue(torch.equal(a, -smoke.sparse_signed(-h)))
        self.assertTrue(torch.equal(smoke.sparse_signed(h, threshold=120), -smoke.sparse_signed(-h, threshold=120)))

    def test_order_and_pair_direction_are_preserved(self):
        x = torch.arange(5632.).reshape(11, 512)
        self.assertTrue(torch.equal(x, x.flatten().reshape(11, 512)))
        self.assertFalse(torch.equal(x.flatten(), x.flip(0).flatten()))
        self.assertEqual(smoke.swap_label(7.1, 8.2), -smoke.swap_label(8.2, 7.1))


if __name__ == "__main__":
    unittest.main()
