import importlib.util
import unittest
from pathlib import Path, PureWindowsPath

P = Path(__file__).resolve().parents[2] / 'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_rms_controller_revision02.py'
spec = importlib.util.spec_from_file_location('revision02_under_test', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def row(epoch, gain):
    return dict(epoch=epoch, gain_db=gain, passed=True, global_step=epoch * 6000, epoch_attempts=6000, epoch_successful_updates=6000, epoch_amp_skips=0, old_fine_resumed=False, checkpoint_path=m.RUN + f'/training/PatrickLeahy/seed42/gain_{gain:+d}/coarse/ep{epoch:04d}/attempt01/workspace/checkpoints/ngp_ep{epoch:04d}.pth')

class Revision02Test(unittest.TestCase):
    def test_posix_path_survives_windows_and_rejects_escape(self):
        p = row(1, 0)['checkpoint_path']
        self.assertEqual(m.remote_checkpoint(p).as_posix(), p)
        with self.assertRaises(AssertionError):
            m.remote_checkpoint(str(PureWindowsPath(p)))
        with self.assertRaises(AssertionError):
            m.remote_checkpoint(m.RUN + '/training/../outside.pth')
        with self.assertRaises(ValueError):
            m.remote_checkpoint('/tmp/ngp_ep0001.pth')

    def test_exact_four_completed_jobs_determine_next_job(self):
        rows = [row(e, g) for e, g in m.ORDER[:4]]
        self.assertEqual(m.validate_prefix(rows), m.ORDER[:4])
        self.assertEqual(m.ORDER[len(rows)], (2, -6))

    def test_gap_duplicate_and_failed_job_rejected(self):
        for rows in ([row(1, 0), row(1, 6)], [row(1, 0), row(1, 0)], [dict(row(1, 0), passed=False)]):
            with self.assertRaises(AssertionError):
                m.validate_prefix(rows)

if __name__ == '__main__':
    unittest.main()
