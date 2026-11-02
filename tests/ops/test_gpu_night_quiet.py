"""Safety and schedule regression tests, without GPU mutation or CUDA work."""
import importlib.util
import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gpu_night_quiet", ROOT / "scripts/ops/gpu_night_quiet.py")
QUIET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUIET)
CONFIG = json.loads((ROOT / "scripts/ops/gpu-night-quiet.json").read_text())


class NightQuietTests(unittest.TestCase):
    def setUp(self):
        self.t = dict(gpu=47, memory=46, watts=25, limit=300, minimum=100, maximum=300)

    def test_cross_midnight_and_timezone(self):
        for iso, watts in [
            ("2026-10-11T22:59:59+08:00", 300),
            ("2026-10-11T23:00:00+08:00", 200),
            ("2026-10-12T07:59:59+08:00", 200),
            ("2026-10-12T08:00:00+08:00", 300),
            ("2026-10-11T15:00:00+00:00", 200),
        ]:
            with self.subTest(iso=iso):
                self.assertEqual(QUIET.policy(CONFIG, self.t, datetime.fromisoformat(iso))[0], watts)

    def test_thermal_guard_applies_even_at_morning_restoration(self):
        now = datetime.fromisoformat("2026-10-12T08:00:00+08:00")
        for gpu, memory, watts in [(75, 60, 150), (50, 78, 150), (80, 60, 100), (50, 82, 100)]:
            self.assertEqual(QUIET.policy(CONFIG, dict(self.t, gpu=gpu, memory=memory), now)[0], watts)

    def test_critical_latch_requires_four_cool_readings(self):
        now = datetime.fromisoformat("2026-10-12T08:00:00+08:00")
        level, count = 2, 0
        for index in range(4):
            target, _, level, count = QUIET.policy(CONFIG, self.t, now, level, count)
            self.assertEqual(target, 300 if index == 3 else 100)
        self.assertEqual(QUIET.policy(CONFIG, dict(self.t, gpu=70), now, 2, 3)[2:], (2, 0))

    def test_identity_invalid_telemetry_and_device_limits_fail_closed(self):
        with patch.object(QUIET.subprocess, "run") as run:
            for output in ["wrong,47,46,25,300,100,300", CONFIG["uuid"] + ",NaN,46,25,300,100,300", CONFIG["uuid"] + ",47,N/A,25,300,100,300"]:
                run.return_value.stdout = output
                with self.assertRaises(ValueError):
                    QUIET.query(CONFIG["uuid"])
        with self.assertRaises(ValueError):
            QUIET.policy(CONFIG, dict(self.t, maximum=180), datetime.fromisoformat("2026-10-11T23:00:00+08:00"))

    def test_power_change_requires_actual_readback(self):
        with patch.object(QUIET.subprocess, "run"), patch.object(QUIET, "query", return_value=self.t):
            with self.assertRaises(RuntimeError):
                QUIET.set_limit(CONFIG["uuid"], 200)


if __name__ == "__main__":
    unittest.main()
