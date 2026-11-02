"""Focused path-binding tests for the shared snapshot application helper."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "audio_enhancement"
               / "fit40_snapshot_spsa_application_attempt04.py")
SPEC = importlib.util.spec_from_file_location("snapshot_spsa_application_attempt04_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlanPinnedNaturalPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.rel = "fit40_source_inputs/source_wav/clip.wav"
        self.path = self.root / self.rel
        self.path.parent.mkdir(parents=True)
        self.raw = b"synthetic-wav-placeholder"
        self.path.write_bytes(self.raw)
        self.plan = {"scientific_input_pins": {"natural_pcm_clip": {
            "path": self.rel,
            "bytes": len(self.raw),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }}}
        self.row = {"sample_id": "clip", "natural_pcm": {"path": str(self.path)}}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_accepts_exact_plan_root_mapping(self) -> None:
        got, pin = MODULE._resolve_plan_pinned_natural_pcm(self.root, self.plan, self.row)
        self.assertEqual(got, self.path)
        self.assertEqual(pin["sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_rejects_byte_identical_unpinned_copy(self) -> None:
        duplicate = self.root / "other" / "clip.wav"
        duplicate.parent.mkdir()
        duplicate.write_bytes(self.raw)
        row = {"sample_id": "clip", "natural_pcm": {"path": str(duplicate)}}
        with self.assertRaisesRegex(MODULE.ApplicationError, "exact plan-root mapping"):
            MODULE._resolve_plan_pinned_natural_pcm(self.root, self.plan, row)

    def test_rejects_symlink_in_plan_path(self) -> None:
        target = self.root / "real"
        target.mkdir()
        target_file = target / "clip.wav"
        target_file.write_bytes(self.raw)
        alias = self.root / "fit40_source_inputs"
        self.path.unlink()
        self.path.parent.rmdir()
        self.path.parent.parent.rmdir()
        alias.symlink_to(target, target_is_directory=True)
        plan = {"scientific_input_pins": {"natural_pcm_clip": {
            "path": "fit40_source_inputs/clip.wav",
            "bytes": len(self.raw),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }}}
        row = {"sample_id": "clip", "natural_pcm": {
            "path": str(self.root / "fit40_source_inputs" / "clip.wav")}}
        with self.assertRaisesRegex(MODULE.ApplicationError, "traverses a symlink"):
            MODULE._resolve_plan_pinned_natural_pcm(self.root, plan, row)


if __name__ == "__main__":
    unittest.main()
