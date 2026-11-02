"""Reject changed manifests/plans before reading any sealed confirmation audio."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[1] / "scripts/experiments/audio200_parent_readback_20261025.py"
spec = importlib.util.spec_from_file_location("parent_audio200", P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BindingGuardTests(unittest.TestCase):
    def test_reject_modified_source_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "selected600_natural_manifest.json").write_text('{"changed":true}', encoding="utf-8")
            (root / "source_split_freeze.json").write_text(json.dumps({"source_manifest_sha256": "0"*64}), encoding="utf-8")
            with self.assertRaises(AssertionError):
                mod.audit(root, "local", root / "no_plan", root / "no_anchor")

    def test_reject_modified_roster(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = b'{"rows":[]}'
            (root / "selected600_natural_manifest.json").write_bytes(raw)
            plan = root / "plan.json"
            plan.write_text('{"changed":true}', encoding="utf-8")
            (root / "source_split_freeze.json").write_text(json.dumps({
                "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "candidate_plan_sha256": "0"*64}), encoding="utf-8")
            with self.assertRaises(AssertionError):
                mod.audit(root, "local", plan, root / "no_anchor")


if __name__ == "__main__":
    unittest.main()
