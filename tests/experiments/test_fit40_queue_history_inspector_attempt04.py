from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
INSPECTOR_PATH = ROOT / ".codex/scratch/fit40_queue_progress_index_attempt04/inspect_fit40_terminal_prefixes_attempt04.py"
_SPEC = importlib.util.spec_from_file_location("fit40_history_inspector_attempt04", INSPECTOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
INSPECTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(INSPECTOR)


def _with_self_hash(record: dict) -> dict:
    record["self_sha256"] = INSPECTOR.self_sha(record, "self_sha256")
    return record


class LaunchReceiptValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.launch_path = self.root / "launch_receipt_block_000_attempt01.json"
        self.intent_path = self.root / "launch_intent_block_000_attempt01.json"
        self.intent_path.write_text('{"schema":"intent"}', encoding="utf-8")
        intent_raw = self.intent_path.read_bytes()
        self.pin_values = {
            str(self.root / "plan.json"): {"bytes": 11, "sha256": "a" * 64},
            str(self.root / "rows.jsonl"): {"bytes": 12, "sha256": "b" * 64},
            str(self.root / "binding.json"): {"bytes": 13, "sha256": "c" * 64},
            str(self.root / "launcher.py"): {"bytes": 14, "sha256": "d" * 64},
            str(self.intent_path): {"bytes": len(intent_raw), "sha256": hashlib.sha256(intent_raw).hexdigest()},
        }
        self.spec = {
            "model_id": "test_model",
            "launch_receipt_schema": "test_detached_receipt_v1",
            "gpu_uuid": "GPU-test",
            "project_root": str(self.root),
            "plan_name": "plan.json",
            "rows_name": "rows.jsonl",
            "plan_bytes": 11,
            "plan_sha256": "a" * 64,
            "rows_bytes": 12,
            "rows_sha256": "b" * 64,
            "runtime_binding_path": str(self.root / "binding.json"),
            "runtime_binding_sha256": "c" * 64,
            "launch_source_pins": {
                "launcher": {"path": str(self.root / "launcher.py"), "bytes": 14, "sha256": "d" * 64},
            },
        }
        self.status = {
            "pid": 1234,
            "result_manifest_path": str(self.root / "output/block_000_attempt01/result_manifest.json"),
        }
        self.plan = {"run_id": "test_run"}
        self.row_ids = [f"row_{index}" for index in range(6)]
        self.launch_doc = {
            "schema": "test_detached_receipt_v1",
            "status": "STARTED_DETACHED",
            "model_id": "test_model",
            "run_id": "test_run",
            "block_index": 0,
            "block_start": 0,
            "block_rows": 6,
            "row_ids": self.row_ids,
            "pid": 1234,
            "output_root": str(self.root / "output/block_000_attempt01"),
            "gpu_uuid": "GPU-test",
            "plan": {"path": str(self.root / "plan.json"), "bytes": 11, "sha256": "a" * 64},
            "planned_rows": {"path": str(self.root / "rows.jsonl"), "bytes": 12, "sha256": "b" * 64},
            "binding": {"path": str(self.root / "binding.json"), "bytes": 13, "sha256": "c" * 64},
            "launcher": {"path": str(self.root / "launcher.py"), "bytes": 14, "sha256": "d" * 64},
            "intent_path": str(self.intent_path),
            "intent_bytes": len(intent_raw),
            "intent_raw_sha256": hashlib.sha256(intent_raw).hexdigest(),
        }

        def pinned(path: Path) -> dict:
            return {"path": str(path), **self.pin_values[str(path)]}

        self.pin_patch = mock.patch.object(INSPECTOR, "pin", side_effect=pinned)
        self.pin_patch.start()
        self.addCleanup(self.pin_patch.stop)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _validate(self, launch_doc: dict) -> list[str]:
        issues: list[str] = []
        INSPECTOR._validate_launch_receipt(
            launch_doc, self.launch_path, self.spec, self.plan, 0, 0,
            self.row_ids, self.status, issues,
        )
        return issues

    def test_receipt_binds_exact_sources_and_intent(self) -> None:
        doc = _with_self_hash(dict(self.launch_doc))
        self.assertEqual(self._validate(doc), [])

    def test_receipt_rejects_tampered_self_hash(self) -> None:
        doc = dict(self.launch_doc)
        doc["self_sha256"] = "0" * 64
        issues = self._validate(doc)
        self.assertIn("detached launch receipt self hash mismatch", issues)

    def test_receipt_rejects_source_pin_mismatch_even_with_valid_self_hash(self) -> None:
        doc = dict(self.launch_doc)
        doc["launcher"] = {**doc["launcher"], "sha256": "0" * 64}
        doc = _with_self_hash(doc)
        issues = self._validate(doc)
        self.assertIn("detached launch receipt launcher source pin mismatch", issues)


if __name__ == "__main__":
    unittest.main()
