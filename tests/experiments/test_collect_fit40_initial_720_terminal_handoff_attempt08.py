from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    collect_fit40_initial_720_terminal_handoff_attempt08 as collector,
    fit_only_stagea_ingest_attempt03 as ingest,
)


_INGEST_FIXTURE_PATH = Path(__file__).with_name(
    "test_fit_only_stagea_ingest_attempt03.py")
_INGEST_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "fit_only_stagea_ingest_attempt03_fixture", _INGEST_FIXTURE_PATH)
assert _INGEST_FIXTURE_SPEC is not None and _INGEST_FIXTURE_SPEC.loader is not None
_INGEST_FIXTURE = importlib.util.module_from_spec(_INGEST_FIXTURE_SPEC)
_INGEST_FIXTURE_SPEC.loader.exec_module(_INGEST_FIXTURE)


class RelativeScientificPinMaterializationTest(unittest.TestCase):
    def _build_producer_shaped_input_pins(self, root: Path):
        fixture_builder = _INGEST_FIXTURE.FitStageAIngestTests()
        _, index = fixture_builder._make_packet(root / "fixture")
        original_files = ingest._file_table(index)
        first_model = next(item for item in index["models"]
                           if item["model_id"] == "model_w2l")
        plan_pin = ingest._required_file(
            original_files, first_model["plan_file_id"], "plan")
        plan = json.loads(ingest._read_pinned(plan_pin).decode("utf-8"))
        input_pins = plan["scientific_input_pins"]
        source_root = root / "staged_project"
        source_root.mkdir()

        rewritten_files = [entry for entry in index["files"]
                           if entry["role"] != "input_pin"]
        file_ids = {entry["file_id"] for entry in rewritten_files}
        input_pin_files = {}
        for pin_name, pin in sorted(input_pins.items()):
            rel = PurePosixPath(pin["path"])
            self.assertFalse(rel.is_absolute())
            self.assertNotIn("..", rel.parts)
            source = source_root.joinpath(*rel.parts)
            source.parent.mkdir(parents=True, exist_ok=True)
            original = ingest._required_file(
                original_files, index["input_pin_files"][pin_name], "input_pin")
            source.write_bytes(ingest._read_pinned(original))
            safe_source = collector._require_inside_output_leaf(
                source, source_root, f"fixture scientific input {pin_name}")
            file_id = f"input_pin:{pin_name}"
            emitted = collector._add_file(
                rewritten_files, file_ids, file_id, "input_pin", pin["path"],
                pin_name=pin_name, source_path=safe_source)
            self.assertEqual(emitted["declared_path"], pin["path"])
            self.assertTrue(Path(emitted["local_path"]).is_absolute())
            self.assertEqual(emitted["bytes"], pin["bytes"])
            self.assertEqual(emitted["sha256"], pin["sha256"])
            input_pin_files[pin_name] = file_id

        index["files"] = rewritten_files
        index["input_pin_files"] = input_pin_files
        index["index_sha256"] = ingest._digest(ingest._canonical(
            {key: value for key, value in index.items() if key != "index_sha256"}))
        materialized = ingest._file_table(index)
        specs = ingest._load_snapshot_points(
            index, materialized, input_pins,
            {item["method_id"]: item for item in plan["methods"]})
        self.assertEqual(tuple(item.source_step for item in specs), (30, 60, 90, 120))
        return index, materialized, input_pins

    def test_collector_relative_declared_path_is_consumable_by_frozen_ingest(self):
        with tempfile.TemporaryDirectory() as temp:
            index, files, input_pins = self._build_producer_shaped_input_pins(Path(temp))
            for pin_name, file_id in index["input_pin_files"].items():
                emitted = files[file_id]
                self.assertEqual(emitted.declared_path, input_pins[pin_name]["path"])
                self.assertTrue(Path(emitted.local_path).is_absolute())

    def test_absolute_declared_path_is_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as temp:
            index, _, input_pins = self._build_producer_shaped_input_pins(Path(temp))
            pin_name = "snapshot_index"
            file_id = index["input_pin_files"][pin_name]
            pin_file = next(item for item in index["files"]
                            if item["file_id"] == file_id)
            pin_file["declared_path"] = str(Path(pin_file["local_path"]))
            files = ingest._file_table(index)
            first_model = next(item for item in index["models"]
                               if item["model_id"] == "model_w2l")
            plan_pin = ingest._required_file(files, first_model["plan_file_id"], "plan")
            plan = json.loads(ingest._read_pinned(plan_pin).decode("utf-8"))
            with self.assertRaisesRegex(ingest.IngestError,
                                       "materialized input pin differs from common plan: snapshot_index"):
                ingest._load_snapshot_points(
                    index, files, input_pins,
                    {item["method_id"]: item for item in plan["methods"]})


if __name__ == "__main__":
    unittest.main()
