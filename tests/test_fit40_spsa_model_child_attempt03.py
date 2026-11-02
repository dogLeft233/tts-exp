from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" / "experiments"
          / "cross_tfg_audio_head_20261003" / "fit40_spsa_model_child_attempt03.py")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_model_child_attempt03_test", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the exact model child source")
CHILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHILD)


class StopBeforeSourceLoading(RuntimeError):
    pass


class ModelChildRequestLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="fit40-model-child03-")
        self.root = Path(self.tempdir.name)
        self.request_path = self.root / "block-request.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def request_bytes() -> bytes:
        body = {
            "schema": CHILD.REQUEST_SCHEMA,
            "status": "READY",
            "model_id": CHILD.MODEL_ORDER[0],
            "point_runner_pin": {},
        }
        body["request_sha256"] = hashlib.sha256(CHILD._canonical(body)).hexdigest()
        return (json.dumps(body, sort_keys=True, ensure_ascii=False,
                           allow_nan=False) + "\n").encode("utf-8")

    def write_valid_request(self) -> Path:
        self.request_path.write_bytes(self.request_bytes())
        return self.request_path

    def assert_request_stops_before_imports(self, invoke) -> None:
        self.write_valid_request()
        with mock.patch.object(CHILD, "_read_pin", side_effect=StopBeforeSourceLoading) as read_pin, \
             mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")) as load_module, \
             mock.patch.object(CHILD, "_adapter_factory_factory",
                               side_effect=AssertionError("adapter factory reached")) as adapter_factory:
            with self.assertRaises(StopBeforeSourceLoading):
                invoke(self.request_path)
        read_pin.assert_called_once()
        load_module.assert_not_called()
        adapter_factory.assert_not_called()

    def test_validate_request_accepts_pathlike_and_string(self) -> None:
        path = self.write_valid_request()
        expected = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(CHILD._validate_request(path), expected)
        self.assertEqual(CHILD._validate_request(str(path)), expected)

    def test_execute_request_loads_absolute_path_before_any_source_or_adapter_load(self) -> None:
        self.assert_request_stops_before_imports(CHILD.execute_request)

    def test_cli_loads_request_before_any_source_or_adapter_load(self) -> None:
        self.write_valid_request()
        with mock.patch.object(CHILD, "_read_pin", side_effect=StopBeforeSourceLoading) as read_pin, \
             mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")) as load_module, \
             mock.patch.object(CHILD, "_adapter_factory_factory",
                               side_effect=AssertionError("adapter factory reached")) as adapter_factory:
            with self.assertRaises(StopBeforeSourceLoading):
                CHILD.main(["--request", str(self.request_path)])
        read_pin.assert_called_once()
        load_module.assert_not_called()
        adapter_factory.assert_not_called()

    def test_malformed_json_fails_before_source_or_adapter_load(self) -> None:
        self.request_path.write_bytes(b"{malformed\n")
        with mock.patch.object(CHILD, "_read_pin", side_effect=AssertionError("source gate reached")), \
             mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")):
            with self.assertRaises(CHILD.ChildLaunchError):
                CHILD.execute_request(self.request_path)

    def test_invalid_request_self_hash_fails_before_source_or_adapter_load(self) -> None:
        body = json.loads(self.request_bytes())
        body["request_sha256"] = "0" * 64
        self.request_path.write_text(json.dumps(body), encoding="utf-8")
        with mock.patch.object(CHILD, "_read_pin", side_effect=AssertionError("source gate reached")), \
             mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")):
            with self.assertRaises(CHILD.ChildLaunchError):
                CHILD.execute_request(self.request_path)

    def test_relative_request_path_is_rejected_before_validation(self) -> None:
        self.write_valid_request()
        original_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            with mock.patch.object(CHILD, "_validate_request",
                                   side_effect=AssertionError("request validation reached")), \
                 mock.patch.object(CHILD, "_read_pin", side_effect=AssertionError("source gate reached")), \
                 mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")):
                with self.assertRaises(CHILD.ChildLaunchError):
                    CHILD.execute_request("block-request.json")
        finally:
            os.chdir(original_cwd)

    def test_symlink_request_path_is_rejected_before_validation(self) -> None:
        self.write_valid_request()
        symlink_path = self.root / "request-link.json"
        try:
            symlink_path.symlink_to(self.request_path)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("symlink creation is unavailable: " + str(exc))
        with mock.patch.object(CHILD, "_validate_request",
                               side_effect=AssertionError("request validation reached")), \
             mock.patch.object(CHILD, "_read_pin", side_effect=AssertionError("source gate reached")), \
             mock.patch.object(CHILD, "_load_module", side_effect=AssertionError("module load reached")):
            with self.assertRaises(CHILD.ChildLaunchError):
                CHILD.execute_request(symlink_path)


if __name__ == "__main__":
    unittest.main()
