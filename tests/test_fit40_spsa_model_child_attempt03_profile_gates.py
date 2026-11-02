"""CPU-only profile and adapter-dispatch checks for child03.

These tests read pinned source and exercise only metadata gates with fake
constructors. They never import a project model, instantiate a renderer, or
score a row.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CHILD_PATH = REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_model_child_attempt03.py"
if CHILD_PATH.stat().st_size != 19_413 or hashlib.sha256(CHILD_PATH.read_bytes()).hexdigest() != (
        "55d6b9c82c181a5bcfa2c8b9a12f70822217212178d6d913d9bd8dace18ee63c"):
    raise RuntimeError("child03 source differs from the pinned request-load fix")
SPEC = importlib.util.spec_from_file_location("fit40_spsa_model_child_attempt03_profile_test", CHILD_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load child03 metadata gates")
CHILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHILD)


ADAPTER_PINS = {
    "ditto": ("scripts/experiments/cross_tfg_audio_head_20261003/ditto_spsa_stagea_adapter_attempt11.py",
              61_793, "474734d9c7d7ec78b97cabbb1fed799254ad21373d66ca0301f5de1779d30fff",
              "DittoSPSAStageAAdapter"),
    "muse": ("scripts/experiments/cross_tfg_audio_head_20261003/musetalk_spsa_stagea_adapter_attempt09.py",
             76_887, "2301a0817a3f94d5130c221d1141247b502bc14975aa93c89006cc73eedb0b67",
             "MuseSPSAStageAAdapter"),
    "w2l": ("scripts/experiments/cross_tfg_audio_head_20261003/w2l_spsa_stagea_adapter_attempt05.py",
            53_822, "890700d4575f39b8a33c965d841d0ad7ab6fa94d70f4542596e1721ca390b1c4",
            "W2LSPSAStageAAdapter"),
}
POINT_RUNNER_PATH = REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_execution_attempt04.py"
POINT_RUNNER_PIN = (67_081, "7747052df91393f1cbf1138dc76523b69f888674d0ab6e07c7fa9cfadfcf1e46")


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def self_pin(value: dict, field: str) -> str:
    return hashlib.sha256(canonical({key: item for key, item in value.items() if key != field})).hexdigest()


def module_class_methods(path: Path, class_name: str) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise AssertionError(f"expected one {class_name} in {path}")
    result = {}
    for node in classes[0].body:
        if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "render_subset"}:
            result[node.name] = tuple(arg.arg for arg in (*node.args.args, *node.args.kwonlyargs))
    return result


class Child03ProfileAndDispatchTests(unittest.TestCase):
    def test_exact_native_adapter_constructor_and_render_signatures_are_pinned(self) -> None:
        expected = {
            "ditto": {
                "__init__": ("self", "core", "ditto_overlay", "point_plan", "point_plan_pin",
                             "resource_helper", "args"),
                "render_subset": ("self", "candidate_rows", "method", "output_root"),
            },
            "muse": {
                "__init__": ("self", "core", "muse_overlay", "cross_frontend", "parent_plan",
                             "runtime_binding", "runtime_binding_raw", "runtime_binding_path",
                             "family_plan_path", "point_plan", "point_plan_path", "point_plan_file_pin",
                             "point_plan_self_sha256", "resource_helper", "args", "strict_receipt_dir"),
                "render_subset": ("self", "candidate_rows", "method", "output_root"),
            },
            "w2l": {
                "__init__": ("self", "core", "family_plan", "family_plan_pin", "point_plan",
                             "point_plan_pin", "args"),
                "render_subset": ("self", "candidate_rows", "output_root"),
            },
        }
        for name, (relative, size, digest, class_name) in ADAPTER_PINS.items():
            path = REPO_ROOT / relative
            raw = path.read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (size, digest))
            self.assertEqual(module_class_methods(path, class_name), expected[name])

    def test_pinned_point_runner_omits_method_only_for_actual_w2l_api(self) -> None:
        raw = POINT_RUNNER_PATH.read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), POINT_RUNNER_PIN)
        tree = ast.parse(raw.decode("utf-8"), filename=str(POINT_RUNNER_PATH))
        execute = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute_point_block"]
        self.assertEqual(len(execute), 1)
        calls = []
        for node in ast.walk(execute[0]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "render_subset"):
                calls.append(tuple(keyword.arg for keyword in node.keywords))
        self.assertCountEqual(calls, [
            ("candidate_rows", "output_root"),
            ("candidate_rows", "method", "output_root"),
        ])

    def test_child_gates_point_package_before_profile_adapter_or_execution(self) -> None:
        tree = ast.parse(CHILD_PATH.read_text(encoding="utf-8"), filename=str(CHILD_PATH))
        execute = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "execute_request"]
        self.assertEqual(len(execute), 1)
        call_nodes = [node for node in ast.walk(execute[0]) if isinstance(node, ast.Call)]
        call_nodes.sort(key=lambda node: (node.lineno, node.col_offset))
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in call_nodes if isinstance(node.func, (ast.Attribute, ast.Name))]
        self.assertLess(calls.index("validate_point_scope"), calls.index("_load_launch_profile"))
        self.assertLess(calls.index("_load_launch_profile"), calls.index("_adapter_factory_factory"))
        self.assertLess(calls.index("_adapter_factory_factory"), calls.index("execute_point_block"))

    def profile_fixture(self, root: Path, model_id: str = CHILD.MODEL_ORDER[0]) -> tuple[dict, Path, dict]:
        import sys

        project = root / "project"
        project.mkdir()
        profile_path = root / "profile.json"
        runtime_path = root / "runtime.json"
        runtime_path.write_text("{}\n", encoding="utf-8")
        runtime_raw = runtime_path.read_bytes()
        runtime_pin = {"path": str(runtime_path), "bytes": len(runtime_raw),
                       "sha256": hashlib.sha256(runtime_raw).hexdigest()}
        environment = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                       "PYTHONNOUSERSITE": "1"}
        profile = {
            "schema": CHILD.PROFILE_SCHEMA,
            "status": "FROZEN",
            "model_id": model_id,
            "interpreter": {
                "invocation_path": os.path.abspath(sys.executable),
                "resolved_path": os.path.realpath(sys.executable),
                "prefix": os.path.realpath(sys.prefix),
                "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            },
            "runtime_binding_pin": runtime_pin,
            "project_root": str(project),
            "python_path_entries": [str(project)],
            "environment": environment,
            "adapter_args": {},
        }
        profile["profile_sha256"] = self_pin(profile, "profile_sha256")
        raw = json.dumps(profile, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        profile_path.write_bytes(raw)
        launch_pin = {"path": str(profile_path), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "self_sha256": profile["profile_sha256"]}
        package_entry = {"source_pins": {"launch_profile": launch_pin},
                         "runtime_binding_pin": runtime_pin}
        context = {"family_plan": {"execution": {"source_model_execution": {
            model_id: {"capacity_stage": "fixture-capacity-stage"}}}}}
        request = {"strict_receipt_dir": str(root / "block" / "strict"),
                   "runtime_binding_path": str(runtime_path),
                   "runtime_binding_sha256": runtime_pin["sha256"],
                   "block_root": str(root / "block")}
        return {"profile": profile, "environment": environment, "project": project,
                "package_entry": package_entry, "runtime_pin": runtime_pin}, profile_path, (context, request)

    def test_launch_profile_accepts_exact_interpreter_runtime_and_clean_environment(self) -> None:
        import sys

        previous_path = list(sys.path)
        try:
            for model_id in CHILD.MODEL_ORDER:
                with self.subTest(model_id=model_id), tempfile.TemporaryDirectory(
                        prefix="fit40-child03-profile-") as directory:
                    fixture, _path, (context, request) = self.profile_fixture(Path(directory), model_id)
                    with mock.patch.dict(os.environ, fixture["environment"], clear=True):
                        loaded, pin = CHILD._load_launch_profile(
                            fixture["package_entry"], model_id, context, request)
                    self.assertEqual(loaded["profile_sha256"], fixture["profile"]["profile_sha256"])
                    self.assertEqual(pin["self_sha256"], fixture["profile"]["profile_sha256"])
        finally:
            sys.path[:] = previous_path

    def test_module_pin_mismatch_fails_before_module_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fit40-child03-module-guard-") as directory:
            path = Path(directory) / "fixture_module.py"
            path.write_text("raise RuntimeError('module must not execute')\n", encoding="utf-8")
            raw = path.read_bytes()
            wrong_pin = {"path": str(path), "bytes": len(raw), "sha256": "0" * 64}
            with self.assertRaisesRegex(CHILD.ChildLaunchError, "raw bytes/SHA"):
                CHILD._load_module(wrong_pin, "fixture module", "never_imported_fit40_fixture")
            self.assertNotIn("never_imported_fit40_fixture", __import__("sys").modules)

    def test_launch_profile_rejects_profile_model_runtime_and_extra_environment(self) -> None:
        import sys

        for mutation, extra_env in (("model", False), ("runtime", False), ("env", True)):
            previous_path = list(sys.path)
            try:
                with tempfile.TemporaryDirectory(prefix="fit40-child03-profile-bad-") as directory:
                    fixture, path, (context, request) = self.profile_fixture(Path(directory))
                    profile = fixture["profile"]
                    package_runtime_pin = dict(fixture["package_entry"]["runtime_binding_pin"])
                    if mutation == "model":
                        profile["model_id"] = CHILD.MODEL_ORDER[1]
                    elif mutation == "runtime":
                        profile["runtime_binding_pin"]["sha256"] = "0" * 64
                    profile["profile_sha256"] = self_pin(profile, "profile_sha256")
                    raw = json.dumps(profile, sort_keys=True, indent=2).encode("utf-8") + b"\n"
                    path.write_bytes(raw)
                    package = dict(fixture["package_entry"])
                    package["runtime_binding_pin"] = package_runtime_pin
                    package["source_pins"] = {"launch_profile": {
                        "path": str(path), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "self_sha256": profile["profile_sha256"],
                    }}
                    env = dict(fixture["environment"])
                    if extra_env:
                        env["UNEXPECTED_PROFILE_ENV"] = "1"
                    with mock.patch.dict(os.environ, env, clear=True):
                        with self.assertRaises(CHILD.ChildLaunchError):
                            CHILD._load_launch_profile(package, CHILD.MODEL_ORDER[0], context, request)
            finally:
                sys.path[:] = previous_path

    def test_model_factory_routes_exact_constructor_kwargs_without_loading_models(self) -> None:
        for model_id in CHILD.MODEL_ORDER:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory(
                    prefix="fit40-child03-factory-") as directory:
                root = Path(directory)
                block_root = root / "block"
                block_root.mkdir()
                runtime_path = root / "runtime.json"
                runtime_path.write_text("{}\n", encoding="utf-8")
                runtime_raw = runtime_path.read_bytes()
                runtime = {"binding_sha256": "b" * 64}
                pin = {"path": str(runtime_path), "bytes": len(runtime_raw),
                       "sha256": hashlib.sha256(runtime_raw).hexdigest()}
                seen = []

                class Capture:
                    def __init__(self, **kwargs):
                        seen.append(kwargs)

                if model_id == CHILD.MODEL_ORDER[0]:
                    cls_name = "W2LSPSAStageAAdapter"
                    source_pins = {}
                    expected = {"core", "family_plan", "family_plan_pin", "point_plan",
                                "point_plan_pin", "args"}
                    loaded = [object(), SimpleNamespace(**{cls_name: Capture})]
                elif model_id == CHILD.MODEL_ORDER[1]:
                    cls_name = "MuseSPSAStageAAdapter"
                    source_pins = {name: {} for name in (
                        "musetalk_block_overlay", "cross_frontend_runner", "resource_helper")}
                    expected = {"core", "muse_overlay", "cross_frontend", "parent_plan",
                                "runtime_binding", "runtime_binding_raw", "runtime_binding_path",
                                "family_plan_path", "point_plan", "point_plan_path", "point_plan_file_pin",
                                "point_plan_self_sha256", "resource_helper", "args", "strict_receipt_dir"}
                    loaded = [object(), SimpleNamespace(**{cls_name: Capture}), object(), object(), object()]
                else:
                    cls_name = "DittoSPSAStageAAdapter"
                    source_pins = {name: {} for name in ("ditto_block_overlay", "resource_helper")}
                    expected = {"core", "ditto_overlay", "point_plan", "point_plan_pin",
                                "resource_helper", "args"}
                    loaded = [object(), SimpleNamespace(**{cls_name: Capture}), object(), object()]

                request = {"model_id": model_id, "strict_receipt_dir": str(block_root / "strict"),
                           "block_root": str(block_root)}
                package = {"runner_pin": {}, "adapter_pin": {}, "source_pins": source_pins,
                           "runtime_binding_pin": pin}
                profile = {"adapter_args": {"fixture_arg": "bound"}, "project_root": str(root)}
                context = {"family_plan": {"execution": {"source_model_execution": {
                    model_id: {"capacity_stage": "exact-capacity-stage"}}}},
                    "family_plan_pin": {"path": str(root / "family.json"), "bytes": 1,
                                        "sha256": "c" * 64},
                    "point_plan": {"point": "plan"},
                    "point_plan_pin": {"path": str(root / "point.json"), "bytes": 2,
                                       "sha256": "d" * 64, "self_sha256": "e" * 64}}
                with mock.patch.object(CHILD, "_load_module", side_effect=loaded) as load_module, \
                     mock.patch.object(CHILD, "_read_pin", return_value=(pin, runtime_raw, runtime)):
                    factory = CHILD._adapter_factory_factory(request, object(), package, profile)
                    factory(context)
                self.assertEqual(len(seen), 1)
                self.assertEqual(set(seen[0]), expected)
                self.assertEqual(seen[0]["args"].fixture_arg, "bound")
                self.assertEqual(load_module.call_count, len(loaded))
                if model_id == CHILD.MODEL_ORDER[0]:
                    self.assertNotIn("method", seen[0])
                if model_id == CHILD.MODEL_ORDER[1]:
                    self.assertTrue((block_root / "strict").is_dir())


if __name__ == "__main__":
    unittest.main()
