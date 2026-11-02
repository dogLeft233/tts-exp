"""Focused import-namespace tests for the native model child launcher."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/experiments/cross_tfg_audio_head_20261003/"
          "fit40_spsa_model_child_attempt04.py")
SPEC = importlib.util.spec_from_file_location("child_attempt04_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
CHILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHILD)


def _pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class MuseNamespaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project_scripts = self.project / "scripts"
        self.project_audio = self.project_scripts / "audio_enhancement"
        self.muse_root = self.root / "musetalk"
        self.muse_scripts = self.muse_root / "scripts"
        self.project_audio.mkdir(parents=True)
        self.muse_scripts.mkdir(parents=True)
        (self.project_audio / "__init__.py").write_text("PACKAGE_ORIGIN = 'project'\n", encoding="utf-8")
        for filename in (
            "audio_io.py", "baselines.py", "pipeline.py", "overdub.py",
            "fit40_snapshot_spsa_application_attempt05.py",
            "fit40_snapshot_spsa_application_attempt07.py",
        ):
            (self.project_audio / filename).write_text("PINNED_SOURCE = " + repr(filename) + "\n", encoding="utf-8")
        (self.muse_scripts / "__init__.py").write_text("MUSE_PACKAGE = True\n", encoding="utf-8")
        self.paths = {
            "shared_audio_enhancement_init": self.project_audio / "__init__.py",
            "audio_io": self.project_audio / "audio_io.py",
            "shared_baselines": self.project_audio / "baselines.py",
            "shared_pipeline": self.project_audio / "pipeline.py",
            "shared_overdub": self.project_audio / "overdub.py",
            "snapshot_application_validator_attempt05": self.project_audio / "fit40_snapshot_spsa_application_attempt05.py",
            "snapshot_shared_application_helper_attempt07": self.project_audio / "fit40_snapshot_spsa_application_attempt07.py",
            "model_source_scripts___init___py": self.muse_scripts / "__init__.py",
        }
        self.package_entry = {"source_pins": {role: _pin(path) for role, path in self.paths.items()}}
        self.profile = {
            "project_root": str(self.project),
            "python_path_entries": [str(self.project), str(self.muse_root)],
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_isolated(self, callback):
        original_path = list(sys.path)
        original_modules = {
            name: module for name, module in tuple(sys.modules.items())
            if name == "scripts" or name.startswith("scripts.")
        }
        try:
            for name in original_modules:
                sys.modules.pop(name, None)
            sys.path[:0] = [str(self.project), str(self.muse_root)]
            for path in (str(self.project), str(self.muse_root)):
                sys.path_importer_cache.pop(path, None)
            importlib.invalidate_caches()
            callback()
        finally:
            for name in tuple(sys.modules):
                if name == "scripts" or name.startswith("scripts."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)
            sys.path[:] = original_path
            importlib.invalidate_caches()

    def test_regular_muse_package_keeps_origin_and_finds_project_audio_namespace(self) -> None:
        def check() -> None:
            import scripts
            self.assertEqual(Path(scripts.__file__).resolve(), self.muse_scripts / "__init__.py")
            self.assertFalse((self.project_scripts / "__init__.py").exists())
            report = CHILD._prepare_muse_project_scripts_namespace(self.package_entry, self.profile)
            self.assertEqual(Path(scripts.__file__).resolve(), self.muse_scripts / "__init__.py")
            self.assertEqual(list(scripts.__path__), [str(self.muse_scripts), str(self.project_scripts)])
            found = importlib.util.find_spec("scripts.audio_enhancement")
            self.assertIsNotNone(found)
            self.assertEqual(Path(found.origin).resolve(), self.project_audio / "__init__.py")
            self.assertEqual(report["muse_scripts_init"], str(self.muse_scripts / "__init__.py"))
            import scripts.audio_enhancement
            self.assertEqual(Path(scripts.audio_enhancement.__file__).resolve(), self.project_audio / "__init__.py")
            CHILD._prepare_muse_project_scripts_namespace(self.package_entry, self.profile)

        self._run_isolated(check)

    def test_rejects_wrong_loaded_scripts_package_origin(self) -> None:
        def check() -> None:
            wrong = self.root / "wrong" / "__init__.py"
            wrong.parent.mkdir(parents=True)
            wrong.write_text("", encoding="utf-8")
            module = types.ModuleType("scripts")
            module.__file__ = str(wrong)
            module.__path__ = [str(wrong.parent)]
            sys.modules["scripts"] = module
            with self.assertRaisesRegex(CHILD.ChildLaunchError, "not the exact pinned Muse package"):
                CHILD._prepare_muse_project_scripts_namespace(self.package_entry, self.profile)

        self._run_isolated(check)

    def test_rejects_preloaded_audio_package_from_unpinned_origin(self) -> None:
        def check() -> None:
            import scripts
            wrong = self.root / "wrong_audio" / "__init__.py"
            wrong.parent.mkdir(parents=True)
            wrong.write_text("", encoding="utf-8")
            module = types.ModuleType("scripts.audio_enhancement")
            module.__file__ = str(wrong)
            module.__path__ = [str(wrong.parent)]
            sys.modules["scripts.audio_enhancement"] = module
            with self.assertRaisesRegex(CHILD.ChildLaunchError, "unexpected search path"):
                CHILD._prepare_muse_project_scripts_namespace(self.package_entry, self.profile)

        self._run_isolated(check)

    def test_rejects_mismatched_audio_source_pin(self) -> None:
        bad = dict(self.package_entry)
        bad["source_pins"] = dict(self.package_entry["source_pins"])
        bad["source_pins"]["snapshot_application_validator_attempt05"] = dict(
            bad["source_pins"]["snapshot_application_validator_attempt05"], sha256="0" * 64)

        def check() -> None:
            with self.assertRaisesRegex(CHILD.ChildLaunchError, "raw bytes/SHA differ"):
                CHILD._prepare_muse_project_scripts_namespace(bad, self.profile)

        self._run_isolated(check)


if __name__ == "__main__":
    unittest.main()
