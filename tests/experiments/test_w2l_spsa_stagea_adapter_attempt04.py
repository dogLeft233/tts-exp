from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "cross_tfg_audio_head_20261003" / "w2l_spsa_stagea_adapter_attempt04.py"
SPEC = importlib.util.spec_from_file_location("w2l_spsa_adapter_attempt04_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load attempt04 source")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class _Samples:
    size = 4


class _Core:
    def __init__(self, path: Path, file_sha: str, pcm_sha: str) -> None:
        self.path = path
        self.file_sha = file_sha
        self.pcm_sha = pcm_sha

    def _reference(self, ref):
        return self.path, _Samples(), {
            "file_sha256": self.file_sha,
            "pcm_sha256": self.pcm_sha,
            "sample_count": 4,
            "sample_rate_hz": 16000,
            "channels": 1,
        }


class W2LSourceNaturalContextTests(unittest.TestCase):
    def _fixture(self, root: Path):
        path = root / "audio.wav"
        raw = b"pinned test audio bytes"
        path.write_bytes(raw)
        file_sha = hashlib.sha256(raw).hexdigest()
        pcm_sha = hashlib.sha256(b"canonical f32 test samples").hexdigest()
        natural = {
            "path": str(path.resolve()),
            "file_sha256": file_sha,
            "pcm_sha256": pcm_sha,
            "sample_count": 4,
            "sample_rate_hz": 16000,
            "channels": 1,
            "subtype": "PCM_16",
        }
        plan = {"scientific_input_pins": {
            "natural_pcm_s01": {"path": "audio.wav", "bytes": len(raw), "sha256": file_sha}
        }}
        return path, natural, plan, _Core(path, file_sha, pcm_sha)

    def test_row_without_byte_count_matches_measured_shared_helper_context(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _path, natural, plan, core = self._fixture(root)
            actual = ADAPTER._validated_source_natural_context(
                core, root, plan, "s01", {"natural_pcm": natural})
            self.assertEqual(actual, {
                "path": str(root / "audio.wav"),
                "bytes": len(b"pinned test audio bytes"),
                "file_sha256": natural["file_sha256"],
                "pcm_sha256": natural["pcm_sha256"],
                "sample_count": 4,
                "sample_rate_hz": 16000,
                "channels": 1,
                "subtype": "PCM_16",
            })

    def test_wrong_input_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _path, natural, plan, core = self._fixture(root)
            plan["scientific_input_pins"]["natural_pcm_s01"]["sha256"] = "0" * 64
            with self.assertRaises(ADAPTER.W2LSPSAAdapterError):
                ADAPTER._validated_source_natural_context(
                    core, root, plan, "s01", {"natural_pcm": natural})


if __name__ == "__main__":
    unittest.main()
