from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py"
SPEC = importlib.util.spec_from_file_location("ditto_fit40_adapter_attempt05_test", ADAPTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load pinned Ditto adapter source for unit coverage")
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


class GeneratedAudioLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "generator_condition_f32.wav"
        self.raw = b"fixture bytes used only to exercise file identity gates"
        self.path.write_bytes(self.raw)
        self.file_sha = hashlib.sha256(self.raw).hexdigest()
        self.pcm_sha = hashlib.sha256(b"canonical float32 pcm fixture").hexdigest()
        self.ref = {
            "path": str(self.path.resolve()),
            "file_sha256": self.file_sha,
            "pcm_sha256": self.pcm_sha,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "sample_count": 8,
            "subtype": "IEEE_FLOAT_32_LE",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_validation(self, ref: dict[str, object]) -> dict[str, object]:
        info = types.SimpleNamespace(
            path=str(self.path.resolve()),
            file_sha256=self.file_sha,
            source_sample_rate=16_000,
            sample_rate=16_000,
            source_channels=1,
            source_frames=8,
            frames=8,
            source_encoding="FLOAT_32",
            pcm_float32_sha256=self.pcm_sha,
        )
        audio_io = types.ModuleType("scripts.audio_enhancement.audio_io")
        audio_io.read_canonical_wav = lambda path: (object(), info)
        with patch.dict(sys.modules, {"scripts.audio_enhancement.audio_io": audio_io}):
            return ADAPTER._validated_audio_ref_from_file(self.path, ref)

    def test_missing_lineage_bytes_are_derived_from_regular_file(self) -> None:
        verified = self.run_validation(self.ref)
        self.assertEqual(verified["bytes"], len(self.raw))
        self.assertEqual(verified["file_sha256"], self.file_sha)
        self.assertEqual(verified["pcm_sha256"], self.pcm_sha)
        self.assertNotIn("bytes", self.ref)

    def test_wrong_declared_file_size_is_rejected(self) -> None:
        ref = dict(self.ref, bytes=len(self.raw) + 1)
        with self.assertRaisesRegex(RuntimeError, "byte count differs"):
            self.run_validation(ref)

    def test_wrong_file_hash_is_rejected(self) -> None:
        ref = dict(self.ref, file_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "bytes, float32 PCM"):
            self.run_validation(ref)

    def test_wrong_pcm_hash_is_rejected(self) -> None:
        ref = dict(self.ref, pcm_sha256="f" * 64)
        with self.assertRaisesRegex(RuntimeError, "bytes, float32 PCM"):
            self.run_validation(ref)


if __name__ == "__main__":
    unittest.main()
