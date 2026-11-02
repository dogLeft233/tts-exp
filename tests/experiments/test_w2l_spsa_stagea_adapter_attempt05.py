from __future__ import annotations

import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[2] / "scripts" / "experiments" /
          "cross_tfg_audio_head_20261003" / "w2l_spsa_stagea_adapter_attempt05.py")
SPEC = importlib.util.spec_from_file_location("w2l_spsa_adapter_attempt05_tested", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load attempt05 source")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def _pcm16_wav() -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    samples = struct.pack("<hhhh", 0, 1200, -1200, 0)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(samples)) + samples)
    return b"RIFF" + struct.pack("<I", len(body)) + body


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
        raw = _pcm16_wav()
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

    def test_source_subtype_and_clock_are_read_from_actual_riff_header(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _path, natural, plan, core = self._fixture(root)
            actual = ADAPTER._validated_source_natural_context(
                core, root, plan, "s01", {"natural_pcm": natural})
            self.assertEqual(actual["subtype"], "PCM_16")
            self.assertEqual(actual["bytes"], len(_pcm16_wav()))
            self.assertEqual(ADAPTER._source_wav_header(_pcm16_wav()), {
                "subtype": "PCM_16", "channels": 1, "sample_rate_hz": 16000,
                "block_align": 2, "sample_count": 4,
            })

    def test_wrong_declared_source_subtype_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _path, natural, plan, core = self._fixture(root)
            natural["subtype"] = "FLOAT"
            with self.assertRaises(ADAPTER.W2LSPSAAdapterError):
                ADAPTER._validated_source_natural_context(
                    core, root, plan, "s01", {"natural_pcm": natural})

    def test_wrong_input_raw_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            _path, natural, plan, core = self._fixture(root)
            plan["scientific_input_pins"]["natural_pcm_s01"]["sha256"] = "0" * 64
            with self.assertRaises(ADAPTER.W2LSPSAAdapterError):
                ADAPTER._validated_source_natural_context(
                    core, root, plan, "s01", {"natural_pcm": natural})

    def test_truncated_riff_is_rejected(self):
        with self.assertRaises(ADAPTER.W2LSPSAAdapterError):
            ADAPTER._source_wav_header(_pcm16_wav()[:-1])


if __name__ == "__main__":
    unittest.main()
