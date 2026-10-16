from __future__ import annotations

import array
import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/experiments/p1_gain_dose_response_20261016/prepare_sources.py"
SPEC = importlib.util.spec_from_file_location("p1_prepare_sources", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_sources)


def make_wav(samples: array.array, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


def make_tar_gz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TestSourcePreparation(unittest.TestCase):
    def test_han_text_rule(self) -> None:
        assert prepare_sources.text_is_han_only("  汉 字\t")
        assert not prepare_sources.text_is_han_only("")
        assert not prepare_sources.text_is_han_only("汉字，")

    def test_wav_quality_gates(self) -> None:
        quiet = array.array("h", [1000, -1000] * 24_000)
        audit = prepare_sources.wav_quality(make_wav(quiet))
        assert audit["duration_s"] == 3.0
        assert audit["finite"] and audit["nonzero"]
        assert audit["raw_unclipped"] and audit["fixed_rms_unclipped"]

        sparse_peak = array.array("h", [0] * 48_000)
        sparse_peak[0] = 32_766
        sparse_audit = prepare_sources.wav_quality(make_wav(sparse_peak))
        assert sparse_audit["raw_unclipped"]
        assert not sparse_audit["fixed_rms_unclipped"]

        rail = array.array("h", [32_767] * 48_000)
        rail_audit = prepare_sources.wav_quality(make_wav(rail))
        assert not rail_audit["raw_unclipped"]
        assert rail_audit["positive_full_scale_samples"] == 48_000

    def test_streamed_archive_extracts_target_and_hashes_full_source(self) -> None:
        inner = make_tar_gz({"train/S0102/BAC009S0102W0121.wav": make_wav(array.array("h", [1000] * 48_000))})
        outer = make_tar_gz({
            "data_aishell/wav/S0102.tar.gz": inner,
            prepare_sources.TRANSCRIPT_MEMBER: "BAC009S0102W0121 中文\n".encode("utf-8"),
            "trailing/after-target.bin": b"outer archive tail" * 100,
        })
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tgz"
            source.write_bytes(outer)
            extracted, transcript = prepare_sources.extract_candidate_archives(
                source,
                root / "inner",
                ("S0102",),
                hashlib.sha256(outer).hexdigest(),
                hashlib.md5(outer).hexdigest(),
                len(outer),
            )
            assert transcript == "BAC009S0102W0121 中文\n".encode("utf-8")
            assert extracted["S0102"]["inner_archive_sha256"] == hashlib.sha256(inner).hexdigest()

    def test_first_qualifying_wav_is_selected_by_name(self) -> None:
        invalid = make_wav(array.array("h", [1000] * 48_000))
        valid = make_wav(array.array("h", [1200, -1200] * 24_000))
        inner = make_tar_gz({
            "train/S0102/BAC009S0102W0121.wav": invalid,
            "train/S0102/BAC009S0102W0122.wav": valid,
        })
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "S0102.tar.gz"
            archive.write_bytes(inner)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            result = prepare_sources.inspect_speaker_archive(
                "S0102",
                archive,
                {"BAC009S0102W0121": "not han-only!", "BAC009S0102W0122": "天 地"},
                audio_dir,
            )
            assert result["utterance_id"] == "BAC009S0102W0122"
            assert result["rejected_before_selection"][0]["reason"] == "transcript_not_han_only"
            assert Path(result["audio_path"]).read_bytes() == valid


if __name__ == "__main__":
    import unittest

    unittest.main()
