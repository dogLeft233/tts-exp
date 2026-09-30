from __future__ import annotations

import importlib.util
import struct
import sys
import wave
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/experiments/qwen_tts_pair_prepare_20260930.py"
SPEC = importlib.util.spec_from_file_location("qwen_pair_prepare_20260930_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)


def test_preferred_voice_name_is_provider_safe_and_deterministic():
    first = qwen.preferred_voice_name("S0052_BAC009S0052W0121")
    assert first == qwen.preferred_voice_name("S0052_BAC009S0052W0121")
    assert len(first) <= 16
    assert first.startswith("ttsxp_")
    assert first.replace("_", "").isalnum()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("InvalidParameter: preferred_name must be 16 characters or fewer", "InvalidParameter: preferred_name must be 16 characters or fewer"),
        ("request failed https://example.invalid/signed/audio", None),
        ("Authorization: Bearer not-a-real-token", None),
        ("payload " + "A" * 80, None),
        ("API key was rejected", None),
        ("参数错误", None),
    ],
)
def test_safe_api_error_detail_whitelists_only_bounded_plain_text(message, expected):
    assert qwen._safe_error_detail({"message": message}, "do-not-print-this-key") == expected


def test_full_decode_count_does_not_trust_streamed_riff_length(tmp_path):
    pytest.importorskip("soundfile")
    path = tmp_path / "streamed.wav"
    frames = 1013
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x00\x01" * frames)
    raw = bytearray(path.read_bytes())
    # Same oversized RIFF/data-length sentinel found in the provider WAVs.
    raw[4:8] = struct.pack("<I", 0x7FFFFFBF)
    raw[40:44] = struct.pack("<I", 0x7FFFFF9B)
    path.write_bytes(raw)

    audit = qwen.full_audio_frame_audit(path, block_frames=127)
    assert audit["declared_frames"] == 1_073_741_773
    assert audit["decoded_frames"] == frames
    assert audit["declared_vs_payload_mismatch"] is True
    assert audit["sample_rate_hz"] == 24000


def test_input_manifest_fingerprint_binds_order_and_text():
    payload = qwen.add_fingerprint(
        {
            "schema_version": 1,
            "manifest_type": "qwen_tts_pair_input_manifest",
            "records": [
                {"id": "id-a", "speaker": "spk-a", "text": "你好", "split": "calibration", "ref_sha256": "abc"},
                {"id": "id-b", "speaker": "spk-b", "text": "早上好", "split": "evaluation", "ref_sha256": "def"},
            ],
        }
    )
    qwen.verify_fingerprint(payload)
    changed = dict(payload)
    changed["records"] = list(payload["records"])
    changed["records"][0] = dict(changed["records"][0], text="再见")
    with pytest.raises(ValueError, match="fingerprint"):
        qwen.verify_fingerprint(changed)

