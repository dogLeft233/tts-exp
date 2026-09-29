"""Contract checks for blind data selection, with no learned models."""
import importlib.util
import io
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

SPEC = importlib.util.spec_from_file_location(
    "prepare_native_holdout", Path(__file__).resolve().parents[2] / "scripts/experiments/prepare_native_holdout.py"
)
holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdout)


def test_lexical_selection_duration_boundaries_and_original_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    monkeypatch.setattr(holdout, "DATA", tmp_path / "data")
    archives = holdout.DATA / "archives"
    archives.mkdir(parents=True)
    raw = {}
    with tarfile.open(archives / "S0002.tar.gz", "w:gz") as archive:
        # Reverse archive order must not influence selection.
        for number, duration in [(4, 4), (3, 8), (2, 3), (1, 2)]:
            uid = f"BAC009S0002W{number:04d}"
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(duration * 16000), 16000, format="WAV", subtype="PCM_16")
            raw[uid] = buffer.getvalue()
            info = tarfile.TarInfo(f"train/S0002/{uid}.wav")
            info.size = len(raw[uid])
            archive.addfile(info, io.BytesIO(raw[uid]))
    selected, decisions = holdout.select("S0002", {uid: "人工转写" for uid in raw}, set())
    assert [row["utterance_id"] for row in selected] == ["BAC009S0002W0002", "BAC009S0002W0003"]
    assert [row["decision"] for row in decisions] == [
        "duration_outside_3_to_8_seconds", "selected", "selected", "not_evaluated_after_first_two_eligible"
    ]
    for row in selected:
        assert (tmp_path / row["audio_path"]).read_bytes() == raw[row["utterance_id"]]


def test_pcm_fingerprint_binds_rate_channels_and_values():
    audio = np.array([[0.0], [0.5]], dtype=np.float64)
    first = holdout.pcm_hash(audio, 16000)
    assert first == holdout.pcm_hash(audio.astype(np.float32), 16000)
    assert first != holdout.pcm_hash(audio, 24000)
    assert first != holdout.pcm_hash(audio.reshape(1, 2), 16000)
    assert first != holdout.pcm_hash(-audio, 16000)
