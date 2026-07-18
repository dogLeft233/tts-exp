import json
import shutil
import sys
import tempfile
import os
from pathlib import Path
from importlib import util

spec = util.spec_from_file_location(
    "librispeech_phn",
    Path(__file__).resolve().parent.parent / "scripts" / "27_download_librispeech_phn.py",
)
mod = util.module_from_spec(spec)


def test_parse_phn_line_basic():
    sample_text = """0 470000 h#
470000 580000 k
580000 700000 ae
700000 1500000 t
1500000 1600000 h#
"""
    with tempfile.NamedTemporaryFile("w", suffix=".phn", delete=False) as f:
        f.write(sample_text)
        path = Path(f.name)
    try:
        spec.loader.exec_module(mod)
        tokens = mod.parse_phn_file(path, sample_rate=16000)
        assert len(tokens) == 5
        assert tokens[0]["token"] == "h#"
        assert abs(tokens[0]["start_s"] - 0.0) < 1e-6
        assert abs(tokens[0]["end_s"] - 470000/16000) < 1e-6
        assert tokens[1]["token"] == "k"
        assert abs(tokens[1]["start_s"] - 470000/16000) < 1e-6
    finally:
        os.unlink(path)


def test_silence_labels_detected():
    sample_text = """0 1000 h#
1000 50000 sil
50000 60000 sp
60000 70000 aa
70000 71000 h#
"""
    with tempfile.NamedTemporaryFile("w", suffix=".phn", delete=False) as f:
        f.write(sample_text)
        path = Path(f.name)
    try:
        spec.loader.exec_module(mod)
        tokens = mod.parse_phn_file(path, sample_rate=16000)
        silences = [t for t in tokens if t["token"] in mod.SILENCE_TOKENS]
        assert len(silences) == 4
    finally:
        os.unlink(path)


def test_arpabet_to_viseme_known_classes():
    spec_a = util.spec_from_file_location(
        "english_alignment",
        Path(__file__).resolve().parent.parent / "scripts" / "28_prepare_english_alignment.py",
    )
    mod_a = util.module_from_spec(spec_a)
    spec_a.loader.exec_module(mod_a)
    assert mod_a.arpabet_to_viseme("P") == "pbmv"
    assert mod_a.arpabet_to_viseme("IY") == "i"
    assert mod_a.arpabet_to_viseme("h#") == "sil"
    assert mod_a.arpabet_to_viseme("UNKNOWN") == "other"


def test_build_manifest_filters_silence_from_viseme():
    fake_phn = """0 10000 h#
10000 50000 K
50000 90000 UW
90000 100000 h#
"""
    tmp = Path(tempfile.mkdtemp())
    fake_phn_path = tmp / "fake-9999-0001.phn"
    fake_phn_path.write_text(fake_phn)
    audio_manifest = tmp / "audio_manifest.json"
    audio_manifest.write_text(json.dumps([{
        "sample_id": 1, "librispeech_id": "fake-9999-0001",
        "text": "FAKE", "duration_s": 6.25
    }]))
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    (audio_dir / "1.wav").write_bytes(b"")

    spec_a = util.spec_from_file_location(
        "english_alignment",
        Path(__file__).resolve().parent.parent / "scripts" / "28_prepare_english_alignment.py",
    )
    mod_a = util.module_from_spec(spec_a)
    spec_a.loader.exec_module(mod_a)
    tokens = mod_a.parse_phn_file(fake_phn_path, sample_rate=16000)
    for t in tokens:
        t["viseme"] = mod_a.arpabet_to_viseme(t["token"])
    assert tokens[0]["viseme"] == "sil"
    assert tokens[1]["viseme"] == "kg"
    assert tokens[2]["viseme"] == "u"
    assert tokens[3]["viseme"] == "sil"
    non_sil = [t for t in tokens if t["viseme"] != "sil"]
    assert len(non_sil) == 2

    shutil.rmtree(tmp)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
