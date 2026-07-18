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


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
