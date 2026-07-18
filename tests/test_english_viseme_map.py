from pathlib import Path
import yaml

VISEME_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "data" / "english_viseme_map.yaml"

# ARPABET phone set used by LibriSpeech (39 phones, TIMIT convention)
LIBRISPEECH_ARPABET = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G",
    "HH",
    "IH", "IY",
    "JH",
    "K", "L",
    "M", "N", "NG",
    "OW", "OY",
    "P", "R",
    "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V",
    "W", "Y",
    "Z", "ZH",
}


def test_viseme_map_exists_and_loads():
    assert VISEME_MAP_PATH.exists(), f"missing: {VISEME_MAP_PATH}"
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "arpabet_to_viseme" in data
    assert "silence_labels" in data


def test_all_arpabet_phones_mapped():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    mapping = data["arpabet_to_viseme"]
    unmapped = [p for p in LIBRISPEECH_ARPABET if p not in mapping]
    assert not unmapped, f"unmapped ARPABET phones: {unmapped}"


def test_thirteen_viseme_classes():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    used = set(data["arpabet_to_viseme"].values())
    expected = {"pbmv", "fv", "th", "cdsz", "kg", "chjsh", "e", "o", "i", "u", "r", "ai", "aw"}
    assert used == expected, f"viseme set mismatch; got {used}"


def test_silence_labels_listed():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    silence_labels = data["silence_labels"]
    for required in ["h#", "sil", "sp", "spn", ""]:
        assert required in silence_labels, f"missing silence label: {required!r}"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
