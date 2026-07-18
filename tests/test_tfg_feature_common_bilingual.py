import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tfg_feature_common import (
    OUTPUT_BASE_EN,
    ENGLISH_STUDY_SAMPLES,
    ENGLISH_SILENCE_LABELS,
)


def test_output_base_en_path():
    assert OUTPUT_BASE_EN.name == "wav2sem_analysis_en"
    assert OUTPUT_BASE_EN.parent.name == "data"


def test_english_study_samples_complete():
    assert ENGLISH_STUDY_SAMPLES == list(range(1, 14))


def test_silence_labels_include_librispeech_markers():
    assert "h#" in ENGLISH_SILENCE_LABELS
    assert "sil" in ENGLISH_SILENCE_LABELS
    assert "sp" in ENGLISH_SILENCE_LABELS


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
