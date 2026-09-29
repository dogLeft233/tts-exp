import numpy as np

from scripts.experiments import lrs3_english_phoneme_transfer as transfer


def test_protocol_keeps_english_condition_transfer_arms() -> None:
    assert transfer.CONDITIONS == (
        "natural",
        "qwen_cloud",
        "qwen_local",
        "index_tts2",
        "cosyvoice2",
    )
    assert transfer.MODELS["hubert"]["layers"] == [0, 6, 11]
    assert transfer.MODELS["hubert"]["probe"] is False
    assert transfer.MODELS["hubert"]["probe_layers"] == []
    assert transfer.MODELS["xlsr"]["layers"] == [10]


def test_tokens_for_level_preserves_exact_phone_labels_and_filters_silence() -> None:
    rows = [
        {"speech": True, "label": "tʰ", "viseme": "T", "sample_id": "1", "embedding": np.array([1.0, 0.0])},
        {"speech": False, "label": "sil", "viseme": "sil", "sample_id": "1", "embedding": np.array([0.0, 1.0])},
        {"speech": True, "label": "t", "viseme": "T", "sample_id": "2", "embedding": np.array([0.0, 1.0])},
    ]

    embeddings, labels, groups = transfer._tokens_for_level(rows, "phoneme")

    assert embeddings.shape == (2, 2)
    assert labels.tolist() == ["tʰ", "t"]
    assert groups.tolist() == ["1", "2"]
