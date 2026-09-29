from __future__ import annotations

import numpy as np

from scripts.experiments.phoneme_tfg_association.features import _support, _valid_speech_tokens


def test_feature_support_filters_silence_invalid_and_zero_vectors():
    rows = [
        {"speech": True, "valid": True, "label": "a", "embedding": np.asarray([1.0, 0.0])},
        {"speech": True, "valid": True, "label": "a", "embedding": np.asarray([0.0, 1.0])},
        {"speech": True, "valid": True, "label": "sil", "embedding": np.asarray([1.0, 0.0])},
        {"speech": False, "valid": True, "label": "b", "embedding": np.asarray([1.0, 0.0])},
        {"speech": True, "valid": True, "label": "b", "embedding": np.asarray([0.0, 0.0])},
        {"speech": True, "valid": False, "label": "c", "embedding": np.asarray([1.0, 0.0])},
    ]
    valid = _valid_speech_tokens(rows)
    assert len(valid) == 2
    support = _support(rows)
    assert support["valid_token_count"] == 2
    assert support["label_count"] == 1
    assert support["repeated_label_count"] == 1
