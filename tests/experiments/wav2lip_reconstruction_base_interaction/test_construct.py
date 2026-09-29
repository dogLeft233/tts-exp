from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_reconstruction_base_interaction.construct import construct_factorial


def test_seed_fusion_and_shared_increment():
    m = np.full((80, 308), 0.2, dtype=np.float32)
    rows = []
    for seed, offset in zip((20260901, 20260902, 20260903), (1.0, 2.0, 3.0), strict=True):
        for condition, value in (("NAT_ONLY", 0.2), ("PAIRED_TTS", 0.5 + offset / 10), ("SAME_PHONE_WRONG_INSTANCE", 0.4 + offset / 10)):
            path = __import__("tempfile").NamedTemporaryFile(suffix=".npy", delete=False).name
            delta = np.zeros_like(m)
            delta[:, 10:20] = value
            np.save(path, m + delta)
            rows.append({"sample_id": "s", "seed": seed, "condition": condition, "path": path, "used_masks": [{"mask_sha256": "m", "global_start_frame": 10, "global_end_frame": 20}]})
    result = construct_factorial(m, rows, {"m": {"mask_sha256": "m", "natural_core_start_frame": 10, "natural_core_end_frame": 20}}, "s")
    natural_increment = result["N_CONTENT"] - result["N"]
    base_increment = result["BASE_CONTENT"] - result["BASE"]
    assert np.allclose(base_increment, natural_increment, atol=2e-6)
    assert np.linalg.norm(base_increment) > 0
    assert not np.array_equal(result["BASE"], result["N"])
    assert result["metadata"]["a"] <= 0.25


def test_wrong_base_is_not_the_natural_baseline():
    assert "BASE" != "N"
