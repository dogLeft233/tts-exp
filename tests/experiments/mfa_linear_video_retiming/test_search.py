from __future__ import annotations

import pytest
import numpy as np

from scripts.experiments.mfa_linear_video_retiming.search_worker import (
    _legacy_curve_from_embeddings,
    final_acceptable,
    select_final_candidate,
)


def _metrics(*, c: float, d: float, d0: float, offset: int = 0, local_d0: float = 4.0, local_offset: int = 0) -> dict:
    return {"sync_c": c, "sync_d": d, "d0": d0, "offset": offset,
            "local_windows": [{"d0": local_d0, "offset": local_offset},
                              {"d0": local_d0, "offset": local_offset}]}


def _search() -> dict:
    return {"minimum_sync_c_gain": 0.1, "minimum_d0_improvement": 0.1,
            "maximum_sync_d_worsening": 0.1, "local_offset_quantile": 0.9,
            "local_sync_c_slack": 0.02}


def test_final_gate_requires_global_and_local_improvement() -> None:
    baseline = _metrics(c=2.0, d=1.0, d0=5.0, local_d0=5.0, local_offset=1)
    candidate = {"candidate_id": "a", "feasible": True, "regularization": 0.1,
                 "metrics": _metrics(c=2.15, d=1.05, d0=4.8, local_d0=4.8, local_offset=0)}

    assert final_acceptable(candidate, baseline, _search())
    candidate["metrics"] = _metrics(c=2.15, d=1.05, d0=4.8, local_d0=5.1, local_offset=0)
    assert not final_acceptable(candidate, baseline, _search())
    candidate["metrics"] = _metrics(c=2.15, d=1.05, d0=4.8, local_d0=4.8, local_offset=2)
    assert not final_acceptable(candidate, baseline, _search())


def test_final_gate_rejects_candidate_that_only_improves_sync_c() -> None:
    baseline = _metrics(c=2.0, d=1.0, d0=5.0, local_d0=5.0)
    candidate = {"candidate_id": "only_c", "feasible": True, "regularization": 0.0,
                 "metrics": _metrics(c=2.5, d=1.2, d0=5.0, local_d0=5.0)}

    assert not final_acceptable(candidate, baseline, _search())


def test_selection_uses_low_regularization_within_sync_c_slack() -> None:
    baseline = _metrics(c=2.0, d=1.0, d0=5.0, local_d0=5.0)
    lower_regularization = {"candidate_id": "b", "feasible": True, "regularization": 0.1,
                            "metrics": _metrics(c=2.20, d=1.0, d0=4.7, local_d0=4.7)}
    highest_sync_c = {"candidate_id": "a", "feasible": True, "regularization": 0.9,
                      "metrics": _metrics(c=2.21, d=1.0, d0=4.7, local_d0=4.7)}

    selected = select_final_candidate([highest_sync_c, lower_regularization], baseline, _search())

    assert selected is lower_regularization


def test_candidate_with_boundary_offset_is_rejected() -> None:
    baseline = _metrics(c=2.0, d=1.0, d0=5.0, local_d0=5.0)
    candidate = {"candidate_id": "edge", "feasible": True, "regularization": 0.0,
                 "metrics": _metrics(c=3.0, d=0.5, d0=4.0, offset=2, local_d0=4.0)}

    assert not final_acceptable(candidate, baseline, _search())
    assert select_final_candidate([candidate], baseline, _search()) is None


def test_legacy_fixed_support_curve_matches_pairwise_distance_reducer() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20260923)
    audio = rng.normal(size=(80, 16)).astype(np.float32)
    visual = np.zeros_like(audio)
    visual[:-3] = audio[3:]
    rows = np.arange(15, 65, dtype=np.int64)

    curve, support_count = _legacy_curve_from_embeddings(visual, audio, rows, torch=torch)

    expected = []
    for lag in range(-15, 16):
        left = torch.from_numpy(visual[rows])
        right = torch.from_numpy(audio[rows + lag])
        expected.append(float(np.mean(torch.nn.functional.pairwise_distance(left, right).numpy())))
    np.testing.assert_allclose(curve, expected, rtol=0, atol=1e-7)
    assert support_count == rows.size
    assert int(np.argmin(curve)) == 18  # visual[t] matches audio[t + 3]
