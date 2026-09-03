from __future__ import annotations

import numpy as np

from scripts.experiments.masked_tts_reconstruction.scaleup_run import (
    MIN_PHONE_GROUPS,
    MIN_PHONE_INSTANCES,
    _apply_phone_support,
    _bootstrap,
    _group_plan,
)
from scripts.experiments.masked_tts_reconstruction.config import EVAL_GROUPS, TRAIN_GROUPS


def test_scaleup_group_plan_is_deterministic_and_fresh() -> None:
    fresh = [
        "70VZ1SOzSnc", "6VnKV1sr5VQ", "7FecGsXFgcQ", "7DLzXAjscXk",
        "7CIq4mtiamY", "6V6p1tgHfm0", "6vLrreR6YOE", "6zVS8HIPUng",
        "73jPh0eRPSY", "79tRTivyMSM", "6ORDQFh0Byw", "6VWPHKABRQA",
        "79zra755WgA", "6yR5OUVb2gY",
    ]
    groups = list(TRAIN_GROUPS) + list(EVAL_GROUPS) + fresh
    by_id = {f"sample_{index}": {"sample_id": f"sample_{index}", "source_group": group} for index, group in enumerate(groups)}
    train, evaluation, candidates = _group_plan(by_id)
    assert len(train) == 12
    assert len(evaluation) == 8
    assert tuple(train[:6]) == tuple(TRAIN_GROUPS)
    assert set(train[6:]) | set(evaluation) == set(fresh)
    assert len(candidates) == 14


def test_phone_support_filter_is_shared_across_arms() -> None:
    masks = []
    for index in range(MIN_PHONE_INSTANCES):
        group = ("g0", "g1", "g2")[index % MIN_PHONE_GROUPS]
        masks.append({
            "mask_sha256": f"a{index}",
            "sample_id": f"train{index}",
            "source_group": group,
            "prototype_split": "train",
            "label": "a",
            "canonical_index": index,
            "operation_index": index,
            "natural_phone_index": index,
            "tts_phone_index": index,
        })
    masks.append({
        "mask_sha256": "b0",
        "sample_id": "eval0",
        "source_group": "e0",
        "prototype_split": "evaluation",
        "label": "b",
        "canonical_index": len(masks),
        "operation_index": 0,
        "natural_phone_index": 0,
        "tts_phone_index": 0,
    })
    result = _apply_phone_support({"masks": masks, "exclusions": [], "counts": {"masks": len(masks), "exclusions": 0}, "readiness": {}})
    assert all(row["label"] == "a" for row in result["masks"])
    assert result["phone_support"]["retained_labels"] == ["a"]
    assert result["phone_support"]["removed_mask_count"] == 1
    assert result["exclusions"][-1]["reason"] == "unsupported_phone"
    assert result["exclusions"][-1]["source_group"] == "e0"


def test_whole_group_bootstrap_is_reproducible() -> None:
    values = np.asarray([0.1, 0.2, 0.4, 0.8, 1.1, 1.2, 1.5, 2.0], dtype=np.float64)
    assert _bootstrap(values, 200, 20260901) == _bootstrap(values, 200, 20260901)
