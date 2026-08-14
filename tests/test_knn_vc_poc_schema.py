from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from knn_vc_poc_schema import (  # noqa: E402
    EXPECTED_CONDITIONS,
    canonical_sha256,
    validate_summary,
)


def _summary():
    keys = ["pair-a"]
    return {
        "schema_version": 1,
        "experiment_type": "wavlm_knn_vc_valid_poc",
        "status": "completed",
        "manifest": {},
        "selection": {
            "split": "valid",
            "speaker_group": "S0765",
            "ordered_paired_keys": keys,
            "ordered_paired_keys_sha256": canonical_sha256(keys),
        },
        "model": {},
        "policies": {},
        "runtime": {},
        "conditions": list(EXPECTED_CONDITIONS),
        "items": [{
            "paired_key": "pair-a",
            "conditions": [{"condition": name, "paired_oracle": True} for name in EXPECTED_CONDITIONS],
        }],
        "heldout_excluded": True,
    }


def test_summary_schema_accepts_valid_payload() -> None:
    validate_summary(_summary())


def test_summary_schema_rejects_extra_key() -> None:
    value = _summary()
    value["extra"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_summary(value)


def test_summary_schema_rejects_selection_hash_mismatch() -> None:
    value = deepcopy(_summary())
    value["selection"]["ordered_paired_keys_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="key hash"):
        validate_summary(value)


def test_summary_schema_rejects_heldout() -> None:
    value = _summary()
    value["heldout_excluded"] = False
    with pytest.raises(ValueError, match="heldout"):
        validate_summary(value)
