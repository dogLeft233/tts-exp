from __future__ import annotations

import copy

import pytest

from scripts.experiments.lrs3_bridge_tts_quality import config
from scripts.experiments.lrs3_bridge_tts_quality.common import (
    ProtocolError,
    sample_ids_sha256,
)
from scripts.experiments.lrs3_bridge_tts_quality.protocol import (
    _record,
    load_frozen_cohort,
)


def test_registered_cohort_has_exact_order_and_denominator() -> None:
    records, selection = load_frozen_cohort()
    assert len(records) == config.EXPECTED_RECORD_COUNT == 22
    assert len({row["source_group"] for row in records}) == 22
    assert sample_ids_sha256([row["sample_id"] for row in records]) == config.EXPECTED_SAMPLE_ID_SHA256
    assert selection["uses_scores"] is False
    assert selection["historical_exposure"] is True


def test_provenance_rejects_text_and_reference_swap() -> None:
    records, _ = load_frozen_cohort()
    parent = records[0]
    # _record receives the already-validated parent-shaped row plus the two
    # provenance rows.  Mutating either join must fail before any model work.
    cloud = {
        "sample_id": parent["sample_id"],
        "source_group": parent["source_group"],
        "transcript": parent["transcript"],
        "tts_transcript": parent["transcript"],
        "reference_audio_sha256": parent["natural_audio"]["sha256"],
        "video_sha256": parent["face_video"]["sha256"],
        "reference_role": "paired_natural_audio",
        "provider": config.CLOUD_PROVIDER,
        "model": config.CLOUD_MODEL,
    }
    historical = {
        "sample_id": parent["sample_id"],
        "source_group": parent["source_group"],
        "tts_audio_sha256": parent["cloud_tts"]["canonical_audio"]["sha256"],
        "natural_audio_sha256": parent["natural_audio"]["sha256"],
        "natural_samples": parent["natural_audio"]["sample_count"],
    }
    # The compact fake is not a valid asset row, so make the direct join test
    # at the earliest identity checks instead of reaching media validation.
    wrong_cloud = copy.deepcopy(cloud)
    wrong_cloud["transcript"] += " WRONG"
    with pytest.raises(ProtocolError):
        _record(parent, wrong_cloud, historical)
    wrong_historical = copy.deepcopy(historical)
    wrong_historical["source_group"] = "other"
    with pytest.raises(ProtocolError):
        _record(parent, cloud, wrong_historical)
