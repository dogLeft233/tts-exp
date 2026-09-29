from __future__ import annotations

from pathlib import Path

from scripts.experiments.mfa_linear_video_retiming.cohort_v2 import SAMPLE_IDS, SPEAKERS, build_cohort
from scripts.experiments.mfa_linear_video_retiming.config import load_config


def test_fixed_64_strict_pairs_and_resampled_tts(tmp_path: Path) -> None:
    config = load_config("scripts/configs/mfa_linear_video_retiming_v2.yaml")
    manifest, tokens, tts = build_cohort(config, Path(config["paths"]["strict_source_manifest"]), tmp_path)
    assert manifest["sample_ids"] == list(SAMPLE_IDS)
    assert len(manifest["records"]) == 64
    assert set(row["speaker_id"] for row in manifest["records"]) == set(SPEAKERS)
    assert all(sum(row["speaker_id"] == speaker for row in manifest["records"]) == 8 for speaker in SPEAKERS)
    assert sum(row["prior_seen"] for row in manifest["records"]) == 25
    assert len(tokens["records"]) == len(tts["results"]) == 64
    for row in manifest["records"]:
        sid = row["sample_id"]
        for source in (tokens["records"][sid], tts["results"][sid]):
            assert (source["paired_key"], source["speaker_id"], source["split"], source["transcript"]) == (
                row["paired_key"], row["speaker_id"], row["split"], row["transcript"])
