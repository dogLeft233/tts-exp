from __future__ import annotations

import numpy as np

from scripts.experiments.lrs3_bridge_tts_quality import config
from scripts.experiments.lrs3_bridge_tts_quality.common import (
    file_sha256,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_bridge_tts_quality.runner import (
    _analysis_destination,
    _bound_files,
)
from scripts.experiments.lrs3_bridge_tts_quality.stats import (
    analyze_score_rows,
    bootstrap_ci,
    spearman_rho,
)


def _score_rows(cloud_bonus: float = 0.06) -> list[dict]:
    rows = []
    for sample_index, sample_id in enumerate(config.EXPECTED_SAMPLE_IDS):
        for repeat in config.REPEATS:
            values = {
                "V_N/A_N": (1.0, 2.0, 0),
                "V_B0/A_N": (1.0, 2.0, 0),
                "V_B_LOCAL/A_N": (1.02, 1.99, 0),
                "V_B_CLOUD/A_N": (1.02 + cloud_bonus, 1.98, 0),
                "V_B_LOCAL/A_B_LOCAL": (1.02, 1.99, 0),
                "V_B_CLOUD/A_B_CLOUD": (1.02 + cloud_bonus, 1.98, 0),
                "V_N/A_N_REV": (0.5, 2.5, 4),
            }
            for cell, (sync_c, sync_d, offset) in values.items():
                rows.append({
                    "sample_id": sample_id,
                    "source_group": f"group-{sample_index}",
                    "render_repeat": repeat,
                    "cell": cell,
                    "score": {"sync_c": sync_c, "sync_d": sync_d, "av_offset": offset},
                })
    return rows


def test_primary_delta_uses_same_n_baseline_and_label_swap_changes_sign() -> None:
    result = analyze_score_rows(_score_rows(), [{"sample_id": sample_id, "arms": {"LOCAL": {"progress": 0.3}, "CLOUD": {"progress": 0.3}}} for sample_id in config.EXPECTED_SAMPLE_IDS])
    assert np.isclose(result["primary"]["deltaC"]["mean"], 0.06)
    assert all(np.isclose(row["deltaC"], row["gC_CLOUD"] - row["gC_LOCAL"]) for row in result["per_record"])
    swapped = analyze_score_rows(_score_rows(cloud_bonus=-0.06), [{"sample_id": sample_id, "arms": {"LOCAL": {"progress": 0.3}, "CLOUD": {"progress": 0.3}}} for sample_id in config.EXPECTED_SAMPLE_IDS])
    assert np.isclose(swapped["primary"]["deltaC"]["mean"], -0.06)


def test_bootstrap_is_record_level_and_spearman_handles_constant_input() -> None:
    interval = bootstrap_ci([0.0] * 22, draws=100, seed=7)
    assert interval["valid_draws"] == 100
    assert interval["ci"] == [0.0, 0.0]
    assert spearman_rho([1.0, 1.0], [1.0, 2.0]) is None


def test_quality_update_selects_versioned_analysis_without_overwriting_parent(tmp_path) -> None:
    paths = config.RunPaths(tmp_path)
    paths.analysis.mkdir(parents=True)
    parent = paths.analysis / "final.json"
    write_self_hashed_json(
        parent,
        {
            "schema_version": 1,
            "stage_id": "07_analysis",
            "protocol_id": config.PROTOCOL_ID,
            "analysis_version": 1,
            "quality_input_sha256": "old-quality",
        },
    )
    paths.quality.mkdir(parents=True)
    write_self_hashed_json(paths.quality / "quality.json", {"status": "ASSESSED"})
    destination, version, parent_hash, existing = _analysis_destination(paths)
    assert destination == paths.analysis / "versions" / "v2"
    assert version == 2
    assert parent_hash == file_sha256(parent)
    assert existing is None


def test_bound_files_ignores_long_transcript_strings() -> None:
    bound: dict[str, dict[str, str]] = {}
    transcript = "THE BASELINE " + ("IF WE HAVE TO PUSH THE ECOSYSTEM BACK TO THE LEFT " * 100)
    _bound_files({"transcript": transcript}, bound)
    assert bound == {}
