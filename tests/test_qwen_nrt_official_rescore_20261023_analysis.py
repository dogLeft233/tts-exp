import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/experiments/qwen_nrt_official_rescore_20261023_analysis.py"
SPEC = importlib.util.spec_from_file_location("nrt_official_analysis", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_three_seeds_are_one_speaker_and_qualification_does_not_change_controls():
    scores = {"N": 5, "R": 6, "T1": 3, "T2": 7, "T3": 8}
    assert MOD.contrasts(scores) == {"T-N": 1, "T-R": 0, "R-N": 1}
    assert MOD.contrasts(scores, ("T2", "T3")) == {"T-N": 2.5, "T-R": 1.5, "R-N": 1}
    with pytest.raises(ValueError, match="No qualifying"):
        MOD.contrasts(scores, ())


def test_paired_protocol_change_cancels_shared_speaker_variation():
    # A difference in pooled scores would mistake the paired constant shift for
    # large between-speaker uncertainty. Its paired interval is exactly +0.25.
    previous = [-4, 1, 7]
    current = [-3.75, 1.25, 7.25]
    s = MOD.summarize([b-a for a,b in zip(previous, current)], replicates=1000)
    assert s["mean"] == 0.25
    assert s["ci97_5"] == [0.25, 0.25]
    assert s["n"] == 3


def test_missing_support_is_explicit_and_nonfinite_values_are_rejected():
    assert MOD.summarize([])["mean"] is None
    assert MOD.summarize([])["n"] == 0
    for values in ([1, float("nan")], [float("inf")], [[1,2]]):
        with pytest.raises(ValueError, match="Finite"):
            MOD.summarize(values)


def test_failed_cell_with_printed_numbers_cannot_enter_primary_speaker_tuple(tmp_path):
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    (run / "scores").mkdir(parents=True)
    video, people = [], []
    protocol = {"bootstrap": {"seed": 20261023, "replicates": 100}}
    (run / "analysis_protocol.json").write_text(json.dumps(protocol), encoding="utf8")
    for i in range(40):
        sid, cohort = f"sample{i:02}", "pilot10" if i < 10 else "new30"
        for role in MOD.ROLES:
            video.append({"sample_id": sid, "role": role, "cohort": cohort})
            row = {"sample_id": sid, "role": role, "cohort": cohort,
                   "sync_c": 1, "sync_d": 2, "av_offset": 0, "tracks": [{}],
                   "status": "FAILED" if i == 10 and role == "T3" else "COMPLETE"}
            (run / "scores" / f"{sid}_{role}.json").write_text(json.dumps(row), encoding="utf8")
        people.append({"sample_id": sid, "speaker_id": f"speaker{i}", "cohort": cohort,
                       "draw_reliability": {r: {"qualified": True} for r in MOD.ROLES[2:]},
                       "policies": {p: {r: {"C": 1} for r in MOD.ROLES}
                                    for p in ("common", "native_full", "equal_count_spread")}})
    (source / "video_manifest.json").write_text(json.dumps({"records": video}), encoding="utf8")
    (source / "analysis.json").write_text(json.dumps({"records": people}), encoding="utf8")
    result = MOD.analyze(run, source, protocol)
    assert result["status"] == "INCOMPLETE_COVERAGE_DESCRIPTIVE"
    assert result["coverage"]["planned_cells"] == 200
    assert result["coverage"]["finite_cells"] == 199
    assert result["coverage"]["complete_speakers"] == 39
    assert result["cohorts"]["new30"]["n"] == 29
    assert result["coverage"]["failures"] == [{"sample_id": "sample10", "role": "T3", "status": "FAILED", "cohort": "new30"}]
