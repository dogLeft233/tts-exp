from __future__ import annotations

import copy

import pytest

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    run_stagea_ditto_fit40_block_attempt02 as adapter,
)


METHODS = [{"method_id": f"method_{index:02d}", "method_config_sha256": f"config-{index}"}
           for index in range(6)]


def make_parent_rows() -> list[dict]:
    rows: list[dict] = []
    for speaker_index in range(20):
        speaker_id = f"S{speaker_index + 52:04d}"
        for clip_index in range(2):
            sample_id = f"clip_{speaker_index:02d}_{clip_index:02d}"
            natural = {
                "channels": 1,
                "file_sha256": f"file-{sample_id}",
                "path": f"/fit/source_wav/{sample_id}.wav",
                "pcm_sha256": f"pcm-{sample_id}",
                "sample_count": 62_639 + speaker_index + clip_index,
                "sample_rate_hz": 16_000,
                "subtype": "PCM_16",
            }
            face = {"file_sha256": adapter.FACE_SHA256,
                    "path": "/fit/portrait/neutral_01.png",
                    "role": "shared_static_portrait_input"}
            source_video = {"file_sha256": None, "fps_den": 1, "fps_num": 25,
                            "path": None, "role": "not_applicable_static_portrait"}
            for method in METHODS:
                method_id = method["method_id"]
                rows.append({
                    "audio_frontend": "wav2lip_official_16k_float32_mel",
                    "checkpoint_sha256": None,
                    "face_id": "primary_face_static_portrait_v1",
                    "generation_settings": {
                        "face_mode": "static_image", "fps_den": 1, "fps_num": 25,
                        "generator": "official_wav2lip_inference.py",
                        "input_audio": "enhanced_generator_condition_only", "seed": 20261003,
                    },
                    "method_config_sha256": method["method_config_sha256"],
                    "method_id": method_id,
                    "model_id": "wav2lip_gan_bac9a81e",
                    "natural_pcm": copy.deepcopy(natural),
                    "row_id": f"parent_{speaker_id}_{sample_id}_{method_id}",
                    "sample_id": sample_id,
                    "score_audio": copy.deepcopy(natural),
                    "seed": 20261003,
                    "source_face": copy.deepcopy(face),
                    "source_media_kind": "static_portrait",
                    "source_video": copy.deepcopy(source_video),
                    "speaker_id": speaker_id,
                    "split": "fit",
                    "wav2lip_native_frame_binding": {
                        "expected_inference_frame_count": 95,
                        "sample_count": natural["sample_count"],
                    },
                })
    return rows


def make_renderer_report(row: dict, *, sample_count: int = 92_225) -> tuple[dict, dict, dict]:
    frame_count = adapter.expected_native_frame_count(sample_count)
    audio_ref = {
        "bytes": 100_000,
        "file_sha256": "generator-wav-sha",
        "pcm_sha256": "generator-pcm-sha",
        "sample_count": sample_count,
        "sample_rate_hz": adapter.SAMPLE_RATE,
        "channels": 1,
    }
    row["ditto_native_frame_binding"] = {
        "sample_count": sample_count,
        "expected_frame_count": frame_count,
        "fps_num": adapter.FPS_NUM,
        "fps_den": adapter.FPS_DEN,
    }
    source_and_weights = {
        "source_commit": adapter.DITTO_COMMIT,
        "source_files": {"inference.py": {"path": "/ditto/inference.py",
                                              "bytes": 3_066, "sha256": "source-sha"}},
        "weights": {"model.pth": {"path": "/weights/model.pth",
                                    "bytes": 100, "sha256": "weight-sha"}},
        "git_head": adapter.DITTO_COMMIT,
        "git_status_porcelain": "",
    }
    face = {"path": "/fit/portrait/neutral_01.png", "bytes": adapter.FACE_BYTES,
            "file_sha256": adapter.FACE_SHA256}
    report = {
        "status": "PASS_ENGINEERING_ONLY",
        "model": "ditto",
        "profile": adapter.RENDERER_RESULT_PROFILE,
        "row_id": row["row_id"],
        "seed": row["seed"],
        "source_and_weights": copy.deepcopy(source_and_weights),
        "face": copy.deepcopy(face),
        "audio": {
            "file_sha256": audio_ref["file_sha256"],
            "pcm_float32_sha256": audio_ref["pcm_sha256"],
            "sample_count": sample_count,
            "sample_rate_hz": adapter.SAMPLE_RATE,
            "channels": 1,
        },
        "silent_video": {
            "path": "/fresh/render.mp4", "bytes": 12_345,
            "file_sha256": "video-sha",
            "probe": {"frame_count": frame_count, "decoded_frame_count": frame_count,
                      "fps": "25/1", "decoded_pts_exact_25fps": True},
        },
    }
    return audio_ref, report, {"source_and_weights": source_and_weights, "face": face}


def test_native_frame_count_uses_ditto_ceil_clock() -> None:
    assert adapter.expected_native_frame_count(92_225) == 145
    assert adapter.expected_native_frame_count(62_639) == 98
    with pytest.raises(ValueError):
        adapter.expected_native_frame_count(0)


def test_projection_preserves_all_40_contexts_and_six_method_order() -> None:
    parent = make_parent_rows()
    original = copy.deepcopy(parent)
    projected = adapter.project_fit_rows(parent, METHODS)

    adapter.validate_fit_parent_rows(projected, METHODS)
    assert len(projected) == len(parent) == 240
    assert len({row["sample_id"] for row in projected}) == 40
    assert len({row["speaker_id"] for row in projected}) == 20
    assert [row["method_id"] for row in projected[:6]] == [m["method_id"] for m in METHODS]
    assert len({row["row_id"] for row in projected}) == 240
    assert parent == original

    changed = {"row_id", "model_id", "audio_frontend", "generation_settings",
               "wav2lip_native_frame_binding", "ditto_native_frame_binding"}
    for source, target in zip(parent, projected, strict=True):
        assert target["model_id"] == adapter.MODEL_ID
        assert target["audio_frontend"] == adapter.FRONTEND
        assert target["natural_pcm"] == source["natural_pcm"]
        assert target["score_audio"] == source["score_audio"]
        assert target["source_face"] == source["source_face"]
        assert target["method_id"] == source["method_id"]
        assert target["speaker_id"] == source["speaker_id"]
        assert target["sample_id"] == source["sample_id"]
        assert target["ditto_native_frame_binding"]["expected_frame_count"] == (
            adapter.expected_native_frame_count(source["natural_pcm"]["sample_count"]))
        for key in source.keys() | target.keys():
            if key not in changed:
                assert target.get(key) == source.get(key)


def test_block_selection_keeps_full_240_row_parent_denominator() -> None:
    rows = adapter.project_fit_rows(make_parent_rows(), METHODS)
    plan = {"rows": rows, "methods": METHODS}
    selected = adapter.validate_block(plan, 6 * 17, 6)
    assert len(selected) == 6
    assert len(plan["rows"]) == 240
    assert {row["sample_id"] for row in selected} == {rows[102]["sample_id"]}
    assert [row["method_id"] for row in selected] == [m["method_id"] for m in METHODS]


@pytest.mark.parametrize("mutation", ["profile", "pcm", "frame_count", "model"])
def test_renderer_report_rejects_wrong_backend_or_source_contract(mutation: str) -> None:
    row = {"row_id": "fit_row", "seed": 20261003}
    audio_ref, report, expected = make_renderer_report(row)
    if mutation == "profile":
        report["profile"] = "wrong-profile"
    elif mutation == "pcm":
        report["audio"]["pcm_float32_sha256"] = "wrong-pcm"
    elif mutation == "frame_count":
        report["silent_video"]["probe"]["decoded_frame_count"] -= 1
    else:
        report["model"] = "musetalk"
    with pytest.raises(RuntimeError):
        adapter.validate_renderer_input_report(
            row, audio_ref=audio_ref, report=report,
            expected_source_and_weights=expected["source_and_weights"],
            expected_face=expected["face"],
        )


def test_renderer_report_accepts_exact_native_identity_and_ceil_clock() -> None:
    row = {"row_id": "fit_row", "seed": 20261003}
    audio_ref, report, expected = make_renderer_report(row)
    adapter.validate_renderer_input_report(
        row, audio_ref=audio_ref, report=report,
        expected_source_and_weights=expected["source_and_weights"],
        expected_face=expected["face"],
    )


def test_parent_rejects_incomplete_roster_or_method_order() -> None:
    rows = make_parent_rows()
    with pytest.raises(ValueError):
        adapter.validate_fit_parent_rows(rows[:-1], METHODS)

    rows = make_parent_rows()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError):
        adapter.validate_fit_parent_rows(rows, METHODS)
