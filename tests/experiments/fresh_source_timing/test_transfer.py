from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.fresh_source_timing.analysis import analyze_records
from scripts.experiments.fresh_source_timing.audio import (
    build_delay_artifacts,
    build_delay_pcm,
)
from scripts.experiments.fresh_source_timing.common import (
    TimingError,
    file_sha256,
    write_json,
)
from scripts.experiments.fresh_source_timing.protocol import input_records, load_a_cells
from scripts.experiments.fresh_source_timing.scoring import score_embeddings
from scripts.experiments.fresh_source_timing.stats import bootstrap_indices
from scripts.experiments.fresh_source_timing.visual import (
    calibrate_visual_shift,
    save_mouth_trajectory,
    synthesize_delayed_trajectory,
    visual_transfer_score,
)


def _ideal_arrays(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    audio = rng.normal(size=(140, 4)).astype(np.float32)
    natural_visual = np.zeros_like(audio)
    natural_visual[:-2] = audio[2:]
    delay_visual = np.zeros_like(audio)
    delay_visual[5:] = natural_visual[:-5]
    trajectory = rng.normal(size=(140, 3))
    delay_trajectory = synthesize_delayed_trajectory(trajectory)["trajectory"]
    return audio, natural_visual, delay_visual, trajectory, delay_trajectory


def test_delay_pcm_preserves_length_and_exact_source_indices() -> None:
    natural = np.arange(12, dtype="<i2")
    delayed, mapping = build_delay_pcm(natural, shift_samples=4)
    assert np.frombuffer(delayed, dtype="<i2").tolist() == [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert mapping.tolist() == [-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7]


def test_syncnet_sign_and_fixed_plus_five_compensation() -> None:
    audio, natural_visual, delay_visual, _trajectory, _delay_trajectory = _ideal_arrays()
    score = score_embeddings(natural_visual, delay_visual, audio)
    assert score["natural"]["best_lag"] == 2
    assert score["delay"]["best_lag"] == -3
    assert score["offset_delta"] == 5
    assert score["d_comp"] < score["d_unc"]
    assert score["compensation_frames"] == 5
    with pytest.raises(TimingError):
        score_embeddings(natural_visual, delay_visual, audio, delay_shift_frames=4)


def test_visual_transfer_static_is_zero_and_calibration_is_fixed() -> None:
    static = np.ones((140, 2), dtype=np.float64)
    score = visual_transfer_score(static, static)
    assert score["e_unc"] == 0.0
    assert score["e_comp"] == 0.0
    assert score["v"] == 0.0
    trajectory = np.column_stack((np.sin(np.arange(140) / 5.0), np.cos(np.arange(140) / 7.0)))
    calibration = calibrate_visual_shift({str(index): trajectory for index in range(12)})
    assert calibration["pass"] is True
    assert calibration["passed_count"] == 12


def test_missing_visual_features_remain_in_denominator() -> None:
    audio, natural_visual, delay_visual, trajectory, delay_trajectory = _ideal_arrays()
    groups = [f"g{index:02d}" for index in range(12)]
    rows = [
        {
            "source_group": group,
            "fresh_forward": True,
            "natural_visual": natural_visual,
            "delay_visual": delay_visual,
            "natural_audio": audio,
            "natural_trajectory": trajectory,
            "delay_trajectory": delay_trajectory,
        }
        for group in groups[:-1]
    ]
    result = analyze_records({"Wav2Lip": rows}, groups=groups, calibration={"pass": True}, draws=64)
    model = result["models"]["Wav2Lip"]
    assert model["outcome"] == "VISUAL_RESPONSE_UNRESOLVED"
    assert model["visual"]["v"]["group_count"] == 12
    assert model["visual"]["v"]["observed_group_count"] == 11
    assert model["visual"]["v"]["missing_group_count"] == 1
    assert model["visual"]["v"]["mean"] is None


def test_bootstrap_is_pcg64_fixed_and_reproducible() -> None:
    groups = ["b", "a", "c"]
    labels_a, first = bootstrap_indices(groups, draws=32)
    labels_b, second = bootstrap_indices(groups, draws=32)
    assert labels_a == labels_b == ["a", "b", "c"]
    assert np.array_equal(first, second)
    expected = np.random.Generator(np.random.PCG64(20260910)).integers(0, 3, size=(32, 3), endpoint=False)
    assert np.array_equal(first, expected)


def test_b_calibration_schema_is_accepted_by_c_gate() -> None:
    groups = [f"g{index:02d}" for index in range(12)]
    result = analyze_records(
        {"Wav2Lip": []},
        groups=groups,
        calibration={"status": "METRIC_CALIBRATED", "calibrated": True},
        draws=8,
    )
    assert result["models"]["Wav2Lip"]["visual"]["gate"]["calibration_pass"] is True


def test_formal_input_selection_never_falls_back_to_smoke() -> None:
    formal = [f"formal_{index:02d}" for index in range(12)]
    records = [{"source_group": group, "sample_id": group} for group in formal[:-1]]
    records.extend({"source_group": f"smoke_{index:02d}", "sample_id": f"smoke_{index:02d}"} for index in range(2))
    with pytest.raises(TimingError):
        input_records({"status": "FROZEN", "records": records}, formal_groups=formal)


def _write_wav(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())


def _a_fixture(tmp_path: Path, *, models: tuple[str, ...] = ("Wav2Lip", "Ditto"), fresh: bool = True) -> tuple[Path, dict[str, object]]:
    natural = tmp_path / "natural.wav"
    _write_wav(natural, np.arange(4_000, dtype="<i2"))
    groups: list[dict[str, object]] = []
    for index in range(12):
        group = f"g{index:02d}"
        sample = f"sample_{index:02d}"
        delayed = tmp_path / f"{sample}.delay.wav"
        source_map = tmp_path / f"{sample}.map.npy"
        artifact = build_delay_artifacts(natural, delayed, source_map)
        groups.append({"source_group": group, "sample_id": sample, **artifact})
    embedding = tmp_path / "embedding.npz"
    values = np.zeros((140, 2), dtype=np.float32)
    np.savez_compressed(embedding, visual=values, audio_embedding=values)
    trajectory = tmp_path / "trajectory.npz"
    save_mouth_trajectory(trajectory, np.zeros((140, 2), dtype=np.float64))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-video")
    rows: list[dict[str, object]] = []
    for model in models:
        for item in groups:
            for arm in ("N", "DELAY"):
                audio = item["natural"] if arm == "N" else item["delay"]
                row: dict[str, object] = {
                    "source_group": item["source_group"],
                    "sample_id": item["sample_id"],
                    "model": model,
                    "arm": arm,
                    "seed": 42,
                    "status": "complete",
                    "fresh_forward": fresh if arm == "DELAY" else True,
                    "video": {"path": str(video), "sha256": file_sha256(video)},
                    "audio": {"path": audio["path"], "sha256": audio["sha256"]},
                    "visual": {"path": str(embedding), "sha256": file_sha256(embedding), "key": "visual"},
                    "audio_embedding": {"path": str(embedding), "sha256": file_sha256(embedding), "key": "audio_embedding"},
                    "features": {"natural": str(trajectory), "delay": str(trajectory)},
                }
                if arm == "DELAY":
                    row["source_index_map"] = {"path": item["source_index_map"]["path"], "sha256": item["source_index_map"]["sha256"]}
                rows.append(row)
    manifest = tmp_path / "run" / "A" / "videos.json"
    write_json(manifest, {"schema_version": 1, "status": "GO", "rows": rows})
    delay_inputs = {"status": "READY", "group_count": 12, "groups": groups}
    return tmp_path, delay_inputs


def test_a_cells_require_both_generators_and_fresh_provenance(tmp_path: Path) -> None:
    root, delay_inputs = _a_fixture(tmp_path, models=("Wav2Lip",))
    with pytest.raises(TimingError, match="models"):
        load_a_cells(root, delay_inputs)

    root, delay_inputs = _a_fixture(tmp_path / "stale", fresh=False)
    records = load_a_cells(root, delay_inputs)
    assert all(row["provenance_valid"] is False for row in records["Wav2Lip"])


def test_ready_validator_requires_final_artifact(tmp_path: Path) -> None:
    _root, delay_inputs = _a_fixture(tmp_path)
    c_root = tmp_path / "run" / "C"
    c_root.mkdir(parents=True, exist_ok=True)
    write_json(c_root / "delay_inputs.json", {**delay_inputs, "protocol_id": "fresh_source_timing_transfer", "protocol_revision": "fresh_source_timing_transfer_v1"})
    from scripts.experiments.fresh_source_timing.validate import validate_run

    result = validate_run(tmp_path)
    assert result["status"] == "NO_GO"
    assert any("final" in error for error in result["errors"])
