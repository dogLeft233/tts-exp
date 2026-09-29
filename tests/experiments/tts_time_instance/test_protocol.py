from __future__ import annotations

import numpy as np

from scripts.experiments import tts_time_instance as experiment


def test_official_matrix_lag_sign_on_known_shift() -> None:
    n = 80
    video = np.eye(n, experiment.EMBEDDING_DIM, dtype=np.float32)
    audio = np.zeros_like(video)
    audio[5:] = video[:-5]
    matrix = experiment.official_distance_matrix(video, audio)
    summary = experiment.curve(matrix, range(15, n - 15))
    assert summary["min_index"] - experiment.VSHIFT == 5
    assert summary["official_offset"] == -5


def test_rank_strict_ties_and_positive_affine_invariance() -> None:
    # The decision uses the strict ordering only; an exact tie is 0.5 and a
    # positive affine transform must preserve all three outcomes.
    pairs = [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]

    def wins(values: list[tuple[float, float]]) -> list[float]:
        result = []
        for d0, dd in values:
            result.append(1.0 if dd > d0 else 0.0 if dd < d0 else 0.5)
        return result

    assert wins(pairs) == wins([(3.0 * d0 + 7.0, 3.0 * dd + 7.0) for d0, dd in pairs]) == [1.0, 0.0, 0.5]


def test_adjacent_lags_are_recorded_as_supplementary_only() -> None:
    assert set(experiment.SUPPLEMENTARY_DELTAS) == {-1, 1}
    assert set(experiment.PRIMARY_DELTAS).isdisjoint(experiment.SUPPLEMENTARY_DELTAS)
    assert set(experiment.ALL_DELTAS) == set(experiment.PRIMARY_DELTAS) | {-1, 1}


def test_phone_event_occurrence_boundaries_and_coverage() -> None:
    tokens = [
        {"index": 0, "label": "sil", "normalized": "sil", "start_s": 0.0, "end_s": 0.1, "silence": True, "unknown": False},
        {"index": 1, "label": "AA", "normalized": "aa", "start_s": 0.1, "end_s": 0.4, "silence": False, "unknown": False},
        {"index": 2, "label": "AA", "normalized": "aa", "start_s": 0.4, "end_s": 0.7, "silence": False, "unknown": False},
        {"index": 3, "label": "B", "normalized": "b", "start_s": 0.7, "end_s": 1.0, "silence": False, "unknown": False},
    ]
    same, same_detail = experiment.classify_event(tokens, 0.2, 0.35)
    repeated, _ = experiment.classify_event(tokens, 0.2, 0.5)
    cross, _ = experiment.classify_event(tokens, 0.2, 0.8)
    excluded, _ = experiment.classify_event(tokens, 0.05, 0.2)
    assert same == "SAME"
    assert repeated == "SAME_LABEL_DIFFERENT_OCCURRENCE"
    assert cross == "CROSS"
    assert excluded == "EXCLUDED"
    assert same_detail["left_sequence"]
    assert all("overlap_fraction" in item for item in same_detail["left_sequence"])


def test_context_donor_key_uses_joint_log_duration_before_match_count() -> None:
    def token(index: int, label: str, start: float, end: float) -> dict[str, object]:
        return {"index": index, "label": label, "normalized": label.lower(), "start_s": start, "end_s": end, "silence": False, "unknown": False}

    def arm_tokens() -> list[dict[str, object]]:
        return [
            token(0, "L", 0.0, 0.1), token(1, "AA", 0.1, 0.2), token(2, "R", 0.2, 0.3),
            token(3, "L", 0.9, 1.0), token(4, "AA", 1.0, 1.1), token(5, "X", 1.1, 1.2),
            token(6, "L", 1.9, 2.0), token(7, "AA", 2.0, 2.2), token(8, "R", 2.2, 2.3),
            token(9, "X", 2.9, 3.0), token(10, "AA", 3.0, 3.1), token(11, "Y", 3.1, 3.2),
        ]

    token_sets = {arm: arm_tokens() for arm in ("N", "T1", "T2")}
    occurrences = [{"occurrence": occurrence, **{arm: token_sets[arm][index] for arm in token_sets}} for occurrence, index in enumerate((1, 4, 7, 10))]
    selected = experiment._select_context_donors(occurrences[0], occurrences, token_sets, {arm: ("l", "r") for arm in token_sets})
    assert selected is not None
    assert selected["plus"]["occurrence"] == 1  # equal duration beats the 2-neighbour tie-break
    assert selected["minus"]["occurrence"] == 3
    assert selected["plus"]["match_count"] == 1


def test_word_then_phone_matching_preserves_occurrence_order() -> None:
    def token(index: int, phone: str, word_index: int, word: str) -> dict[str, object]:
        return {"index": index, "label": phone, "normalized": phone.lower(), "start_s": index * 0.1, "end_s": (index + 1) * 0.1, "silence": False, "unknown": False, "word_index": word_index, "word_label": word, "word_normalized": word.lower()}

    arms = {
        "N": [token(0, "AA", 0, "one"), token(1, "B", 1, "two")],
        "T1": [token(0, "AA", 0, "one"), token(1, "B", 1, "two")],
        "T2": [token(0, "AA", 0, "one"), token(1, "AA", 1, "two"), token(2, "B", 1, "two")],
    }
    matches, meta = experiment.match_occurrences(arms)
    assert [row["occurrence"] for row in matches] == [0, 1]
    assert meta["word_matches"] == 2


def test_phase_interpolation_and_unit_identity() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    assert np.allclose(experiment.interpolation(values, 1.5), [3.0, 4.0])
    assert experiment.interpolation(values, -0.01) is None
    unit = experiment.l2_unit(experiment.interpolation(values, 1.5))
    assert unit is not None and np.isclose(np.linalg.norm(unit), 1.0)
    assert np.allclose(experiment.interpolation(values, 1.0), values[1])
    assert experiment._support_clear(1.5, [(0.0, 1.0)])
    assert not experiment._support_clear(1.2, [(0.0, 1.0)])


def test_two_by_two_interaction_identity_and_zero_case() -> None:
    rng = np.random.default_rng(11)
    v1, v2, a1, a2 = rng.normal(size=(4, 16))

    def d(v: np.ndarray, a: np.ndarray) -> float:
        return float(np.sum((v - a) ** 2))

    interaction = 0.5 * (d(v1, a2) + d(v2, a1) - d(v1, a1) - d(v2, a2))
    assert np.isclose(interaction, float(np.dot(v1 - v2, a1 - a2)))
    assert np.isclose(0.5 * (d(v1, a2) + d(v1, a1) - d(v1, a1) - d(v1, a2)), 0.0)


def test_bootstrap_counts_groups_and_fixed_draws() -> None:
    insufficient = experiment.bootstrap_summary({"g1": 1.0, "g2": -1.0}, metric="x", min_groups=3)
    assert insufficient["status"] == "NOT_ESTIMABLE"
    summary = experiment.bootstrap_summary({"g1": 1.0, "g2": -1.0}, metric="x", min_groups=1)
    assert summary["status"] == "COMPLETE"
    assert summary["group_count"] == 2
    assert summary["draws"] == 20_000
    assert summary["seed"] == 20260917


def test_missing_mfa_is_reported_without_fabricating_alignment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiment, "MFA", tmp_path / "missing-mfa")
    result = experiment.run_mfa(
        [{"sample_id": 1, "transcript": "HELLO", "audio": str(tmp_path / "audio.wav")}],
        ("audio",),
        tmp_path / "out",
        prefix="X",
    )
    assert result["status"] == "MFA_UNAVAILABLE"


def test_cached_tts_contract_rejects_changed_audio(tmp_path) -> None:
    output = tmp_path / "T1.wav"
    pcm = np.asarray([0, 1, -2, 3], dtype=np.int16)
    audio_hash = experiment._audio_write(output, pcm)
    raw = tmp_path / "T1.raw.npy"
    np.save(raw, np.asarray([0.0, 0.1], dtype=np.float32), allow_pickle=False)
    record = {"sample_id": 1, "source_group": "g", "transcript": "HELLO", "transcript_sha256": "text", "N_audio": str(tmp_path / "ref.wav"), "N_pcm_sha256": "ref"}
    meta = {"sample_id": 1, "source_group": "g", "arm": "T1", "seed": 42, "text": "HELLO", "text_sha256": "text", "reference_audio": str((tmp_path / "ref.wav").resolve()), "reference_pcm_sha256": "ref", "model_id": experiment.TTS_MODEL_ID, "strict_backend": True, "language": experiment.TTS_LANGUAGE, "max_new_tokens": experiment.TTS_MAX_NEW_TOKENS, "status": "COMPLETE", "audio": str(output.resolve()), "audio_sha256": audio_hash, "raw_audio": str(raw.resolve()), "raw_audio_sha256": experiment.sha256_file(raw), "sample_count": len(pcm), "peak": 3, "pre_quant_clipping_fraction": 0.0}
    assert experiment._cached_tts_row_valid(meta, record, "T1", output)
    output.write_bytes(output.read_bytes() + b"tamper")
    assert not experiment._cached_tts_row_valid(meta, record, "T1", output)


def test_b_dependency_block_is_explicit(tmp_path) -> None:
    b_root = tmp_path / "B"
    b_root.mkdir()
    (b_root / "video_manifest.json").write_text('{"status":"DEPENDENCY_BLOCKED"}\n', encoding="utf-8")
    (b_root / "feature_manifest.json").write_text('{"status":"DEPENDENCY_BLOCKED"}\n', encoding="utf-8")
    result = experiment.analyze_b(tmp_path, {})
    assert result["status"] == "DEPENDENCY_BLOCKED"
    assert "incomplete" in result["reason"]
