from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, load_self_hashed, write_json


def _score(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("score", row)
    if not isinstance(value, dict):
        raise ProtocolError("score row is not an object")
    return value


def _arrays(
    score: dict[str, Any], expected_rows: int | None = 88
) -> tuple[np.ndarray, np.ndarray]:
    worker = load_self_hashed(Path(str(score["worker"])))
    visual_path, audio_path = (
        Path(str(worker["visual"])),
        Path(str(worker["audio_embedding"])),
    )
    if file_sha256(visual_path) != worker.get("visual_sha256") or file_sha256(
        audio_path
    ) != worker.get("audio_embedding_sha256"):
        raise ProtocolError("embedding hash mismatch")
    visual, audio = (
        np.asarray(np.load(visual_path, allow_pickle=False), dtype=np.float32),
        np.asarray(np.load(audio_path, allow_pickle=False), dtype=np.float32),
    )
    if (
        visual.ndim != 2
        or visual.shape[1] != 1024
        or (expected_rows is not None and visual.shape != (expected_rows, 1024))
        or audio.shape != visual.shape
        or not np.isfinite(visual).all()
        or not np.isfinite(audio).all()
    ):
        raise ProtocolError("invalid embedding array")
    return visual, audio


def _full_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    padded = np.pad(audio.astype(np.float64), ((15, 15), (0, 0)))
    result = np.empty((visual.shape[0], config.MATRIX_COLUMNS), dtype=np.float64)
    for row in range(visual.shape[0]):
        difference = (
            visual[row : row + 1].astype(np.float64)
            - padded[row : row + config.MATRIX_COLUMNS]
            + 1e-6
        )
        result[row] = np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float64))
    return result


def _full_metrics(matrix: np.ndarray) -> dict[str, Any]:
    curve = np.mean(matrix, axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {
        "curve": [float(value) for value in curve],
        "min_index": index,
        "C": float(np.median(curve) - curve[index]),
        "D": float(curve[index]),
        "offset": 15 - index,
    }


def _score_metrics_from_raw(score: dict[str, Any], label: str) -> dict[str, Any]:
    """Recompute a score row from raw embeddings and its matrix artifact."""
    visual, audio = _arrays(score, expected_rows=None)
    matrix_path = Path(str(score.get("matrix", "")))
    if not matrix_path.is_file() or file_sha256(matrix_path) != str(
        score.get("matrix_sha256")
    ):
        raise ProtocolError(f"matrix hash mismatch: {label}")
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    rebuilt = _full_matrix(visual, audio)
    if matrix.shape != rebuilt.shape or float(np.max(np.abs(matrix - rebuilt))) > 1e-4:
        raise ProtocolError(f"matrix cannot be independently rebuilt: {label}")
    expected = _full_metrics(rebuilt)
    reported = score.get("U")
    if not isinstance(reported, dict):
        raise ProtocolError(f"U metrics missing: {label}")
    selected = rebuilt[np.asarray(config.U_ROWS, dtype=np.int64)]
    expected_u = _full_metrics(selected)
    if (
        int(reported.get("min_index", -1)) != int(expected_u["min_index"])
        or int(reported.get("offset", -(10**9))) != int(expected_u["offset"])
        or abs(float(reported.get("C", math.nan)) - float(expected_u["C"])) > 1e-6
        or abs(float(reported.get("D", math.nan)) - float(expected_u["D"])) > 1e-6
    ):
        raise ProtocolError(f"U endpoint mismatch: {label}")
    reported_curve = np.asarray(reported.get("curve", []), dtype=np.float64)
    if (
        reported_curve.shape != (config.MATRIX_COLUMNS,)
        or float(np.max(np.abs(reported_curve - expected_u["curve"]))) > 1e-6
    ):
        raise ProtocolError(f"U curve mismatch: {label}")
    return {
        "visual": visual,
        "audio": audio,
        "matrix": matrix,
        "full": expected,
        "U": expected_u,
    }


def _lookup_score(
    manifest: dict[str, Any], sample_id: str, video_arm: str, audio_arm: str
) -> dict[str, Any]:
    for row in manifest.get("rows", []):
        if (
            str(row.get("sample_id")) == sample_id
            and str(row.get("video_arm")) == video_arm
            and str(row.get("audio_arm")) == audio_arm
        ):
            return _score(row)
    raise ProtocolError(f"score cell missing: {sample_id}/{video_arm}/{audio_arm}")


_STAT_TOLERANCE = 1e-6


def _assert_stat_equal(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for key in (
        "mean",
        "ci99",
        "ci95",
        "group_labels",
        "group_means",
        "group_positive_count",
        "draws",
        "seed",
        "indices_sha256",
    ):
        if key not in actual or key not in expected:
            raise ProtocolError(f"stat field missing: {label}/{key}")
        if key == "group_labels":
            if actual[key] != expected[key]:
                raise ProtocolError(f"stat mismatch: {label}/{key}")
        elif isinstance(expected[key], list):
            if len(actual[key]) != len(expected[key]) or any(
                abs(float(left) - float(right)) > _STAT_TOLERANCE
                for left, right in zip(actual[key], expected[key], strict=True)
            ):
                raise ProtocolError(f"stat mismatch: {label}/{key}")
        elif isinstance(expected[key], dict):
            if set(actual[key]) != set(expected[key]) or any(
                abs(float(actual[key][name]) - float(expected[key][name]))
                > _STAT_TOLERANCE
                for name in expected[key]
            ):
                raise ProtocolError(f"stat mismatch: {label}/{key}")
        elif key == "mean":
            if abs(float(actual[key]) - float(expected[key])) > _STAT_TOLERANCE:
                raise ProtocolError(f"stat mismatch: {label}/{key}")
        elif actual[key] != expected[key]:
            raise ProtocolError(f"stat mismatch: {label}/{key}")


def _validate_factorial_statistics(
    run_root: Path,
    protocol: dict[str, Any],
    control_scores: dict[str, Any],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild all four cells and the interaction without trusting analysis.json."""
    parent = load_self_hashed(config.CONTROL_SCORES)
    q = load_self_hashed(config.Q_CANDIDATES)
    if (
        len(parent.get("rows", [])) != 36
        or len(q.get("rows", [])) < config.RECORD_COUNT
    ):
        raise ProtocolError("fixed F0 manifests are incomplete")
    candidate_rows = candidates.get("rows", [])
    if len(candidate_rows) != config.RECORD_COUNT:
        raise ProtocolError("F46_C candidate count is not 16")
    candidate_ids = {str(row.get("sample_id")) for row in candidate_rows}
    expected_ids = {str(row.get("sample_id")) for row in protocol.get("records", [])}
    if candidate_ids != expected_ids or any(
        str(row.get("video_arm")) != "F46_C" or str(row.get("audio_arm")) != "N"
        for row in candidate_rows
    ):
        raise ProtocolError("F46_C candidate identity mismatch")
    f46_n_rows = [
        row
        for row in control_scores.get("rows", [])
        if str(row.get("video_arm")) == "F46_N" and str(row.get("audio_arm")) == "N"
    ]
    if len(f46_n_rows) != config.RECORD_COUNT:
        raise ProtocolError("fresh F46_N cell count is not 16")
    f46_n_ids = {str(row.get("sample_id")) for row in f46_n_rows}
    if f46_n_ids != expected_ids:
        raise ProtocolError("fresh F46_N identity mismatch")
    candidate_by_id = {str(row["sample_id"]): _score(row) for row in candidate_rows}
    fresh_by_id = {str(row["sample_id"]): _score(row) for row in f46_n_rows}
    groups: list[str] = []
    values: dict[str, list[float]] = {"g0": [], "g1": [], "I": []}
    records: list[dict[str, Any]] = []
    for record in sorted(
        protocol["records"],
        key=lambda row: (str(row["source_group"]), str(row["sample_id"])),
    ):
        sid, group = str(record["sample_id"]), str(record["source_group"])
        n0 = _score_metrics_from_raw(
            _lookup_score(parent, sid, "N", "N"), f"F0_N/{sid}"
        )
        c0 = _score_metrics_from_raw(
            _lookup_score(q, sid, "CORRECT", "N"), f"F0_C/{sid}"
        )
        n1 = _score_metrics_from_raw(fresh_by_id[sid], f"F46_N/{sid}")
        c1 = _score_metrics_from_raw(candidate_by_id[sid], f"F46_C/{sid}")
        k0 = int(n0["U"]["min_index"])
        g0 = float(n0["U"]["curve"][k0] - c0["U"]["curve"][k0])
        g1 = float(n1["U"]["curve"][k0] - c1["U"]["curve"][k0])
        interaction = float(g1 - g0)
        groups.append(group)
        values["g0"].append(g0)
        values["g1"].append(g1)
        values["I"].append(interaction)
        records.append(
            {
                "sample_id": sid,
                "source_group": group,
                "k0": k0,
                "g0": g0,
                "g1": g1,
                "I": interaction,
                "F46_within_reference": {
                    "C": float(n1["U"]["C"] - c1["U"]["C"]),
                    "D": float(c1["U"]["D"] - n1["U"]["D"]),
                    "A": float(n1["U"]["curve"][k0] - c1["U"]["curve"][k0]),
                },
            }
        )
    expected_stats = {
        name: _bootstrap(
            values[name],
            groups,
            seed=config.BOOTSTRAP_SEED,
            draws=config.BOOTSTRAP_DRAWS,
        )
        for name in ("g0", "g1", "I")
    }
    interaction = expected_stats["I"]
    passed = bool(
        (interaction["ci99"][0] > 0 or interaction["ci99"][1] < 0)
        and abs(interaction["mean"]) > 0.05
        and max(
            interaction["group_positive_count"],
            config.GROUP_COUNT - interaction["group_positive_count"],
        )
        >= 7
    )
    decision = (
        "REFERENCE_DEPENDENT_RESPONSE"
        if passed
        else "NO_REFERENCE_INTERACTION_ESTABLISHED"
    )
    analysis = load_self_hashed(run_root / "analysis.json")
    for name, expected_name in (("g0", "g0"), ("g1", "g1"), ("interaction", "I")):
        _assert_stat_equal(analysis.get(name, {}), expected_stats[expected_name], name)
    if analysis.get("scientific_decision") != decision or bool(
        analysis.get("within_reference_negative")
    ) != bool(float(np.mean(values["g1"])) < 0):
        raise ProtocolError("factorial decision mismatch")
    reported_records = analysis.get("records", [])
    if len(reported_records) != len(records):
        raise ProtocolError("factorial record count mismatch")
    for actual, expected in zip(reported_records, records, strict=True):
        if (
            actual.get("sample_id") != expected["sample_id"]
            or actual.get("source_group") != expected["source_group"]
            or actual.get("k0") != expected["k0"]
        ):
            raise ProtocolError(
                f"factorial record identity mismatch: {expected['sample_id']}"
            )
        for key in ("g0", "g1", "I"):
            if abs(float(actual.get(key, math.nan)) - expected[key]) > _STAT_TOLERANCE:
                raise ProtocolError(
                    f"factorial record mismatch: {expected['sample_id']}/{key}"
                )
        for key in ("C", "D", "A"):
            if (
                abs(
                    float(actual.get("F46_within_reference", {}).get(key, math.nan))
                    - expected["F46_within_reference"][key]
                )
                > _STAT_TOLERANCE
            ):
                raise ProtocolError(
                    f"factorial within-reference mismatch: {expected['sample_id']}/{key}"
                )
    return {
        "scientific_decision": decision,
        "records": records,
        "stats": expected_stats,
    }


def _fixed_input_errors(drivers: dict[str, Any], refs: dict[str, Any]) -> list[str]:
    paths = {
        "drivers": config.DRIVERS,
        "control_scores": config.CONTROL_SCORES,
        "q_candidates": config.Q_CANDIDATES,
        "old_control_scores": config.OLD_CONTROL_SCORES,
        "old_protocol": config.OLD_PROTOCOL,
        "old_drivers": config.OLD_DRIVERS,
        "old_reference": config.OLD_REFERENCE,
        "checkpoint": config.WAV2LIP_CHECKPOINT,
        "syncnet": config.SYNCNET_MODEL,
    }
    errors = []
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != config.FIXED_HASHES[name]:
            errors.append(f"fixed_asset:{name}")
    for row in refs.get("rows", []):
        sid = str(row.get("sample_id"))
        for name in ("source_video", "box", "F0", "F46", "historical_F46"):
            binding = row.get(name)
            if not isinstance(binding, dict):
                errors.append(f"binding_missing:{sid}/{name}")
                continue
            path = Path(str(binding.get("path", "")))
            if not path.is_file() or file_sha256(path) != str(binding.get("sha256")):
                errors.append(f"binding_changed:{sid}/{name}")
    for row in drivers.get("rows", []):
        sid = str(row.get("sample_id"))
        for name, binding in (
            ("natural_audio", row.get("natural_audio")),
            ("F0", row.get("F0")),
            ("F46", row.get("F46")),
            ("N", row.get("mel", {}).get("N")),
            ("CORRECT", row.get("mel", {}).get("CORRECT")),
        ):
            if not isinstance(binding, dict):
                errors.append(f"driver_binding_missing:{sid}/{name}")
                continue
            path = Path(str(binding.get("path", "")))
            if not path.is_file() or file_sha256(path) != str(binding.get("sha256")):
                errors.append(f"driver_binding_changed:{sid}/{name}")
    return errors


def _domain(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> dict[str, Any]:
    values = []
    for row in config.U_ROWS:
        indices = (
            int(row) + np.arange(config.MATRIX_COLUMNS, dtype=np.int64) + int(lag_start)
        )
        if int(indices.min()) < 0 or int(indices.max()) >= audio.shape[0]:
            raise ProtocolError("domain lacks real support")
        difference = (
            visual[row : row + 1].astype(np.float64)
            - audio[indices].astype(np.float64)
            + 1e-6
        )
        values.append(
            np.sqrt(np.sum(difference * difference, axis=1, dtype=np.float64))
        )
    curve = np.mean(np.asarray(values), axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {
        "curve": curve.tolist(),
        "min_index": index,
        "offset": -int(lag_start) - index,
        "D": float(curve[index]),
        "C": float(np.median(curve) - curve[index]),
        "lag_start": int(lag_start),
    }


def _bootstrap(
    values: list[float], groups: list[str], *, seed: int, draws: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("expected eight source groups x two records")
    means = np.asarray(
        [np.mean(grouped[label], dtype=np.float64) for label in labels],
        dtype=np.float64,
    )
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 8, size=(draws, 8), endpoint=False
    )
    estimates = means[indices].mean(axis=1, dtype=np.float64)
    return {
        "mean": float(means.mean()),
        "ci99": [
            float(np.quantile(estimates, 0.005, method="linear")),
            float(np.quantile(estimates, 0.995, method="linear")),
        ],
        "ci95": [
            float(np.quantile(estimates, 0.025, method="linear")),
            float(np.quantile(estimates, 0.975, method="linear")),
        ],
        "group_labels": labels,
        "group_means": {
            label: float(value) for label, value in zip(labels, means, strict=True)
        },
        "group_positive_count": int(np.sum(means > 0)),
        "seed": seed,
        "draws": draws,
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def _historical_gate() -> tuple[bool, dict[str, Any]]:
    old = load_self_hashed(config.OLD_CONTROL_SCORES)
    by_key = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): _score(
            row
        )
        for row in old.get("rows", [])
    }
    records = load_self_hashed(config.OLD_PROTOCOL).get("records", [])
    if len(records) != 16:
        raise ProtocolError("historical protocol record count")
    offsets, damages, groups, detail = [], [], [], []
    for record in records:
        sid, group = str(record["sample_id"]), str(record["source_group"])
        n = by_key[(sid, "F46_N", "N")]
        d = by_key[(sid, "F46_N", "A_DELAY")]
        vn, an = _arrays(n)
        vd, ad = _arrays(d)
        if not np.array_equal(vn, vd):
            raise ProtocolError(f"historical delayed visual differs: {sid}")
        natural, matched, legacy = (
            _domain(vn, an, config.NATURAL_LAG_START),
            _domain(vn, ad, config.DELAY_LAG_START),
            _domain(vn, ad, config.NATURAL_LAG_START),
        )
        delta = int(matched["offset"]) - int(natural["offset"])
        damage = float(
            legacy["curve"][natural["min_index"]]
            - natural["curve"][natural["min_index"]]
        )
        offsets.append(delta)
        damages.append(damage)
        groups.append(group)
        detail.append(
            {
                "sample_id": sid,
                "source_group": group,
                "offset_difference": delta,
                "offset_pass": bool(-6 <= delta <= -4),
                "damage_at_natural_k0": damage,
            }
        )
    stat = _bootstrap(damages, groups, seed=20260909, draws=10_000)
    passed = bool(
        sum(-6 <= value <= -4 for value in offsets) >= 14
        and stat["ci95"][0] > 0
        and stat["group_positive_count"] >= 7
    )
    return passed, {
        "offset_pass_count": int(sum(-6 <= value <= -4 for value in offsets)),
        "legacy_damage": stat,
        "records": detail,
        "control_pass": passed,
    }


def _fresh_delay_gate(control_scores: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    rows = control_scores.get("rows", [])
    by_key = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): _score(
            row
        )
        for row in rows
    }
    if len(rows) != 38:
        raise ProtocolError("fresh control score count is not 38")
    offsets, damages, groups = [], [], []
    for row in rows:
        if row.get("video_arm") == "F46_N" and row.get("audio_arm") == "N":
            sid = str(row["sample_id"])
            n = by_key[(sid, "F46_N", "N")]
            d = by_key[(sid, "F46_N", "A_DELAY")]
            vn, an = _arrays(n)
            vd, ad = _arrays(d)
            if not np.array_equal(vn, vd):
                raise ProtocolError(f"fresh delayed visual differs: {sid}")
            natural, matched, legacy = (
                _domain(vn, an, config.NATURAL_LAG_START),
                _domain(vn, ad, config.DELAY_LAG_START),
                _domain(vn, ad, config.NATURAL_LAG_START),
            )
            delta = int(matched["offset"]) - int(natural["offset"])
            offsets.append(delta)
            damages.append(
                float(
                    legacy["curve"][natural["min_index"]]
                    - natural["curve"][natural["min_index"]]
                )
            )
            groups.append(str(row["source_group"]))
    if len(offsets) != 16:
        raise ProtocolError("fresh F46 delay cells missing")
    stat = _bootstrap(damages, groups, seed=20260909, draws=10_000)
    return int(sum(-6 <= value <= -4 for value in offsets)), {
        "offset_pass_count": int(sum(-6 <= value <= -4 for value in offsets)),
        "legacy_damage": stat,
        "pass": bool(
            sum(-6 <= value <= -4 for value in offsets) >= 14
            and stat["ci95"][0] > 0
            and stat["group_positive_count"] >= 7
        ),
    }


def _fresh_replay_checks(
    control_scores: dict[str, Any], parent_scores: dict[str, Any]
) -> tuple[bool, bool]:
    """Check the two fresh replay cells without trusting producer flags."""
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        decode_video_frames,
    )

    fresh = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row
        for row in control_scores.get("rows", [])
    }
    parent = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row
        for row in parent_scores.get("rows", [])
    }
    replay = [
        row
        for row in control_scores.get("rows", [])
        if row.get("video_arm") == "N_REPLAY"
    ]
    repeats = [
        row
        for row in control_scores.get("rows", [])
        if row.get("video_arm") == "F46_N_REPEAT"
    ]
    replay_pass = len(replay) == 2
    for row in replay:
        old = parent.get((str(row["sample_id"]), "N", "N"))
        if old is None:
            replay_pass = False
            continue
        score, prior = _score(row), _score(old)
        new_matrix, old_matrix = (
            np.asarray(np.load(score["matrix"], allow_pickle=False)),
            np.asarray(np.load(prior["matrix"], allow_pickle=False)),
        )
        try:
            new_frames = decode_video_frames(Path(str(score["media"])))
            old_frames = decode_video_frames(Path(str(prior["media"])))
            pixels = len(new_frames) == len(old_frames) and all(
                np.array_equal(a, b)
                for a, b in zip(new_frames, old_frames, strict=True)
            )
        except Exception:  # noqa: BLE001
            pixels = False
        endpoint_error = max(
            (
                abs(float(score["U"][key]) - float(prior["U"][key]))
                for key in ("C", "D")
            ),
            default=float("inf"),
        )
        replay_pass = (
            replay_pass
            and pixels
            and score.get("source_pcm_sha256") == prior.get("source_pcm_sha256")
            and new_matrix.shape == old_matrix.shape
            and float(np.max(np.abs(new_matrix - old_matrix))) <= 1e-4
            and endpoint_error <= 1e-6
            and int(score["U"]["offset"]) == int(prior["U"]["offset"])
        )
    repeat_pass = len(repeats) == 2
    for row in repeats:
        base = fresh.get((str(row["sample_id"]), "F46_N", "N"))
        if base is None:
            repeat_pass = False
            continue
        score, prior = _score(row), _score(base)
        new_matrix, old_matrix = (
            np.asarray(np.load(score["matrix"], allow_pickle=False)),
            np.asarray(np.load(prior["matrix"], allow_pickle=False)),
        )
        try:
            new_frames = decode_video_frames(Path(str(score["media"])))
            old_frames = decode_video_frames(Path(str(prior["media"])))
            pixels = len(new_frames) == len(old_frames) and all(
                np.array_equal(a, b)
                for a, b in zip(new_frames, old_frames, strict=True)
            )
        except Exception:  # noqa: BLE001
            pixels = False
        endpoint_error = max(
            (
                abs(float(score["U"][key]) - float(prior["U"][key]))
                for key in ("C", "D")
            ),
            default=float("inf"),
        )
        repeat_pass = (
            repeat_pass
            and pixels
            and score.get("source_pcm_sha256") == prior.get("source_pcm_sha256")
            and new_matrix.shape == old_matrix.shape
            and float(np.max(np.abs(new_matrix - old_matrix))) <= 1e-4
            and endpoint_error <= 1e-6
            and int(score["U"]["offset"]) == int(prior["U"]["offset"])
        )
    return replay_pass, repeat_pass


def _parity_check(
    control_scores: dict[str, Any], parent_scores: dict[str, Any]
) -> bool:
    fresh = {
        (str(row.get("video_arm")), str(row.get("sample_id"))): _score(row)
        for row in control_scores.get("rows", [])
        if str(row.get("video_arm", "")).startswith("PARITY_")
    }
    expected = {
        (str(row.get("video_arm")), str(row.get("sample_id"))): row
        for row in parent_scores.get("rows", [])
        if str(row.get("video_arm", "")).startswith("PARITY_")
    }
    if len(fresh) != 2 or len(expected) != 2:
        return False
    for key, score in fresh.items():
        row = expected.get(key)
        if row is None:
            return False
        reference = np.asarray(
            np.load(row["parity"]["reference_matrix"], allow_pickle=False)
        )
        actual = np.asarray(np.load(score["matrix"], allow_pickle=False))
        visual, audio = _arrays(score, expected_rows=None)
        rebuilt = _full_matrix(visual, audio)
        if (
            actual.shape != reference.shape
            or float(np.max(np.abs(actual - reference))) > 1e-4
            or rebuilt.shape != actual.shape
            or float(np.max(np.abs(rebuilt - actual))) > 1e-4
        ):
            return False
        actual_endpoint, reference_endpoint = (
            _full_metrics(rebuilt),
            _full_metrics(reference),
        )
        if max(
            abs(float(actual_endpoint[key]) - float(reference_endpoint[key]))
            for key in ("C", "D")
        ) > 1e-6 or int(actual_endpoint["offset"]) != int(reference_endpoint["offset"]):
            return False
    return True


def validate(run_root: Path, stage: str = "all") -> dict[str, Any]:
    errors: list[str] = []
    scientific = "not_available"
    try:
        protocol = load_self_hashed(run_root / "protocol.json")
        drivers = load_self_hashed(run_root / "drivers/manifest.json")
        refs = load_self_hashed(run_root / "reference_manifest.json")
        errors.extend(_fixed_input_errors(drivers, refs))
        driver_groups = [
            str(row.get("source_group")) for row in drivers.get("rows", [])
        ]
        ref_groups = [str(row.get("source_group")) for row in refs.get("rows", [])]
        if (
            protocol.get("record_count") != 16
            or protocol.get("source_group_count") != 8
            or len(drivers.get("rows", [])) != 16
            or len(refs.get("rows", [])) != 16
            or len(set(driver_groups)) != 8
            or any(driver_groups.count(group) != 2 for group in set(driver_groups))
            or sorted(driver_groups) != sorted(ref_groups)
        ):
            errors.append("cohort_count")
        for row in refs.get("rows", []):
            path = Path(str(row["F46"]["path"]))
            value = np.asarray(np.load(path, allow_pickle=False))
            if (
                file_sha256(path) != row["F46"].get("sha256")
                or value.shape != (224, 224, 3)
                or value.dtype != np.uint8
            ):
                errors.append(f"reference:{row.get('sample_id')}")
        if stage in {"controls", "candidates", "analyze", "all"}:
            hist_pass, _hist = _historical_gate()
            control = load_self_hashed(run_root / "control_analysis.json")
            scientific = str(control.get("scientific_decision", "not_available"))
            if (
                bool(
                    control.get("historical_control", {}).get("control_pass", hist_pass)
                )
                != hist_pass
            ):
                errors.append("historical_control_mismatch")
            if hist_pass:
                if (
                    control.get("fresh_video_count") != 20
                    or control.get("fresh_score_count") != 38
                ):
                    errors.append("control_budget")
                fresh = load_self_hashed(run_root / "control_scores/manifest.json")
                count, gate = _fresh_delay_gate(fresh)
                expected_ids = {
                    str(row.get("sample_id")) for row in protocol.get("records", [])
                }
                fresh_ids = {
                    str(row.get("sample_id"))
                    for row in fresh.get("rows", [])
                    if str(row.get("video_arm")) == "F46_N"
                    and str(row.get("audio_arm")) == "N"
                }
                if (
                    len(
                        [
                            row
                            for row in fresh.get("rows", [])
                            if str(row.get("video_arm")) == "F46_N"
                            and str(row.get("audio_arm")) == "N"
                        ]
                    )
                    != 16
                    or fresh_ids != expected_ids
                ):
                    errors.append("fresh_f46_identity")
                if count != int(
                    control.get("delay_control", {}).get("offset_pass_count", -1)
                ) or bool(gate["pass"]) != bool(
                    control.get("delay_control", {}).get("pass")
                ):
                    errors.append("fresh_delay_gate_mismatch")
                parent = load_self_hashed(config.CONTROL_SCORES)
                replay_pass, repeat_pass = _fresh_replay_checks(fresh, parent)
                if (
                    not replay_pass
                    or not repeat_pass
                    or not _parity_check(fresh, parent)
                ):
                    errors.append("fresh_control_replay_or_parity")
            elif (run_root / "candidate_scores/manifest.json").is_file():
                errors.append("candidate_present_after_control_failure")
        if stage in {"candidates", "analyze", "all"}:
            control = load_self_hashed(run_root / "control_analysis.json")
            if control.get("control_pass"):
                candidates = load_self_hashed(
                    run_root / "candidate_scores/manifest.json"
                )
                if len(candidates.get("rows", [])) != 16:
                    errors.append("candidate_count")
                for row in candidates.get("rows", []):
                    if row.get("video_arm") != "F46_C" or row.get("audio_arm") != "N":
                        errors.append(f"candidate_binding:{row.get('sample_id')}")
                if stage in {"analyze", "all"} and not errors:
                    _validate_factorial_statistics(
                        run_root,
                        protocol,
                        load_self_hashed(run_root / "control_scores/manifest.json"),
                        candidates,
                    )
            elif stage == "candidates":
                errors.append("candidate_stage_without_control")
        if stage in {"analyze", "all"} and not (run_root / "analysis.json").is_file():
            errors.append("analysis_missing")
        if (run_root / "analysis.json").is_file():
            scientific = str(
                load_self_hashed(run_root / "analysis.json").get(
                    "scientific_decision", scientific
                )
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"exception:{type(exc).__name__}:{exc}")
    result = {
        "schema_version": 1,
        "protocol_id": "wav2lip_reference_conditioning_interaction",
        "stage": stage,
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
        "independent": True,
        "scientific_decision": scientific,
        "replacement_confirmed": False,
        "waveform_head_authorized": False,
        "generalization_established": False,
        "historical_shift_gate_repaired": False,
    }
    output = run_root / (
        "control_validation.json" if stage == "controls" else "validation.json"
    )
    return write_json(output, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", default="all")
    args = parser.parse_args(argv)
    result = validate(args.run_root, args.stage)
    print(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
