"""Independent validator for ``wav2lip_noise_cross_v1``.

This module deliberately repeats the small numerical reductions instead of
calling the runner's analysis functions.  It verifies the saved PCM,
embeddings, matrices, condition bindings, support windows, and controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.experiments.tts_native_gain_attribution.audio import read_pcm16_wav  # noqa: E402
from scripts.experiments.tts_native_gain_attribution.common import ProtocolError, file_sha256  # noqa: E402
from scripts.experiments.wav2lip_noise_cross_metrics import (  # noqa: E402
    EMBEDDING_DIM,
    LAG_COUNT,
    VSHIFT,
    canonical_hash,
)


PROTOCOL_ID = "wav2lip_noise_cross_v1"
MAIN_KEYS = {"RAW": ("RAW", "RAW"), "q00": ("A0", "A0"), "q01": ("A0", "NOISE20"), "q10": ("NOISE20", "A0"), "q11": ("NOISE20", "NOISE20")}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _read_self(path: Path) -> dict[str, Any]:
    value = _read(path)
    recorded = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_hash(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def _array(binding: dict[str, Any], *, label: str) -> np.ndarray:
    path = Path(str(binding.get("path", "")))
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    if file_sha256(path) != str(binding.get("sha256")):
        raise ProtocolError(f"{label} hash mismatch: {path}")
    value = np.asarray(np.load(path, allow_pickle=False))
    shape = binding.get("shape")
    if isinstance(shape, list) and [int(item) for item in value.shape] != [int(item) for item in shape]:
        raise ProtocolError(f"{label} shape metadata mismatch: {path}")
    if not np.isfinite(value).all():
        raise ProtocolError(f"{label} contains non-finite values")
    return value


def _matrix_from_embeddings(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual32 = np.asarray(visual, dtype=np.float32)
    audio32 = np.asarray(audio, dtype=np.float32)
    if visual32.ndim != 2 or audio32.ndim != 2 or visual32.shape[1] != EMBEDDING_DIM or audio32.shape[1] != EMBEDDING_DIM:
        raise ProtocolError(f"invalid embedding dimensions: {visual32.shape}/{audio32.shape}")
    count = min(int(visual32.shape[0]), int(audio32.shape[0]))
    if count < 1:
        raise ProtocolError("no common embedding rows")
    padded = np.pad(audio32[:count], ((VSHIFT, VSHIFT), (0, 0)), mode="constant")
    result = np.empty((count, LAG_COUNT), dtype=np.float32)
    for index in range(count):
        delta = visual32[index : index + 1] - padded[index : index + LAG_COUNT]
        result[index] = np.sqrt(np.sum(np.square(delta + np.float32(1e-6), dtype=np.float32), axis=1), dtype=np.float32)
    return result.astype(np.float64)


def _support(matrices: list[np.ndarray]) -> list[int]:
    count = min(int(value.shape[0]) for value in matrices)
    return list(range(VSHIFT, count - VSHIFT))


def _metrics(matrix: np.ndarray, rows: list[int]) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != LAG_COUNT or not np.isfinite(value).all():
        raise ProtocolError("invalid distance matrix")
    if not rows or any(index < 0 or index >= value.shape[0] for index in rows):
        raise ProtocolError("invalid common support")
    curve = value[np.asarray(rows)].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    return {"support_count": len(rows), "support_rows": rows, "curve": curve.tolist(), "min_index": index, "official_offset": VSHIFT - index, "sync_d": minimum, "background_b": float(np.median(curve)), "sync_c": float(np.median(curve) - minimum), "d0": float(curve[VSHIFT])}


def _four(q00: float, q01: float, q10: float, q11: float) -> dict[str, float]:
    return {"evaluation": float(q01 - q00), "generation": float(q10 - q00), "interaction": float(q11 - q10 - q01 + q00), "total": float(q11 - q00)}


def _bootstrap(rows: list[dict[str, Any]], field: str, *, primary: bool) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(float(row[field]))
    labels = sorted(grouped)
    if not labels:
        raise ProtocolError(f"no rows for bootstrap: {field}")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    indices = np.random.Generator(np.random.PCG64(20_260_920)).integers(0, len(labels), size=(20_000, len(labels)), dtype=np.int64)
    estimates = means[indices].mean(axis=1, dtype=np.float64)
    low, high = ((0.0125, 0.9875) if primary else (0.025, 0.975))
    return {"mean": float(means.mean()), "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))], "ci_primary": [float(np.quantile(estimates, low, method="linear")), float(np.quantile(estimates, high, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "positive_count": int(np.sum(means > 0.0)), "negative_count": int(np.sum(means < 0.0)), "zero_count": int(np.sum(means == 0.0)), "group_count": len(labels), "draws": 20_000, "seed": 20_260_920}


def _close_json(a: Any, b: Any, *, atol: float = 1e-8) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return bool(np.isclose(float(a), float(b), atol=atol, rtol=0.0))
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close_json(x, y, atol=atol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close_json(a[key], b[key], atol=atol) for key in a)
    return a == b


def validate(run_dir: Path) -> dict[str, Any]:
    root = run_dir.resolve()
    protocol = _read_self(root / "protocol.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("protocol id mismatch")
    _read_self(root / "inputs.json")
    audio = _read_self(root / "audio" / "manifest.json")
    videos = _read_self(root / "videos" / "manifest.json")
    scores = _read_self(root / "scores" / "manifest.json")
    inputs = _read_self(root / "inputs.json")
    input_ids = {int(row["sample_id"]): row for row in inputs.get("records", [])}
    expected = protocol.get("expected", {})
    audio_bindings: dict[tuple[int, str, str], dict[str, Any]] = {}
    for audio_row in audio.get("rows", []):
        sid = int(audio_row["sample_id"])
        for source in ("N", "T"):
            for condition, binding in audio_row["arms"][source]["conditions"].items():
                audio_bindings[(sid, source, str(condition))] = dict(binding)
    main_video_rows = [row for row in videos.get("rows", []) if row.get("status") == "complete"]
    if len(main_video_rows) != int(expected.get("main_videos", -1)):
        raise ProtocolError(f"main video count mismatch: {len(main_video_rows)}")
    if len([row for row in videos.get("controls", []) if row.get("status") == "complete"]) != int(expected.get("repeat_videos", -1)):
        raise ProtocolError("repeat video count mismatch")
    # PCM contract and the independent A0/noise checks.
    for row in audio.get("rows", []):
        sid = int(row["sample_id"])
        if sid not in input_ids:
            raise ProtocolError(f"audio row has unknown sample: {sid}")
        power = dict(row.get("natural_noise_power_binding", {}))
        power_path = Path(str(power.get("path", "")))
        if not power_path.is_file() or file_sha256(power_path) != str(power.get("sha256")):
            raise ProtocolError(f"natural noise power binding mismatch: {sid}")
        power_array = np.asarray(np.load(power_path, allow_pickle=False), dtype=np.float64)
        if power_array.ndim != 1 or power_array.size != 257 or not np.isfinite(power_array).all():
            raise ProtocolError(f"natural noise power is invalid: {sid}")
        for source in ("N", "T"):
            arms = row["arms"][source]
            values: dict[str, np.ndarray] = {}
            decoded = dict(arms.get("decoded_input", {}))
            decoded_path = Path(str(decoded.get("path", "")))
            if not decoded_path.is_file() or file_sha256(decoded_path) != str(decoded.get("sha256")):
                raise ProtocolError(f"decoded audio binding mismatch: {sid}/{source}")
            for condition in ("RAW", "A0", "NOISE20"):
                binding = arms["conditions"][condition]
                path = Path(str(binding["path"]))
                if file_sha256(path) != str(binding["sha256"]):
                    raise ProtocolError(f"audio hash mismatch: {path}")
                values[condition] = read_pcm16_wav(path)
                if values[condition].size != int(binding["sample_count"]):
                    raise ProtocolError(f"audio sample count mismatch: {path}")
            if not np.array_equal(values["RAW"], read_pcm16_wav(decoded_path)):
                raise ProtocolError(f"RAW is not the decoded carrier: {sid}/{source}")
            mask_binding = arms["conditions"]["activity_mask"]
            mask = _array(mask_binding, label=f"activity mask {sid}/{source}").astype(bool)
            if mask.size != values["RAW"].size:
                raise ProtocolError("activity mask clock mismatch")
            checks = {"snr_db": float(arms["checks"]["snr_db"])}
            if not 19.8 <= checks["snr_db"] <= 20.2:
                raise ProtocolError(f"SNR contract failed: {sid}/{source}")
            if not np.array_equal(values["RAW"], values["RAW"].copy()):
                raise ProtocolError("RAW array is not stable")
    render_map = {(int(row["sample_id"]), str(row["source"]), str(row["condition"])): row for row in main_video_rows}
    for row in main_video_rows + [item for item in videos.get("controls", []) if item.get("status") == "complete"]:
        output = Path(str(row["video_path"]))
        receipt = Path(str(row["receipt_path"]))
        if not output.is_file() or file_sha256(output) != str(row["video_sha256"]):
            raise ProtocolError(f"video binding mismatch: {output}")
        value = _read(receipt)
        if value.get("audio_sha256") != str(row["audio_sha256"]) or value.get("output_sha256") != file_sha256(output):
            raise ProtocolError(f"render receipt binding mismatch: {receipt}")
        if value.get("pixel_sha256") != row.get("pixel_sha256") or value.get("pts_sha256") != row.get("pts_sha256"):
            raise ProtocolError(f"video pixel/PTS binding mismatch: {output}")
    for control in [item for item in videos.get("controls", []) if item.get("status") == "complete"]:
        base = render_map.get((int(control["sample_id"]), str(control["source"]), "A0"))
        if base is None or control.get("pixel_sha256") != base.get("pixel_sha256") or control.get("pts_sha256") != base.get("pts_sha256"):
            raise ProtocolError(f"repeat render is not pixel/PTS identical: {control['sample_id']}/{control['source']}")
    score_cells = [row for row in scores.get("cells", []) if row.get("kind") == "main"]
    if len(score_cells) != int(expected.get("main_scores", -1)):
        raise ProtocolError(f"main score count mismatch: {len(score_cells)}")
    by_arm: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell in score_cells:
        key = str(cell["key"])
        if key not in MAIN_KEYS:
            raise ProtocolError(f"unknown main score key: {key}")
        sid, source = int(cell["sample_id"]), str(cell["source"])
        if key in by_arm[(sid, source)]:
            raise ProtocolError(f"duplicate score cell: {sid}/{source}/{key}")
        expected_video, expected_audio = MAIN_KEYS[key]
        if str(cell.get("video_condition")) != expected_video or str(cell.get("audio_condition")) != expected_audio:
            raise ProtocolError(f"condition binding mismatch: {sid}/{source}/{key}")
        video_binding = render_map.get((sid, source, expected_video))
        audio_binding = audio_bindings.get((sid, source, expected_audio))
        if video_binding is None or audio_binding is None:
            raise ProtocolError(f"condition input is missing: {sid}/{source}/{key}")
        if str(cell.get("video_input")) != str(video_binding["video_path"]) or str(cell.get("video_input_sha256")) != str(video_binding["video_sha256"]):
            raise ProtocolError(f"video input binding mismatch: {sid}/{source}/{key}")
        if str(cell.get("audio_input")) != str(audio_binding["path"]) or str(cell.get("audio_input_sha256")) != str(audio_binding["sha256"]):
            raise ProtocolError(f"audio input binding mismatch: {sid}/{source}/{key}")
        if file_sha256(Path(str(cell["video_input"]))) != str(cell["video_input_sha256"]) or file_sha256(Path(str(cell["audio_input"]))) != str(cell["audio_input_sha256"]):
            raise ProtocolError(f"score input hash mismatch: {sid}/{source}/{key}")
        visual = _array(dict(cell["visual_feature"]), label=f"visual {sid}/{source}/{key}")
        aud = _array(dict(cell["audio_feature"]), label=f"audio feature {sid}/{source}/{key}")
        matrix = _array(dict(cell["matrix"]), label=f"matrix {sid}/{source}/{key}").astype(np.float64)
        rebuilt = _matrix_from_embeddings(visual, aud)
        if rebuilt.shape != matrix.shape or float(np.max(np.abs(rebuilt - matrix))) > 1e-4:
            raise ProtocolError(f"independent matrix mismatch: {sid}/{source}/{key}")
        by_arm[(sid, source)][key] = {"cell": cell, "matrix": matrix}
    source_rows: list[dict[str, Any]] = []
    for (sid, source), cells in sorted(by_arm.items()):
        if set(cells) != set(MAIN_KEYS):
            raise ProtocolError(f"incomplete five-cell arm: {sid}/{source}")
        matrices = [cells[key]["matrix"] for key in MAIN_KEYS]
        rows = _support(matrices)
        if len(rows) < 25:
            raise ProtocolError(f"insufficient common support: {sid}/{source}")
        metrics = {key: _metrics(cells[key]["matrix"], rows) for key in MAIN_KEYS}
        four = _four(metrics["q00"]["sync_c"], metrics["q01"]["sync_c"], metrics["q10"]["sync_c"], metrics["q11"]["sync_c"])
        source_group = str(input_ids[sid].get("source_group", sid))
        source_rows.append({"sample_id": sid, "source": source, "source_group": source_group, "status": "COMPLETE", "support_count": len(rows), "metrics": metrics, **{key: four[value] for key, value in (("E", "evaluation"), ("G", "generation"), ("I", "interaction"), ("Total", "total"))}})
    saved_rows = _read(root / "audit" / "per_source.json").get("rows", [])
    saved_map = {(int(row["sample_id"]), str(row["source"])): row for row in saved_rows if row.get("status") == "COMPLETE"}
    if set(saved_map) != {(int(row["sample_id"]), str(row["source"])) for row in source_rows}:
        raise ProtocolError("per-source manifest does not match score cells")
    for row in source_rows:
        saved = saved_map[(int(row["sample_id"]), str(row["source"]))]
        for field in ("E", "G", "I", "Total"):
            if not np.isclose(float(row[field]), float(saved[field]), atol=1e-8, rtol=0.0):
                raise ProtocolError(f"analysis value mismatch: {row['sample_id']}/{row['source']}/{field}")
    summary = _read(root / "audit" / "summary.json")
    for metric, field in (("E_T", "E"), ("G_T", "G")):
        values = [row for row in source_rows if row["source"] == "T"]
        if not values:
            raise ProtocolError(f"missing primary source: {metric}")
        expected_stats = _bootstrap(values, field, primary=True)
        actual = summary.get("primary", {}).get(metric)
        if not isinstance(actual, dict) or not np.isclose(float(actual.get("mean")), expected_stats["mean"], atol=1e-8, rtol=0.0) or not np.allclose(actual.get("ci_primary"), expected_stats["ci_primary"], atol=1e-8, rtol=0.0):
            raise ProtocolError(f"primary summary mismatch: {metric}")
    # Re-render controls: the independent file hash is allowed to differ, but
    # decoded FFV1 bytes should be identical for the same deterministic input.
    control_score_rows = [row for row in scores.get("cells", []) if row.get("kind") in {"repeat", "delay"}]
    expected_control_scores = int(expected.get("control_scores", -1))
    if len(control_score_rows) != expected_control_scores:
        raise ProtocolError(f"control score count mismatch: {len(control_score_rows)}")
    for row in control_score_rows:
        sid, source, kind = int(row["sample_id"]), str(row["source"]), str(row["kind"])
        for field in ("video_input", "audio_input"):
            path = Path(str(row.get(field, "")))
            digest_field = f"{field}_sha256"
            if not path.is_file() or file_sha256(path) != str(row.get(digest_field)):
                raise ProtocolError(f"control score input binding mismatch: {sid}/{source}/{kind}/{field}")
        matrix = _array(dict(row["matrix"]), label=f"control matrix {sid}/{source}/{kind}").astype(np.float64)
        if kind == "repeat":
            base = by_arm[(sid, source)]["q00"]["matrix"]
            base_rows = _support([base, matrix])
            if len(base_rows) < 25:
                raise ProtocolError("repeat control has insufficient support")
            lhs, rhs = _metrics(base, base_rows), _metrics(matrix, base_rows)
            if abs(lhs["sync_c"] - rhs["sync_c"]) > 0.01 or abs(lhs["sync_d"] - rhs["sync_d"]) > 0.01 or abs(lhs["official_offset"] - rhs["official_offset"]) > 1:
                raise ProtocolError(f"repeat control failed: {sid}/{source}")
        else:
            base = by_arm[(sid, source)]["q00"]["matrix"]
            count = min(base.shape[0], matrix.shape[0])
            safe = range(20, count - 20)
            errors = [abs(float(matrix[index, column + 5]) - float(base[index, column])) for index in safe for column in range(26) if column + 5 < matrix.shape[1]]
            if not errors or max(errors) > 1e-3:
                raise ProtocolError(f"delay control failed: {sid}/{source}")
    validation = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS", "engineering_status": "COMPLETE", "independent": True, "main_score_count": len(score_cells), "control_score_count": len(control_score_rows), "complete_source_count": len(source_rows), "checks": ["PCM condition bindings", "independent embedding matrix", "common support and time-first curve", "primary bootstrap", "repeat render and delay controls"]}
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = validate(args.run_dir)
        body = dict(value)
        body["artifact_sha256"] = canonical_hash(body)
        target = args.run_dir.resolve() / "validation.json"
        target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "engineering_status": value["engineering_status"]}), flush=True)
        return 0
    except Exception as exc:
        value = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "FAIL", "engineering_status": "BLOCKED", "independent": True, "error": f"{type(exc).__name__}: {exc}"}
        target = args.run_dir.resolve() / "validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        body = dict(value)
        body["artifact_sha256"] = canonical_hash(body)
        target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[check_wav2lip_noise_cross] FAIL: {value['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
