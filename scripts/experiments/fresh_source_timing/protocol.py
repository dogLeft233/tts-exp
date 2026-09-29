"""Frozen input and A-manifest bindings for the C branch.

Only immutable P/A artifacts are consumed here.  This module intentionally
does not import ``fresh_source_inputs.runner`` or any cohort-selection gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .audio import build_delay_artifacts
from .common import (
    DELAY_FRAMES,
    DELAY_SAMPLES,
    EXPECTED_GROUP_COUNT,
    PROTOCOL_ID,
    PROTOCOL_REVISION,
    TimingError,
    file_sha256,
    load_self_hashed,
    read_json,
    resolve_path,
    write_json,
)
from .scoring import load_embedding_array
from .visual import load_mouth_trajectory

REQUIRED_MODELS = ("Wav2Lip", "Ditto")


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_artifact(path: str | Path, *, self_hashed: bool = True) -> dict[str, Any]:
    return load_self_hashed(path) if self_hashed else read_json(path)


def cohort_is_ready(cohort: Mapping[str, Any]) -> bool:
    status = str(cohort.get("status", ""))
    formal = cohort.get("formal")
    if status not in {"COHORT_READY", "GO", "READY"} or not isinstance(formal, list):
        return False
    required = int(cohort.get("required_formal", EXPECTED_GROUP_COUNT) or EXPECTED_GROUP_COUNT)
    groups = [str(item.get("source_group", "")) for item in formal if isinstance(item, Mapping)]
    return required == EXPECTED_GROUP_COUNT and len(groups) == EXPECTED_GROUP_COUNT and all(groups) and len(set(groups)) == EXPECTED_GROUP_COUNT


def input_is_frozen(inputs: Mapping[str, Any]) -> bool:
    rows = inputs.get("records")
    if not isinstance(rows, list):
        rows = inputs.get("formal")
    return str(inputs.get("status", "")) in {"INPUTS_FROZEN", "FROZEN", "GO", "READY", "complete"} and isinstance(rows, list)


def input_records(inputs: Mapping[str, Any], *, formal_groups: Sequence[str] | None = None) -> list[dict[str, Any]]:
    raw = inputs.get("records")
    if not isinstance(raw, list):
        raw = inputs.get("formal")
    if not isinstance(raw, list):
        raise TimingError("frozen inputs records are missing")
    result = [dict(row) for row in raw if isinstance(row, Mapping)]
    if not result:
        raise TimingError("frozen inputs contain no records")
    if formal_groups is not None:
        wanted = {str(group) for group in formal_groups if str(group)}
        if len(wanted) != EXPECTED_GROUP_COUNT:
            raise TimingError(f"frozen cohort has {len(wanted)} formal groups, expected {EXPECTED_GROUP_COUNT}")
        selected = [row for row in result if str(row.get("source_group", "")) in wanted]
        if len(selected) != EXPECTED_GROUP_COUNT:
            raise TimingError("frozen inputs do not contain exactly the frozen formal groups")
        result = selected
    elif len(result) != EXPECTED_GROUP_COUNT:
        raise TimingError(f"frozen inputs contain {len(result)} records; refusing smoke/formal mixing")
    result = sorted(result, key=lambda row: (str(row.get("source_group", "")), str(row.get("sample_id", ""))))
    groups = [str(row.get("source_group", "")) for row in result]
    if len(groups) != EXPECTED_GROUP_COUNT or any(not group for group in groups) or len(set(groups)) != EXPECTED_GROUP_COUNT:
        raise TimingError("frozen input source groups are not unique")
    return result


def _audio_path(row: Mapping[str, Any], *, base: Path | None = None) -> Path:
    candidate = row.get("natural_audio")
    if isinstance(candidate, Mapping):
        candidate = candidate.get("path")
    candidate = candidate or row.get("natural_audio_path") or row.get("audio_path") or row.get("audio")
    if isinstance(candidate, Mapping):
        candidate = candidate.get("path")
    if candidate is None:
        raise TimingError(f"natural audio path is missing: {row.get('sample_id', row.get('source_group'))}")
    return resolve_path(candidate, base=base)


def _expected_audio_sha(row: Mapping[str, Any]) -> str | None:
    candidate = row.get("natural_audio")
    if isinstance(candidate, Mapping) and candidate.get("sha256"):
        return str(candidate["sha256"])
    for key in ("natural_audio_sha256", "audio_sha256", "natural_pcm_sha256"):
        value = row.get(key)
        if value and key != "natural_pcm_sha256":
            return str(value)
    return None


def _locate_signed_artifact(root: Path, name: str, supplied: Mapping[str, Any] | None, *, label: str) -> tuple[Path, dict[str, Any]]:
    """Locate the immutable P artifact and verify a supplied payload agrees."""

    candidates = (root / name, root / "run" / "shared" / name)
    for path in candidates:
        if not path.is_file():
            continue
        value = load_self_hashed(path)
        if supplied is not None and supplied.get("artifact_sha256") != value.get("artifact_sha256"):
            raise TimingError(f"{label} payload differs from frozen artifact: {path}")
        return path.resolve(), value
    raise TimingError(f"{label} artifact is missing under {root}")


def build_delay_inputs(
    run_root: str | Path,
    *,
    inputs: Mapping[str, Any] | None = None,
    cohort: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct all 12 fixed DELAY audio assets and bind them to P inputs."""

    root = Path(run_root).resolve()
    cohort_path, cohort_value = _locate_signed_artifact(root, "cohort.json", cohort, label="cohort")
    if not cohort_is_ready(cohort_value):
        raise TimingError(f"cohort is not COHORT_READY: {cohort_value.get('status')}")
    if inputs is None:
        input_path_used, inputs_value = _locate_signed_artifact(root, "inputs.json", None, label="inputs")
    else:
        input_path_used, inputs_value = _locate_signed_artifact(root, "inputs.json", inputs, label="inputs")
    if not input_is_frozen(inputs_value):
        raise TimingError(f"inputs are not frozen: {inputs_value.get('status')}")
    formal_groups = [str(item.get("source_group")) for item in cohort_value.get("formal", []) if isinstance(item, Mapping)]
    rows = input_records(inputs_value, formal_groups=formal_groups or None)
    output_root = root / "run" / "C"
    groups: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", row.get("id", "")))
        source_group = str(row.get("source_group", ""))
        if not sample_id or not source_group:
            raise TimingError("frozen input is missing sample_id/source_group")
        natural_path = _audio_path(row, base=root)
        if not natural_path.is_file():
            raise TimingError(f"natural audio is missing: {natural_path}")
        expected_hash = _expected_audio_sha(row)
        if expected_hash is not None and expected_hash != file_sha256(natural_path):
            raise TimingError(f"natural audio hash changed: {sample_id}")
        delay_path = output_root / "audio" / f"{sample_id}__DELAY.wav"
        map_path = output_root / "audio" / f"{sample_id}__DELAY.source_index.npy"
        artifact = build_delay_artifacts(natural_path, delay_path, map_path)
        artifact.update({"sample_id": sample_id, "source_group": source_group, "status": "complete"})
        groups.append(artifact)
    groups.sort(key=lambda item: (str(item["source_group"]), str(item["sample_id"])))
    group_ids = [str(item["source_group"]) for item in groups]
    if len(groups) != EXPECTED_GROUP_COUNT or len(set(group_ids)) != EXPECTED_GROUP_COUNT:
        raise TimingError(f"C requires exactly {EXPECTED_GROUP_COUNT} unique formal groups, got {len(groups)}")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "status": "READY",
        "groups": groups,
        "group_count": len(groups),
        "delay_samples": DELAY_SAMPLES,
        "delay_frames": DELAY_FRAMES,
        "sample_rate": 16_000,
        "fps": 25,
        "formula": "DELAY[n]=N[n-3200], out-of-range zero, original length L",
        "source_index_sentinel": -1,
        "input_binding": {
            "path": str(input_path_used.resolve()),
            "sha256": file_sha256(input_path_used) if input_path_used.is_file() else None,
            "artifact_sha256": inputs_value.get("artifact_sha256"),
        },
        "cohort_binding": {
            "path": str(cohort_path),
            "sha256": file_sha256(cohort_path),
            "artifact_sha256": cohort_value.get("artifact_sha256"),
        },
        "fresh_forward_required": True,
        "scientific_cells": 0,
        "thread_limit": 2,
    }
    return write_json(output_root / "delay_inputs.json", payload)


def load_delay_inputs(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    return load_self_hashed(root / "run" / "C" / "delay_inputs.json")


def discover_a_manifest(run_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(run_root).resolve() / "run" / "A"
    candidates = (root / "videos.json", root / "videos_manifest.json", root / "manifest.json")
    for path in candidates:
        if path.is_file():
            return path, load_self_hashed(path)
    raise TimingError("A videos manifest is missing")


def _manifest_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("rows", "videos", "cells"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, Mapping):
            result: list[Mapping[str, Any]] = []
            for cell_key, row in rows.items():
                if isinstance(row, Mapping):
                    item = dict(row)
                    item.setdefault("key", str(cell_key))
                    result.append(item)
            return result
    raise TimingError("A videos manifest rows are missing")


def canonical_model(value: Any) -> str:
    name = str(value or "").strip()
    lowered = name.lower().replace("_", "").replace("-", "")
    if lowered in {"wav2lip", "wav2lipgan", "w2l"}:
        return "Wav2Lip"
    if lowered in {"ditto", "dittotrta", "dittopytorch"}:
        return "Ditto"
    return name or "unknown"


def canonical_arm(value: Any) -> str:
    name = str(value or "").upper().replace("-", "_")
    if name in {"N", "N42", "NATURAL", "NATURAL42"}:
        return "N"
    if name in {"DELAY", "DELAY42", "DELAY_200", "A_DELAY"}:
        return "DELAY"
    return name


def _load_json_binding(value: Any, *, base: Path) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = resolve_path(value, base=base)
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                return load_self_hashed(path)
            except TimingError:
                return read_json(path)
    return None


def _find_embedding_binding(row: Mapping[str, Any], names: Sequence[str], *, base: Path) -> Any:
    direct = _first(row, names)
    if direct is not None:
        return direct
    for container_name in ("embedding", "embeddings", "score", "worker", "syncnet"):
        container = _load_json_binding(row.get(container_name), base=base)
        if isinstance(container, Mapping):
            value = _first(container, names)
            if value is not None:
                return value
    return None


def _find_feature_binding(row: Mapping[str, Any], names: Sequence[str], *, base: Path) -> Any:
    direct = _first(row, names)
    if direct is not None:
        return direct
    for container_name in ("features", "mouth_features", "visual_features", "trajectory"):
        container = row.get(container_name)
        if isinstance(container, Mapping):
            value = _first(container, names)
            if value is not None:
                return value
        elif container is not None and container_name in names:
            return container
    return None


def _row_is_complete(row: Mapping[str, Any]) -> bool:
    return str(row.get("status", "complete")).lower() in {"complete", "completed", "go", "ok", "ready"}


def load_a_cells(run_root: str | Path, delay_inputs: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Bind exactly one fresh N42 and DELAY42 row per model/group.

    The returned arrays are loaded only after their artifact hashes are checked;
    DELAY rows must explicitly carry ``new_forward``/``fresh_forward`` so a
    scorer-only reindex cannot masquerade as generator response.
    """

    manifest_path, payload = discover_a_manifest(run_root)
    base = manifest_path.parent
    rows = _manifest_rows(payload)
    # A keeps media/video provenance in videos.json and SyncNet arrays in
    # scores.json.  Join them by the immutable cell key before validating the
    # C contract; C must never answer from an audio re-index alone.
    score_path = manifest_path.parent / "scores.json"
    score_cells: dict[str, Mapping[str, Any]] = {}
    if score_path.is_file():
        score_payload = load_self_hashed(score_path)
        raw_scores = score_payload.get("cells")
        if isinstance(raw_scores, Mapping):
            score_cells = {str(key): value for key, value in raw_scores.items() if isinstance(value, Mapping)}
    joined_rows: list[Mapping[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        score = score_cells.get(str(row.get("key", "")))
        if isinstance(score, Mapping) and str(score.get("status")) == "complete":
            embedding_path = score.get("embedding_path")
            embedding_sha = score.get("embedding_sha256")
            if embedding_path and embedding_sha:
                embedding = {"path": str(embedding_path), "sha256": str(embedding_sha)}
                row["visual"] = {**embedding, "key": "visual"}
                row["audio_embedding"] = {**embedding, "key": "audio"}
                row["score_key"] = str(score.get("key", row.get("key", "")))
                row["score_sha256"] = file_sha256(score_path)
        joined_rows.append(row)
    rows = joined_rows
    expected = {
        (str(item["source_group"]), str(item["sample_id"]))
        for item in delay_inputs.get("groups", [])
        if isinstance(item, Mapping)
    }
    if len(expected) != int(delay_inputs.get("group_count", 0)):
        raise TimingError("delay input groups are malformed")
    by_model: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for row in rows:
        if not _row_is_complete(row):
            continue
        model = canonical_model(_first(row, ("model", "generator", "model_name")))
        arm = canonical_arm(_first(row, ("arm", "video_arm", "condition", "audio_arm")))
        seed = int(row.get("seed", 42) or 42)
        if seed != 42 or arm not in {"N", "DELAY"}:
            continue
        group = str(row.get("source_group", ""))
        sample_id = str(row.get("sample_id", row.get("id", "")))
        key = (group, sample_id)
        if key not in expected:
            continue
        indexed = by_model.setdefault(model, {})
        full_key = (group, arm)
        if full_key in indexed:
            raise TimingError(f"duplicate A video cell: {model}/{group}/{arm}")
        indexed[full_key] = row
    if not by_model:
        raise TimingError("A has no completed seed42 N/DELAY rows")
    missing_models = sorted(set(REQUIRED_MODELS) - set(by_model))
    if missing_models:
        raise TimingError(f"A manifest is missing required generator models: {missing_models}")
    result: dict[str, list[dict[str, Any]]] = {}
    delay_by_group = {(str(item["source_group"]), str(item["sample_id"])): item for item in delay_inputs.get("groups", []) if isinstance(item, Mapping)}
    for model, indexed in by_model.items():
        records: list[dict[str, Any]] = []
        for group, sample_id in sorted(expected):
            n_row = indexed.get((group, "N"))
            d_row = indexed.get((group, "DELAY"))
            if n_row is None or d_row is None:
                records.append({"sample_id": sample_id, "source_group": group, "fresh_forward": False, "provenance_valid": False, "status": "missing"})
                continue
            fresh = bool(d_row.get("fresh_forward", d_row.get("new_forward", d_row.get("forward", False))))
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "source_group": group,
                "fresh_forward": fresh,
                "new_forward": fresh,
                "model": model,
                "natural_control_valid": bool(n_row.get("natural_control_valid", n_row.get("control_valid", True))),
                "provenance_valid": False,
            }
            try:
                delay_input = delay_by_group[(group, sample_id)]
                for source_row, expected_audio, label in (
                    (n_row, delay_input["natural"], "N42 audio"),
                    (d_row, delay_input["delay"], "DELAY audio"),
                ):
                    audio_binding = _first(source_row, ("audio", "source_audio", "audio_path", "source_audio_path"))
                    if isinstance(audio_binding, Mapping):
                        audio_path_value = audio_binding.get("path")
                        audio_sha = audio_binding.get("sha256") or audio_binding.get("container_sha256")
                    else:
                        audio_path_value = audio_binding
                        audio_sha = source_row.get("audio_sha256") or source_row.get("audio_pcm_sha256")
                    if audio_path_value is None:
                        raise TimingError(f"{label} path binding is missing")
                    audio_path = resolve_path(audio_path_value, base=base)
                    expected_path = resolve_path(expected_audio["path"])
                    if audio_path != expected_path:
                        raise TimingError(f"{label} is not bound to frozen C audio")
                    if audio_sha is not None and str(audio_sha) != str(expected_audio.get("sha256")):
                        raise TimingError(f"{label} container hash binding differs")
                n_visual_binding = _find_embedding_binding(n_row, ("visual", "visual_embedding", "visual_path"), base=base)
                n_audio_binding = _find_embedding_binding(n_row, ("audio_embedding", "audio", "audio_embedding_path"), base=base)
                d_visual_binding = _find_embedding_binding(d_row, ("visual", "visual_embedding", "visual_path"), base=base)
                if n_visual_binding is None or n_audio_binding is None or d_visual_binding is None:
                    raise TimingError("A row lacks visual/N audio embedding binding")
                record["natural_visual"] = load_embedding_array(n_visual_binding, name="N42 visual", base=base)
                record["natural_audio"] = load_embedding_array(n_audio_binding, name="N42 audio", base=base)
                record["delay_visual"] = load_embedding_array(d_visual_binding, name="DELAY visual", base=base)
                record["a_manifest"] = {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path), "n_row": dict(n_row), "delay_row": dict(d_row)}
                # A's DELAY audio binding is provenance only; C does not score
                # against it.  Bind it so the validator can reject a missing or
                # mislabelled fresh forward.
                delay_audio_binding = _find_embedding_binding(d_row, ("audio_embedding", "audio", "audio_embedding_path"), base=base)
                if delay_audio_binding is None:
                    raise TimingError("DELAY fresh row lacks its own forward audio embedding")
                record["delay_audio"] = load_embedding_array(delay_audio_binding, name="DELAY audio", base=base)
                feature_n = _find_feature_binding(n_row, ("natural", "N", "canonical_mouth", "trajectory", "path"), base=base)
                feature_d = _find_feature_binding(d_row, ("delay", "DELAY", "canonical_mouth", "trajectory", "path"), base=base)
                if isinstance(feature_n, Mapping) and (feature_n.get("path") or feature_n.get("file")):
                    record["natural_trajectory"] = load_mouth_trajectory(resolve_path(feature_n.get("path") or feature_n.get("file"), base=base))["trajectory"]
                elif isinstance(feature_n, (str, Path)):
                    record["natural_trajectory"] = load_mouth_trajectory(resolve_path(feature_n, base=base))["trajectory"]
                elif feature_n is not None:
                    record["natural_trajectory"] = feature_n
                if isinstance(feature_d, Mapping) and (feature_d.get("path") or feature_d.get("file")):
                    record["delay_trajectory"] = load_mouth_trajectory(resolve_path(feature_d.get("path") or feature_d.get("file"), base=base))["trajectory"]
                elif isinstance(feature_d, (str, Path)):
                    record["delay_trajectory"] = load_mouth_trajectory(resolve_path(feature_d, base=base))["trajectory"]
                elif feature_d is not None:
                    record["delay_trajectory"] = feature_d
                record["status"] = "complete"
                record["provenance_valid"] = bool(fresh)
            except Exception as exc:  # noqa: BLE001 - preserve row for missingness report
                record["status"] = "unresolved"
                record["binding_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
        result[model] = records
    return result
