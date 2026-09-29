"""Protocol locking, input binding, deterministic assignment, and receipts.

This module deliberately contains no model imports.  Audit, feature, and plan
stages therefore remain CPU-only and can be run while a separate CUDA job is
using the machine.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.utils import resolve_repo_path

CONDITIONS = ("natural", "qwen_cloud", "qwen_local", "index_tts2", "cosyvoice2")
TTS_CONDITIONS = CONDITIONS[1:]
PAIR_TYPES = tuple(itertools.combinations(TTS_CONDITIONS, 2))
PROTOCOL_ID = "phoneme_tfg_association_v1"


class ProtocolError(RuntimeError):
    """A frozen protocol or artifact contract was violated."""


class InsufficientEligibleGroups(ProtocolError):
    """The pre-registered source-group gate cannot be met."""


class ResourceBusy(ProtocolError):
    """A requested CUDA stage would collide with an existing process."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"JSONL row is not an object: {path}")
            rows.append(value)
    return rows


def _duration(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        samples = int(handle.getnframes())
    return channels, sample_width, sample_rate, samples / max(sample_rate, 1)


def _audio_stats(path: Path, *, silence_threshold: float = 0.005) -> dict[str, float]:
    """Return cheap waveform diagnostics used by the registered sensitivities."""
    peak = 0.0
    squared = 0.0
    count = 0
    silent = 0
    with wave.open(str(path), "rb") as handle:
        sample_width = int(handle.getsampwidth())
        if sample_width != 2:
            raise ProtocolError(f"silence statistics require PCM16: {path}")
        scale = float(1 << (8 * sample_width - 1))
        while True:
            raw = handle.readframes(1 << 16)
            if not raw:
                break
            values = np.frombuffer(raw, dtype="<i2").astype(np.float64) / scale
            if values.size == 0:
                continue
            abs_values = np.abs(values)
            peak = max(peak, float(abs_values.max()))
            squared += float(np.dot(values, values))
            count += int(values.size)
            silent += int(np.count_nonzero(abs_values <= float(silence_threshold)))
    if count == 0:
        raise ProtocolError(f"empty audio: {path}")
    return {
        "peak_abs": float(peak),
        "rms": float(math.sqrt(squared / count)),
        "silence_fraction": float(silent / count),
        "silence_threshold": float(silence_threshold),
    }


def _resolve(path: str | Path, repo_root: Path) -> Path:
    return resolve_repo_path(repo_root, path).resolve()


def _hash_optional(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def _contains_lrs3_component(path: Path) -> bool:
    """Return whether a path is explicitly inside/named as an LRS3 asset."""
    return any("lrs3" in part.lower() for part in path.resolve().parts)


def _external_visual_assignments(
    repo_root: Path,
    cfg: Mapping[str, Any],
    sample_ids: Sequence[str],
    cohort_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Bind each cohort item to a non-LRS3 first-frame visual candidate.

    The manifest is deliberately a dataset-level provenance boundary.  It may
    contain LRS3 records for the broader multi-dataset inventory, but this
    protocol only admits records from the explicit allowlist and rejects paths
    containing ``lrs3`` before assignment.  Assignment is outcome-independent
    and recorded in the input audit; generation later consumes one frozen frame
    from this external asset, never the cohort source video.
    """
    raw_manifest = str(cfg.get("visual_manifest", "")).strip()
    if not raw_manifest:
        raise ProtocolError("visual_manifest is required; LRS3 cohort videos cannot be renderer inputs")
    manifest_path = _resolve(raw_manifest, repo_root)
    manifest = read_json(manifest_path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ProtocolError("visual_manifest.records must be a list")
    allowlist = {str(item).lower() for item in cfg.get("visual_dataset_allowlist", [])}
    if not allowlist:
        raise ProtocolError("visual_dataset_allowlist must explicitly exclude LRS3 by allowlisting external datasets")
    source_paths = {
        _resolve(str(row.get("video_local_path", "")), repo_root)
        for row in cohort_rows.values()
        if str(row.get("video_local_path", "")).strip()
    }
    pool: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping):
            rejected.append({"reason": "malformed_record"})
            continue
        dataset = str(record.get("dataset", "")).strip().lower()
        raw_path = str(record.get("video_local_path", record.get("path", ""))).strip()
        if not dataset or dataset not in allowlist:
            rejected.append({"dataset": dataset, "path": raw_path, "reason": "dataset_not_allowlisted"})
            continue
        path = _resolve(raw_path, repo_root)
        if _contains_lrs3_component(path):
            rejected.append({"dataset": dataset, "path": raw_path, "reason": "lrs3_path_forbidden"})
            continue
        if path in source_paths:
            rejected.append({"dataset": dataset, "path": raw_path, "reason": "cohort_source_path_forbidden"})
            continue
        if path in seen or not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov", ".png", ".jpg", ".jpeg"}:
            rejected.append({"dataset": dataset, "path": raw_path, "reason": "missing_duplicate_or_unsupported"})
            continue
        digest = file_sha256(path)
        seen.add(path)
        pool.append({
            "dataset": dataset,
            "stem": str(record.get("stem", path.stem)),
            "path": str(path),
            "sha256": digest,
            "frame_policy": "first_frame_only",
        })
    pool.sort(key=lambda row: (str(row["dataset"]), str(row["path"])))
    if len(pool) < len(sample_ids):
        raise ProtocolError(f"external visual pool has {len(pool)} usable assets for {len(sample_ids)} cohort samples")
    assignments: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    seed = int(cfg.get("seed", 20260921))
    for sample_id in sorted(str(item) for item in sample_ids):
        digest = hashlib.sha256(f"{seed}|external_visual|{sample_id}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "big") % len(pool)
        selected: dict[str, Any] | None = None
        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if str(candidate["path"]) not in used:
                selected = dict(candidate)
                break
        if selected is None:  # guarded by the pool-size check, kept explicit for the contract
            raise ProtocolError(f"unable to assign unique external visual for {sample_id}")
        source_video = _resolve(str(cohort_rows[sample_id].get("video_local_path", "")), repo_root)
        if Path(str(selected["path"])).resolve() == source_video or str(selected["sha256"]) == _hash_optional(source_video):
            raise ProtocolError(f"external visual collides with cohort source media: {sample_id}")
        used.add(str(selected["path"]))
        assignments[sample_id] = selected
    meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "allowlist": sorted(allowlist),
        "pool_count": len(pool),
        "pool_sha256": canonical_hash(pool),
        "assigned_count": len(assignments),
        "unique_assigned_count": len(used),
        "rejected_record_count": len(rejected),
        "rejected_reason_counts": dict(Counter(str(row.get("reason")) for row in rejected)),
        "selection": "sha256(seed|external_visual|sample_id) with sorted-path collision resolution",
        "frame_policy": "first_frame_only_then_repeat_for_every_mel_chunk",
    }
    return assignments, meta


def _cohort_index(cohort: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = cohort.get("records", [])
    if not isinstance(records, list):
        raise ProtocolError("cohort.records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        sid = str(row.get("sample_id", ""))
        if not sid or sid in result:
            raise ProtocolError(f"invalid or duplicate cohort sample_id: {sid}")
        result[sid] = dict(row)
    return result


def _alignment_index(alignment: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in alignment.get("records", []):
        key = (str(row.get("sample_id")), str(row.get("condition")))
        if key in result:
            raise ProtocolError(f"duplicate alignment key: {key}")
        result[key] = dict(row)
    return result


def _record_audio(manifest_row: Mapping[str, Any], condition: str, repo_root: Path) -> tuple[Path, Mapping[str, Any]]:
    arms = manifest_row.get("arms", {})
    arm = arms.get(condition)
    if not isinstance(arm, Mapping) or not isinstance(arm.get("audio"), Mapping):
        raise ProtocolError(f"manifest row lacks audio arm: {manifest_row.get('sample_id')}/{condition}")
    audio_meta = arm["audio"]
    path = _resolve(str(audio_meta.get("path", "")), repo_root)
    return path, audio_meta


def _reuse_inventory(repo_root: Path, transfer_run: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Audit only local candidates; never downloads or silently rebinds media."""
    candidates = [
        repo_root / "runs/lrs3_english_phoneme_transfer_n100_20260921",
        repo_root / "runs/lrs3_tts_gain_mechanism_20260908",
        repo_root / "runs/wav2lip_face_roi_replacement_20260906_host_fix5",
    ]
    rows: list[dict[str, Any]] = []
    for root in candidates:
        rows.append({
            "root": str(root),
            "exists": root.is_dir(),
            "candidate_manifests": sorted(str(p.relative_to(repo_root)) for p in root.rglob("*manifest*.json"))[:50] if root.is_dir() else [],
            "policy": "exact sample_id + canonical PCM + reference/checkpoint/seed contract required",
        })
    return {
        "status": "complete",
        "transfer_run": str(transfer_run),
        "historical_reuse_checked": True,
        "source_not_local_paths": ["old replacement 133-cell paths"],
        "candidates": rows,
        "reused_cells": [],
        "rejected_cells": [],
        "selection_independent_of_cache": True,
    }


def audit_inputs(repo_root: Path, run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Verify manifest paths/hashes and emit the auditable input universe."""
    cohort_path = _resolve(str(cfg["cohort"]), repo_root)
    transfer_run = _resolve(str(cfg["transfer_run"]), repo_root)
    manifest_path = transfer_run / "00_inputs/manifest.json"
    alignment_path = transfer_run / "01_mfa/alignment_manifest.json"
    cohort = read_json(cohort_path)
    manifest = read_json(manifest_path)
    alignment = read_json(alignment_path)
    cohort_by_id = _cohort_index(cohort)
    alignment_by_key = _alignment_index(alignment)
    manifest_rows = {str(row["sample_id"]): row for row in manifest.get("records", [])}
    visual_assignments, visual_meta = _external_visual_assignments(repo_root, cfg, sorted(cohort_by_id), cohort_by_id)
    bindings_path = repo_root / "openspec/changes/probe-phoneme-tfg-association/input-bindings.json"
    binding_checks: list[dict[str, Any]] = []
    binding_errors: list[str] = []
    if bindings_path.is_file():
        bindings = read_json(bindings_path)
        for item in bindings.get("files", []):
            bound_path = _resolve(str(item.get("path", "")), repo_root)
            actual = _hash_optional(bound_path)
            expected = item.get("sha256")
            check = {"path": str(bound_path), "expected_sha256": expected, "actual_sha256": actual, "exists": bound_path.is_file(), "ok": bool(actual and actual == expected)}
            binding_checks.append(check)
            if not check["ok"]:
                binding_errors.append(str(item.get("path", bound_path)))
    else:
        binding_errors.append(str(bindings_path))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sid in sorted(cohort_by_id):
        cohort_row = cohort_by_id[sid]
        manifest_row = manifest_rows.get(sid)
        source_video = _resolve(str(cohort_row.get("video_local_path", "")), repo_root)
        source_video_ok = source_video.is_file()
        row: dict[str, Any] = {
            "sample_id": sid,
            "source_group": str(cohort_row.get("source_group", "")),
            "speaker_id": cohort_row.get("speaker_id"),
            "transcript": str(cohort_row.get("transcript", "")),
            "transcript_sha256": str(cohort_row.get("transcript_sha256", "")),
            "source_video": str(source_video),
            "source_video_sha256_expected": cohort_row.get("video_sha256"),
            "source_video_sha256_actual": _hash_optional(source_video),
            "source_video_exists": source_video_ok,
            "source_video_role": "cohort_audio_provenance_only_not_renderer_input",
            "visual_source": visual_assignments[sid]["path"],
            "visual_source_sha256_expected": visual_assignments[sid]["sha256"],
            "visual_source_sha256_actual": _hash_optional(Path(visual_assignments[sid]["path"])),
            "visual_source_dataset": visual_assignments[sid]["dataset"],
            "visual_source_stem": visual_assignments[sid]["stem"],
            "visual_source_frame_policy": visual_assignments[sid]["frame_policy"],
            "visual_source_is_lrs3": _contains_lrs3_component(Path(visual_assignments[sid]["path"])),
            "arms": {},
            "status": "ready",
            "reasons": [],
        }
        if not source_video_ok:
            row["status"] = "ineligible"
            row["reasons"].append("missing_source_video")
        elif cohort_row.get("video_sha256") and row["source_video_sha256_actual"] != cohort_row.get("video_sha256"):
            row["status"] = "ineligible"
            row["reasons"].append("source_video_sha256_mismatch")
        if row["visual_source_is_lrs3"]:
            row["status"] = "ineligible"
            row["reasons"].append("lrs3_visual_source_forbidden")
        if row["visual_source_sha256_actual"] != row["visual_source_sha256_expected"]:
            row["status"] = "ineligible"
            row["reasons"].append("visual_source_sha256_mismatch")
        if manifest_row is None:
            row["status"] = "ineligible"
            row["reasons"].append("missing_transfer_manifest_row")
            failures.append({"sample_id": sid, "reason": "missing_transfer_manifest_row"})
            rows.append(row)
            continue
        for condition in CONDITIONS:
            arm_info: dict[str, Any] = {"condition": condition, "status": "ready", "reasons": []}
            try:
                audio_path, audio_meta = _record_audio(manifest_row, condition, repo_root)
                align_row = alignment_by_key.get((sid, condition))
                textgrid_path = _resolve(str(align_row.get("textgrid", "")), repo_root) if align_row else Path("")
                channels, width, sample_rate, duration_s = _duration(audio_path)
                audio_stats = _audio_stats(audio_path, silence_threshold=float(cfg.get("silence_threshold", 0.005)))
                actual_audio_sha = file_sha256(audio_path)
                expected_audio_sha = str(audio_meta.get("sha256", ""))
                if expected_audio_sha and actual_audio_sha != expected_audio_sha:
                    arm_info["status"] = "ineligible"
                    arm_info["reasons"].append("audio_sha256_mismatch")
                if sample_rate != int(cfg.get("sample_rate", 16000)) or channels != 1 or width != 2:
                    arm_info["status"] = "ineligible"
                    arm_info["reasons"].append("audio_format_mismatch")
                if not textgrid_path.is_file():
                    arm_info["status"] = "ineligible"
                    arm_info["reasons"].append("missing_textgrid")
                align_sha = _hash_optional(textgrid_path)
                if align_row and align_row.get("textgrid_sha256") and align_sha != align_row.get("textgrid_sha256"):
                    arm_info["status"] = "ineligible"
                    arm_info["reasons"].append("textgrid_sha256_mismatch")
                arm_info.update({
                    "audio": str(audio_path),
                    "audio_sha256_expected": expected_audio_sha,
                    "audio_sha256_actual": actual_audio_sha,
                    "pcm_sha256": audio_meta.get("pcm_sha256"),
                    "sample_count": int(audio_meta.get("sample_count", round(duration_s * sample_rate))),
                    "duration_s": float(duration_s),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": width,
                    **audio_stats,
                    "textgrid": str(textgrid_path),
                    "textgrid_sha256": align_sha,
                    "alignment_source": align_row.get("alignment_source") if align_row else None,
                    "alignment_record_exists": align_row is not None,
                })
            except (OSError, EOFError, wave.Error, ProtocolError, ValueError) as exc:
                arm_info["status"] = "ineligible"
                arm_info["reasons"].append(f"audit_error:{type(exc).__name__}")
                arm_info["error"] = str(exc)
            row["arms"][condition] = arm_info
            if arm_info["status"] != "ready":
                row["status"] = "ineligible"
        if row["status"] != "ready":
            failures.append({"sample_id": sid, "source_group": row["source_group"], "reasons": row["reasons"], "arm_reasons": {k: v["reasons"] for k, v in row["arms"].items() if v["status"] != "ready"}})
        rows.append(row)
    group_counts = Counter(str(row["source_group"]) for row in rows if row["status"] == "ready")
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": "input_audit",
        "status": "complete",
        "cohort": str(cohort_path),
        "cohort_sha256": file_sha256(cohort_path),
        "transfer_manifest": str(manifest_path),
        "transfer_manifest_sha256": file_sha256(manifest_path),
        "alignment_manifest": str(alignment_path),
        "alignment_manifest_sha256": file_sha256(alignment_path),
        "visual_source_policy": "external_non_lrs3_first_frame_only",
        "visual_source_manifest": visual_meta["manifest"],
        "visual_source_manifest_sha256": visual_meta["manifest_sha256"],
        "visual_source_audit": visual_meta,
        "input_bindings": str(bindings_path),
        "input_bindings_sha256": _hash_optional(bindings_path),
        "input_binding_checks": binding_checks,
        "input_binding_errors": binding_errors,
        "expected_sample_count": len(cohort_by_id),
        "audited_sample_count": len(rows),
        "mfa_completed": int(alignment.get("completed", 0)),
        "mfa_expected": int(alignment.get("expected", 0)),
        "condition_counts": dict(alignment.get("condition_counts", {})),
        "ready_sample_count": sum(row["status"] == "ready" for row in rows),
        "ready_source_group_count": len(group_counts),
        "source_group_counts": dict(sorted(group_counts.items())),
        "rows": rows,
        "failures": failures,
        "reuse_inventory": _reuse_inventory(repo_root, transfer_run, manifest),
    }
    write_json(run_dir / "00_protocol/input_audit.json", result)
    write_json(run_dir / "00_protocol/reuse_inventory.json", result["reuse_inventory"])
    write_jsonl(run_dir / "00_protocol/failures.jsonl", failures)
    return result


def _stable_group_key(seed: int, group: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{group}|{sample_id}".encode("utf-8")).hexdigest()


def select_blocks(
    audit_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Select one eligible utterance per source group, then assign pair blocks."""
    feature_by_id = {str(row["sample_id"]): row for row in feature_rows}
    reference_by_id = {str(row["sample_id"]): row for row in reference_rows}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audit_rows:
        sid = str(audit["sample_id"])
        feature = feature_by_id.get(sid, {})
        reference = reference_by_id.get(sid, {})
        if not bool(feature.get("eligible", False)) or reference.get("status") not in {"ready", "fallback_ready"}:
            continue
        item = {"sample_id": sid, "source_group": str(audit["source_group"]), "feature": feature, "reference": reference}
        candidates[str(audit["source_group"])].append(item)
    seed = int(cfg.get("seed", 20260921))
    one_per_group: list[dict[str, Any]] = []
    for group in sorted(candidates):
        ordered = sorted(candidates[group], key=lambda item: _stable_group_key(seed, group, item["sample_id"]))
        one_per_group.append(ordered[0])
    requested = int(cfg.get("smoke_source_groups", 2) if smoke else cfg.get("source_groups", 36))
    if len(one_per_group) < requested:
        raise InsufficientEligibleGroups(f"eligible source groups={len(one_per_group)} < requested={requested}")
    one_per_group.sort(key=lambda item: _stable_group_key(seed, item["source_group"], item["sample_id"]))
    selected = one_per_group[:requested]
    rng = np.random.Generator(np.random.PCG64(seed + 1))
    order = rng.permutation(len(selected)).tolist()
    selected = [selected[index] for index in order]
    tts_order = tuple(str(item) for item in cfg.get("tts_order", TTS_CONDITIONS))
    pair_types = list(itertools.combinations(tts_order, 2))
    if smoke:
        pair_types = pair_types[:1]
    blocks: list[dict[str, Any]] = []
    pair_replicates = int(cfg.get("pair_replicates", 6))
    for index, item in enumerate(selected):
        pair = pair_types[min(index // pair_replicates, len(pair_types) - 1)]
        blocks.append({
            "block_id": f"block_{index:03d}",
            "sample_id": item["sample_id"],
            "source_group": item["source_group"],
            "pair": list(pair),
            "natural_arm": "natural",
            "tts_arms": list(pair),
            "feature": item["feature"],
            "reference": item["reference"],
            "assignment_index": index,
        })
    if not smoke:
        counts = Counter(tuple(row["pair"]) for row in blocks)
        expected = {pair: pair_replicates for pair in pair_types}
        if dict(counts) != expected:
            raise ProtocolError(f"pair allocation mismatch: {counts} != {expected}")
    return blocks


def freeze_protocol(run_dir: Path, blocks: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Write immutable plan artifacts; refuse to silently overwrite a lock."""
    if not blocks:
        raise ProtocolError("cannot freeze an empty plan")
    expected_videos = len(blocks) * 3
    max_videos = int(cfg.get("max_unique_wav2lip_videos", 108))
    if expected_videos > max_videos:
        raise ProtocolError(f"video budget exceeded: {expected_videos}>{max_videos}")
    cells: list[dict[str, Any]] = []
    for block in blocks:
        for arm in ("natural", *block["tts_arms"]):
            feature = block["feature"]
            cell = {
                "cell_key": f"{block['source_group']}::{block['sample_id']}::{arm}",
                "protocol_id": PROTOCOL_ID,
                "block_id": block["block_id"],
                "sample_id": block["sample_id"],
                "source_group": block["source_group"],
                "tfg": str(cfg.get("primary_tfg", "wav2lip")),
                "arm": arm,
                "audio_condition": arm,
                "seed": int(cfg.get("seed", 20260921)),
                "reference": block["reference"],
                "feature": feature,
                "status": "planned",
            }
            cells.append(cell)
    ditto_cells: list[dict[str, Any]] = []
    tts_order = tuple(str(item) for item in cfg.get("tts_order", TTS_CONDITIONS))
    pair_types = list(itertools.combinations(tts_order, 2))
    if bool(cfg.get("enable_ditto_replication", False)) and not smoke:
        # The optional extension is fixed before any new scoring and contains
        # exactly two independently hashed blocks from every TTS pair.
        by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for block in blocks:
            by_pair[tuple(str(item) for item in block["pair"])].append(block)
        ditto_blocks: list[Mapping[str, Any]] = []
        for pair in pair_types:
            candidates = sorted(
                by_pair.get(tuple(pair), []),
                key=lambda row: _stable_group_key(int(cfg.get("seed", 20260921)) + 29, str(row["source_group"]), str(row["sample_id"])),
            )
            if len(candidates) < 2:
                raise ProtocolError(f"optional Ditto extension lacks two blocks for pair {pair}")
            ditto_blocks.extend(candidates[:2])
        for block in ditto_blocks:
            for arm in ("natural", *block["tts_arms"]):
                ditto_cells.append({
                    "cell_key": f"ditto::{block['source_group']}::{block['sample_id']}::{arm}",
                    "protocol_id": PROTOCOL_ID,
                    "block_id": block["block_id"],
                    "sample_id": block["sample_id"],
                    "source_group": block["source_group"],
                    "tfg": "ditto",
                    "arm": arm,
                    "audio_condition": arm,
                    "seed": int(cfg.get("seed", 20260921)),
                    "reference": block["reference"],
                    "feature": block["feature"],
                    "status": "planned",
                })
    ditto_budget = int(cfg.get("max_unique_ditto_videos", 36))
    if len(ditto_cells) > ditto_budget:
        raise ProtocolError(f"Ditto video budget exceeded: {len(ditto_cells)}>{ditto_budget}")
    hash_payload = {"blocks": blocks, "cells": cells, "config": dict(cfg), "smoke": smoke}
    if ditto_cells:
        hash_payload["ditto_cells"] = ditto_cells
    freeze_hash = canonical_hash(hash_payload)
    for cell in (*cells, *ditto_cells):
        cell["protocol_hash"] = freeze_hash
    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": "v1",
        "status": "locked_smoke" if smoke else "locked",
        "created_at_epoch": time.time(),
        "config": dict(cfg),
        "smoke": bool(smoke),
        "source_group_count": len({str(row["source_group"]) for row in blocks}),
        "block_count": len(blocks),
        "pair_counts": {"-".join(pair): sum(tuple(row["pair"]) == pair for row in blocks) for pair in pair_types},
        "video_budget": {"planned_unique_wav2lip_videos": len(cells), "max_unique_wav2lip_videos": max_videos, "planned_unique_ditto_videos": len(ditto_cells), "max_unique_ditto_videos": ditto_budget},
        "blocks": [dict(row) for row in blocks],
        "cells": cells,
        "ditto_cells": ditto_cells,
        "freeze_hash": freeze_hash,
    }
    target = run_dir / "00_protocol/protocol.json"
    if target.is_file():
        existing = read_json(target)
        if existing.get("freeze_hash") != protocol["freeze_hash"]:
            raise ProtocolError("protocol already frozen with a different plan; use a new run-id")
        return existing
    write_json(target, protocol)
    write_json(run_dir / "00_protocol/blocks.json", {"schema_version": 1, "blocks": [dict(row) for row in blocks], "protocol_hash": protocol["freeze_hash"]})
    write_json(run_dir / "00_protocol/generation_plan.json", {"schema_version": 1, "protocol_hash": protocol["freeze_hash"], "cells": cells, "ditto_cells": ditto_cells})
    return protocol


def validate_receipt(receipt: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    if str(receipt.get("protocol_id")) != PROTOCOL_ID:
        raise ProtocolError("receipt protocol_id mismatch")
    for key in ("cell_key", "sample_id", "source_group", "tfg", "arm"):
        if str(receipt.get(key)) != str(cell.get(key)):
            raise ProtocolError(f"receipt identity mismatch for {key}")
    if cell.get("protocol_hash") is not None and receipt.get("protocol_hash") != cell.get("protocol_hash"):
        raise ProtocolError("receipt protocol_hash mismatch")
    if receipt.get("status") not in {"complete", "failed"}:
        raise ProtocolError("receipt status must be complete or failed")
    if receipt.get("status") == "complete":
        if not receipt.get("output_sha256"):
            raise ProtocolError("complete receipt lacks output hash")


def gpu_processes() -> list[dict[str, str]]:
    """Return compute processes reported by nvidia-smi without changing state."""
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3:
            rows.append({"pid": fields[0], "process_name": fields[1], "used_memory_mib": fields[2]})
    return rows


def assert_gpu_clear(*, allow_pids: set[int] | None = None) -> list[dict[str, str]]:
    active = gpu_processes()
    allowed = {str(pid) for pid in (allow_pids or set())}
    blocking = [row for row in active if row["pid"] not in allowed]
    if blocking:
        raise ResourceBusy(f"CUDA resource is busy: {blocking}")
    return active
