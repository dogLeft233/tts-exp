"""Freeze a full-audio/center-32 v2 fit plan from the historical 8-row fit set.

The builder changes only fit rows/runtime context. It reads the existing
FIT_ONLY_FROZEN short-window plan as provenance, verifies the original full
PCM files and a separate remote full-audio transfer manifest, then writes a
new plan without touching the old plan or any dev/confirmation audio.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audio_enhancement.audio_io import read_canonical_wav


EXPECTED_BASE_SCHEMA = "cross_tfg_audio_head_historical_fit_subset_v1"
EXPECTED_PLAN_SCHEMA = "cross_tfg_audio_head_historical_fullcontext_fit_v2"
EXPECTED_ROWS = 8
EXPECTED_SPEAKERS = {"S0052", "S0053", "S0054", "S0055"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _pcm16_sha(audio: np.ndarray) -> str:
    if audio.dtype != np.float32 or audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("historical source must be finite mono float32")
    values = np.rint(np.clip(audio.astype(np.float64) * 32768.0, -32768.0, 32767.0)).astype("<i2")
    return _sha256_bytes(np.ascontiguousarray(values).tobytes())


def _read_audio_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"audio manifest JSONL row {line_no} is not an object")
            rows.append(item)
        payload = {"rows": rows}
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("full-audio transfer manifest must be a JSON object with rows[]")
    records: dict[str, dict[str, Any]] = {}
    for row in payload["rows"]:
        if not isinstance(row, dict):
            raise ValueError("full-audio transfer manifest rows must be objects")
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in records:
            raise ValueError(f"missing or duplicate full-audio row_id: {row_id!r}")
        records[row_id] = row
    embedded = payload.get("manifest_sha256")
    if embedded is not None:
        without = dict(payload)
        without.pop("manifest_sha256", None)
        if str(embedded).lower() != _canonical_sha(without):
            raise ValueError("full-audio transfer manifest embedded canonical SHA mismatch")
    return records, digest, str(embedded).lower() if embedded is not None else None


def _check_sha(name: str, value: Any) -> str:
    text = str(value or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return text


def build_plan(*, base_path: Path, full_audio_manifest: Path, output_path: Path,
               cache_root: str, fit_seed: int, epochs: int, steps: int) -> dict[str, Any]:
    base_raw = base_path.read_bytes()
    base_sha = _sha256_bytes(base_raw)
    base = json.loads(base_raw.decode("utf-8"))
    if not isinstance(base, dict) or base.get("schema") != EXPECTED_BASE_SCHEMA or base.get("status") != "FIT_ONLY_FROZEN":
        raise ValueError("base plan is not the frozen historical fit-only subset")
    if str(base.get("plan_payload_sha256", "")).lower() != _canonical_sha(
        {key: value for key, value in base.items() if key != "plan_payload_sha256"}
    ):
        raise ValueError("base fit plan embedded canonical SHA mismatch")
    fit_rows = [row for row in base.get("rows", []) if row.get("split") == "fit"]
    if len(fit_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_ROWS} inherited fit rows, got {len(fit_rows)}")
    speaker_ids = {str(row.get("speaker_id")) for row in fit_rows}
    if speaker_ids != EXPECTED_SPEAKERS:
        raise ValueError(f"historical fit speaker set changed: {sorted(speaker_ids)}")
    if any(row.get("historical_exposed") is not True for row in fit_rows):
        raise ValueError("historical exposure must remain explicitly marked for all inherited fit rows")

    mapped_rows, manifest_raw_sha, manifest_payload_sha = _read_audio_manifest(full_audio_manifest)
    if not PurePosixPath(cache_root).is_absolute():
        raise ValueError("remote SyncNet cache root must be an absolute POSIX path")
    expected_ids = {str(row["row_id"]) for row in fit_rows}
    if set(mapped_rows) != expected_ids:
        missing = sorted(expected_ids - set(mapped_rows))
        extra = sorted(set(mapped_rows) - expected_ids)
        raise ValueError(f"full-audio row identity mismatch; missing={missing}, extra={extra}")

    result_rows = []
    source_count_total = 0
    for source_row in fit_rows:
        row = copy.deepcopy(source_row)
        row_id = str(row["row_id"])
        source = row["source_pcm"]
        remote = mapped_rows[row_id]
        remote_path = str(remote.get("remote_path", ""))
        if not PurePosixPath(remote_path).is_absolute() or "\x00" in remote_path:
            raise ValueError(f"remote full-audio path must be an absolute POSIX path: {row_id}")
        file_sha = _check_sha(f"{row_id}.file_sha256", remote.get("file_sha256"))
        pcm_sha = _check_sha(f"{row_id}.pcm_float32_sha256", remote.get("pcm_float32_sha256"))
        pcm16_sha = _check_sha(f"{row_id}.direct_source_pcm16_sha256", remote.get("direct_source_pcm16_sha256"))
        source_file_sha = _check_sha(f"{row_id}.source_pcm.file_sha256", source.get("file_sha256"))
        source_pcm_sha = _check_sha(f"{row_id}.source_pcm.pcm_sha256", source.get("pcm_sha256"))
        source_pcm16_sha = _check_sha(f"{row_id}.source_pcm.pcm16_sha256", source.get("pcm16_sha256"))
        sample_count = int(remote.get("sample_count", -1))
        if (
            file_sha != source_file_sha
            or pcm_sha != source_pcm_sha
            or pcm16_sha != source_pcm16_sha
            or sample_count != int(source.get("sample_count", -2))
            or int(remote.get("sample_rate_hz", -1)) != 16_000
            or int(remote.get("channels", -1)) != 1
        ):
            raise ValueError(f"remote transfer metadata disagrees with frozen full source row: {row_id}")
        duration = sample_count / 16_000.0
        if not (3.2 <= duration <= 8.0):
            raise ValueError(f"historical full source falls outside the frozen 3.2--8.0s fit window: {row_id}")

        local_source_path = Path(str(source["path"])).resolve(strict=True)
        actual_file_sha = _sha256_file(local_source_path)
        if actual_file_sha != source_file_sha:
            raise ValueError(f"local historical full source file hash mismatch: {row_id}")
        audio, info = read_canonical_wav(local_source_path)
        actual_pcm16_sha = _pcm16_sha(audio)
        if (
            info.sample_rate != 16_000
            or info.source_channels != 1
            or int(info.frames) != sample_count
            or info.pcm_float32_sha256 != pcm_sha
            or actual_pcm16_sha != pcm16_sha
        ):
            raise ValueError(f"local historical full source PCM metadata/hash mismatch: {row_id}")

        prior_crop = row.get("natural_pcm")
        row["legacy_short_window_natural_pcm"] = prior_crop
        row["legacy_short_window_selection"] = row.pop("training_window", None)
        row["legacy_row_id"] = row_id
        row["row_id"] = f"{row['sample_id']}__full_natural_context_v2"
        row["natural_pcm"] = {
            "path": remote_path,
            "file_sha256": file_sha,
            "pcm_sha256": pcm_sha,
            "direct_source_pcm16_sha256": pcm16_sha,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "sample_count": sample_count,
            "source_encoding": "original AISHELL 16 kHz mono WAV represented as canonical float32 PCM by read_canonical_wav",
            "source_file_path": str(source["path"]),
            "source_archive_path": source.get("source_archive_path"),
            "source_archive_sha256": source.get("source_archive_sha256"),
            "source_archive_member": source.get("source_archive_member"),
        }
        if remote.get("transfer_local_path"):
            row["natural_pcm"]["transfer_local_path"] = str(remote["transfer_local_path"])
        row["full_context_fit"] = {
            "audio_scope": "complete original utterance; no 1.4s crop",
            "source_sample_count": sample_count,
            "source_duration_seconds": duration,
            "source_clock_hz": "25/1",
            "video_frame_count": 32,
            "visual_query_count": 27,
            "audio_query_rule": "q0=floor((N_audio_embeddings-27)/2); map the center 27 visual windows to audio q0..q0+26",
            "lag_offsets_frames": list(range(-15, 16)),
            "lag_padding": "none; every lag candidate must be in the full natural-audio embedding sequence",
        }
        result_rows.append(row)
        source_count_total += sample_count

    if len({row["row_id"] for row in result_rows}) != EXPECTED_ROWS:
        raise ValueError("full-context row IDs are not unique")
    if len({row["sample_id"] for row in result_rows}) != EXPECTED_ROWS:
        raise ValueError("duplicate historical sample ID in full-context fit subset")

    plan = copy.deepcopy(base)
    plan["schema"] = EXPECTED_PLAN_SCHEMA
    plan["schema_version"] = 2
    plan["run_id"] = "cross_tfg_audio_head_fullcontext_fit_20261003_attempt01"
    plan["status"] = "FIT_ONLY_FROZEN"
    plan["rows"] = result_rows
    plan["historical_fit_subset"] = True
    plan["full_context_fit_v2"] = {
        "source_plan_path": str(base_path.resolve()),
        "source_plan_raw_sha256": base_sha,
        "full_audio_manifest_path": str(full_audio_manifest.resolve()),
        "full_audio_manifest_raw_sha256": manifest_raw_sha,
        "full_audio_manifest_payload_sha256": manifest_payload_sha,
        "fit_rows": EXPECTED_ROWS,
        "fit_speakers": sorted(EXPECTED_SPEAKERS),
        "fit_seed": int(fit_seed),
        "planned_optimizer_steps": steps,
        "confirmation_audio_opened": False,
        "development_audio_opened": False,
        "model_scoring_or_effects_read": False,
    }
    plan["fit_config"] = copy.deepcopy(base["fit_config"])
    plan["fit_config"].update({
        "epochs": epochs,
        "steps_requested": steps,
        "fit_seed": int(fit_seed),
        "source_sample_count_total": source_count_total,
        "selected_rows": [row["row_id"] for row in result_rows],
        "window_samples": None,
        "context": "full original utterance; center 32 Wav2Lip frames; 27 visual queries; all 31 SyncNet lags valid without padding",
        "proxy_loss": {
            "kind": "fullaudio_center32_median_lag_distance_minus_normalized_softmin",
            "lag_count": 31,
            "support": "27 center video queries against full natural-audio embeddings; exact source-clock alignment; no lag padding",
            "temperature": 0.1,
        },
        "status_scope": "exploratory historical fit-only runtime comparison; no dev/confirmation scoring or candidate selection",
    })
    plan["runtime"] = copy.deepcopy(base["runtime"])
    plan["runtime"]["source_context_is_training_proxy_only"] = True
    plan["runtime"]["context"]["training_context"] = (
        "full original natural audio; fixed portrait crop; central contiguous 32-frame Wav2Lip window; "
        "all full-audio SyncNet windows cached; 27 center queries use complete unpadded 31-lag support"
    )
    plan["runtime"]["full_context"] = {
        "schema": "cross_tfg_wav2lip_fullcontext_fit_runtime_v2",
        "audio_embedding_cache_root": cache_root,
        "video_frame_count": 32,
        "visual_query_count": 27,
        "lag_radius_frames": 15,
        "padding_policy": "none",
        "audio_embedding_batch_size": 32,
        "video_frame_batch_size": 4,
        "syncnet_batch_size": 16,
        "head_n_fft": int(base["fit_config"]["transform"]["n_fft"]),
        "head_hop_length": int(base["fit_config"]["transform"]["hop_length"]),
        "head_hidden_channels": int(base["fit_config"]["transform"]["hidden_channels"]),
        "head_max_residual_db": float(base["fit_config"]["transform"]["max_residual_db"]),
        "head_level_normalize": True,
        "source_sample_count_total": source_count_total,
    }
    plan["inherited_split_manifest_status"] = base.get("split_manifest_status")
    plan["fit_subset_row_status"] = "FULL_CONTEXT_ROWS_FROZEN_FROM_HISTORICAL_FIT_SUBSET"
    plan["plan_payload_sha256"] = _canonical_sha({key: value for key, value in plan.items() if key != "plan_payload_sha256"})
    output_path = output_path.resolve()
    receipt_path = output_path.with_name("fullcontext_fit_plan_build_receipt.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite full-context v2 plan: {output_path}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite fit plan receipt: {receipt_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output_path.write_bytes(raw)
    receipt = {
        "schema": "cross_tfg_audio_head_fullcontext_fit_plan_build_receipt_v2",
        "status": "FIT_ONLY_FROZEN",
        "plan_path": str(output_path),
        "plan_bytes": len(raw),
        "plan_raw_sha256": _sha256_bytes(raw),
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "base_plan_raw_sha256": base_sha,
        "full_audio_manifest_raw_sha256": manifest_raw_sha,
        "full_audio_manifest_payload_sha256": manifest_payload_sha,
        "row_count": len(result_rows),
        "speaker_count": len({row["speaker_id"] for row in result_rows}),
        "sample_count_total": source_count_total,
        "optimizer_steps_requested": steps,
        "epochs_requested": epochs,
        "confirmation_audio_opened": False,
        "development_audio_opened": False,
        "effects_or_scores_read": False,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--full-audio-manifest", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--fit-seed", type=int, default=20261003)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps <= 0 or args.fit_seed < 0:
        parser.error("epochs/steps must be positive and fit seed non-negative")
    receipt = build_plan(
        base_path=args.base_plan.resolve(strict=True),
        full_audio_manifest=args.full_audio_manifest.resolve(strict=True),
        output_path=args.output_plan,
        cache_root=args.cache_root,
        fit_seed=args.fit_seed,
        epochs=args.epochs,
        steps=args.steps,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
