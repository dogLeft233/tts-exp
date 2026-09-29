"""Stage 02: common MFA-linear alignment and WavLM/HiFi-GAN targets."""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16
from .common import (
    ProtocolError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .gpu import gpu_lease, release_torch_memory


def _normalized_text(text: str) -> str:
    from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.mfa3_alignment import (
        normalize_mfa3_transcript,
    )

    return normalize_mfa3_transcript(str(text))


def _write_lab(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return file_sha256(path)


def _copy_canonical_audio(source: Path, destination: Path, expected_hash: str) -> str:
    from .common import copy_verified

    copy_verified(source, destination, expected_hash)
    with wave.open(str(destination), "rb") as handle:
        if handle.getframerate() != config.SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ProtocolError(f"MFA input is not canonical PCM16: {destination}")
    return file_sha256(destination)


def _extract_features_covering_audio(adapter: Any, waveform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract one feature frame for every 320-sample audio interval.

    The frozen kNN-VC WavLM wrapper drops the final feature frame when its
    input ends at the audio boundary.  The minimum right context needed to
    expose that boundary is determined only by the audio sample-count
    remainder and the frozen 400-sample WavLM frontend.  The generated
    waveform is still cropped or padded only to the original sample count
    downstream.
    """
    import torch

    from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
        HOP_SAMPLES,
    )

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size < config.MIN_AUDIO_SAMPLES or not np.isfinite(values).all():
        raise ProtocolError("feature input must be finite and contain enough samples")
    remainder = int(values.size % HOP_SAMPLES)
    right_pad_samples = 80 if remainder == 0 else 400 - remainder
    padded = np.pad(values, (0, right_pad_samples)).astype(np.float32, copy=False)
    features = adapter.extract(torch.from_numpy(padded).unsqueeze(0)).cpu().numpy()
    expected_frames = int(np.ceil(values.size / HOP_SAMPLES))
    if features.ndim != 2 or features.shape[0] != expected_frames or features.shape[1] != 1024:
        raise ProtocolError(
            "feature extraction did not cover the complete audio tail: "
            f"got {tuple(features.shape)}, expected ({expected_frames}, 1024)"
        )
    return features, {
        "original_samples": int(values.size),
        "feature_stride_samples": int(HOP_SAMPLES),
        "right_pad_samples": int(right_pad_samples),
        "model_input_samples": int(padded.size),
        "feature_frames": int(features.shape[0]),
    }


def _prepare_corpus(
    records: Sequence[Mapping[str, Any]],
    tts_rows: Mapping[str, Mapping[str, Any]],
    stage_dir: Path,
    side: str,
) -> list[dict[str, Any]]:
    corpus = stage_dir / "mfa_input" / side
    corpus.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if side == "N":
            source = Path(str(record["natural_audio"]["path"]))
            expected_hash = str(record["natural_audio"]["sha256"])
        else:
            row = tts_rows[sample_id]
            source = Path(str(row["canonical_audio"]))
            expected_hash = str(row["canonical_audio_sha256"])
        destination = corpus / f"{sample_id}.wav"
        _copy_canonical_audio(source, destination, expected_hash)
        normalized = _normalized_text(str(record["transcript"]))
        lab_hash = _write_lab(corpus / f"{sample_id}.lab", normalized)
        entries.append(
            {
                "sample_id": sample_id,
                "audio": str(destination.resolve()),
                "audio_sha256": file_sha256(destination),
                "lab": str((corpus / f"{sample_id}.lab").resolve()),
                "lab_sha256": lab_hash,
                "normalized_transcript": normalized,
            }
        )
    return entries


def _mfa_command(input_dir: Path, output_dir: Path) -> list[str]:
    return [
        str(config.MFA_EXECUTABLE),
        "align",
        "--clean",
        "--overwrite",
        "--single_speaker",
        str(input_dir),
        config.MFA_DICTIONARY,
        config.MFA_ACOUSTIC_MODEL,
        str(output_dir),
    ]


def _run_mfa(input_dir: Path, output_dir: Path, log_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _mfa_command(input_dir, output_dir)
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(config.MFA_ROOT_DIR.resolve())
    environment["PATH"] = os.pathsep.join([str(config.MFA_EXECUTABLE.parent.resolve()), environment.get("PATH", "")])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=str(config.REPO), env=environment, capture_output=True, text=True, check=False)
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise ProtocolError(f"MFA alignment failed for {input_dir.name}: exit {result.returncode}")
    expected_ids = [path.stem for path in sorted(input_dir.glob("*.wav"))]
    missing = [sample_id for sample_id in expected_ids if not (output_dir / f"{sample_id}.TextGrid").is_file()]
    retry_records: list[dict[str, Any]] = []
    log_parts = [result.stdout, "\n--- STDERR ---\n", result.stderr]
    if missing:
        # MFA can report a successful batch while omitting an otherwise valid
        # utterance.  Retry only the missing, already-frozen inputs with the
        # same dictionary, acoustic model, and command policy.  This never
        # regenerates TTS audio or selects an output by a downstream score.
        retry_root = output_dir.parent / "mfa_single_sample_retries" / output_dir.name
        retry_root.mkdir(parents=True, exist_ok=True)
        for sample_id in missing:
            retry_input = retry_root / sample_id / "input"
            retry_output = retry_root / sample_id / "output"
            if retry_input.parent.exists():
                shutil.rmtree(retry_input.parent)
            retry_input.mkdir(parents=True, exist_ok=True)
            for suffix in (".wav", ".lab"):
                source = input_dir / f"{sample_id}{suffix}"
                if not source.is_file():
                    raise ProtocolError(f"missing MFA retry input: {source}")
                shutil.copy2(source, retry_input / source.name)
            retry_command = _mfa_command(retry_input, retry_output)
            retry_result = subprocess.run(
                retry_command,
                cwd=str(config.REPO),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            log_parts.extend(
                [
                    f"\n--- SINGLE SAMPLE RETRY {sample_id} ---\n",
                    "COMMAND: " + __import__("json").dumps(retry_command, ensure_ascii=False) + "\n",
                    retry_result.stdout,
                    "\n--- RETRY STDERR ---\n",
                    retry_result.stderr,
                ]
            )
            retry_grid = retry_output / f"{sample_id}.TextGrid"
            if retry_result.returncode != 0 or not retry_grid.is_file():
                log_path.write_text("".join(log_parts), encoding="utf-8")
                if retry_result.returncode != 0:
                    raise ProtocolError(f"MFA single-sample retry failed: {sample_id}; exit {retry_result.returncode}")
                raise ProtocolError(f"MFA single-sample retry produced no TextGrid: {sample_id}")
            destination = output_dir / retry_grid.name
            shutil.copy2(retry_grid, destination)
            retry_records.append(
                {
                    "sample_id": sample_id,
                    "input_dir": str(retry_input.resolve()),
                    "output_dir": str(retry_output.resolve()),
                    "textgrid": str(destination.resolve()),
                    "textgrid_sha256": file_sha256(destination),
                    "command": retry_command,
                    "command_sha256": __import__("hashlib").sha256(
                        __import__("json").dumps(retry_command, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "returncode": int(retry_result.returncode),
                }
            )
    log_path.write_text("".join(log_parts), encoding="utf-8")
    missing_after_retry = [sample_id for sample_id in expected_ids if not (output_dir / f"{sample_id}.TextGrid").is_file()]
    if missing_after_retry:
        raise ProtocolError(f"MFA alignment produced missing TextGrids: {missing_after_retry[:5]}")
    return {
        "command": command,
        "command_sha256": __import__("hashlib").sha256(__import__("json").dumps(command, separators=(",", ":")).encode()).hexdigest(),
        "returncode": int(result.returncode),
        "log": str(log_path.resolve()),
        "textgrid_count": len(expected_ids),
        "mfa_root_dir": str(config.MFA_ROOT_DIR.resolve()),
        "missing_after_batch": missing,
        "single_sample_retries": retry_records,
    }


def _alignment_failure_payload(stage_dir: Path, failures: list[dict[str, Any]], *, message: str | None = None) -> None:
    paths = config.RunPaths(stage_dir.parent)
    payload = {
        "schema_version": 1,
        "stage_id": "02_targets",
        "protocol_id": config.PROTOCOL_ID,
        "status": "blocked",
        "record_count": 0,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_textgrids": config.EXPECTED_RECORD_COUNT * 3,
        "rows": [],
        "failures": failures,
        "message": message,
    }
    write_self_hashed_json(paths.targets / "alignment_manifest.json", payload)


def _load_tts_rows(tts: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if tts.get("status") != "complete":
        raise ProtocolError("TTS manifest is incomplete")
    local: dict[str, Mapping[str, Any]] = {}
    cloud: dict[str, Mapping[str, Any]] = {}
    for row in tts.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or not isinstance(row.get("LOCAL"), Mapping) or not isinstance(row.get("CLOUD"), Mapping):
            raise ProtocolError(f"malformed TTS paired row: {sample_id}")
        local[sample_id] = row["LOCAL"]
        cloud[sample_id] = row["CLOUD"]
    if len(local) != config.EXPECTED_RECORD_COUNT or len(cloud) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("TTS provider coverage is incomplete")
    return local, cloud


def _trace_rows(mapping: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "natural_frame_index": int(item.natural_frame_index),
            "natural_token_index": int(item.natural_token_index),
            "natural_label": str(item.natural_label),
            "natural_silence": bool(item.natural_silence),
            "mapping_type": str(item.mapping_type),
            "tts_token_index": None if item.tts_token_index is None else int(item.tts_token_index),
            "left_frame_index": int(item.left_frame_index),
            "right_frame_index": int(item.right_frame_index),
            "interpolation_alpha": float(item.interpolation_alpha),
            "fallback_reason": item.fallback_reason,
        }
        for item in mapping
    ]


def _load_alignment_tokens(path: Path) -> list[dict[str, Any]]:
    from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
        parse_textgrid,
    )

    return parse_textgrid(path)


def _prepare_alignment(
    run_root: Path,
    records: Sequence[Mapping[str, Any]],
    local_rows: Mapping[str, Mapping[str, Any]],
    cloud_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    paths.targets.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    corpora: dict[str, list[dict[str, Any]]] = {}
    try:
        corpora["N"] = _prepare_corpus(records, {}, paths.targets, "N")
        corpora["LOCAL"] = _prepare_corpus(records, local_rows, paths.targets, "LOCAL")
        corpora["CLOUD"] = _prepare_corpus(records, cloud_rows, paths.targets, "CLOUD")
        grids: dict[str, Path] = {}
        for side in ("N", "LOCAL", "CLOUD"):
            grids[side] = paths.targets / "textgrids" / side
            _run_mfa(paths.targets / "mfa_input" / side, grids[side], paths.targets / "logs" / f"mfa_{side}.log")
    except Exception as exc:
        failures.append({"stage": "alignment", "error_type": type(exc).__name__, "error": str(exc)})
        # Preserve failures at record granularity even when MFA stops early.
        for record in records:
            sample_id = str(record["sample_id"])
            for side in ("N", "LOCAL", "CLOUD"):
                grid = paths.targets / "textgrids" / side / f"{sample_id}.TextGrid"
                if not grid.is_file():
                    failures.append({"sample_id": sample_id, "side": side, "error": "missing TextGrid after MFA failure"})
        _alignment_failure_payload(paths.targets, failures, message=str(exc))
        raise ProtocolError(f"alignment stage blocked: {exc}") from exc

    from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
        parsed_token_hash,
    )
    from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.mfa3_alignment import (
        NORMALIZATION_POLICY_ID,
    )

    rows: list[dict[str, Any]] = []
    normalized_by_id = {str(entry["sample_id"]): str(entry["normalized_transcript"]) for entry in corpora["N"]}
    for record in records:
        sample_id = str(record["sample_id"])
        row: dict[str, Any] = {"sample_id": sample_id, "source_group": str(record["source_group"]), "normalized_transcript": normalized_by_id[sample_id]}
        try:
            for side in ("N", "LOCAL", "CLOUD"):
                grid_path = paths.targets / "textgrids" / side / f"{sample_id}.TextGrid"
                tokens = _load_alignment_tokens(grid_path)
                row[side] = {
                    "textgrid": str(grid_path.resolve()),
                    "textgrid_sha256": file_sha256(grid_path),
                    "tokens": tokens,
                    "token_sha256": parsed_token_hash(tokens),
                    "token_count": len(tokens),
                }
            if len({str(row["normalized_transcript"]) for row in (corpora["N"] + corpora["LOCAL"] + corpora["CLOUD"]) if str(row["sample_id"]) == sample_id}) != 1:
                raise ProtocolError(f"normalized transcript differs across MFA inputs: {sample_id}")
            row["normalization_policy_id"] = NORMALIZATION_POLICY_ID
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - preserve each alignment failure
            failures.append({"sample_id": sample_id, "stage": "parse_textgrid", "error_type": type(exc).__name__, "error": str(exc)})
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT
    payload = {
        "schema_version": 1,
        "stage_id": "02_targets",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_textgrids": config.EXPECTED_RECORD_COUNT * 3,
        "mfa": {
            "version": config.MFA_VERSION,
            "dictionary": config.MFA_DICTIONARY,
            "acoustic_model": config.MFA_ACOUSTIC_MODEL,
            "root_dir": str(config.MFA_ROOT_DIR.resolve()),
            "corpus_entries": corpora,
        },
        "rows": rows,
        "failures": failures,
    }
    write_self_hashed_json(paths.targets / "alignment_manifest.json", payload)
    if not complete:
        raise ProtocolError(f"alignment parse incomplete: {len(rows)}/{config.EXPECTED_RECORD_COUNT}")
    return payload


def _target_audio_row(
    *,
    record: Mapping[str, Any],
    provider: str,
    tts_row: Mapping[str, Any],
    alignment: Mapping[str, Any],
    adapter: Any,
    output_dir: Path,
    natural_features: np.ndarray | None = None,
    natural_feature_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
        candidate_from_features,
        canonical_pcm_s16le,
        validate_wavlm_interface,
    )
    from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
        build_frame_mapping,
        extend_final_token_for_feature_tail,
        parsed_token_hash,
    )
    sample_id = str(record["sample_id"])
    natural, _natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
    tts, _tts_meta = read_pcm16(Path(str(tts_row["canonical_audio"])))
    tts_float = tts.astype(np.float32) / 32768.0
    if natural_features is None:
        natural_features, natural_feature_metadata = _extract_features_covering_audio(
            adapter, natural.astype(np.float32) / 32768.0
        )
    if natural_feature_metadata is None:
        raise ProtocolError(f"natural feature metadata is missing: {sample_id}")
    tts_features, tts_feature_metadata = _extract_features_covering_audio(adapter, tts_float)
    natural_tokens = alignment["N"]["tokens"]
    tts_tokens = alignment[provider]["tokens"]
    natural_tokens_extended, natural_extension = extend_final_token_for_feature_tail(
        natural_tokens,
        natural_features.shape[0],
        max_extension_s=config.MAX_FEATURE_TAIL_EXTENSION_S,
    )
    tts_tokens_extended, tts_extension = extend_final_token_for_feature_tail(
        tts_tokens,
        tts_features.shape[0],
        max_extension_s=config.MAX_FEATURE_TAIL_EXTENSION_S,
    )
    mapping, mapping_stats = build_frame_mapping(
        natural_features.shape[0],
        tts_features.shape[0],
        natural_tokens_extended,
        tts_tokens_extended,
    )
    candidate, candidate_meta = candidate_from_features(
        natural_features=natural_features,
        tts_features=tts_features,
        mapping=mapping,
        natural_audio_samples=natural.size,
        vocode=lambda conditioning: adapter.vocode(torch.from_numpy(conditioning)).cpu().numpy(),
    )
    adjustment = int(candidate_meta["length_adjustment"]["adjustment_samples"])
    if abs(adjustment) >= 320:
        raise ProtocolError(f"candidate tail adjustment is outside <320 samples: {sample_id}/{provider}")
    pcm, pcm_qc = canonical_pcm_s16le(candidate, natural.size)
    output = output_dir / "audio" / provider / f"{sample_id}.wav"
    trace = output_dir / "traces" / provider / f"{sample_id}.json"
    sidecar = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and trace.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        trace_prior = verify_self_hashed_json(trace)
        if (
            prior.get("target_audio_sha256") == file_sha256(output)
            and prior.get("tts_audio_sha256") == tts_row["canonical_audio_sha256"]
            and trace_prior.get("sample_id") == sample_id
            and trace_prior.get("provider") == provider
        ):
            return prior
        raise ProtocolError(f"existing target identity changed: {sample_id}/{provider}")
    if output.exists() or trace.exists() or sidecar.exists():
        raise ProtocolError(f"partial target cannot be resumed: {sample_id}/{provider}")
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(pcm)
    interface = validate_wavlm_interface(
        adapter.metadata(),
        revision=config.KNN_VC_REVISION,
        wavlm_checkpoint_sha256=config.WAVLM_CHECKPOINT_SHA256,
        vocoder_checkpoint_sha256=config.VOCODER_CHECKPOINT_SHA256,
    )
    trace_payload = {
        "schema_version": 1,
        "stage_id": "02_targets",
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "provider": provider,
        "natural_token_sha256": parsed_token_hash(natural_tokens_extended),
        "tts_token_sha256": parsed_token_hash(tts_tokens_extended),
        "natural_token_extension_s": natural_extension,
        "tts_token_extension_s": tts_extension,
        "mapping_stats": mapping_stats,
        "frames": _trace_rows(mapping),
    }
    write_self_hashed_json(trace, trace_payload)
    target_hash = file_sha256(output)
    target_row = {
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "provider": provider,
        "natural_audio": str(record["natural_audio"]["path"]),
        "natural_audio_sha256": str(record["natural_audio"]["sha256"]),
        "tts_audio": str(tts_row["canonical_audio"]),
        "tts_audio_sha256": str(tts_row["canonical_audio_sha256"]),
        "target_audio": str(output.resolve()),
        "target_audio_sha256": target_hash,
        "decoded_pcm_sha256": read_pcm16(output)[1]["decoded_pcm_sha256"],
        "natural_samples": int(natural.size),
        "target_samples": int(natural.size),
        "textgrid": alignment[provider]["textgrid"],
        "textgrid_sha256": alignment[provider]["textgrid_sha256"],
        "natural_textgrid": alignment["N"]["textgrid"],
        "natural_textgrid_sha256": alignment["N"]["textgrid_sha256"],
        "trace": str(trace.resolve()),
        "trace_sha256": file_sha256(trace),
        "mapping": mapping_stats,
        "candidate": candidate_meta,
        "pcm_qc": pcm_qc,
        "feature_extraction": {
            "natural": dict(natural_feature_metadata),
            "tts": dict(tts_feature_metadata),
        },
        "model_interface": interface,
        "tail_extension": {"natural_s": natural_extension, "tts_s": tts_extension, "max_s": config.MAX_FEATURE_TAIL_EXTENSION_S},
        "status": "ok",
    }
    write_self_hashed_json(sidecar, target_row)
    return target_row


def run_targets_stage(
    run_root: Path,
    cohort: Mapping[str, Any],
    tts: Mapping[str, Any],
) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    records = cohort.get("records", [])
    if cohort.get("status") != "complete" or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete before target stage")
    local_rows, cloud_rows = _load_tts_rows(tts)
    alignment = _prepare_alignment(run_root, records, local_rows, cloud_rows)
    align_by_id = {str(row["sample_id"]): row for row in alignment["rows"]}
    targets: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    gpu_held = False
    try:
        with gpu_lease("wavlm_targets"):
            gpu_held = True

            from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

            adapter = WavLMKNNVCAdapter.load_pretrained(device="cuda", source=config.KNN_VC_SOURCE, revision=config.KNN_VC_REVISION)
            for index, record in enumerate(records, 1):
                sample_id = str(record["sample_id"])
                row = align_by_id[sample_id]
                generated: dict[str, Any] = {}
                try:
                    natural_values, _ = read_pcm16(Path(str(record["natural_audio"]["path"])))
                    natural_features, natural_feature_metadata = _extract_features_covering_audio(
                        adapter, natural_values.astype(np.float32) / 32768.0
                    )
                    generated["LOCAL"] = _target_audio_row(
                        record=record,
                        provider="LOCAL",
                        tts_row=local_rows[sample_id],
                        alignment=row,
                        adapter=adapter,
                        output_dir=paths.targets,
                        natural_features=natural_features,
                        natural_feature_metadata=natural_feature_metadata,
                    )
                    generated["CLOUD"] = _target_audio_row(
                        record=record,
                        provider="CLOUD",
                        tts_row=cloud_rows[sample_id],
                        alignment=row,
                        adapter=adapter,
                        output_dir=paths.targets,
                        natural_features=natural_features,
                        natural_feature_metadata=natural_feature_metadata,
                    )
                    targets.extend(generated.values())
                    print(f"TARGET {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
                except Exception as exc:  # noqa: BLE001 - preserve each failed target record
                    failures.append({"sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - preserve stage failure in manifest
        failures.append({"sample_id": None, "error_type": type(exc).__name__, "error": str(exc), "gpu_stage": True})
    finally:
        if gpu_held:
            release_torch_memory()
    target_ids = {(str(row["sample_id"]), str(row["provider"])) for row in targets}
    complete = not failures and len(targets) == config.EXPECTED_RECORD_COUNT * 2 and len(target_ids) == config.EXPECTED_RECORD_COUNT * 2
    manifest = {
        "schema_version": 1,
        "stage_id": "02_targets",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len({str(row["sample_id"]) for row in targets}),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "target_count": len(targets),
        "expected_target_count": config.EXPECTED_RECORD_COUNT * 2,
        "providers": ["LOCAL", "CLOUD"],
        "rows": targets,
        "failures": failures,
        "alignment_manifest_sha256": file_sha256(paths.targets / "alignment_manifest.json"),
        "tts_manifest_sha256": file_sha256(paths.tts / "tts_manifest.json"),
    }
    write_self_hashed_json(paths.targets / "targets_manifest.json", manifest)
    write_self_hashed_json(paths.targets / "failures.json", {"schema_version": 1, "stage_id": "02_targets", "protocol_id": config.PROTOCOL_ID, "status": manifest["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"target generation incomplete: {len(targets)}/{config.EXPECTED_RECORD_COUNT * 2}")
    return manifest


__all__ = ["run_targets_stage"]
