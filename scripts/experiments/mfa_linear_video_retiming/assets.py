from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.io import wavfile

from .common import ProtocolError, file_sha256, path_binding, read_json, verify_json, write_json


EXPECTED_WAV2LIP_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
EXPECTED_PORTRAIT_3_RGB_SHA256 = "bd5659ec3560bea57c34aa98d97ec9f60916a85a9414e70a3cf7a538db3f3903"


def _records_by_id(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get("sample_id", row.get("id", "")))
        if not key or key in output:
            raise ProtocolError(f"{label} has a missing or duplicate sample ID: {key!r}")
        output[key] = row
    return output


def read_pcm16_mono(path: str | Path) -> tuple[int, bytes, int]:
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as handle:
            if handle.getcomptype() != "NONE" or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ProtocolError(f"audio must be mono PCM16 WAV: {target}")
            sample_rate = int(handle.getframerate())
            pcm = handle.readframes(handle.getnframes())
            sample_count = int(handle.getnframes())
        if sample_rate != 16000 or len(pcm) != sample_count * 2:
            raise ProtocolError(f"audio must be 16 kHz mono PCM16: {target}")
        return sample_rate, pcm, sample_count
    except (wave.Error, EOFError):
        # Python's wave module intentionally rejects WAVE_FORMAT_IEEE_FLOAT.
        # scipy preserves those samples and their native rate/length.
        pass
    try:
        sample_rate, samples = wavfile.read(target)
    except (OSError, ValueError, EOFError) as exc:
        raise ProtocolError(f"invalid WAV: {target}: {exc}") from exc
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ProtocolError(f"audio must be mono WAV: {target}")
    sample_count = int(samples.shape[0])
    if samples.dtype != np.dtype("float32"):
        raise ProtocolError(f"unsupported non-PCM16 WAV encoding: {target}")
    if not np.isfinite(samples).all() or np.any(samples < -1.0) or np.any(samples > 1.0):
        raise ProtocolError(f"IEEE float WAV samples must be finite and normalized to [-1, 1]: {target}")
    # Keep the original float WAV as model/scorer input. This stable PCM16
    # view is used only for identity diagnostics and sample-count checks.
    quantized = np.rint(samples.astype(np.float64) * 32768.0)
    quantized = np.clip(quantized, -32768, 32767).astype("<i2")
    pcm = np.ascontiguousarray(quantized).tobytes()
    if int(sample_rate) != 16000 or len(pcm) != sample_count * 2:
        raise ProtocolError(f"audio must be 16 kHz mono IEEE float32 WAV: {target}")
    return sample_rate, pcm, sample_count


def validate_audio_pair(
    natural_path: str | Path,
    mfa_path: str | Path,
    *,
    natural_sha256: str | None = None,
    mfa_sha256: str | None = None,
) -> dict[str, Any]:
    natural = Path(natural_path).resolve()
    mfa = Path(mfa_path).resolve()
    if not natural.is_file() or not mfa.is_file():
        raise ProtocolError(f"missing N/M audio pair: N={natural.is_file()} M={mfa.is_file()}")
    n_rate, n_pcm, n_count = read_pcm16_mono(natural)
    m_rate, m_pcm, m_count = read_pcm16_mono(mfa)
    if n_rate != m_rate or n_count != m_count:
        raise ProtocolError(f"N/M sample rate or exact length mismatch: {(n_rate, n_count)} != {(m_rate, m_count)}")
    n_hash = file_sha256(natural)
    m_hash = file_sha256(mfa)
    if natural_sha256 and n_hash != natural_sha256:
        raise ProtocolError(f"natural source SHA mismatch: {natural}")
    if mfa_sha256 and m_hash != mfa_sha256:
        raise ProtocolError(f"MFA-linear output SHA mismatch: {mfa}")
    return {
        "sample_rate": n_rate,
        "sample_count": n_count,
        "duration_seconds": n_count / n_rate,
        "pcm_hash_encoding": "normalized_signed_pcm16_le; float32 rounded ties-to-even at 32768 scale",
        "natural": {"path": str(natural), "sha256": n_hash, "pcm_sha256": hashlib.sha256(n_pcm).hexdigest()},
        "mfa_linear": {"path": str(mfa), "sha256": m_hash, "pcm_sha256": hashlib.sha256(m_pcm).hexdigest()},
        "pcm_identity": n_pcm == m_pcm,
    }


def _portrait_record(registry: Mapping[str, Any], portrait_id: str) -> Mapping[str, Any]:
    portraits = registry.get("portraits")
    if not isinstance(portraits, Mapping) or not isinstance(portraits.get(portrait_id), Mapping):
        raise ProtocolError(f"portrait registry has no portrait {portrait_id}")
    row = portraits[portrait_id]
    if str(row.get("portrait_id")) != portrait_id:
        raise ProtocolError(f"portrait registry key/id mismatch for {portrait_id}")
    return row


def audit_portraits(config: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = Path(config["paths"]["portrait_registry"]).resolve()
    registry = read_json(registry_path)
    result: dict[str, Any] = {}
    for portrait_id in ("3", "6", "9"):
        row = _portrait_record(registry, portrait_id)
        image_path = Path(str(row.get("path", ""))).resolve()
        if not image_path.is_file():
            raise ProtocolError(f"portrait PNG is missing: {image_path}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        container_sha = file_sha256(image_path)
        expected_rgb = str(row.get("rgb_pixel_sha256", ""))
        expected_container = str(row.get("container_sha256", ""))
        if rgb_sha != expected_rgb or container_sha != expected_container:
            raise ProtocolError(f"portrait image fingerprint mismatch: {portrait_id}")
        if [width, height] != [int(row.get("width", -1)), int(row.get("height", -1))]:
            raise ProtocolError(f"portrait dimensions differ from registry: {portrait_id}")
        box = [int(value) for value in row.get("generation_box_xyxy", [])]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ProtocolError(f"invalid generation box for portrait {portrait_id}")
        if box == [0, 0, width, height]:
            raise ProtocolError(f"full-frame Wav2Lip fallback is forbidden for portrait {portrait_id}")
        score = row.get("score_box")
        if not isinstance(score, Mapping) or score.get("order") != "x1,y1,x2,y2; unbounded square with zero padding":
            raise ProtocolError(f"invalid score-box coordinate contract for portrait {portrait_id}")
        score_box = [int(value) for value in score.get("box", [])]
        if len(score_box) != 4 or score_box[2] - score_box[0] <= 0 or score_box[3] - score_box[1] != score_box[2] - score_box[0]:
            raise ProtocolError(f"invalid square score box for portrait {portrait_id}")
        detector = row.get("detector")
        if not isinstance(detector, Mapping) or not isinstance(detector.get("selected"), Mapping):
            raise ProtocolError(f"portrait {portrait_id} has no detector provenance")
        if row.get("geometry_contract") != "sfd_selected_face_v1" or row.get("source_frame_indices") != [0]:
            raise ProtocolError(f"portrait {portrait_id} does not satisfy the frozen geometry contract")
        if portrait_id == "3" and (rgb_sha != EXPECTED_PORTRAIT_3_RGB_SHA256 or box != [138, 90, 357, 387] or score_box != [33, 18, 462, 447]):
            raise ProtocolError("portrait 3 differs from the explicitly frozen geometry")
        result[portrait_id] = {
            "portrait_id": portrait_id,
            "path": str(image_path),
            "container_sha256": container_sha,
            "rgb_pixel_sha256": rgb_sha,
            "width": width,
            "height": height,
            "generation_box_xyxy": box,
            "score_box": dict(score),
            "detector": dict(detector),
            "geometry_contract": row["geometry_contract"],
        }
    return {"registry": path_binding(registry_path), "portraits": result}


def freeze_inputs(
    config: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    sample_ids: Sequence[str | int] | None = None,
    cohort_dir: str | Path | None = None,
) -> dict[str, Any]:
    if config.get("protocol") == "mfa_linear_video_retiming_v2":
        return _freeze_inputs_v2(config, output_path=output_path, sample_ids=sample_ids, cohort_dir=cohort_dir)
    paths = config["paths"]
    summary_path = Path(paths["mfa_summary"]).resolve()
    clean_path = Path(paths["clean_manifest"]).resolve()
    mini_path = Path(paths["mini_manifest"]).resolve()
    legacy_root = Path(paths["legacy_eval_dir"]).resolve()
    summary = read_json(summary_path)
    clean = read_json(clean_path)
    mini = read_json(mini_path)
    if summary.get("arm") != "mfa_linear" or int(summary.get("samples_total", -1)) != 3 or int(summary.get("samples_ok", -1)) != 3 or summary.get("failures"):
        raise ProtocolError("MFA-linear summary is not the frozen clean three-sample repair")
    if Path(str(summary.get("cohort_manifest", ""))).resolve() != mini_path:
        raise ProtocolError("MFA summary is bound to a different mini cohort manifest")
    if file_sha256(mini_path) != summary.get("cohort_manifest_sha256"):
        raise ProtocolError("MFA summary cohort manifest hash mismatch")
    tokens_path = Path(str(summary.get("tokens_path", ""))).resolve()
    if not tokens_path.is_file() or file_sha256(tokens_path) != summary.get("tokens_sha256"):
        raise ProtocolError("MFA token source is missing or changed")
    if not isinstance(clean.get("natural_source"), Mapping) or not isinstance(clean.get("transcripts"), Mapping):
        raise ProtocolError("clean cohort manifest is missing N/transcript maps")
    if mini.get("manifest_type") != "mfa3_clean_cohort_v1" and "records" not in mini:
        raise ProtocolError("unexpected mini cohort manifest schema")
    mini_rows = _records_by_id(mini.get("records", []), label="mini manifest")
    summary_rows = summary.get("results")
    if not isinstance(summary_rows, Mapping):
        raise ProtocolError("MFA summary results must be keyed by sample ID")
    selected = [str(value) for value in (sample_ids if sample_ids is not None else config["sample_ids"])]
    if selected not in (["1"], ["1", "101", "201"]):
        raise ProtocolError("only the fixed formal cohort or sample 1 smoke cohort is allowed")

    legacy_records_path = legacy_root / "records.json"
    legacy_summary_path = legacy_root / "summary.json"
    if not legacy_records_path.is_file() or not legacy_summary_path.is_file():
        raise ProtocolError("legacy natural/MFA evaluation manifest is missing")
    legacy_summary = read_json(legacy_summary_path)
    legacy_payload = read_json(legacy_records_path)
    legacy_records = legacy_payload.get("records", [])
    if not isinstance(legacy_records, list):
        raise ProtocolError("legacy evaluation records are malformed")

    record_rows: list[dict[str, Any]] = []
    for sample_id in selected:
        try:
            mfa_row = summary_rows[sample_id]
            mini_row = mini_rows[sample_id]
            natural_path = Path(str(clean["natural_source"][sample_id])).resolve()
        except (KeyError, TypeError) as exc:
            raise ProtocolError(f"cohort is missing fixed sample {sample_id}") from exc
        paired_key = str(mini_row.get("paired_key", ""))
        speaker_id = str(mini_row.get("speaker_id", ""))
        transcript = str(mini_row.get("transcript", ""))
        if not paired_key or not speaker_id or not transcript:
            raise ProtocolError(f"cohort identity fields are incomplete for sample {sample_id}")
        if str(mfa_row.get("paired_key")) != paired_key or str(mfa_row.get("speaker_id")) != speaker_id:
            raise ProtocolError(f"paired_key/speaker mismatch for sample {sample_id}")
        if str(mini_row.get("audio_path", "")).strip() != str(natural_path):
            raise ProtocolError(f"natural path mismatch between cohort manifests for sample {sample_id}")
        if str(clean["transcripts"].get(sample_id, "")) != transcript:
            raise ProtocolError(f"transcript mismatch between cohort manifests for sample {sample_id}")
        if Path(str(mfa_row.get("audio_path", ""))).resolve().parent != summary_path.parent:
            raise ProtocolError(f"MFA WAV escaped the frozen clean repair directory for sample {sample_id}")
        if str(mfa_row.get("sample_id")) != sample_id or mfa_row.get("exact_natural_length") is not True:
            raise ProtocolError(f"MFA output record is malformed for sample {sample_id}")
        if int(mfa_row.get("natural_samples", -1)) != int(mfa_row.get("output_samples", -2)):
            raise ProtocolError(f"MFA summary N/M sample counts differ for sample {sample_id}")
        if str(mini_row.get("natural_source_sha256", "")) == "":
            raise ProtocolError(f"clean cohort has no natural source hash for sample {sample_id}")

        matching_old = [
            row for row in legacy_records
            if str(row.get("sample_id")) == sample_id
            and str(row.get("speaker_id")) == speaker_id
            and str(row.get("arm")) == "natural_raw"
            and Path(str(row.get("audio", ""))).resolve() == natural_path
        ]
        if len(matching_old) != 1:
            raise ProtocolError(f"legacy evaluation does not uniquely bind the N source by path and speaker for {sample_id}")
        old = matching_old[0]
        if old.get("paired_key") not in (None, "", paired_key):
            raise ProtocolError(f"legacy paired_key disagrees with clean cohort for sample {sample_id}")
        natural_sha = str(mini_row["natural_source_sha256"])
        if str(old.get("audio_sha256", "")) != natural_sha:
            raise ProtocolError(f"legacy natural SHA differs from clean cohort for sample {sample_id}")
        if str(clean["natural_source"].get(sample_id)) != str(natural_path):
            raise ProtocolError(f"sample {sample_id} is not linked through the frozen natural-source map")

        audio = validate_audio_pair(
            natural_path,
            str(mfa_row["audio_path"]),
            natural_sha256=natural_sha,
            mfa_sha256=str(mfa_row.get("audio_sha256", "")),
        )
        if audio["sample_count"] != int(mfa_row["natural_samples"]):
            raise ProtocolError(f"decoded N/M sample count differs from MFA summary for {sample_id}")
        record_rows.append({
            "sample_id": sample_id,
            "paired_key": paired_key,
            "speaker_id": speaker_id,
            "split": str(mini_row.get("split", "")),
            "transcript": transcript,
            "audio": audio,
            "mfa_linear_source": {
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "tokens_path": str(tokens_path),
                "tokens_sha256": file_sha256(tokens_path),
                "mfa_linear_meta": mfa_row.get("mfa_linear_meta"),
            },
            "legacy_natural_binding": {
                "records_path": str(legacy_records_path),
                "records_sha256": file_sha256(legacy_records_path),
                "legacy_summary_path": str(legacy_summary_path),
                "legacy_summary_sha256": file_sha256(legacy_summary_path),
                "paired_key_present": bool(old.get("paired_key")),
                "matched_by": ["sample_id", "speaker_id", "audio_path", "audio_sha256"],
            },
        })

    portraits = audit_portraits(config)
    model_bindings = {
        "wav2lip_checkpoint": path_binding(paths["wav2lip_checkpoint"]),
        "syncnet_model": path_binding(paths["syncnet_model"]),
        "wav2lip_checkpoint_expected_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_expected_sha256": EXPECTED_SYNCNET_SHA256,
    }
    if model_bindings["wav2lip_checkpoint"]["sha256"] != EXPECTED_WAV2LIP_SHA256:
        raise ProtocolError("Wav2Lip checkpoint does not match the frozen model")
    if model_bindings["syncnet_model"]["sha256"] != EXPECTED_SYNCNET_SHA256:
        raise ProtocolError("SyncNet checkpoint does not match the frozen model")
    result = {
        "schema_version": 1,
        "manifest_type": "mfa_linear_video_retiming_frozen_inputs",
        "protocol": config["protocol"],
        "engineering_only": selected == ["1"],
        "sample_ids": selected,
        "records": record_rows,
        "source_bindings": {
            "mfa_summary": path_binding(summary_path),
            "clean_manifest": path_binding(clean_path),
            "mini_manifest": path_binding(mini_path),
            "tokens": path_binding(tokens_path),
        },
        "model_bindings": model_bindings,
        "portrait_bindings": portraits,
        "legacy_evaluation_status": legacy_summary.get("status"),
        "legacy_paired_key_limitation": "legacy records omit paired_key; natural source is matched by paired_key-derived sample identity plus exact speaker, path, and source SHA",
    }
    if output_path is not None:
        write_json(output_path, result, self_hash=True)
        return read_json(output_path)
    return result


def _freeze_inputs_v2(config: Mapping[str, Any], *, output_path: str | Path | None,
                      sample_ids: Sequence[str | int] | None, cohort_dir: str | Path | None) -> dict[str, Any]:
    from .cohort_v2 import SAMPLE_IDS, _verify_completed
    if cohort_dir is None:
        raise ProtocolError("v2 freeze requires an explicit 00_cohort directory")
    directory = Path(cohort_dir).resolve()
    cohort_path = directory / "cohort_v2.json"
    tokens_path = directory / "tokens_v2.json"
    tts_path = directory / "tts_meta_v2.json"
    summary_path = directory / "mfa_summary_v2.json"
    cohort, tokens, tts, summary = (verify_json(path, self_hash=True) for path in
                                    (cohort_path, tokens_path, tts_path, summary_path))
    if cohort.get("source_manifest_sha256") != file_sha256(config["paths"]["strict_source_manifest"]):
        raise ProtocolError("v2 strict source manifest changed")
    if list(cohort.get("sample_ids", [])) != list(SAMPLE_IDS) or len(cohort.get("records", [])) != 64:
        raise ProtocolError("v2 cohort must have the fixed 64 sample IDs")
    if set(tokens.get("records", {})) != set(SAMPLE_IDS) or set(tts.get("results", {})) != set(SAMPLE_IDS):
        raise ProtocolError("v2 token/TTS cohort incomplete")
    _verify_completed(config, cohort, summary)
    selected = [str(value) for value in (sample_ids if sample_ids is not None else SAMPLE_IDS)]
    if selected not in (["1"], list(SAMPLE_IDS)):
        raise ProtocolError("v2 allows only sample 1 smoke or the fixed 64-sample cohort")
    record_rows = []
    for row in cohort["records"]:
        sid = str(row["sample_id"])
        if sid not in selected:
            continue
        side = tokens["records"][sid]
        tts_row = tts["results"][sid]
        m = summary["results"][sid]
        if any(str(side.get(field)) != str(row[field]) for field in ("sample_id", "paired_key", "speaker_id", "split", "transcript")):
            raise ProtocolError(f"v2 token identity mismatch: {sid}")
        if any(str(tts_row.get(field)) != str(row[field]) for field in ("sample_id", "paired_key", "speaker_id", "split", "transcript")):
            raise ProtocolError(f"v2 TTS identity mismatch: {sid}")
        if side["natural"]["audio_sha256"] != row["natural_source_sha256"] or side["tts"]["audio_sha256"] != row["tts_source_sha256"] or tts_row["source_audio_sha256"] != row["tts_source_sha256"]:
            raise ProtocolError(f"v2 audio token/TTS hash mismatch: {sid}")
        if (file_sha256(tts_row["source_audio"]) != row["tts_source_sha256"]
                or file_sha256(tts_row["canonical_16k_audio"]) != tts_row["canonical_audio_sha256"]):
            raise ProtocolError(f"v2 TTS source or canonical conversion changed: {sid}")
        audio = validate_audio_pair(row["audio_path"], m["audio_path"],
                                    natural_sha256=row["natural_source_sha256"], mfa_sha256=m["audio_sha256"])
        record_rows.append({"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"],
                            "split": row["split"], "transcript": row["transcript"], "prior_seen": bool(row["prior_seen"]),
                            "audio": audio, "mfa_linear_source": {"summary_path": str(summary_path),
                                                                     "summary_sha256": file_sha256(summary_path),
                                                                     "tokens_path": str(tokens_path), "tokens_sha256": file_sha256(tokens_path),
                                                                     "generator_bindings": summary["generator_bindings"],
                                                                     "model": summary["model"],
                                                                     "mfa_linear_meta": m.get("mfa_linear_meta")}})
    portraits = audit_portraits(config)
    paths = config["paths"]
    model_bindings = {"wav2lip_checkpoint": path_binding(paths["wav2lip_checkpoint"]),
                      "syncnet_model": path_binding(paths["syncnet_model"]),
                      "wav2lip_checkpoint_expected_sha256": EXPECTED_WAV2LIP_SHA256,
                      "syncnet_model_expected_sha256": EXPECTED_SYNCNET_SHA256}
    if model_bindings["wav2lip_checkpoint"]["sha256"] != EXPECTED_WAV2LIP_SHA256 or model_bindings["syncnet_model"]["sha256"] != EXPECTED_SYNCNET_SHA256:
        raise ProtocolError("v2 frozen model checkpoint changed")
    result = {"schema_version": 2, "manifest_type": "mfa_linear_video_retiming_frozen_inputs",
              "protocol": config["protocol"], "engineering_only": selected == ["1"],
              "sample_ids": selected, "fixed_cohort_sample_ids": list(SAMPLE_IDS), "records": record_rows,
              "source_bindings": {"cohort": path_binding(cohort_path), "tokens": path_binding(tokens_path),
                                  "tts_meta": path_binding(tts_path), "mfa_summary": path_binding(summary_path),
                                  "strict_source": path_binding(paths["strict_source_manifest"])},
              "model_bindings": model_bindings, "portrait_bindings": portraits}
    if output_path is not None:
        return write_json(output_path, result, self_hash=True)
    return result
