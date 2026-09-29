"""Independent listening package and blinded quality analysis."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import quantize_pcm16, read_pcm16, waveform_qc, write_pcm16
from .common import (
    ProtocolError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)

RATINGS_COLUMNS = ("rater_id", "stimulus_id", "quality", "misread", "artifact")


def _listening_copy(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    waveform = np.asarray(values, dtype=np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(waveform * waveform)))
    if rms <= 0.0 or not np.isfinite(rms):
        raise ProtocolError("cannot prepare a silent listening stimulus")
    target_rms = float(10.0 ** (-26.0 / 20.0))
    rms_gain = target_rms / rms
    adjusted = waveform * rms_gain
    peak_before_attenuation = float(np.max(np.abs(adjusted)))
    peak_gain = 1.0
    if peak_before_attenuation >= config.PEAK_LIMIT:
        peak_gain = config.PEAK_LIMIT / peak_before_attenuation
        adjusted = adjusted * peak_gain
    if float(np.max(np.abs(adjusted))) >= 1.0:
        raise ProtocolError("listening copy remains clipped after peak attenuation")
    output = quantize_pcm16(adjusted, scale=32768.0)
    return output, {
        "input_rms": rms,
        "target_rms_dbfs": -26.0,
        "rms_gain": rms_gain,
        "peak_before_attenuation": peak_before_attenuation,
        "peak_gain": peak_gain,
        "output_qc": waveform_qc(output.astype(np.float64) / 32768.0),
        "enters_model_or_syncnet": False,
    }


def _source_audio_rows(tts: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[tuple[str, str], Path]:
    if tts.get("status") != "complete" or targets.get("status") != "complete":
        raise ProtocolError("quality package requires complete TTS and target stages")
    result: dict[tuple[str, str], Path] = {}
    for row in tts.get("rows", []):
        sample_id = str(row["sample_id"])
        for provider in ("LOCAL", "CLOUD"):
            provider_row = row.get(provider)
            if not isinstance(provider_row, Mapping):
                raise ProtocolError(f"missing TTS quality input: {sample_id}/{provider}")
            result[(sample_id, f"T_{provider}")] = Path(str(provider_row["canonical_audio"]))
    for row in targets.get("rows", []):
        sample_id = str(row["sample_id"])
        provider = str(row["provider"])
        if provider not in ("LOCAL", "CLOUD"):
            raise ProtocolError(f"unknown target provider: {provider}")
        result[(sample_id, f"M_{provider}")] = Path(str(row["target_audio"])
        )
    expected = config.EXPECTED_RECORD_COUNT * 4
    if len(result) != expected:
        raise ProtocolError(f"quality input coverage is {len(result)}/{expected}")
    return result


def _balanced_blind_mapping(sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED))
    entries: list[dict[str, Any]] = []
    for stage in ("raw", "target"):
        order = rng.permutation(2)
        for ordinal, sample_id in enumerate(sample_ids):
            if ordinal % 2 == 0:
                position_order = [int(order[0]), int(order[1])]
            else:
                position_order = [int(order[1]), int(order[0])]
            for position, source_index in enumerate(position_order, 1):
                provider = ("LOCAL", "CLOUD")[source_index]
                entries.append({
                    "stimulus_id": f"stimulus_{len(entries) + 1:04d}",
                    "sample_id": sample_id,
                    "stage": stage,
                    "provider": provider,
                    "source_arm": f"{('T' if stage == 'raw' else 'M')}_{provider}",
                    "position": position,
                })
    return entries


def _questionnaire() -> str:
    return """# LRS3 TTS 音质听评问卷

本问卷只评价语音本身的整体听感。每个编号对应一段匿名音频；请不要根据文件名、顺序或猜测来源判断答案。参考文本只用于判断是否漏读、重读或错读。

## 评分

请为每段音频给出“整体听感质量”分数：

1. 很差  2. 较差  3. 一般  4. 较好  5. 很好

评分时综合考虑自然度、清晰度和可闻失真。另填两个独立标记：是否有漏读/重读/错读，是否有明显伪影。标记不参与质量分数加权。

原始语音和目标语音分开完成；同一记录的两种来源在同一场次由同一位评审者完成。不要填写模型名称、来源名称或同步分数。
"""


def run_quality_pack(run_root: Path, cohort: Mapping[str, Any], tts: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    paths.quality.mkdir(parents=True, exist_ok=True)
    existing = paths.quality / "listening_manifest.json"
    if existing.is_file() and (paths.quality / "blind_mapping.json").is_file() and (paths.quality / "quality.json").is_file():
        return verify_self_hashed_json(existing)
    sample_ids = [str(record["sample_id"]) for record in cohort.get("records", [])]
    if cohort.get("status") != "complete" or sample_ids != list(config.EXPECTED_SAMPLE_IDS):
        raise ProtocolError("quality package must use the ordered frozen cohort")
    source_audio = _source_audio_rows(tts, targets)
    mapping = _balanced_blind_mapping(sample_ids)
    public_rows: list[dict[str, Any]] = []
    for entry in mapping:
        source = source_audio[(entry["sample_id"], entry["source_arm"])]
        values, source_meta = read_pcm16(source)
        listening, copy_meta = _listening_copy(values)
        output = paths.quality / "audio" / f"{entry['stimulus_id']}.wav"
        write_pcm16(output, listening)
        public_rows.append({
            "stimulus_id": entry["stimulus_id"],
            "record_ordinal": sample_ids.index(entry["sample_id"]) + 1,
            "session": entry["stage"],
            "path": str(output.resolve()),
            "sha256": file_sha256(output),
            "sample_count": int(listening.size),
            "copy_qc": copy_meta,
            "source_is_experiment_audio": True,
            "enters_model_or_syncnet": False,
            "source_decoded_pcm_sha256": source_meta["decoded_pcm_sha256"],
        })
    write_self_hashed_json(paths.quality / "blind_mapping.json", {
        "schema_version": 1,
        "stage_id": "04_quality",
        "protocol_id": config.PROTOCOL_ID,
        "seed": config.BOOTSTRAP_SEED,
        "private": True,
        "entries": mapping,
    })
    listening_manifest = {
        "schema_version": 1,
        "stage_id": "04_quality",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "stimulus_count": len(public_rows),
        "expected_stimulus_count": config.EXPECTED_QUALITY_STIMULI,
        "blind": True,
        "public_fields_exclude_provider_labels": True,
        "source_audio_is_not_modified": True,
        "listening_copy_policy": "one RMS gain to -26 dBFS, then one peak attenuation if needed",
        "rows": public_rows,
    }
    write_self_hashed_json(paths.quality / "listening_manifest.json", listening_manifest)
    (paths.quality / "questionnaire.md").write_text(_questionnaire(), encoding="utf-8")
    with (paths.quality / "ratings_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATINGS_COLUMNS)
        writer.writeheader()
    quality = {
        "schema_version": 1,
        "stage_id": "04_quality",
        "protocol_id": config.PROTOCOL_ID,
        "status": "QUALITY_NOT_ASSESSED",
        "raw": {"status": "QUALITY_NOT_ASSESSED"},
        "target": {"status": "QUALITY_NOT_ASSESSED"},
        "ratings_present": False,
        "minimum_common_raters": config.MIN_RATERS_PER_PAIR,
        "ratings_enter_model_or_syncnet": False,
        "listening_manifest_sha256": file_sha256(paths.quality / "listening_manifest.json"),
        "blind_mapping_sha256": file_sha256(paths.quality / "blind_mapping.json"),
    }
    write_self_hashed_json(paths.quality / "quality.json", quality)
    return listening_manifest


def _read_ratings(path: Path, stimulus_ids: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"ratings file is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"rater_id", "stimulus_id", "quality"}
        if not required <= fields:
            raise ProtocolError(f"ratings CSV must contain {sorted(required)}")
        if fields & {"provider", "arm", "source", "model", "winner"}:
            raise ProtocolError("ratings CSV must stay blind and cannot contain provider/model labels")
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for number, raw in enumerate(reader, 2):
            rater = str(raw.get("rater_id", "")).strip()
            stimulus = str(raw.get("stimulus_id", "")).strip()
            if not rater or stimulus not in stimulus_ids:
                raise ProtocolError(f"invalid rater/stimulus at CSV line {number}")
            key = (rater, stimulus)
            if key in seen:
                raise ProtocolError(f"duplicate rating at CSV line {number}")
            seen.add(key)
            try:
                quality_value = float(raw.get("quality", ""))
            except ValueError as exc:
                raise ProtocolError(f"quality is not numeric at CSV line {number}") from exc
            if not math.isfinite(quality_value) or quality_value < 1.0 or quality_value > 5.0:
                raise ProtocolError(f"quality must be in [1,5] at CSV line {number}")
            rows.append({"rater_id": rater, "stimulus_id": stimulus, "quality": quality_value, "line": number})
    return rows


def _paired_quality_rows(ratings: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, Any]], stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_stimulus = {str(row["stimulus_id"]): row for row in mapping if str(row["stage"]) == stage}
    by_record: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in ratings:
        entry = by_stimulus.get(str(row["stimulus_id"]))
        if entry is not None:
            by_record[str(entry["sample_id"])][str(row["rater_id"])][str(entry["provider"])] = float(row["quality"])
    expected_ids = sorted({str(row["sample_id"]) for row in mapping if str(row["stage"]) == stage})
    paired: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        raters = by_record.get(sample_id, {})
        common = sorted(rater for rater, values in raters.items() if "LOCAL" in values and "CLOUD" in values)
        if len(common) < config.MIN_RATERS_PER_PAIR:
            missing.append({"sample_id": sample_id, "stage": stage, "common_rater_count": len(common), "required": config.MIN_RATERS_PER_PAIR})
            continue
        differences = [raters[rater]["CLOUD"] - raters[rater]["LOCAL"] for rater in common]
        paired.append({"sample_id": sample_id, "common_raters": common, "common_rater_count": len(common), "difference": float(np.mean(differences)), "rater_differences": differences})
    return paired, missing


def _bootstrap_quality(paired: Sequence[Mapping[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    if not paired:
        return {"mean": None, "ci95": None, "draws": draws, "valid_draws": 0}
    rng = np.random.Generator(np.random.PCG64(seed))
    values: list[float] = []
    for _ in range(draws):
        selected = rng.integers(0, len(paired), size=len(paired))
        record_values: list[float] = []
        for index in selected:
            differences = np.asarray(paired[int(index)].get("rater_differences", []), dtype=np.float64)
            if differences.size:
                rater_indices = rng.integers(0, differences.size, size=differences.size)
                record_values.append(float(np.mean(differences[rater_indices])))
        if record_values:
            values.append(float(np.mean(record_values)))
    point = float(np.mean([float(row["difference"]) for row in paired]))
    interval = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
    return {"mean": point, "ci95": interval, "draws": draws, "valid_draws": len(values), "seed": seed, "unit": "record_then_common_rater"}


def analyze_ratings(run_root: Path, ratings_path: Path) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    listening = verify_self_hashed_json(paths.quality / "listening_manifest.json")
    mapping = verify_self_hashed_json(paths.quality / "blind_mapping.json")
    stimulus_ids = {str(row["stimulus_id"]) for row in listening.get("rows", [])}
    ratings = _read_ratings(ratings_path, stimulus_ids)
    output: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": "04_quality",
        "protocol_id": config.PROTOCOL_ID,
        "ratings_present": True,
        "ratings_sha256": file_sha256(ratings_path),
        "minimum_common_raters": config.MIN_RATERS_PER_PAIR,
        "raw": {},
        "target": {},
        "association_input_is_human_quality": True,
        "ratings_enter_model_or_syncnet": False,
    }
    for stage in ("raw", "target"):
        paired, missing = _paired_quality_rows(ratings, mapping["entries"], stage)
        if len(paired) != config.EXPECTED_RECORD_COUNT or missing:
            output[stage] = {"status": "NOT_ASSESSED", "paired_record_count": len(paired), "missing": missing, "paired": paired}
        else:
            output[stage] = {"status": "ASSESSED", "paired_record_count": len(paired), "missing": [], "paired": paired, "statistics": _bootstrap_quality(paired, config.BOOTSTRAP_DRAWS, config.QUALITY_BOOTSTRAP_SEED)}
    raw_stats = output["raw"].get("statistics", {})
    target_stats = output["target"].get("statistics", {})
    raw_positive = bool(raw_stats.get("ci95") and raw_stats["ci95"][0] > 0.0)
    target_positive = bool(target_stats.get("ci95") and target_stats["ci95"][0] > 0.0)
    if raw_positive and target_positive:
        status = "CLOUD_QUALITY_ADVANTAGE_OBSERVED"
    elif raw_positive:
        status = "QUALITY_ADVANTAGE_UNCONFIRMED_AT_TARGET"
    elif output["raw"].get("status") == "ASSESSED" and output["target"].get("status") == "ASSESSED":
        status = "QUALITY_ORDER_UNRESOLVED"
    else:
        status = "QUALITY_NOT_ASSESSED"
    output["status"] = status
    output["listening_manifest_sha256"] = file_sha256(paths.quality / "listening_manifest.json")
    output["blind_mapping_sha256"] = file_sha256(paths.quality / "blind_mapping.json")
    write_self_hashed_json(paths.quality / "quality.json", output)
    return output


__all__ = ["analyze_ratings", "run_quality_pack"]
