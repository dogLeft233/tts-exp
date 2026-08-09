#!/usr/bin/env python3
"""Build a stable-ID, two-arm MFA manifest for MDC English pairs."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import validate_manifest
from prepare_mdc_english_pairs import EXPECTED_SAMPLE_IDS, sha256_file

INTERVAL_START_RE = re.compile(r"intervals\s*\[\s*(\d+)\s*\]\s*:", re.IGNORECASE)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FIELD_RE = {
    "xmin": re.compile(r"^\s*xmin\s*=\s*([^\s]+)\s*$", re.MULTILINE | re.IGNORECASE),
    "xmax": re.compile(r"^\s*xmax\s*=\s*([^\s]+)\s*$", re.MULTILINE | re.IGNORECASE),
    "text": re.compile(r'^\s*text\s*=\s*"((?:\\.|[^"\\])*)"\s*$', re.MULTILINE),
}
SILENCE_LABELS = frozenset({"", "sil", "sp", "spn", "<eps>", "<sil>", "h#"})
NOISE_LABELS = frozenset({
    "noise", "<noise>", "[noise]", "nsn", "br", "laugh", "<laugh>",
    "vocalization", "<vocalization>", "cough", "<cough>", "click", "<click>",
})
UNKNOWN_LABELS = frozenset({"unk", "<unk>", "unknown", "<unknown>"})

# English MFA output can use IPA and X-SAMPA-like diphthongs.
PHONE_TO_VISEME = {
    "p": "pbmv", "b": "pbmv", "m": "pbmv",
    "f": "fv", "v": "fv", "θ": "th", "ð": "th",
    "t": "cdsz", "d": "cdsz", "s": "cdsz", "z": "cdsz",
    "n": "cdsz", "l": "cdsz", "ɫ": "cdsz", "ɾ": "cdsz",
    "k": "kg", "g": "kg", "ɡ": "kg", "ŋ": "kg", "ɲ": "kg",
    "ʃ": "chjsh", "ʒ": "chjsh", "tʃ": "chjsh", "dʒ": "chjsh",
    "ɛ": "e", "æ": "e", "ʌ": "e", "ə": "e", "ɐ": "e", "a": "e", "ɜ": "e",
    "ɑ": "o", "ɒ": "o", "ɔ": "o", "o": "o", "ʊ": "o",
    "i": "i", "ɪ": "i", "e": "i", "u": "u", "ʉ": "u", "w": "u",
    "ɹ": "r", "ɚ": "r", "ɝ": "r", "aɪ": "ai", "eɪ": "ai",
    "aʊ": "aw", "oʊ": "o", "ɔɪ": "o", "h": "e", "ɦ": "e",
    "ʔ": "e", "j": "i", "ʎ": "i",
    "aj": "ai", "ej": "ai", "aw": "aw", "ow": "o", "oj": "o",
}
DIACRITICS = re.compile(r"[̪̺̻̼̟̠̬ʰʲːˑ̩ʷˠˤ̃ˈˌ]")


def normalize_phone(phone: str) -> str:
    """Normalize case and MFA diacritics without changing IPA symbols."""
    return DIACRITICS.sub("", phone.strip()).casefold()


def _label_kind(phone: str) -> str:
    normalized = normalize_phone(phone)
    if normalized in SILENCE_LABELS:
        return "silence"
    if normalized in NOISE_LABELS:
        return "noise"
    if normalized in UNKNOWN_LABELS:
        return "unknown"
    return "speech"


def phone_to_viseme(phone: str) -> str:
    normalized = normalize_phone(phone)
    if _label_kind(normalized) != "speech":
        return "sil"
    return PHONE_TO_VISEME.get(normalized, "other")


def _parse_float(value: str, path: Path, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field} value in {path}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {field} value in {path}: {value!r}")
    return parsed


def _field(block: str, name: str, path: Path, index: int) -> str:
    matches = FIELD_RE[name].findall(block)
    if len(matches) != 1:
        raise ValueError(f"interval {index} has malformed {name} field in {path}")
    return matches[0]


def parse_textgrid_phones(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    marker = re.search(r'name\s*=\s*"phones"', raw, re.IGNORECASE)
    if marker is None:
        raise ValueError(f"phones tier missing: {path}")
    body = raw[marker.end():]
    next_tier = re.search(r"\n\s*item\s*\[\s*\d+\s*\]\s*:", body, re.IGNORECASE)
    if next_tier:
        body = body[:next_tier.start()]

    first_interval = INTERVAL_START_RE.search(body)
    if first_interval is None:
        raise ValueError(f"phones tier has no intervals: {path}")
    prefix = body[:first_interval.start()]
    tier_xmins = re.findall(r"^\s*xmin\s*=\s*([^\s]+)\s*$", prefix, re.MULTILINE | re.IGNORECASE)
    tier_xmaxs = re.findall(r"^\s*xmax\s*=\s*([^\s]+)\s*$", prefix, re.MULTILINE | re.IGNORECASE)
    if len(tier_xmins) != 1 or len(tier_xmaxs) != 1:
        raise ValueError(f"phones tier bounds missing or duplicated: {path}")
    tier_start = _parse_float(tier_xmins[0], path, "tier xmin")
    tier_end = _parse_float(tier_xmaxs[0], path, "tier xmax")
    if tier_end < tier_start:
        raise ValueError(f"phones tier bounds are reversed: {path}")

    size_match = re.search(r"intervals\s*:\s*size\s*=\s*(\d+)", body, re.IGNORECASE)
    declared_size = int(size_match.group(1)) if size_match else None
    starts = list(INTERVAL_START_RE.finditer(body))
    if declared_size is not None and declared_size != len(starts):
        raise ValueError(
            f"phones interval count mismatch in {path}: declared {declared_size}, found {len(starts)}"
        )

    tokens: list[dict[str, Any]] = []
    for position, match in enumerate(starts, start=1):
        index = int(match.group(1))
        if index != position:
            raise ValueError(f"phones interval indices are not contiguous in {path}")
        end_of_block = starts[position].start() if position < len(starts) else len(body)
        block = body[match.end():end_of_block]
        start = _parse_float(_field(block, "xmin", path, index), path, "xmin")
        end = _parse_float(_field(block, "xmax", path, index), path, "xmax")
        raw_phone = _field(block, "text", path, index).replace(r'\"', '"').replace(r"\\", "\\")
        kind = _label_kind(raw_phone)
        if end < start:
            raise ValueError(f"negative phone interval in {path}: {raw_phone!r}")
        if start < tier_start - 1e-6 or end > tier_end + 1e-6:
            raise ValueError(f"phone interval exceeds phones tier bounds in {path}: {raw_phone!r}")
        if end == start and kind == "speech":
            raise ValueError(f"zero-duration speech interval in {path}: {raw_phone!r}")
        tokens.append(
            {
                "token": raw_phone,
                "phoneme": normalize_phone(raw_phone),
                "viseme": phone_to_viseme(raw_phone),
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "duration_s": round(end - start, 6),
                "is_silence": kind == "silence",
                "is_noise": kind == "noise",
                "is_unknown": kind == "unknown",
                "is_non_speech": kind != "speech",
                "alignment_confidence": 1.0,
            }
        )
    if not tokens:
        raise ValueError(f"phones tier has no intervals: {path}")
    return tokens


def validate_tokens(tokens: list[dict[str, Any]], audio_path: Path) -> dict[str, Any]:
    info = sf.info(audio_path)
    audio_duration = float(info.duration)
    previous_end = 0.0
    for index, token in enumerate(tokens):
        try:
            start = float(token["start_s"])
            end = float(token["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed token {index} in {audio_path}") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"non-finite phone interval in {audio_path}")
        if start < -1e-6 or end < start - 1e-6:
            raise ValueError(f"invalid phone interval in {audio_path}")
        if index == 0 and start > 1e-5:
            raise ValueError(f"phone intervals do not start at zero in {audio_path}")
        if start < previous_end - 1e-5:
            raise ValueError(f"overlapping phone intervals in {audio_path}")
        if start > previous_end + 1e-5:
            raise ValueError(f"gap between phone intervals in {audio_path}")
        if end > audio_duration + 1e-5:
            raise ValueError(
                f"phone interval exceeds audio duration in {audio_path}: {end:.3f} > {audio_duration:.3f}"
            )
        if end <= start + 1e-8 and not bool(token.get("is_non_speech", False)):
            raise ValueError(f"zero-duration speech interval in {audio_path}")
        previous_end = end
    if previous_end < audio_duration - 1e-5:
        raise ValueError(
            f"phone intervals do not cover audio duration in {audio_path}: "
            f"{previous_end:.3f} < {audio_duration:.3f}"
        )
    non_speech = [token for token in tokens if bool(token.get("is_non_speech", token.get("is_silence", False)))]
    speech = [token for token in tokens if token not in non_speech]
    if not speech:
        raise ValueError(f"alignment contains only non-speech intervals: {audio_path}")
    return {
        "duration_s": round(audio_duration, 6),
        "token_count": len(tokens),
        "phone_count": len(speech),
        "silence_count": sum(bool(token.get("is_silence")) for token in tokens),
        "noise_count": sum(bool(token.get("is_noise")) for token in tokens),
        "unknown_count": sum(bool(token.get("is_unknown")) for token in tokens),
        "non_speech_count": len(non_speech),
        "other_count": sum(token.get("viseme") == "other" for token in speech),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _records_by_key(records: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"invalid {label} records")
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict) or not row.get("sample_key"):
            raise ValueError(f"invalid {label} record")
        if "paired_key" not in row:
            raise ValueError(f"{label} paired_key missing")
        key = str(row["sample_key"])
        if key in result:
            raise ValueError(f"duplicate {label} sample_key: {key}")
        if str(row["paired_key"]) != key:
            raise ValueError(f"{label} paired_key mismatch for {key}")
        result[key] = row
    return result


def _load_tts_results(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    results = payload.get("results", {})
    if isinstance(results, list):
        return _records_by_key(results, "TTS")
    if isinstance(results, dict):
        rows = []
        for key, value in results.items():
            if not isinstance(value, dict):
                raise ValueError(f"invalid TTS record: {key}")
            row = dict(value)
            if row.get("sample_key", str(key)) != str(key):
                raise ValueError(f"TTS result key mismatch for {key}")
            row.setdefault("sample_key", str(key))
            rows.append(row)
        return _records_by_key(rows, "TTS")
    raise ValueError(f"invalid TTS results: {path}")


def _repo_path(repo: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {label} path")
    root = repo.resolve()
    path = Path(value)
    resolved = (path if path.is_absolute() else repo / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository: {value}") from exc
    return resolved


def _relative_path(repo: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _validate_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing {label} SHA-256 for {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def _validate_source_provenance(
    repo: Path,
    pair_manifest: dict[str, Any],
) -> dict[str, Any]:
    source_manifest_value = pair_manifest.get("source_manifest")
    if not isinstance(source_manifest_value, str) or not source_manifest_value.strip():
        raise ValueError("pair manifest source manifest provenance is missing")
    source_path = _repo_path(repo, source_manifest_value, "source manifest")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    _validate_hash(
        source_path,
        pair_manifest.get("source_manifest_sha256"),
        "source manifest",
    )
    source_payload = _load_json(source_path)
    source_samples = source_payload.get("samples")
    if not isinstance(source_samples, list):
        raise ValueError(f"source manifest samples missing: {source_path}")
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_samples:
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"invalid source manifest row: {source_path}")
        sample_id = str(row["sample_id"])
        if sample_id in source_by_id:
            raise ValueError(f"duplicate source manifest sample_id: {sample_id}")
        source_by_id[sample_id] = row
    pair_records = pair_manifest.get("records")
    if not isinstance(pair_records, list):
        raise ValueError("pair manifest records missing")
    for pair in pair_records:
        if not isinstance(pair, dict):
            raise ValueError("invalid pair record")
        sample_key = str(pair.get("sample_key", ""))
        if not sample_key:
            raise ValueError("pair sample_key missing")
        source_sample_id = pair.get("source_sample_id")
        if not isinstance(source_sample_id, str) or not source_sample_id.strip():
            raise ValueError(f"pair source_sample_id missing for {sample_key}")
        if source_sample_id != sample_key:
            raise ValueError(f"pair source_sample_id mismatch for {sample_key}")
        source_row = source_by_id.get(source_sample_id)
        if source_row is None:
            raise ValueError(f"source manifest row missing for {sample_key}")
        if str(source_row.get("language_code")) != "en":
            raise ValueError(f"source manifest language mismatch for {sample_key}")
        pair_source_path = _repo_path(
            repo, pair.get("source_audio_path"), f"pair source audio for {sample_key}"
        )
        manifest_source_path = _repo_path(
            repo, source_row.get("file_path"), f"source manifest audio for {sample_key}"
        )
        if pair_source_path != manifest_source_path:
            raise ValueError(f"source audio path mismatch for {sample_key}")
        if not pair_source_path.is_file():
            raise FileNotFoundError(pair_source_path)
        source_hash = _validate_hash(
            pair_source_path, source_row.get("sha256"), "source audio"
        )
        if pair.get("source_audio_sha256") != source_hash:
            raise ValueError(f"pair source audio hash mismatch for {sample_key}")
        if pair.get("text") != source_row.get("text"):
            raise ValueError(f"source text mismatch for {sample_key}")
    return source_payload


def _clear_mfa_staging(arm_dir: Path) -> None:
    if arm_dir.is_symlink():
        raise ValueError(f"MFA staging arm must not be a symlink: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    for child in arm_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            raise ValueError(f"unexpected directory in MFA staging: {child}")
        child.unlink()


def _resolve_license(repo: Path, pair_manifest: dict[str, Any]) -> str:
    license_value = pair_manifest.get("license")
    source_manifest = pair_manifest.get("source_manifest")
    if not license_value and source_manifest:
        source_path = _repo_path(repo, source_manifest, "source manifest")
        source_payload = _load_json(source_path)
        language_meta = source_payload.get("languages", {}).get("en", {})
        if isinstance(language_meta, dict):
            license_value = language_meta.get("license")
    if not isinstance(license_value, str) or not license_value.strip():
        raise ValueError("MDC English license provenance is missing")
    return license_value


def _validate_tts_provenance(
    repo: Path,
    pair_manifest_path: Path,
    pair_manifest: dict[str, Any],
    tts_payload: dict[str, Any],
    tts_records: dict[str, dict[str, Any]],
) -> None:
    expected_hash = sha256_file(pair_manifest_path)
    if tts_payload.get("source_pair_manifest_sha256") != expected_hash:
        raise ValueError("TTS source pair-manifest SHA-256 does not match pair manifest")
    declared_path = tts_payload.get("source_pair_manifest")
    if declared_path is not None and _repo_path(repo, declared_path, "TTS source pair manifest") != pair_manifest_path.resolve():
        raise ValueError("TTS source pair-manifest path does not match pair manifest")
    expected_dataset = pair_manifest.get("dataset", "mdc_tts")
    if tts_payload.get("dataset") != expected_dataset:
        raise ValueError("TTS dataset does not match pair manifest")
    if tts_payload.get("language_code") != "en":
        raise ValueError("TTS language_code must be en")
    pair_records = _records_by_key(pair_manifest.get("records"), "pair")
    pair_keys = set(pair_records)
    tts_keys = set(tts_records)
    if tts_keys != pair_keys:
        missing = sorted(pair_keys - tts_keys)
        extra = sorted(tts_keys - pair_keys)
        raise ValueError(f"TTS/pair sample-key mismatch (missing={missing}, extra={extra})")
    provider = tts_payload.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("TTS provider provenance is missing")
    for sample_key, row in tts_records.items():
        if row.get("provider") != provider:
            raise ValueError(f"TTS provider mismatch for {sample_key}")
        if row.get("dataset", expected_dataset) != expected_dataset:
            raise ValueError(f"TTS dataset mismatch for {sample_key}")
        if row.get("paired_key") != sample_key:
            raise ValueError(f"TTS paired_key mismatch for {sample_key}")


def prepare_mfa_inputs(
    repo: Path, pair_manifest_path: Path, tts_meta_path: Path, output_dir: Path,
    sample_ids: list[str],
) -> None:
    pair_manifest = _load_json(pair_manifest_path)
    pair_records = _records_by_key(pair_manifest.get("records"), "pair")
    tts_payload = _load_json(tts_meta_path)
    tts_records = _load_tts_results(tts_meta_path)
    _validate_source_provenance(repo, pair_manifest)
    _validate_tts_provenance(repo, pair_manifest_path, pair_manifest, tts_payload, tts_records)
    for condition in ("natural", "tts"):
        _clear_mfa_staging(output_dir / condition)
    expected_provider = str(tts_payload["provider"])
    sample_ids = _validate_sample_ids(sample_ids)
    _validate_pair_inventory(pair_records, sample_ids)
    for sample_key in sample_ids:
        if sample_key not in pair_records:
            raise ValueError(f"pair record missing for {sample_key}")
        pair = pair_records[sample_key]
        text = str(pair.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty pair text for {sample_key}")
        for condition in ("natural", "tts"):
            audio_path, _ = _audio_for_condition(
                repo, pair, tts_records.get(sample_key), condition, expected_provider
            )
            arm_dir = output_dir / condition
            link = arm_dir / f"{sample_key}.wav"
            link.symlink_to(audio_path.resolve())
            (arm_dir / f"{sample_key}.lab").write_text(text + "\n", encoding="utf-8")


def run_mfa(input_dir: Path, output_dir: Path, dictionary: str, acoustic: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["mfa", "align", "--clean", "--overwrite", str(input_dir), dictionary, acoustic, str(output_dir)],
        check=True,
    )


def _validate_sample_ids(sample_ids: list[str]) -> list[str]:
    normalized = [str(value) for value in sample_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate sample IDs requested")
    if any(value not in EXPECTED_SAMPLE_IDS for value in normalized):
        raise ValueError("MDC English IDs must be exactly en_001 through en_050")
    return normalized


def _validate_pair_inventory(
    pair_records: dict[str, dict[str, Any]], sample_ids: list[str]
) -> None:
    pair_keys = set(pair_records)
    requested_keys = set(sample_ids)
    missing = sorted(requested_keys - pair_keys)
    if missing:
        raise ValueError(f"pair records missing requested sample IDs: {missing}")
    if requested_keys == set(EXPECTED_SAMPLE_IDS) and pair_keys != requested_keys:
        extra = sorted(pair_keys - requested_keys)
        raise ValueError(f"full MDC pair inventory has unexpected sample IDs: {extra}")


def _audio_for_condition(
    repo: Path,
    pair: dict[str, Any],
    tts: dict[str, Any] | None,
    condition: str,
    expected_provider: str | None = None,
) -> tuple[Path, str]:
    sample_key = str(pair["sample_key"])
    natural_path = _repo_path(repo, pair.get("audio_path"), f"natural audio for {sample_key}")
    if not natural_path.is_file():
        raise FileNotFoundError(natural_path)
    natural_hash = _validate_hash(
        natural_path, pair.get("canonical_qc", {}).get("sha256"), "natural audio"
    )
    if condition == "natural":
        return natural_path, str(pair["text"]).strip()
    if not tts or tts.get("status") != "ok":
        raise ValueError(f"TTS output missing or failed for {sample_key}")
    if tts.get("condition") != "tts":
        raise ValueError(f"TTS condition mismatch for {sample_key}")
    if tts.get("paired_key") != sample_key:
        raise ValueError(f"TTS paired_key mismatch for {sample_key}")
    if str(tts.get("text", "")).strip() != str(pair["text"]).strip():
        raise ValueError(f"TTS text mismatch for {sample_key}")
    if str(tts.get("utterance_id", "")) != f"mdc_tts/{sample_key}/tts":
        raise ValueError(f"TTS utterance_id mismatch for {sample_key}")
    tts_provider = tts.get("provider")
    if not isinstance(tts_provider, str) or not tts_provider.strip():
        raise ValueError(f"TTS provider provenance missing for {sample_key}")
    if expected_provider is not None and tts_provider != expected_provider:
        raise ValueError(f"TTS provider mismatch for {sample_key}")
    if str(tts.get("reference_audio", "")):
        reference_path = _repo_path(repo, tts["reference_audio"], f"TTS reference audio for {sample_key}")
        if reference_path != natural_path:
            raise ValueError(f"TTS reference audio mismatch for {sample_key}")
    if tts.get("reference_sha256") != natural_hash:
        raise ValueError(f"TTS reference audio hash mismatch for {sample_key}")
    generated_path = _repo_path(repo, tts.get("generated_audio"), f"TTS generated audio for {sample_key}")
    if generated_path == natural_path:
        raise ValueError(f"TTS generated audio aliases natural audio for {sample_key}")
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)
    generated_hash = _validate_hash(
        generated_path, tts.get("canonical_qc", {}).get("sha256"), "TTS generated audio"
    )
    try:
        if generated_path.stat().st_dev == natural_path.stat().st_dev and generated_path.stat().st_ino == natural_path.stat().st_ino:
            raise ValueError(f"TTS generated audio shares inode with natural audio for {sample_key}")
    except FileNotFoundError:
        raise
    if generated_hash == natural_hash:
        raise ValueError(f"TTS generated audio has identical content to natural audio for {sample_key}")
    return generated_path, str(pair["text"]).strip()


def build_alignment_manifest(
    repo: Path,
    pair_manifest_path: Path,
    tts_meta_path: Path,
    natural_tg_dir: Path,
    tts_tg_dir: Path,
    output_path: Path,
    sample_ids: list[str],
) -> dict[str, Any]:
    sample_ids = _validate_sample_ids(sample_ids)
    pair_manifest = _load_json(pair_manifest_path)
    pair_records = _records_by_key(pair_manifest.get("records"), "pair")
    _validate_pair_inventory(pair_records, sample_ids)
    tts_payload = _load_json(tts_meta_path)
    tts_records = _load_tts_results(tts_meta_path)
    _validate_source_provenance(repo, pair_manifest)
    _validate_tts_provenance(repo, pair_manifest_path, pair_manifest, tts_payload, tts_records)
    license_value = _resolve_license(repo, pair_manifest)
    expected_provider = str(tts_payload["provider"])
    output_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample_key in sample_ids:
        pair = pair_records.get(sample_key)
        if pair is None:
            failures.extend({"sample_key": sample_key, "condition": condition, "error": "pair missing"} for condition in ("natural", "tts"))
            continue
        if pair.get("condition") != "natural" or str(pair.get("paired_key")) != sample_key:
            raise ValueError(f"pair metadata mismatch for {sample_key}")
        for condition, tg_dir in (("natural", natural_tg_dir), ("tts", tts_tg_dir)):
            try:
                tg_path = tg_dir / f"{sample_key}.TextGrid"
                tts = tts_records.get(sample_key)
                audio_path, transcript = _audio_for_condition(
                    repo, pair, tts, condition, expected_provider
                )
                if not tg_path.is_file():
                    raise FileNotFoundError(tg_path)
                tokens = parse_textgrid_phones(tg_path)
                qc = validate_tokens(tokens, audio_path)
                if condition == "tts":
                    if tts is None:
                        raise ValueError(f"TTS output missing or failed for {sample_key}")
                    speaker_id = f"tts:{tts['provider']}"
                else:
                    speaker_id = pair.get("speaker_id")
                output_records.append(
                    {
                        "schema_version": 1,
                        "dataset": "mdc_tts",
                        "sample_id": sample_key,
                        "sample_key": sample_key,
                        "paired_key": sample_key,
                        "utterance_id": f"mdc_tts/{sample_key}/{condition}",
                        "condition": condition,
                        "variant": "raw",
                        "tts_provider": tts.get("provider") if condition == "tts" and tts else None,
                        "speaker_id": speaker_id,
                        "speaker_scope": "provider_condition" if condition == "tts" else "source_author",
                        "speaker_semantics": "provider_label_not_verified_voice_identity" if condition == "tts" else "source_author_identity",
                        "reference_speaker_id": tts.get("reference_speaker_id") if condition == "tts" and tts else None,
                        "generation_mode": tts.get("generation_mode") if condition == "tts" and tts else None,
                        "source_author_id": pair.get("source_author_id"),
                        "transcript": transcript,
                        "text": transcript,
                        "audio_path": _relative_path(repo, audio_path),
                        "split": str(pair.get("split", "analysis")),
                        "license": license_value,
                        "alignment_source": "mfa",
                        "alignment_file": _relative_path(repo, tg_path),
                        "duration_s": qc["duration_s"],
                        "alignment_qc": qc,
                        "tokens": tokens,
                    }
                )
            except (OSError, ValueError, KeyError) as exc:
                failures.append({"sample_key": sample_key, "condition": condition, "error": str(exc)})
    validate_manifest(output_records)
    expected = len(sample_ids) * 2
    result = {
        "schema_version": 1,
        "dataset": "mdc_tts",
        "language_code": "en",
        "sample_keys": sample_ids,
        "entries_expected": expected,
        "entries_written": len(output_records),
        "complete": len(output_records) == expected and not failures,
        "failures": failures,
        "records": output_records,
        "manifest": output_records,
        "source_pair_manifest": _relative_path(repo, pair_manifest_path),
        "source_pair_manifest_sha256": sha256_file(pair_manifest_path),
        "tts_meta": _relative_path(repo, tts_meta_path),
        "tts_meta_sha256": sha256_file(tts_meta_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-manifest", default="")
    parser.add_argument("--tts-meta", default="")
    parser.add_argument("--natural-tg", default="")
    parser.add_argument("--tts-tg", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--prepare-mfa-input", action="store_true")
    parser.add_argument("--run-mfa", action="store_true")
    parser.add_argument("--dictionary", default="english_mfa")
    parser.add_argument("--acoustic", default="english_mfa")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        parser.error("run-id must contain only letters, digits, '.', '_' or '-'")
    sample_ids = [value.strip() for value in args.sample_ids.split(",") if value.strip()] or list(EXPECTED_SAMPLE_IDS)
    pair_manifest = _repo_path(
        repo,
        args.pair_manifest or f"runs/{args.run_id}/00_pairs/pair_manifest.json",
        "pair manifest",
    )
    tts_meta = _repo_path(
        repo,
        args.tts_meta or f"runs/{args.run_id}/02_tts/tts_meta.json",
        "TTS metadata",
    )
    align_root = _repo_path(repo, f"runs/{args.run_id}/03_alignment", "alignment root")
    natural_tg = _repo_path(
        repo, args.natural_tg or str(align_root / "mfa_out" / "natural"), "natural TextGrid directory"
    )
    tts_tg = _repo_path(
        repo, args.tts_tg or str(align_root / "mfa_out" / "tts"), "TTS TextGrid directory"
    )
    if args.prepare_mfa_input:
        prepare_mfa_inputs(repo, pair_manifest, tts_meta, align_root / "mfa_input", sample_ids)
        if not args.run_mfa:
            return 0
    if args.run_mfa:
        run_mfa(align_root / "mfa_input" / "natural", natural_tg, args.dictionary, args.acoustic)
        run_mfa(align_root / "mfa_input" / "tts", tts_tg, args.dictionary, args.acoustic)
    output = _repo_path(
        repo,
        args.output or f"runs/{args.run_id}/03_alignment/alignment.json",
        "alignment output",
    )
    result = build_alignment_manifest(repo, pair_manifest, tts_meta, natural_tg, tts_tg, output, sample_ids)
    print(f"[mdc-en-align] {result['entries_written']}/{result['entries_expected']} entries; complete={result['complete']}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
