from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SAMPLE_RATE = 16_000
FRAME_STRIDE_SAMPLES = 320
SILENCE_LABELS = {"", "sil", "sp"}
UNKNOWN_LABELS = {"spn", "<unk>", "unk", "unknown", "oov", "<oov>"}
_INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:?\s*xmin\s*=\s*([0-9.eE+-]+)\s*xmax\s*=\s*([0-9.eE+-]+)\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class PhoneToken:
    label: str
    start_s: float
    end_s: float
    silence: bool

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class FrameOwner:
    frame_index: int
    token_index: int
    label: str
    silence: bool
    relative_position: float


@dataclass(frozen=True)
class FrameMapping:
    natural_frame_index: int
    natural_token_index: int
    natural_label: str
    natural_silence: bool
    mapping_type: str
    tts_token_index: int | None
    left_frame_index: int
    right_frame_index: int
    interpolation_alpha: float
    fallback_reason: str | None


def normalize_transcript(text: str) -> str:
    normalized = " ".join(str(text).upper().split())
    if not normalized:
        raise AlignmentError("transcript is empty after normalization")
    return normalized


def transcript_sha256(text: str) -> str:
    return hashlib.sha256(normalize_transcript(text).encode("utf-8")).hexdigest()


def parse_textgrid(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    marker = 'name = "phones"'
    if marker not in raw:
        raise AlignmentError(f"phones tier missing: {path}")
    start = raw.index(marker)
    try:
        end = raw.index("item [", start + len(marker))
    except ValueError:
        end = len(raw)
    section = raw[start:end]
    tokens: list[dict[str, Any]] = []
    for match in _INTERVAL_RE.finditer(section):
        start_s = float(match.group(1))
        end_s = float(match.group(2))
        label = match.group(3).strip()
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise AlignmentError("TextGrid contains an invalid phone interval")
        normalized_label = label.lower()
        if normalized_label in UNKNOWN_LABELS:
            raise AlignmentError(f"unknown/OOV phone is not an allowed silence label: {label!r}")
        tokens.append({
            "token": label,
            "label": normalized_label,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "silence": normalized_label in SILENCE_LABELS,
        })
    validate_tokens(tokens)
    return tokens


def validate_tokens(tokens: Sequence[Mapping[str, Any]]) -> list[PhoneToken]:
    if not tokens:
        raise AlignmentError("phone alignment is empty")
    result: list[PhoneToken] = []
    previous_end = 0.0
    has_speech = False
    for index, token in enumerate(tokens):
        try:
            label = str(token.get("label", token.get("token", ""))).strip().lower()
            start_s = float(token["start_s"])
            end_s = float(token["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AlignmentError(f"malformed token {index}") from exc
        if label in UNKNOWN_LABELS:
            raise AlignmentError(f"unknown/OOV phone at token {index}: {label!r}")
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
            raise AlignmentError(f"invalid token span {index}")
        if index == 0 and abs(start_s) > 1e-6:
            raise AlignmentError("alignment must start at zero")
        if start_s < previous_end - 1e-7:
            raise AlignmentError(f"overlapping token span {index}")
        silence = label in SILENCE_LABELS
        has_speech |= not silence
        result.append(PhoneToken(label, start_s, end_s, silence))
        previous_end = end_s
    if not has_speech:
        raise AlignmentError("alignment contains no speech phone")
    return result


def frame_owners(frame_count: int, tokens: Sequence[Mapping[str, Any]], *, frame_stride_samples: int = FRAME_STRIDE_SAMPLES, sample_rate: int = SAMPLE_RATE) -> list[FrameOwner]:
    if frame_count <= 0 or frame_stride_samples <= 0 or sample_rate <= 0:
        raise AlignmentError("frame count, stride, and sample rate must be positive")
    normalized = validate_tokens(tokens)
    owners: list[FrameOwner] = []
    token_index = 0
    for frame_index in range(frame_count):
        centre = (frame_index + 0.5) * frame_stride_samples / sample_rate
        while token_index + 1 < len(normalized) and centre >= normalized[token_index].end_s - 1e-9:
            token_index += 1
        token = normalized[token_index]
        if not token.start_s - 1e-7 <= centre < token.end_s + 1e-7:
            raise AlignmentError(f"frame {frame_index} has no phone owner")
        relative = min(1.0, max(0.0, (centre - token.start_s) / token.duration_s))
        owners.append(FrameOwner(frame_index, token_index, token.label, token.silence, relative))
    return owners


def parsed_token_hash(tokens: Sequence[Mapping[str, Any]]) -> str:
    import json

    canonical = json.dumps(list(tokens), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _matching_token_map(natural: Sequence[Mapping[str, Any]], tts: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    natural_labels = [str(token.get("label", token.get("token", ""))).lower() for token in natural]
    tts_labels = [str(token.get("label", token.get("token", ""))).lower() for token in tts]
    natural_silence = [label in SILENCE_LABELS for label in natural_labels]
    n_rows = len(natural_labels)
    n_cols = len(tts_labels)
    scores: list[list[tuple[int, int]]] = [
        [(0, 0) for _ in range(n_cols + 1)] for _ in range(n_rows + 1)
    ]
    choices: list[list[str | None]] = [
        [None for _ in range(n_cols + 1)] for _ in range(n_rows + 1)
    ]
    for natural_index in range(n_rows - 1, -1, -1):
        for tts_index in range(n_cols - 1, -1, -1):
            best = scores[natural_index + 1][tts_index]
            choice = "skip_natural"
            skip_tts = scores[natural_index][tts_index + 1]
            if skip_tts > best:
                best = skip_tts
                choice = "skip_tts"
            if natural_labels[natural_index] == tts_labels[tts_index]:
                speech_score = 0 if natural_silence[natural_index] else 1
                match = scores[natural_index + 1][tts_index + 1]
                match = (match[0] + speech_score, match[1] + 1)
                if match >= best:
                    best = match
                    choice = "match"
            scores[natural_index][tts_index] = best
            choices[natural_index][tts_index] = choice

    mapping: dict[int, int] = {}
    natural_index = 0
    tts_index = 0
    while natural_index < n_rows and tts_index < n_cols:
        choice = choices[natural_index][tts_index]
        if choice == "match":
            mapping[natural_index] = tts_index
            natural_index += 1
            tts_index += 1
        elif choice == "skip_tts":
            tts_index += 1
        else:
            natural_index += 1
    return mapping


def _frame_position(time_s: float, frame_count: int, *, frame_stride_samples: int, sample_rate: int) -> float:
    position = time_s * sample_rate / frame_stride_samples - 0.5
    return min(float(frame_count - 1), max(0.0, position))


def build_frame_mapping(
    natural_frame_count: int,
    tts_frame_count: int,
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    *,
    frame_stride_samples: int = FRAME_STRIDE_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[list[FrameMapping], dict[str, int]]:
    natural_owners = frame_owners(natural_frame_count, natural_tokens, frame_stride_samples=frame_stride_samples, sample_rate=sample_rate)
    tts_owners = frame_owners(tts_frame_count, tts_tokens, frame_stride_samples=frame_stride_samples, sample_rate=sample_rate)
    token_map = _matching_token_map(natural_tokens, tts_tokens)
    tts_silence_frames = [owner.frame_index for owner in tts_owners if owner.silence]
    rows: list[FrameMapping] = []
    matched_frames = 0
    silence_fallback_frames = 0
    speech_fallback_frames = 0
    for owner in natural_owners:
        tts_token_index = token_map.get(owner.token_index)
        fallback_reason: str | None = None
        if tts_token_index is not None:
            tts_token = tts_tokens[tts_token_index]
            mapped_time = float(tts_token["start_s"]) + owner.relative_position * (float(tts_token["end_s"]) - float(tts_token["start_s"]))
            position = _frame_position(mapped_time, tts_frame_count, frame_stride_samples=frame_stride_samples, sample_rate=sample_rate)
            mapping_type = "matched_phone"
            matched_frames += 1
        elif owner.silence:
            if not tts_silence_frames:
                raise AlignmentError(f"natural silence frame {owner.frame_index} has no TTS silence frame")
            global_position = owner.frame_index / max(1, natural_frame_count - 1)
            frame_index = min(tts_silence_frames, key=lambda index: (abs(index / max(1, tts_frame_count - 1) - global_position), index))
            position = float(frame_index)
            mapping_type = "silence_fallback"
            fallback_reason = "unmatched_natural_silence"
            silence_fallback_frames += 1
        else:
            speech_fallback_frames += 1
            raise AlignmentError(f"unmatched natural speech frame {owner.frame_index}: {owner.label}")
        left = int(position)
        right = min(tts_frame_count - 1, left + 1)
        rows.append(FrameMapping(owner.frame_index, owner.token_index, owner.label, owner.silence, mapping_type, tts_token_index, left, right, position - left, fallback_reason))
    return rows, {
        "natural_frame_count": natural_frame_count,
        "tts_frame_count": tts_frame_count,
        "matched_frames": matched_frames,
        "silence_fallback_frames": silence_fallback_frames,
        "speech_fallback_frames": speech_fallback_frames,
        "fallback_frames": silence_fallback_frames + speech_fallback_frames,
    }


def trace_rows(mapping: Sequence[FrameMapping]) -> list[dict[str, Any]]:
    return [
        {
            "natural_frame_index": row.natural_frame_index,
            "natural_token_index": row.natural_token_index,
            "natural_label": row.natural_label,
            "natural_silence": row.natural_silence,
            "mapping_type": row.mapping_type,
            "tts_token_index": row.tts_token_index,
            "left_frame_index": row.left_frame_index,
            "right_frame_index": row.right_frame_index,
            "interpolation_alpha": row.interpolation_alpha,
            "fallback_reason": row.fallback_reason,
        }
        for row in mapping
    ]


def build_mfa_command(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dictionary: str = "english_us_mfa",
    acoustic_model: str = "english_mfa",
    mfa_executable: str = "mfa",
) -> list[str]:
    return [
        mfa_executable,
        "align",
        "--clean",
        "--overwrite",
        str(input_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
    ]


def run_mfa_alignment(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    dictionary: str = "english_us_mfa",
    acoustic_model: str = "english_mfa",
    mfa_executable: str = "mfa",
    expected_sample_ids: Sequence[str],
    runner=subprocess.run,
) -> dict[str, Any]:
    command = build_mfa_command(input_dir, output_dir, dictionary=dictionary, acoustic_model=acoustic_model, mfa_executable=mfa_executable)
    if len(expected_sample_ids) != 24 or len(set(str(value) for value in expected_sample_ids)) != 24:
        raise AlignmentError("MFA execution requires the ordered 24-record cohort")
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MFA alignment failed with exit {result.returncode}: {result.stderr[-2000:]}")
    missing = [f"{sample_id}.TextGrid" for sample_id in expected_sample_ids if not (Path(output_dir) / f"{sample_id}.TextGrid").is_file()]
    if missing:
            raise AlignmentError(f"MFA completed without all expected TextGrids: {missing[:5]}")
    return {"command": command, "returncode": int(result.returncode), "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def prepare_mfa_corpus(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    audio_key: str,
    transcript_key: str = "transcript",
    expected_records: Sequence[Mapping[str, Any]],
    audio_hash_key: str,
    sample_rate: int = SAMPLE_RATE,
) -> list[dict[str, Any]]:
    target = Path(output_dir)
    if expected_records is None or len(expected_records) != 24 or len({str(row.get("sample_id", "")) for row in expected_records}) != 24:
        raise AlignmentError("MFA corpus preparation requires the ordered 24-record cohort")
    if not isinstance(audio_hash_key, str) or not audio_hash_key:
        raise AlignmentError("MFA corpus preparation requires an audio hash field")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"MFA corpus directory is non-empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    if expected_records is not None:
        expected_ids = [str(row.get("sample_id", "")) for row in expected_records]
        actual_ids = [str(row.get("sample_id", "")) for row in records]
        if actual_ids != expected_ids:
            raise AlignmentError("MFA corpus records do not match the ordered frozen cohort")
    entries: list[dict[str, Any]] = []
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        audio_value = row.get(audio_key)
        if not sample_id or not isinstance(audio_value, str) or not audio_value:
            raise AlignmentError(f"record has no {audio_key}: {sample_id}")
        audio_path = Path(audio_value)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if audio_hash_key is not None:
            expected_hash = str(row.get(audio_hash_key, ""))
            actual_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            if not expected_hash or actual_hash != expected_hash:
                raise AlignmentError(f"audio hash mismatch for {sample_id}")
        try:
            with wave.open(str(audio_path), "rb") as handle:
                if handle.getframerate() != sample_rate or handle.getnchannels() != 1:
                    raise AlignmentError(f"audio format mismatch for {sample_id}")
        except (wave.Error, OSError) as exc:
            raise AlignmentError(f"audio is not a readable WAV for {sample_id}") from exc
        transcript = normalize_transcript(str(row.get(transcript_key, "")))
        destination = target / f"{sample_id}.wav"
        shutil.copy2(audio_path, destination)
        lab_path = target / f"{sample_id}.lab"
        lab_path.write_text(transcript + "\n", encoding="utf-8")
        entries.append({"sample_id": sample_id, "audio_path": str(destination), "lab_path": str(lab_path), "transcript": transcript, "transcript_sha256": transcript_sha256(transcript)})
    return entries


def extend_final_token_for_feature_tail(tokens: Sequence[Mapping[str, Any]], frame_count: int, *, frame_stride_samples: int = FRAME_STRIDE_SAMPLES, sample_rate: int = SAMPLE_RATE, max_extension_s: float | None = None) -> tuple[list[dict[str, Any]], float]:
    normalized = validate_tokens(tokens)
    if frame_count <= 0:
        raise AlignmentError("frame count must be positive")
    required_end = (frame_count - 0.5) * frame_stride_samples / sample_rate
    extension = max(0.0, required_end - normalized[-1].end_s)
    limit = frame_stride_samples / sample_rate if max_extension_s is None else max_extension_s
    if extension > limit + 1e-7:
        raise AlignmentError(f"feature-tail extension exceeds bound: {extension:.9f}s")
    extended = [
        {"token": token.label, "label": token.label, "start_s": token.start_s, "end_s": token.end_s, "duration_s": token.duration_s, "silence": token.silence}
        for token in normalized
    ]
    if extension:
        extended[-1]["end_s"] = required_end
        extended[-1]["duration_s"] = required_end - normalized[-1].start_s
    return extended, extension
