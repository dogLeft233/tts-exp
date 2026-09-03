"""Transcript normalization, LRS3 timing parsing, and deterministic word alignment."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NORMALIZER_VERSION = "nfkc_ascii_apostrophe_upper_az_apostrophe_v2"
_NONLEXICAL_TOKENS = ("{LG}", "{LAUGHTER}", "[LAUGHTER]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'").upper()
    for token in _NONLEXICAL_TOKENS:
        text = text.replace(token, " ")
    text = re.sub(r"[^A-Z']", " ", text)
    return " ".join(text.split())


def normalized_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def parse_lrs3_txt(path_or_text: str | Path) -> dict[str, Any]:
    text = Path(path_or_text).read_text(encoding="utf-8") if isinstance(path_or_text, Path) else path_or_text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text_line = next((line[5:].strip() for line in lines if line.upper().startswith("TEXT:")), None)
    conf_line = next((line[5:].strip() for line in lines if line.upper().startswith("CONF:")), None)
    header_index = next((i for i, line in enumerate(lines) if line.upper().startswith("WORD START END")), None)
    if text_line is None or header_index is None:
        raise ValueError("LRS3 txt is missing Text: or WORD START END header")
    rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    previous_end = 0.0
    for line in lines[header_index + 1 :]:
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            start, end = float(fields[1]), float(fields[2])
            score = float(fields[3]) if len(fields) > 3 else None
        except ValueError as exc:
            raise ValueError(f"invalid LRS3 timing row: {line}") from exc
        if not (0 <= start < end) or start < previous_end:
            raise ValueError(f"invalid or non-monotonic timing row: {line}")
        previous_end = end
        word = normalize_text(fields[0])
        if not word:
            ignored_rows.append({"raw_word": fields[0], "start_s": start, "end_s": end, "reason": "empty_after_normalization"})
            continue
        rows.append({"word": word, "raw_word": fields[0], "start_s": start, "end_s": end, "ascore": score})
    if not rows:
        raise ValueError("LRS3 txt has no timing rows")
    return {
        "text": normalize_text(text_line),
        "confidence": float(conf_line) if conf_line is not None else None,
        "words": rows,
        "ignored_words": ignored_rows,
        "normalizer_version": NORMALIZER_VERSION,
    }


def validate_official_timing(parsed: Mapping[str, Any], transcript: str) -> None:
    expected = normalized_words(transcript)
    actual = [str(row["word"]) for row in parsed["words"]]
    if parsed.get("text") != normalize_text(transcript):
        raise ValueError("Text: line disagrees with canonical transcript")
    if actual != expected:
        raise ValueError("official timing words disagree with canonical transcript")


def _span_from_timing(span: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if span is None:
        return None
    start = float(span["start_s"])
    end = float(span["end_s"])
    if not (0 <= start < end):
        raise ValueError("word span must be finite, nonnegative, and half-open")
    return start, end


def align_words(reference: Sequence[str], prediction: Sequence[str]) -> list[dict[str, Any]]:
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            equal_cost = dp[i - 1][j - 1] if reference[i - 1] == prediction[j - 1] else dp[i - 1][j - 1] + 1
            dp[i][j] = min(equal_cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    reversed_ops: list[dict[str, Any]] = []
    i, j = n, m
    while i or j:
        if i and j and reference[i - 1] == prediction[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            operation = "equal"
            i -= 1
            j -= 1
            ref_index, pred_index = i, j
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            operation = "substitution"
            i -= 1
            j -= 1
            ref_index, pred_index = i, j
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            operation = "deletion"
            i -= 1
            ref_index, pred_index = i, None
        elif j and dp[i][j] == dp[i][j - 1] + 1:
            operation = "insertion"
            j -= 1
            ref_index, pred_index = None, j
        else:
            raise RuntimeError("edit-distance backtrace lost a minimum-cost predecessor")
        reversed_ops.append({
            "operation_id": "",
            "operation": operation,
            "reference_index": ref_index,
            "prediction_index": pred_index,
            "reference_word": reference[ref_index] if ref_index is not None else None,
            "prediction_word": prediction[pred_index] if pred_index is not None else None,
        })
    operations = list(reversed(reversed_ops))
    for index, operation in enumerate(operations):
        operation["operation_id"] = f"op_{index:06d}"
    return operations


def _union_span(spans: Iterable[tuple[float, float] | None]) -> tuple[float, float] | None:
    values = [span for span in spans if span is not None]
    if not values:
        return None
    return min(span[0] for span in values), max(span[1] for span in values)


def assign_error_spans(
    operations: Sequence[Mapping[str, Any]],
    reference_spans: Sequence[Mapping[str, Any] | None],
    prediction_spans: Sequence[Mapping[str, Any] | None],
    *,
    audio_duration_s: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atomic: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for operation in operations:
        if operation["operation"] == "equal":
            if current:
                blocks.append(_block(current))
                current = []
            continue
        ref_i = operation.get("reference_index")
        pred_i = operation.get("prediction_index")
        ref_span = _span_from_timing(reference_spans[ref_i]) if ref_i is not None else None
        pred_span = _span_from_timing(prediction_spans[pred_i]) if pred_i is not None else None
        if operation["operation"] == "deletion":
            span = ref_span
        elif operation["operation"] == "insertion":
            span = pred_span
        else:
            if ref_span is None or pred_span is None:
                raise ValueError("substitution requires both reference and prediction spans")
            span = _union_span((ref_span, pred_span))
        if span is None:
            raise ValueError(f"missing timing span for {operation['operation']}")
        if audio_duration_s is not None and span[1] > audio_duration_s + 1e-9:
            raise ValueError("error span extends beyond audio duration")
        atomic.append({**dict(operation), "start_s": span[0], "end_s": span[1]})
        current.append(atomic[-1])
    if current:
        blocks.append(_block(current))

    merged: list[dict[str, Any]] = []
    for block in blocks:
        if merged and block["start_s"] <= merged[-1]["end_s"] + 1e-9:
            previous = merged[-1]
            previous["end_s"] = max(previous["end_s"], block["end_s"])
            previous["operation_ids"].extend(block["operation_ids"])
            previous["reference_indices"].extend(block["reference_indices"])
            previous["prediction_indices"].extend(block["prediction_indices"])
        else:
            merged.append(block)
    return atomic, merged


def _block(operations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "block_id": f"edit_block_{len(operations):06d}_{operations[0]['operation_id']}",
        "start_s": min(float(row["start_s"]) for row in operations),
        "end_s": max(float(row["end_s"]) for row in operations),
        "operation_ids": [str(row["operation_id"]) for row in operations],
        "reference_indices": [row["reference_index"] for row in operations if row.get("reference_index") is not None],
        "prediction_indices": [row["prediction_index"] for row in operations if row.get("prediction_index") is not None],
    }


def error_alignment(
    reference: Sequence[str],
    prediction: Sequence[str],
    reference_spans: Sequence[Mapping[str, Any] | None],
    prediction_spans: Sequence[Mapping[str, Any] | None],
    *,
    audio_duration_s: float | None = None,
) -> dict[str, Any]:
    if len(reference_spans) != len(reference) or len(prediction_spans) != len(prediction):
        raise ValueError("word span counts do not match word sequences")
    operations = align_words(reference, prediction)
    atomic, merged = assign_error_spans(operations, reference_spans, prediction_spans, audio_duration_s=audio_duration_s)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for operation in operations:
        if operation["operation"] == "substitution":
            counts["substitutions"] += 1
        elif operation["operation"] == "deletion":
            counts["deletions"] += 1
        elif operation["operation"] == "insertion":
            counts["insertions"] += 1
    denominator = len(reference)
    return {
        "operations": operations,
        "atomic_error_spans": atomic,
        "error_spans": merged,
        "counts": counts,
        "word_error_rate": (sum(counts.values()) / denominator) if denominator else None,
        "reference_word_count": denominator,
        "normalizer_version": NORMALIZER_VERSION,
    }
