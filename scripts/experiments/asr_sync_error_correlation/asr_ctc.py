"""Uncorrected Wav2Vec2-CTC decoding and per-arm reference alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .io import atomic_write_json, atomic_write_npz, canonical_hash, file_sha256
from .word_errors import normalize_text, normalized_words


def _frame_span(frame_indices: Sequence[int], frame_stride_s: float, audio_duration_s: float | None) -> tuple[float, float]:
    if not frame_indices:
        raise ValueError("empty token frame contribution")
    start = min(frame_indices) * frame_stride_s
    end = (max(frame_indices) + 1) * frame_stride_s
    if audio_duration_s is not None:
        start = min(start, audio_duration_s)
        end = min(end, audio_duration_s)
    if not start < end:
        raise ValueError("token span is empty after audio-duration clipping")
    return float(start), float(end)


def collapse_ctc(
    frame_ids: Sequence[int] | np.ndarray,
    *,
    blank_id: int,
    delimiter_id: int,
    frame_stride_s: float,
    audio_duration_s: float | None = None,
    id_to_token: Callable[[int], str] | None = None,
) -> list[dict[str, Any]]:
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    tokens: list[dict[str, Any]] = []
    previous: int | None = None
    contribution: list[int] = []
    for frame, token_id in enumerate(ids.tolist() + [blank_id]):
        if previous is not None and token_id != previous:
            if previous != blank_id:
                start_s, end_s = _frame_span(contribution, frame_stride_s, audio_duration_s)
                tokens.append({
                    "token_id": int(previous),
                    "token": id_to_token(previous) if id_to_token else str(previous),
                    "frames": list(contribution),
                    "start_s": start_s,
                    "end_s": end_s,
                    "is_delimiter": previous == delimiter_id,
                })
            contribution = []
        if token_id != blank_id:
            contribution.append(frame)
        previous = int(token_id)
    return tokens


def tokens_to_words(
    tokens: Sequence[Mapping[str, Any]],
    *,
    delimiter_id: int,
    log_probs: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = normalize_text("".join(str(token["token"]) for token in current))
        if not text:
            current.clear()
            return
        start_s = min(float(token["start_s"]) for token in current)
        end_s = max(float(token["end_s"]) for token in current)
        confidence = None
        if log_probs is not None:
            frame_values: list[float] = []
            for token in current:
                token_id = int(token["token_id"])
                frame_values.extend(float(log_probs[frame, token_id]) for frame in token["frames"])
            if frame_values:
                confidence = float(np.exp(np.mean(np.asarray(frame_values, dtype=np.float64))))
        words.append({
            "word": text,
            "start_s": start_s,
            "end_s": end_s,
            "confidence": confidence,
            "token_ids": [int(token["token_id"]) for token in current],
            "frames": [frame for token in current for frame in token["frames"]],
        })
        current.clear()

    for token in tokens:
        if int(token["token_id"]) == delimiter_id or bool(token.get("is_delimiter")):
            flush()
        else:
            current.append(token)
    flush()
    return words


def greedy_decode(
    log_probs: np.ndarray,
    *,
    blank_id: int,
    delimiter_id: int,
    frame_stride_s: float,
    audio_duration_s: float | None = None,
    id_to_token: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    values = np.asarray(log_probs, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("log_probs must be a finite [time, vocab] array")
    frame_ids = np.argmax(values, axis=1).astype(np.int64)
    tokens = collapse_ctc(
        frame_ids,
        blank_id=blank_id,
        delimiter_id=delimiter_id,
        frame_stride_s=frame_stride_s,
        audio_duration_s=audio_duration_s,
        id_to_token=id_to_token,
    )
    words = tokens_to_words(tokens, delimiter_id=delimiter_id, log_probs=values)
    transcript = " ".join(str(word["word"]) for word in words)
    return {
        "frame_argmax_ids": frame_ids,
        "tokens": tokens,
        "words": words,
        "transcript": transcript,
        "frame_count": int(values.shape[0]),
        "vocab_size": int(values.shape[1]),
    }


def encode_reference(tokenizer: Any, transcript: str, *, blank_id: int) -> tuple[str, list[str], np.ndarray, int]:
    normalized = normalize_text(transcript)
    encoded = tokenizer(normalized, add_special_tokens=False)
    ids = np.asarray(encoded["input_ids"], dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("reference transcript encodes to no tokens")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and np.any(ids == int(unk_id)):
        raise ValueError("reference transcript contains tokenizer unknown token")
    delimiter_token = getattr(tokenizer, "word_delimiter_token", "|")
    delimiter_id = getattr(tokenizer, "word_delimiter_token_id", None)
    if delimiter_id is None:
        vocab = tokenizer.get_vocab()
        if delimiter_token not in vocab:
            raise ValueError("tokenizer has no word delimiter")
        delimiter_id = int(vocab[delimiter_token])
    if int(blank_id) in ids:
        raise ValueError("reference encoding contains CTC blank")
    token_text = [str(value) for value in tokenizer.convert_ids_to_tokens(ids.tolist())]
    return normalized, token_text, ids, int(delimiter_id)


def forced_align_reference(
    log_probs: np.ndarray,
    tokenizer: Any,
    transcript: str,
    *,
    blank_id: int,
    frame_stride_s: float,
    audio_duration_s: float,
) -> dict[str, Any]:
    import torch
    import torchaudio

    normalized, token_text, target_ids, delimiter_id = encode_reference(tokenizer, transcript, blank_id=blank_id)
    emissions = torch.from_numpy(np.asarray(log_probs, dtype=np.float32))[None, ...]
    targets = torch.from_numpy(target_ids.astype(np.int64))[None, ...]
    with torch.inference_mode():
        aligned, scores = torchaudio.functional.forced_align(
            emissions, targets, input_lengths=torch.tensor([emissions.shape[1]]),
            target_lengths=torch.tensor([targets.shape[1]]), blank=int(blank_id),
        )
    aligned_ids = aligned[0].cpu().numpy().astype(np.int64)
    aligned_scores = scores[0].cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(aligned_scores)):
        raise ValueError("forced-alignment scores are non-finite")
    target_spans: list[dict[str, Any]] = []
    cursor = 0
    for target_id in target_ids.tolist():
        while cursor < aligned_ids.size and int(aligned_ids[cursor]) != int(target_id):
            cursor += 1
        if cursor >= aligned_ids.size:
            raise ValueError(f"forced alignment omitted target token {target_id}")
        start_frame = cursor
        while cursor + 1 < aligned_ids.size and int(aligned_ids[cursor + 1]) == int(target_id):
            cursor += 1
        frames = np.arange(start_frame, cursor + 1, dtype=np.int64)
        cursor += 1
        start_s, end_s = _frame_span(frames.tolist(), frame_stride_s, audio_duration_s)
        target_spans.append({
            "token_id": int(target_id),
            "token": token_text[len(target_spans)],
            "start_s": start_s,
            "end_s": end_s,
            "frames": frames.tolist(),
            "mean_score": float(np.mean(aligned_scores[frames])),
            "min_score": float(np.min(aligned_scores[frames])),
        })

    words: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    reference_word_list = normalized_words(normalized)
    for span in target_spans:
        if span["token_id"] == delimiter_id:
            if current:
                words.append(_reference_word(current, reference_word_list[len(words)]))
                current = []
        else:
            current.append(span)
    if current:
        words.append(_reference_word(current, reference_word_list[len(words)]))
    if len(words) != len(reference_word_list):
        raise ValueError("forced alignment word count disagrees with reference")
    starts = [float(word["start_s"]) for word in words]
    ends = [float(word["end_s"]) for word in words]
    if any(start_b < start_a for start_a, start_b in zip(starts, starts[1:])) or any(end_b < end_a for end_a, end_b in zip(ends, ends[1:])):
        raise ValueError("forced reference spans are not monotonic")
    return {
        "transcript": normalized,
        "tokens": target_spans,
        "words": words,
        "aligned_token_ids": aligned_ids,
        "aligned_scores": aligned_scores,
        "delimiter_id": delimiter_id,
        "blank_id": int(blank_id),
    }


def _reference_word(tokens: Sequence[Mapping[str, Any]], word: str) -> dict[str, Any]:
    return {
        "word": word,
        "start_s": min(float(token["start_s"]) for token in tokens),
        "end_s": max(float(token["end_s"]) for token in tokens),
        "mean_score": float(np.mean([float(token["mean_score"]) for token in tokens])),
        "min_score": float(min(float(token["min_score"]) for token in tokens)),
        "token_ids": [int(token["token_id"]) for token in tokens],
        "frames": [frame for token in tokens for frame in token["frames"]],
    }


def infer_waveform(waveform: np.ndarray, processor: Any, model: Any, *, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("waveform must be non-empty finite float32")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        model_inputs = {key: value.to(device=device) for key, value in inputs.items() if hasattr(value, "to")}
        logits = model(**model_inputs).logits.to(dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
    return log_probs, {
        "input_samples": int(audio.size),
        "output_frames": int(log_probs.shape[0]),
        "vocab_size": int(log_probs.shape[1]),
        "device": str(device),
        "dtype": "float32",
    }


def export_emissions(path: str | Path, log_probs: np.ndarray, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(log_probs, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("emissions must be finite [time, vocab] float32")
    frame_ids = np.argmax(values, axis=1).astype(np.int64)
    atomic_write_npz(path, {"log_probs": values, "frame_argmax_ids": frame_ids})
    return {
        "path": str(Path(path)),
        "sha256": file_sha256(path),
        "shape": list(values.shape),
        "provenance": dict(provenance),
        "binding_hash": canonical_hash(provenance),
    }


def compare_official_timings(forced_words: Sequence[Mapping[str, Any]], official_words: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(forced_words) != len(official_words):
        raise ValueError("official and forced word counts disagree")
    starts = [float(a["start_s"]) - float(b["start_s"]) for a, b in zip(forced_words, official_words)]
    ends = [float(a["end_s"]) - float(b["end_s"]) for a, b in zip(forced_words, official_words)]
    mids = [
        (float(a["start_s"]) + float(a["end_s"]) - float(b["start_s"]) - float(b["end_s"])) / 2
        for a, b in zip(forced_words, official_words)
    ]
    def summary(values: Sequence[float]) -> dict[str, float]:
        array = np.abs(np.asarray(values, dtype=np.float64))
        return {"median_abs_s": float(np.median(array)), "p95_abs_s": float(np.percentile(array, 95))}
    return {"start": summary(starts), "end": summary(ends), "midpoint": summary(mids)}


def save_asr_record(path: str | Path, record: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(record))
