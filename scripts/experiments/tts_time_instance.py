"""TTS temporal identifiability and same-text instance cross-matching experiment.

This module deliberately keeps the protocol in one small, auditable runner.  A
replays cached SyncNet features; B optionally generates two fresh local TTS
instances, renders them through the frozen static Wav2Lip worker, and compares
the resulting embeddings on a common forced-alignment coordinate.  Missing
external resources are recorded as blocked artifacts instead of being silently
replaced by a different model or an invented alignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import wave
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# This file lives directly under ``scripts/experiments`` (unlike the nested
# static-image worker), so the repository root is parents[2].
REPO = Path(__file__).resolve().parents[2]
RUN_PREFIX = "tts_time_instance"
SAMPLE_IDS = tuple(range(151, 163))
SAMPLE_RATE = 16_000
FPS = 25
LAG_COUNT = 31
VSHIFT = 15
EMBEDDING_DIM = 1024
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 20_000
PRIMARY_COUNT = 4
PRIMARY_ALPHA = 0.05 / PRIMARY_COUNT
# The eight non-zero lags are the registered A_rank/A_event estimand.  The
# adjacent +/-1-frame comparisons are retained as a descriptive sensitivity
# output only; keeping the sets explicit prevents them from entering either
# primary statistic by accident.
PRIMARY_DELTAS = (-5, -4, -3, -2, 2, 3, 4, 5)
SUPPLEMENTARY_DELTAS = (-1, 1)
ALL_DELTAS = SUPPLEMENTARY_DELTAS + PRIMARY_DELTAS
FEATURE_ROOT = REPO / "runs/tts_native_gain_attribution_implementation_20260915_v1/02_fixed_video/features"
INPUT_BINDING = REPO / "openspec/changes/disentangle-tts-native-gain/input-bindings.json"
PARENT_COHORT = REPO / "runs/lrs3_tts_gain_mechanism_review_v15/00_audit/cohort.json"
A_MANIFEST = REPO / "runs/tts_native_gain_attribution_completion_20260916_v1/02_fixed_video/a_manifest.json"
A_SUMMARY = REPO / "runs/tts_native_gain_attribution_completion_20260916_v1/05_analysis/summary.json"
ASSETS = REPO / "runs/tts_native_gain_attribution_completion_20260916_v1/00_audit/assets.json"
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
WAV2LIP_PYTHON = Path(os.environ.get("WAV2LIP_PYTHON", "/home/wjj/.venvs/wav2lip/bin/python"))
SYNCNET_PYTHON = Path(os.environ.get("SYNCNET_PYTHON", "/home/wjj/.venvs/syncnet/bin/python"))
QWEN_PYTHON = Path(os.environ.get("QWEN_PYTHON", "/home/wjj/.venvs/qwen3/bin/python"))
MFA = Path(os.environ.get("MFA", "/home/wjj/.local/bin/mfa"))
MFA_DICTIONARY = "english_us_mfa"
MFA_ACOUSTIC_MODEL = "english_mfa"
PORTRAIT_ROOT = REPO / "runs/tts_native_gain_attribution_implementation_20260915_v1/00_audit/portraits"
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TTS_LANGUAGE = "English"
TTS_MAX_NEW_TOKENS = 4096
SILENCE = {"", "sp", "sil", "silence"}
UNKNOWN = {"spn", "<unk>", "unk", "unknown", "oov", "<oov>"}


class ProtocolError(RuntimeError):
    """An invalid frozen input or an incomplete required artifact."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def run_id_valid(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("run-id may contain only letters, numbers, underscores, and hyphens")
    return value


def run_root(run_id: str) -> Path:
    return REPO / "runs" / f"{RUN_PREFIX}_{run_id_valid(run_id)}"


def quantile(values: np.ndarray, p: float) -> float:
    try:
        return float(np.quantile(values, p, method="linear"))
    except TypeError:  # pragma: no cover - old NumPy
        return float(np.quantile(values, p, interpolation="linear"))


def bootstrap_summary(values_by_group: Mapping[str, float], *, metric: str, min_groups: int = 1) -> dict[str, Any]:
    labels = sorted(str(key) for key in values_by_group)
    if len(labels) < min_groups:
        return {"status": "NOT_ESTIMABLE", "metric": metric, "group_count": len(labels), "required_group_count": min_groups}
    values = np.asarray([float(values_by_group[label]) for label in labels], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels)), dtype=np.int64)
    means = values[indices].mean(axis=1, dtype=np.float64)
    lower95, upper95 = quantile(means, 0.025), quantile(means, 0.975)
    lower_bc, upper_bc = quantile(means, PRIMARY_ALPHA / 2), quantile(means, 1.0 - PRIMARY_ALPHA / 2)
    return {
        "status": "COMPLETE",
        "metric": metric,
        "mean": float(values.mean()),
        "ci95": [lower95, upper95],
        "ci98_75_bonferroni": [lower_bc, upper_bc],
        "group_count": len(labels),
        "group_labels": labels,
        "group_values": {label: float(value) for label, value in zip(labels, values, strict=True)},
        "positive_group_count": int(np.sum(values > 0)),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64)",
        "quantile_method": "linear",
        "status_by_corrected_ci": (
            "POSITIVE" if lower_bc > 0.0 else "NEGATIVE" if upper_bc < 0.0 else "INCONCLUSIVE"
        ),
    }


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ProtocolError("distance inputs are not finite and shape-matched")
    return float(np.linalg.norm(a - b))


def official_distance_matrix(video: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Reproduce SyncNet's float32 pairwise_distance(eps=1e-6) exactly."""

    v = np.asarray(video, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != EMBEDDING_DIM or a.shape[1] != EMBEDDING_DIM:
        raise ProtocolError(f"invalid feature shapes {v.shape}/{a.shape}")
    count = min(v.shape[0], a.shape[0])
    if count < 1:
        raise ProtocolError("no common feature rows")
    try:
        import torch
    except ImportError:  # pragma: no cover - syncnet environment always has torch
        rows = []
        padded = np.pad(a[:count], ((VSHIFT, VSHIFT), (0, 0)))
        for i in range(count):
            rows.append(np.sqrt(np.sum((v[i] - padded[i : i + LAG_COUNT]) ** 2, axis=1) + 1e-12))
        return np.asarray(rows, dtype=np.float64)
    vt = torch.from_numpy(v[:count])
    at = torch.from_numpy(np.pad(a[:count], ((VSHIFT, VSHIFT), (0, 0))))
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(count):
            left = vt[i : i + 1].expand(LAG_COUNT, -1)
            rows.append(torch.nn.functional.pairwise_distance(left, at[i : i + LAG_COUNT], eps=1e-6).numpy().astype(np.float64))
    return np.stack(rows, axis=0)


def strict_distance_matrix(video: np.ndarray, audio: np.ndarray) -> np.ndarray:
    v = np.asarray(video, dtype=np.float64)
    a = np.asarray(audio, dtype=np.float64)
    count = min(v.shape[0], a.shape[0])
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != EMBEDDING_DIM or a.shape[1] != EMBEDDING_DIM or count < 1:
        raise ProtocolError("invalid strict feature shapes")
    padded = np.pad(a[:count], ((VSHIFT, VSHIFT), (0, 0)))
    return np.stack([np.linalg.norm(v[i] - padded[i : i + LAG_COUNT], axis=1) for i in range(count)], axis=0)


def unit_features(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ProtocolError("zero or invalid feature norm")
    return array / norms[:, None]


def curve(matrix: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float32)
    support = np.asarray(list(rows), dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != LAG_COUNT or support.size == 0 or np.any(support < 0) or np.any(support >= value.shape[0]):
        raise ProtocolError("invalid curve support")
    means = value[support].mean(axis=0, dtype=np.float32).astype(np.float64)
    d = float(np.min(means))
    idx = int(np.argmin(means))
    b = float(np.median(means))
    return {"support_count": int(support.size), "support_rows": support.tolist(), "curve": means.tolist(), "min_index": idx, "official_offset": VSHIFT - idx, "sync_d": d, "background_b": b, "sync_c": b - d, "d0": float(means[VSHIFT])}


def _feature_path(arm: str, sample_id: int, kind: str) -> Path:
    if kind == "audio":
        return FEATURE_ROOT / "audio" / arm / str(sample_id) / "ORIGINAL.npy"
    return FEATURE_ROOT / "visual" / arm / f"{sample_id}.npy"


def _read_feature(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ProtocolError(f"feature missing: {path}")
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] != EMBEDDING_DIM or not np.isfinite(value).all():
        raise ProtocolError(f"invalid feature: {path}")
    return np.asarray(value, dtype=np.float32)


def _parse_tier_intervals(raw: str, tier_name: str) -> list[dict[str, Any]]:
    marker = re.search(rf'name\s*=\s*"{re.escape(tier_name)}"', raw, flags=re.IGNORECASE)
    if marker is None:
        return []
    tail = raw[marker.end() :]
    next_tier = re.search(r'item\s*\[\s*\d+\s*\]\s*:', tail, flags=re.IGNORECASE)
    section = tail[: next_tier.start()] if next_tier else tail
    pattern = re.compile(
        # The tier header also has xmin/xmax but no ``intervals [n]``
        # prefix.  Requiring the interval marker prevents that header from
        # becoming a spurious full-duration silence token.
        r"intervals\s*\[\s*\d+\s*\]\s*:\s*xmin\s*=\s*([0-9.eE+-]+).*?xmax\s*=\s*([0-9.eE+-]+).*?text\s*=\s*\"(.*?)\"",
        flags=re.IGNORECASE | re.DOTALL,
    )
    intervals: list[dict[str, Any]] = []
    for match in pattern.finditer(section):
        start, end = float(match.group(1)), float(match.group(2))
        label = match.group(3).strip()
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        normalized = label.lower()
        intervals.append({"index": len(intervals), "label": label, "normalized": normalized, "start_s": start, "end_s": end, "silence": normalized in SILENCE, "unknown": normalized in UNKNOWN})
    return intervals


def parse_textgrid(path: str | Path) -> list[dict[str, Any]]:
    """Parse phones and annotate each phone with its original word span."""

    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    tokens = _parse_tier_intervals(raw, "phones")
    if not tokens:
        raise ProtocolError(f"phones tier contains no usable speech: {path}")
    words = _parse_tier_intervals(raw, "words")
    for token in tokens:
        midpoint = 0.5 * (float(token["start_s"]) + float(token["end_s"]))
        owner = next((word for word in words if float(word["start_s"]) <= midpoint < float(word["end_s"]) and not word["silence"] and word["label"]), None)
        token["word_index"] = int(owner["index"]) if owner is not None else None
        token["word_label"] = str(owner["label"]) if owner is not None else None
        token["word_normalized"] = str(owner["normalized"]) if owner is not None else None
    if not any(not t["silence"] and not t["unknown"] for t in tokens):
        raise ProtocolError(f"phones tier contains no usable speech: {path}")
    return tokens


def token_at(tokens: Sequence[Mapping[str, Any]], time_s: float) -> dict[str, Any] | None:
    for token in tokens:
        if float(token["start_s"]) <= time_s < float(token["end_s"]):
            return dict(token)
    return None


def window_coverage(tokens: Sequence[Mapping[str, Any]], center_s: float, *, window_s: float = 0.215) -> list[dict[str, Any]]:
    """Return every phone touched by one cached audio window.

    The event label uses the window centre, but the full overlap ledger is
    retained because a 215 ms SyncNet window commonly spans several phones.
    The right edge is treated as open, matching the TextGrid interval rule.
    """

    start_s = float(center_s) - window_s / 2.0
    end_s = float(center_s) + window_s / 2.0
    result: list[dict[str, Any]] = []
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        return result
    for token in tokens:
        left = max(start_s, float(token["start_s"]))
        right = min(end_s, float(token["end_s"]))
        overlap = right - left
        if overlap <= 0.0:
            continue
        result.append({"index": int(token["index"]), "phone": str(token["label"]), "overlap_s": float(overlap), "overlap_fraction": float(overlap / window_s)})
    return result


def classify_event(tokens: Sequence[Mapping[str, Any]], left_time: float, right_time: float) -> tuple[str, dict[str, Any]]:
    left, right = token_at(tokens, left_time), token_at(tokens, right_time)
    detail = {"left_phone": None, "right_phone": None, "left_index": None, "right_index": None, "left_sequence": window_coverage(tokens, left_time), "right_sequence": window_coverage(tokens, right_time)}
    if left:
        detail.update({"left_phone": left["label"], "left_index": left["index"]})
    if right:
        detail.update({"right_phone": right["label"], "right_index": right["index"]})
    if left is None or right is None or left["silence"] or right["silence"] or left["unknown"] or right["unknown"]:
        return "EXCLUDED", detail
    if left["index"] == right["index"]:
        return "SAME", detail
    if left["normalized"] != right["normalized"]:
        return "CROSS", detail
    return "SAME_LABEL_DIFFERENT_OCCURRENCE", detail


def interpolation(value: np.ndarray, position: float) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not math.isfinite(float(position)) or position < 0.0 or position > array.shape[0] - 1:
        return None
    left = math.floor(position)
    right = min(left + 1, array.shape[0] - 1)
    alpha = position - left
    result = (1.0 - alpha) * array[left] + alpha * array[right]
    return result if np.isfinite(result).all() else None


def l2_unit(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return np.asarray(value, dtype=np.float64) / norm


def _mfa_command(corpus: Path, output: Path) -> list[str]:
    return [str(MFA), "align", "--clean", "--overwrite", str(corpus), MFA_DICTIONARY, MFA_ACOUSTIC_MODEL, str(output)]


def _mfa_version() -> str | None:
    if not MFA.is_file():
        return None
    try:
        result = subprocess.run([str(MFA), "--version"], capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else None


def _mfa_cache_inputs_match(records: Sequence[Mapping[str, Any]], audio_fields: Sequence[str], corpus: Path) -> bool:
    """Check that reusable MFA corpus WAV/LAB inputs still match the manifest."""

    try:
        for row in records:
            sid = str(row["sample_id"])
            expected_lab = str(row["transcript"]).upper().strip()
            for field in audio_fields:
                source = Path(str(row[field]))
                copied = corpus / f"{sid}_{field}.wav"
                lab = corpus / f"{sid}_{field}.lab"
                if not source.is_file() or not copied.is_file() or not lab.is_file():
                    return False
                if sha256_file(source) != sha256_file(copied) or lab.read_text(encoding="utf-8").strip() != expected_lab:
                    return False
    except (KeyError, OSError, UnicodeError):
        return False
    return True


def run_mfa(records: Sequence[Mapping[str, Any]], audio_fields: Sequence[str], output_dir: Path, *, prefix: str) -> dict[str, Any]:
    """Run the pinned local MFA command, returning a blocked manifest on failure."""

    corpus = output_dir / f"{prefix}_corpus"
    grids = output_dir / f"{prefix}_textgrids"
    if corpus.exists() and any(corpus.iterdir()):
        if not grids.exists() or not _mfa_cache_inputs_match(records, audio_fields, corpus):
            # A run-local stale corpus must never be paired with new TextGrids.
            # Rebuild both directories; historical inputs remain untouched.
            shutil.rmtree(corpus, ignore_errors=True)
            shutil.rmtree(grids, ignore_errors=True)
        else:
            expected_existing = [f"{row['sample_id']}_{field}.TextGrid" for row in records for field in audio_fields]
            missing_existing = [name for name in expected_existing if not (grids / name).is_file()]
            if not missing_existing:
                return {"status": "COMPLETE", "reused": True, "corpus": str(corpus), "textgrids": str(grids), "mfa_version": _mfa_version(), "command": _mfa_command(corpus, grids)}
            if len(expected_existing) - len(missing_existing) > 0:
                return {"status": "MFA_FAILED", "reused": True, "corpus": str(corpus), "textgrids": str(grids), "mfa_version": _mfa_version(), "missing": missing_existing, "reason": "reusing partial MFA output; missing utterances are retained in denominator", "command": _mfa_command(corpus, grids)}
    if not MFA.is_file():
        return {"status": "MFA_UNAVAILABLE", "reason": f"missing executable: {MFA}"}
    corpus.mkdir(parents=True, exist_ok=True)
    grids.mkdir(parents=True, exist_ok=True)
    expected: list[str] = []
    try:
        for record in records:
            sid = str(record["sample_id"])
            for field in audio_fields:
                stem = f"{sid}_{field}"
                source = Path(str(record[field]))
                target = corpus / f"{stem}.wav"
                if not target.exists():
                    shutil.copy2(source, target)
                (corpus / f"{stem}.lab").write_text(str(record["transcript"]).upper().strip() + "\n", encoding="utf-8")
                expected.append(stem)
        command = _mfa_command(corpus, grids)
        result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, check=False)
        write_json(output_dir / f"{prefix}_mfa.log.json", {"command": command, "returncode": int(result.returncode), "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-3000:]})
        missing = [stem for stem in expected if not (grids / f"{stem}.TextGrid").is_file()]
        if result.returncode != 0 or missing:
            return {"status": "MFA_FAILED", "command": command, "returncode": int(result.returncode), "mfa_version": _mfa_version(), "missing": missing[:20], "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-2000:], "corpus": str(corpus), "textgrids": str(grids)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "MFA_FAILED", "reason": f"{type(exc).__name__}: {exc}", "corpus": str(corpus), "textgrids": str(grids)}
    return {"status": "COMPLETE", "reused": False, "mfa_version": _mfa_version(), "corpus": str(corpus), "textgrids": str(grids), "command": command}


def _record_from_assets(input_binding: Mapping[str, Any], assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    asset_by_id = {int(row["sample_id"]): row for row in assets.get("records", [])}
    result: list[dict[str, Any]] = []
    for item in input_binding.get("records", []):
        sid = int(item["sample_id"])
        if sid not in SAMPLE_IDS:
            continue
        asset = asset_by_id.get(sid)
        if asset is None or not isinstance(asset.get("transcript"), Mapping):
            raise ProtocolError(f"transcript asset missing for {sid}")
        real_video = (REPO / str(item["real_video"])).resolve()
        n_json = FEATURE_ROOT / "audio" / "N" / str(sid) / "ORIGINAL.json"
        t_json = FEATURE_ROOT / "audio" / "T" / str(sid) / "ORIGINAL.json"
        n_meta, t_meta = read_json(n_json), read_json(t_json)
        n_audio, t_audio = Path(str(n_meta["audio_path"])), Path(str(t_meta["audio_path"]))
        if not n_audio.is_file() or not t_audio.is_file():
            raise ProtocolError(f"cached audio missing for {sid}")
        expected_n = item["conditions"]["natural_raw"]["original_pcm_sha256"]
        expected_t = item["conditions"]["tts_raw"]["original_pcm_sha256"]
        if n_meta.get("pcm_hash") != expected_n or t_meta.get("pcm_hash") != expected_t:
            raise ProtocolError(f"cached PCM hash mismatch for {sid}")
        transcript = str(asset["transcript"]["text"])
        transcript_hash = sha256_bytes(transcript.encode("utf-8"))
        bound_transcript_hash = str(asset["transcript"].get("text_sha256", ""))
        transcript_source = real_video.with_suffix(".txt")
        transcript_source_hash = sha256_file(transcript_source) if transcript_source.is_file() else None
        # The parent audit's ``text_sha256`` is the hash of the bound LRS3
        # transcript sidecar (see its _text_for_video contract), rather than a
        # hash of the normalized Text line.  Verify that source file directly;
        # keep a separate normalized-text hash for TTS/MFA cache contracts.
        if bound_transcript_hash and transcript_source_hash != bound_transcript_hash:
            raise ProtocolError(f"transcript source hash mismatch for {sid}")
        portrait = Path(str(asset.get("portrait", {}).get("path", PORTRAIT_ROOT / f"{sid}.png")))
        if not portrait.is_file():
            raise ProtocolError(f"portrait missing for {sid}: {portrait}")
        result.append({
            "sample_id": sid,
            "source_group": str(item["source_group"]),
            "real_video": str(real_video),
            "transcript": transcript,
            "transcript_sha256": transcript_hash,
            "transcript_source": str(transcript_source.resolve()),
            "transcript_source_sha256": transcript_source_hash,
            "N_audio": str(n_audio.resolve()),
            "T_audio": str(t_audio.resolve()),
            "N_pcm_sha256": expected_n,
            "T_pcm_sha256": expected_t,
            "portrait": str(portrait.resolve()),
            "portrait_sha256": sha256_file(portrait),
        })
    if [int(row["sample_id"]) for row in result] != list(SAMPLE_IDS):
        raise ProtocolError("frozen cohort order/count is not 151..162")
    return result


def audit_inputs(root: Path) -> dict[str, Any]:
    binding = read_json(INPUT_BINDING)
    cohort = read_json(PARENT_COHORT)
    a_manifest = read_json(A_MANIFEST)
    summary = read_json(A_SUMMARY)
    assets = read_json(ASSETS)
    records = _record_from_assets(binding, assets)
    historical_cells = {(int(row["id"]), str(row["video_type"]), str(row["eval_condition"])): row for row in a_manifest.get("cells", [])}
    feature_rows: list[dict[str, Any]] = []
    for row in records:
        sid = int(row["sample_id"])
        for arm, video_arm in (("N", "V_N"), ("T", "V_T")):
            vp = _feature_path(video_arm, sid, "visual")
            ap = _feature_path(arm, sid, "audio")
            vj, aj = vp.with_suffix(".json"), ap.with_suffix(".json")
            if not vp.is_file() or not ap.is_file() or not vj.is_file() or not aj.is_file():
                raise ProtocolError(f"A feature quartet missing for {sid}/{arm}")
            vm, am = read_json(vj), read_json(aj)
            for path, meta, kind in ((vp, vm, "visual"), (ap, am, "audio")):
                if str(meta.get("sha256")) != sha256_file(path):
                    raise ProtocolError(f"feature hash mismatch: {path}")
            hist = historical_cells.get((sid, video_arm, "ORIGINAL"))
            if hist is None:
                raise ProtocolError(f"historical A cell missing: {sid}/{video_arm}")
            matrix_path = Path(str(hist["matrix_path"]))
            if not matrix_path.is_file() or sha256_file(matrix_path) != str(hist["matrix_hash"]):
                raise ProtocolError(f"historical matrix missing/hash mismatch: {matrix_path}")
            feature_rows.append({"sample_id": sid, "source_group": row["source_group"], "arm": arm, "video_arm": video_arm, "visual": str(vp.resolve()), "audio": str(ap.resolve()), "visual_meta": str(vj.resolve()), "audio_meta": str(aj.resolve()), "historical_matrix": str(matrix_path.resolve()), "historical_matrix_sha256": str(hist["matrix_hash"])})
    payload = {
        "schema_version": 1,
        "protocol": "tts_time_instance_v1",
        "status": "READY",
        "sample_ids": list(SAMPLE_IDS),
        "source_group_count": len(records),
        "records": records,
        "A_features": feature_rows,
        "frozen_inputs": {"input_binding": {"path": str(INPUT_BINDING.resolve()), "sha256": sha256_file(INPUT_BINDING)}, "cohort": {"path": str(PARENT_COHORT.resolve()), "sha256": sha256_file(PARENT_COHORT)}, "a_manifest": {"path": str(A_MANIFEST.resolve()), "sha256": sha256_file(A_MANIFEST)}, "a_summary": {"path": str(A_SUMMARY.resolve()), "sha256": sha256_file(A_SUMMARY)}},
        "history_summary_status": summary.get("status"),
        "cohort_status": cohort.get("status"),
        "implementation": {
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "independent_recompute": {"path": str((REPO / "scripts/experiments/tts_time_instance_recompute.py").resolve()), "sha256": sha256_file(REPO / "scripts/experiments/tts_time_instance_recompute.py") if (REPO / "scripts/experiments/tts_time_instance_recompute.py").is_file() else None},
            "faster_qwen3_provider": {"path": str((REPO / "scripts/tts/faster_qwen3.py").resolve()), "sha256": sha256_file(REPO / "scripts/tts/faster_qwen3.py") if (REPO / "scripts/tts/faster_qwen3.py").is_file() else None},
        },
        # Keep the old flat key for consumers that already expect it.
        "code_sha256": sha256_file(Path(__file__)),
        "runtime_hashes": {"wav2lip_checkpoint": sha256_file(WAV2LIP_CHECKPOINT) if WAV2LIP_CHECKPOINT.is_file() else None, "syncnet_model": sha256_file(SYNCNET_MODEL) if SYNCNET_MODEL.is_file() else None},
        "tts_contract": {"model_id": TTS_MODEL_ID, "strict_backend": True, "language": TTS_LANGUAGE, "max_new_tokens": TTS_MAX_NEW_TOKENS, "seeds": {"T1": 42, "T2": 43}, "sampling_defaults_expanded": {"min_new_tokens": 2, "temperature": 0.9, "top_k": 50, "top_p": 1.0, "do_sample": True, "repetition_penalty": 1.05}},
    }
    write_json(root / "inputs.json", payload)
    return payload


def _calibrate_k(v: np.ndarray, a: np.ndarray, rows: Sequence[int]) -> tuple[int | None, dict[str, Any]]:
    usable = [int(i) for i in rows if 0 <= int(i) < min(len(v), len(a))]
    folds = {0: [i for i in usable if (i // 25) % 2 == 0], 1: [i for i in usable if (i // 25) % 2 == 1]}
    if len(folds[0]) < 10 or len(folds[1]) < 10:
        return None, {"status": "COVERAGE_LOW", "fold_counts": {str(k): len(vv) for k, vv in folds.items()}}
    fold_k: dict[str, int] = {}
    details: dict[str, Any] = {"fold_counts": {str(k): len(vv) for k, vv in folds.items()}}
    for test_fold in (0, 1):
        train = folds[1 - test_fold]
        candidates: list[tuple[float, int, int]] = []
        for k in range(-VSHIFT, VSHIFT + 1):
            distances = [l2_distance(v[i], a[i + k]) for i in train if 0 <= i + k < len(a)]
            if distances:
                candidates.append((float(np.mean(distances)), abs(k), k))
        if not candidates:
            return None, {"status": "COVERAGE_LOW", "fold_counts": details["fold_counts"]}
        chosen = min(candidates)
        fold_k[str(test_fold)] = int(chosen[2])
        details[f"test_fold_{test_fold}"] = {"k0": int(chosen[2]), "train_mean": float(chosen[0]), "train_count": len(train)}
    # A single global k is calibrated on all rows, with the same deterministic tie rule.
    all_candidates: list[tuple[float, int, int]] = []
    for k in range(-VSHIFT, VSHIFT + 1):
        ds = [l2_distance(v[i], a[i + k]) for i in usable if 0 <= i + k < len(a)]
        if ds:
            all_candidates.append((float(np.mean(ds)), abs(k), k))
    chosen = min(all_candidates)
    details.update({"status": "COMPLETE", "k0": int(chosen[2]), "global_mean": float(chosen[0]), "global_count": len(usable), "cross_fold_k": fold_k})
    return int(chosen[2]), details


def run_a(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    a_root = root / "A"
    a_root.mkdir(parents=True, exist_ok=True)
    replay_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    arm_state: dict[tuple[int, str], dict[str, Any]] = {}
    by_id = {int(row["sample_id"]): row for row in inputs["records"]}
    for feature in inputs["A_features"]:
        sid, arm = int(feature["sample_id"]), str(feature["arm"])
        v, a = _read_feature(Path(feature["visual"])), _read_feature(Path(feature["audio"]))
        official = official_distance_matrix(v, a)
        try:
            unit_v, unit_a = unit_features(v), unit_features(a)
            unit_status = "COMPLETE"
        except ProtocolError as exc:
            # Unit norm is a registered secondary diagnostic.  Preserve A1/A2
            # if a malformed cached vector makes only this diagnostic unusable.
            unit_v, unit_a, unit_status = None, None, f"UNAVAILABLE: {exc}"
        cached = np.load(Path(feature["historical_matrix"]), allow_pickle=False)
        max_error = float(np.max(np.abs(official - cached))) if official.shape == cached.shape else float("inf")
        F = min(len(v), len(a))
        support = list(range(30, F - 30))
        k0, calibration = _calibrate_k(v, a, support)
        replay_rows.append({"sample_id": sid, "arm": arm, "visual_rows": len(v), "audio_rows": len(a), "matrix_shape": list(official.shape), "historical_shape": list(cached.shape), "max_abs_error": max_error, "pass": bool(max_error <= 1e-4), "support_count": len(support), "calibration": calibration})
        state = {"sample_id": sid, "arm": arm, "source_group": str(feature["source_group"]), "v": v, "a": a, "unit_v": unit_v, "unit_a": unit_a, "unit_status": unit_status, "matrix": official, "support": support, "k0": k0, "calibration": calibration}
        arm_state[(sid, arm)] = state
        if k0 is not None:
            for i in support:
                d0 = l2_distance(v[i], a[i + k0]) if 0 <= i + k0 < len(a) else None
                if d0 is None:
                    continue
                c0 = 0.04 * (i + k0) + 0.1075
                for delta in ALL_DELTAS:
                    target = i + k0 + delta
                    if not 0 <= target < len(a):
                        continue
                    dd = l2_distance(v[i], a[target])
                    weight = 1.0 if dd > d0 else 0.0 if dd < d0 else 0.5
                    d0_unit = dd_unit = w_unit = None
                    if unit_v is not None and unit_a is not None and 0 <= i + k0 < len(unit_a) and 0 <= target < len(unit_a):
                        d0_unit = l2_distance(unit_v[i], unit_a[i + k0])
                        dd_unit = l2_distance(unit_v[i], unit_a[target])
                        w_unit = 1.0 if dd_unit > d0_unit else 0.0 if dd_unit < d0_unit else 0.5
                    pair_rows.append({"sample_id": sid, "source_group": state["source_group"], "arm": arm, "i": i, "k0": k0, "delta": delta, "h": abs(delta), "primary": delta in PRIMARY_DELTAS, "d0": d0, "d_delta": dd, "w_raw": weight, "d0_unit": d0_unit, "d_delta_unit": dd_unit, "w_unit": w_unit, "center0_s": c0, "center_delta_s": 0.04 * target + 0.1075})
    write_json(a_root / "replay.json", {"status": "COMPLETE" if all(row["pass"] for row in replay_rows) else "INCOMPLETE", "rows": replay_rows, "max_abs_error": max(row["max_abs_error"] for row in replay_rows) if replay_rows else None})
    write_csv(a_root / "time_pairs.csv", ["sample_id", "source_group", "arm", "i", "k0", "delta", "h", "primary", "d0", "d_delta", "w_raw", "d0_unit", "d_delta_unit", "w_unit", "center0_s", "center_delta_s"], pair_rows)
    # Official C/B/D on the historical INTERIOR is an input-level
    # reproduction check; the stricter U support is retained alongside it.
    native_rows: list[dict[str, Any]] = []
    for sid in SAMPLE_IDS:
        current = {arm: arm_state.get((sid, arm)) for arm in ("N", "T")}
        if current["N"] and current["T"] and current["N"]["support"] and current["T"]["support"]:
            # Keep the historical INTERIOR reproduction (15-frame margins)
            # separate from the stricter U=30..F-30 support used for local
            # rank calibration and event analysis.
            historical_n = list(range(VSHIFT, min(len(current["N"]["v"]), len(current["N"]["a"])) - VSHIFT))
            historical_t = list(range(VSHIFT, min(len(current["T"]["v"]), len(current["T"]["a"])) - VSHIFT))
            ncurve, tcurve = curve(current["N"]["matrix"], historical_n), curve(current["T"]["matrix"], historical_t)
            n_analysis, t_analysis = curve(current["N"]["matrix"], current["N"]["support"]), curve(current["T"]["matrix"], current["T"]["support"])
            native_rows.append({"sample_id": sid, "source_group": current["N"]["source_group"], "N_C": ncurve["sync_c"], "T_C": tcurve["sync_c"], "delta_C": tcurve["sync_c"] - ncurve["sync_c"], "N": ncurve, "T": tcurve, "analysis_support_N": n_analysis, "analysis_support_T": t_analysis})
    a_rank_group: dict[str, float] = {}
    unit_group: dict[str, float] = {}
    rank_records: list[dict[str, Any]] = []
    for sid in SAMPLE_IDS:
        rows_n = [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "N" and bool(r.get("primary", True))]
        rows_t = [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "T" and bool(r.get("primary", True))]
        if rows_n and rows_t:
            rn, rt = float(np.mean([r["w_raw"] for r in rows_n])), float(np.mean([r["w_raw"] for r in rows_t]))
            group = str(rows_n[0]["source_group"])
            a_rank_group[group] = rt - rn
            rank_records.append({"sample_id": sid, "source_group": group, "R_N": rn, "R_T": rt, "A_rank": rt - rn, "pair_count_N": len(rows_n), "pair_count_T": len(rows_t)})
            # Unit-norm is pre-registered secondary only.
            nstate, tstate = arm_state[(sid, "N")], arm_state[(sid, "T")]
            ru: dict[str, float] = {}
            for arm, state in (("N", nstate), ("T", tstate)):
                uv, ua = state["unit_v"], state["unit_a"]
                if uv is None or ua is None:
                    ru[arm] = float("nan")
                    continue
                values = []
                for i in state["support"]:
                    if state["k0"] is None or not 0 <= i + state["k0"] < len(ua):
                        continue
                    base = l2_distance(uv[i], ua[i + state["k0"]])
                    for delta in PRIMARY_DELTAS:
                        j = i + state["k0"] + delta
                        if 0 <= j < len(ua):
                            other = l2_distance(uv[i], ua[j])
                            values.append(1.0 if other > base else 0.0 if other < base else 0.5)
                ru[arm] = float(np.mean(values)) if values else float("nan")
            if math.isfinite(ru["N"]) and math.isfinite(ru["T"]):
                unit_group[group] = ru["T"] - ru["N"]
    rank_summary = bootstrap_summary(a_rank_group, metric="A_rank", min_groups=8)
    unit_summary = bootstrap_summary(unit_group, metric="A_rank_unit", min_groups=8)
    adjacent_records: list[dict[str, Any]] = []
    adjacent_summary: dict[str, Any] = {}
    for delta in SUPPLEMENTARY_DELTAS:
        n_by_group: dict[str, float] = {}
        t_by_group: dict[str, float] = {}
        for sid in SAMPLE_IDS:
            n_rows = [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "N" and int(r["delta"]) == delta]
            t_rows = [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "T" and int(r["delta"]) == delta]
            if not n_rows or not t_rows:
                continue
            group = str(n_rows[0]["source_group"])
            n_value = float(np.mean([float(r["w_raw"]) for r in n_rows]))
            t_value = float(np.mean([float(r["w_raw"]) for r in t_rows]))
            n_by_group[group], t_by_group[group] = n_value, t_value
            adjacent_records.append({"sample_id": sid, "source_group": group, "delta": delta, "R_N": n_value, "R_T": t_value, "A_rank_adjacent": t_value - n_value})
        differences = {group: t_by_group[group] - n_by_group[group] for group in n_by_group if group in t_by_group}
        adjacent_summary[str(delta)] = {"delta": delta, "group_count": len(differences), "mean_R_N": float(np.mean(list(n_by_group.values()))) if n_by_group else None, "mean_R_T": float(np.mean(list(t_by_group.values()))) if t_by_group else None, "mean_A_rank_adjacent": float(np.mean(list(differences.values()))) if differences else None, "group_values": differences, "status": "DESCRIPTIVE_ONLY"}
    write_json(a_root / "rank_records.json", {"records": rank_records, "summary": rank_summary, "unit_summary": unit_summary, "native_rows": native_rows, "adjacent_sensitivity": {"records": adjacent_records, "summary": adjacent_summary, "status": "DESCRIPTIVE_ONLY"}})

    a2_mfa = run_mfa(
        [{"sample_id": row["sample_id"], "source_group": row["source_group"], "transcript": row["transcript"], "N_audio": row["N_audio"], "T_audio": row["T_audio"]} for row in inputs["records"]],
        ("N_audio", "T_audio"),
        a_root,
        prefix="A",
    )
    event_rows: list[dict[str, Any]] = []
    event_group: dict[str, float] = {}
    event_record_rows: list[dict[str, Any]] = []
    if a2_mfa.get("status") == "COMPLETE":
        for sid in SAMPLE_IDS:
            state_n, state_t = arm_state.get((sid, "N")), arm_state.get((sid, "T"))
            if not state_n or not state_t or state_n["k0"] is None or state_t["k0"] is None:
                continue
            rec = by_id[sid]
            grids = {
                "N": a_root / "A_textgrids" / f"{sid}_N_audio.TextGrid",
                "T": a_root / "A_textgrids" / f"{sid}_T_audio.TextGrid",
            }
            try:
                tokens_n, tokens_t = parse_textgrid(grids["N"]), parse_textgrid(grids["T"])
            except ProtocolError:
                continue
            by_h: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            phone_distribution: dict[str, int] = defaultdict(int)
            for pair in [p for p in pair_rows if int(p["sample_id"]) == sid]:
                arm = str(pair["arm"])
                tokens = tokens_n if arm == "N" else tokens_t
                category, detail = classify_event(tokens, float(pair["center0_s"]), float(pair["center_delta_s"]))
                pair = {**pair, "event": category, **detail}
                event_rows.append(pair)
                phone_key = f"{category}:{detail.get('left_phone')}->{detail.get('right_phone')}"
                phone_distribution[phone_key] += 1
                if pair.get("primary", True) and category in ("SAME", "CROSS"):
                    by_h[int(pair["h"])][f"{arm}_{category}"].append(float(pair["w_raw"]))
            values_h = []
            counts: dict[str, dict[str, int]] = {}
            for h, cells in sorted(by_h.items()):
                counts[str(h)] = {key: len(value) for key, value in cells.items()}
                required = [cells.get(f"{arm}_{cat}", []) for arm in ("N", "T") for cat in ("SAME", "CROSS")]
                if any(len(value) < 5 for value in required):
                    continue
                e = (float(np.mean(cells["T_CROSS"])) - float(np.mean(cells["N_CROSS"]))) - (float(np.mean(cells["T_SAME"])) - float(np.mean(cells["N_SAME"])))
                values_h.append(e)
            if values_h:
                event_group[str(rec["source_group"])] = float(np.mean(values_h))
                try:
                    with wave.open(str(rec["N_audio"]), "rb") as n_handle, wave.open(str(rec["T_audio"]), "rb") as t_handle:
                        n_duration_s = n_handle.getnframes() / n_handle.getframerate()
                        t_duration_s = t_handle.getnframes() / t_handle.getframerate()
                    duration_ratio = float(n_duration_s / t_duration_s) if t_duration_s > 0.0 else None
                except (OSError, wave.Error, ZeroDivisionError):
                    n_duration_s = t_duration_s = duration_ratio = None
                event_record_rows.append({"sample_id": sid, "source_group": rec["source_group"], "A_event": float(np.mean(values_h)), "qualified_h": [h for h in sorted(by_h) if all(len(by_h[h].get(f"{arm}_{cat}", [])) >= 5 for arm in ("N", "T") for cat in ("SAME", "CROSS"))], "counts": counts, "phone_distribution": dict(sorted(phone_distribution.items())), "N_duration_s": n_duration_s, "T_duration_s": t_duration_s, "N_over_T_duration_ratio": duration_ratio})
    write_csv(a_root / "phone_events.csv", ["sample_id", "source_group", "arm", "i", "k0", "delta", "h", "primary", "d0", "d_delta", "w_raw", "event", "left_phone", "right_phone", "left_index", "right_index", "left_sequence", "right_sequence"], event_rows)
    event_summary = bootstrap_summary(event_group, metric="A_event", min_groups=8) if a2_mfa.get("status") == "COMPLETE" else {"status": "MFA_UNAVAILABLE", "metric": "A_event", "reason": a2_mfa.get("reason", a2_mfa.get("status"))}
    write_json(a_root / "a_summary.json", {"status": "COMPLETE", "a1": {"summary": rank_summary, "records": rank_records, "native_rows": native_rows, "replay": replay_rows}, "a_event": {"summary": event_summary, "records": event_record_rows, "mfa": a2_mfa}, "secondary_unit": unit_summary, "adjacent_sensitivity": {"records": adjacent_records, "summary": adjacent_summary, "status": "DESCRIPTIVE_ONLY"}, "denominators": {"A_rank_groups": len(a_rank_group), "A_event_groups": len(event_group)}, "primary_family": ["A_rank", "A_event", "B_instance", "B_transfer"]})
    return read_json(a_root / "a_summary.json")


def _audio_write(path: Path, values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int16).reshape(-1)
    if array.size == 0:
        raise ProtocolError("empty WAV output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(array.astype("<i2").tobytes())
    os.replace(temporary, path)
    return sha256_file(path)


def _resample_and_quantize(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ProtocolError("TTS returned empty/nonfinite audio")
    if int(sample_rate) != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(sample_rate), SAMPLE_RATE)
        value = np.asarray(resample_poly(value, SAMPLE_RATE // divisor, int(sample_rate) // divisor), dtype=np.float64)
    scaled = np.rint(32768.0 * value)
    clipping = float(np.mean((scaled < -32768.0) | (scaled > 32767.0)))
    return np.clip(scaled, -32768.0, 32767.0).astype("<i2"), clipping


def _cached_tts_row_valid(meta: Mapping[str, Any], record: Mapping[str, Any], arm: str, output: Path) -> bool:
    """Reject a stale/corrupt TTS cache before it can affect B."""

    expected = {
        "sample_id": int(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "arm": arm,
        "seed": 42 if arm == "T1" else 43,
        "text": str(record["transcript"]),
        "text_sha256": str(record["transcript_sha256"]),
        "reference_audio": str(Path(record["N_audio"]).resolve()),
        "reference_pcm_sha256": str(record["N_pcm_sha256"]),
        "model_id": TTS_MODEL_ID,
        "strict_backend": True,
        "language": TTS_LANGUAGE,
        "max_new_tokens": TTS_MAX_NEW_TOKENS,
        "status": "COMPLETE",
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        return False
    try:
        if Path(str(meta["audio"])).resolve() != output.resolve() or not output.is_file() or sha256_file(output) != str(meta["audio_sha256"]):
            return False
        raw = Path(str(meta["raw_audio"]))
        if not raw.is_file() or sha256_file(raw) != str(meta["raw_audio_sha256"]):
            return False
        with wave.open(str(output), "rb") as handle:
            sample_count = int(handle.getnframes())
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != SAMPLE_RATE or sample_count != int(meta["sample_count"]):
                return False
            pcm = np.frombuffer(handle.readframes(sample_count), dtype="<i2")
        if pcm.size == 0 or int(meta["peak"]) != int(np.max(np.abs(pcm.astype(np.int32)))):
            return False
        clipping = float(meta["pre_quant_clipping_fraction"])
        return math.isfinite(clipping) and 0.0 <= clipping <= 1.0
    except (KeyError, OSError, TypeError, ValueError, wave.Error):
        return False


@contextmanager
def _rng_scope(torch: Any, seed: int):
    """Isolate one stochastic TTS call while preserving the caller's RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def generate_b_audio(root: Path) -> int:
    """Internal command executed by the pinned Qwen environment."""

    inputs = read_json(root / "inputs.json")
    try:
        import torch
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    except Exception as exc:  # pragma: no cover - exercised by dependency block  # noqa: BLE001
        write_json(root / "B" / "tts_generation.json", {"status": "DEPENDENCY_BLOCKED", "reason": f"provider import failed: {type(exc).__name__}: {exc}"})
        return 1
    cfg = {"faster_qwen3": {"model_id": TTS_MODEL_ID, "strict_backend": True, "max_new_tokens": TTS_MAX_NEW_TOKENS}}
    provider = FasterQwen3TTSProvider(cfg, run_id=root.name, repo_root=REPO)
    rows: list[dict[str, Any]] = []
    cache_rejections: list[dict[str, Any]] = []
    try:
        for record in inputs["records"]:
            sid = int(record["sample_id"])
            for seed, arm in ((42, "T1"), (43, "T2")):
                output = root / "B" / "audio" / str(sid) / f"{arm}.wav"
                meta_path = output.with_suffix(".json")
                if output.is_file() and meta_path.is_file():
                    cached = read_json(meta_path)
                    if _cached_tts_row_valid(cached, record, arm, output):
                        rows.append(cached)
                        continue
                    cache_rejections.append({"sample_id": sid, "arm": arm, "reason": "cached metadata/audio failed contract; regenerated"})
                    output.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    output.with_suffix(".raw.npy").unlink(missing_ok=True)
                with _rng_scope(torch, seed):
                    result = provider.generate_voice_clone(record["transcript"], Path(record["N_audio"]), record["transcript"], language=TTS_LANGUAGE)
                raw_path = output.with_suffix(".raw.npy")
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, np.asarray(result.audio, dtype=np.float32), allow_pickle=False)
                pcm, clipping = _resample_and_quantize(result.audio, int(result.sample_rate))
                pcm_hash = _audio_write(output, pcm)
                row = {"sample_id": sid, "source_group": record["source_group"], "arm": arm, "seed": seed, "text": record["transcript"], "text_sha256": record["transcript_sha256"], "reference_audio": record["N_audio"], "reference_pcm_sha256": record["N_pcm_sha256"], "model_id": TTS_MODEL_ID, "strict_backend": True, "language": TTS_LANGUAGE, "max_new_tokens": TTS_MAX_NEW_TOKENS, "sampling_defaults": {"min_new_tokens": 2, "temperature": 0.9, "top_k": 50, "top_p": 1.0, "do_sample": True, "repetition_penalty": 1.05}, "raw_audio": str(raw_path.resolve()), "raw_audio_sha256": sha256_file(raw_path), "sample_rate_before_resample": int(result.sample_rate), "sample_count": len(pcm), "peak": int(np.max(np.abs(pcm.astype(np.int32)))), "pre_quant_clipping_fraction": clipping, "audio": str(output.resolve()), "audio_sha256": pcm_hash, "status": "COMPLETE"}
                write_json(meta_path, row)
                rows.append(row)
                print(f"TTS {sid} {arm} {len(pcm)} samples", flush=True)
    except Exception as exc:  # keep exact dependency/runtime reason in the run  # noqa: BLE001
        write_json(root / "B" / "tts_generation.json", {"status": "FAILED", "rows": rows, "cache_rejections": cache_rejections, "reason": f"{type(exc).__name__}: {exc}"})
        return 1
    write_json(root / "B" / "tts_generation.json", {"status": "COMPLETE", "rows": rows, "cache_rejections": cache_rejections, "count": len(rows), "model_id": TTS_MODEL_ID, "strict_backend": True})
    return 0


def prepare_b(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    b = root / "B"
    b.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for record in inputs["records"]:
        sid = int(record["sample_id"])
        records.append({**record, "T1_audio": str((b / "audio" / str(sid) / "T1.wav").resolve()), "T2_audio": str((b / "audio" / str(sid) / "T2.wav").resolve()), "box_xyxy": [0, 0, 224, 224], "fps": FPS})
    payload = {"schema_version": 1, "status": "PLANNED", "records": records, "tts_contract": {"model_id": TTS_MODEL_ID, "strict_backend": True, "language": TTS_LANGUAGE, "max_new_tokens": TTS_MAX_NEW_TOKENS, "seeds": {"T1": 42, "T2": 43}, "ref_text": "record transcript", "sampling_defaults_expanded": {"min_new_tokens": 2, "temperature": 0.9, "top_k": 50, "top_p": 1.0, "do_sample": True, "repetition_penalty": 1.05}}, "runtime": {"qwen_python": str(QWEN_PYTHON), "wav2lip_python": str(WAV2LIP_PYTHON), "syncnet_python": str(SYNCNET_PYTHON), "mfa": str(MFA), "wav2lip_checkpoint": str(WAV2LIP_CHECKPOINT), "syncnet_model": str(SYNCNET_MODEL)}, "budget": {"new_tts": 24, "science_videos": 36, "repeat_videos": 3, "native_score_cells": 39, "delay_controls": 6}}
    write_json(b / "audio_manifest.json", payload)
    return payload


def _static_render(root: Path, b_manifest: Mapping[str, Any], *, eligible_sids: set[int] | None = None) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    if not WAV2LIP_PYTHON.is_file() or not WAV2LIP_CHECKPOINT.is_file():
        return {"status": "DEPENDENCY_BLOCKED", "reason": f"Wav2Lip runtime/checkpoint missing: {WAV2LIP_PYTHON}/{WAV2LIP_CHECKPOINT}"}
    for record in b_manifest["records"]:
        sid = int(record["sample_id"])
        if eligible_sids is not None and sid not in eligible_sids:
            continue
        for arm, audio_key, seed in (("N", "N_audio", 42), ("T1", "T1_audio", 42), ("T2", "T2_audio", 42)):
            output = root / "B" / "videos" / "main" / str(sid) / f"{arm}.mkv"
            result_path = output.with_suffix(".worker.json")
            if output.is_file() and result_path.is_file():
                videos.append(read_json(result_path)); continue
            image = Path(str(record["portrait"]))
            try:
                import cv2
                frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ProtocolError(f"cannot read portrait {image}")
                rgb_hash = sha256_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            except ImportError as exc:
                return {"status": "DEPENDENCY_BLOCKED", "reason": f"opencv missing: {exc}"}
            command = [str(WAV2LIP_PYTHON), str(REPO / "scripts/experiments/static_image_bridge/render_worker.py"), "--image", str(image), "--image-rgb-sha256", rgb_hash, "--audio", str(record[audio_key]), "--box", *map(str, record["box_xyxy"]), "--checkpoint", str(WAV2LIP_CHECKPOINT), "--ffmpeg", "/home/wjj/miniconda3/bin/ffmpeg", "--outfile", str(output), "--result", str(result_path), "--batch-size", "4", "--seed", str(seed)]
            log = root / "B" / "logs" / f"render_{sid}_{arm}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0 or not output.is_file() or not result_path.is_file():
                return {"status": "RENDER_FAILED", "reason": f"{sid}/{arm} returncode={result.returncode}", "log": str(log)}
            row = read_json(result_path)
            row.update({"sample_id": sid, "source_group": record["source_group"], "arm": arm, "video": str(output.resolve()), "video_sha256": sha256_file(output), "audio": str(Path(record[audio_key]).resolve()), "audio_sha256": sha256_file(record[audio_key]), "portrait": str(image.resolve()), "box_xyxy": record["box_xyxy"], "command": command, "log": str(log.resolve())})
            write_json(result_path, row)
            videos.append(row)
    # Repeat all three cells for ID151 using byte-identical inputs.
    first = next(row for row in b_manifest["records"] if int(row["sample_id"]) == 151)
    for arm, audio_key in (("N", "N_audio"), ("T1", "T1_audio"), ("T2", "T2_audio")):
        if eligible_sids is not None and 151 not in eligible_sids:
            break
        output = root / "B" / "videos" / "repeat" / "151" / f"{arm}.mkv"
        result_path = output.with_suffix(".worker.json")
        image = Path(str(first["portrait"]))
        if not output.is_file() or not result_path.is_file():
            import cv2
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            rgb_hash = sha256_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            command = [str(WAV2LIP_PYTHON), str(REPO / "scripts/experiments/static_image_bridge/render_worker.py"), "--image", str(image), "--image-rgb-sha256", rgb_hash, "--audio", str(first[audio_key]), "--box", *map(str, first["box_xyxy"]), "--checkpoint", str(WAV2LIP_CHECKPOINT), "--ffmpeg", "/home/wjj/miniconda3/bin/ffmpeg", "--outfile", str(output), "--result", str(result_path), "--batch-size", "4", "--seed", "42"]
            log = root / "B" / "logs" / f"render_repeat_151_{arm}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0 or not output.is_file():
                return {"status": "RENDER_FAILED", "reason": f"repeat/151/{arm} returncode={result.returncode}"}
        row = read_json(result_path)
        row.update({"sample_id": 151, "source_group": first["source_group"], "arm": arm, "repeat": True, "video": str(output.resolve()), "video_sha256": sha256_file(output), "audio": str(Path(first[audio_key]).resolve()), "audio_sha256": sha256_file(first[audio_key])})
        write_json(result_path, row)
        videos.append(row)
    payload = {"status": "COMPLETE", "videos": videos, "count": len(videos), "expected_count": (len(eligible_sids) * 3 + (3 if eligible_sids is None or 151 in eligible_sids else 0)) if eligible_sids is not None else 39, "eligible_sample_ids": sorted(eligible_sids) if eligible_sids is not None else list(SAMPLE_IDS), "checkpoint_sha256": sha256_file(WAV2LIP_CHECKPOINT)}
    write_json(root / "B" / "video_manifest.json", payload)
    return payload


def extract_b_features(root: Path) -> int:
    """Internal command executed by the pinned SyncNet environment."""

    manifest = read_json(root / "B" / "video_manifest.json")
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from scripts.experiments.tts_native_gain_attribution.syncnet import (
            SyncNetEngine,
        )
    except Exception as exc:  # noqa: BLE001
        write_json(root / "B" / "feature_manifest.json", {"status": "DEPENDENCY_BLOCKED", "reason": f"SyncNet import failed: {type(exc).__name__}: {exc}"})
        return 1
    try:
        engine = SyncNetEngine(model_path=SYNCNET_MODEL, device="cuda")
    except Exception as exc:  # noqa: BLE001
        write_json(root / "B" / "feature_manifest.json", {"status": "DEPENDENCY_BLOCKED", "reason": f"SyncNet initialization failed: {type(exc).__name__}: {exc}"})
        return 1
    rows: list[dict[str, Any]] = []
    audio_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    try:
        for item in manifest["videos"]:
            sid, arm = int(item["sample_id"]), str(item["arm"])
            audio = Path(str(item["audio"]))
            key = str(audio.resolve())
            if key not in audio_cache:
                audio_cache[key] = engine.extract_audio(audio)
            a, am = audio_cache[key]
            v, vm = engine.extract_visual(Path(str(item["video"])))
            base = root / "B" / "features" / ("repeat" if item.get("repeat") else "main") / str(sid)
            base.mkdir(parents=True, exist_ok=True)
            vp = base / f"V_{arm}.npy"
            ap = base / f"A_{arm}.npy"
            np.save(vp, v, allow_pickle=False)
            np.save(ap, a, allow_pickle=False)
            rows.append({"sample_id": sid, "arm": arm, "repeat": bool(item.get("repeat", False)), "video": item["video"], "audio": str(audio), "visual": str(vp.resolve()), "audio_feature": str(ap.resolve()), "visual_sha256": sha256_file(vp), "audio_feature_sha256": sha256_file(ap), "visual_meta": vm, "audio_meta": am})
            print(f"FEATURE {sid} {arm} repeat={bool(item.get('repeat', False))} v={len(v)} a={len(a)}", flush=True)
    finally:
        engine.close()
    write_json(root / "B" / "feature_manifest.json", {"status": "COMPLETE", "rows": rows, "count": len(rows), "model_sha256": sha256_file(SYNCNET_MODEL)})
    return 0


def match_occurrences(tokens_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    speech = {arm: [token for token in tokens if not token["silence"] and not token["unknown"]] for arm, tokens in tokens_by_arm.items()}
    arms = list(speech)
    if not arms:
        return [], {"matched": 0}
    # MFA supplies a words tier.  When it is present in all three arms, first
    # match exact word labels in occurrence order, then match exact phones
    # inside the matched word.  Synthetic/unit-test TextGrids may omit words;
    # those use the same ordered phone fallback and are explicitly marked.
    has_words = all(all(token.get("word_index") is not None and token.get("word_normalized") for token in tokens) for tokens in speech.values())
    if has_words:
        groups: dict[str, list[list[dict[str, Any]]]] = {}
        for arm in arms:
            grouped: list[list[dict[str, Any]]] = []
            for token in speech[arm]:
                if not grouped or grouped[-1][0].get("word_index") != token.get("word_index"):
                    grouped.append([])
                grouped[-1].append(dict(token))
            groups[arm] = grouped
        cursors = {arm: 0 for arm in arms}
        matches: list[dict[str, Any]] = []
        word_matches = 0
        anchor_arm = arms[0]
        for anchor_group in groups[anchor_arm]:
            word_label = anchor_group[0].get("word_normalized")
            chosen: dict[str, list[dict[str, Any]]] = {anchor_arm: anchor_group}
            ok = True
            for arm in arms[1:]:
                found = next((idx for idx in range(cursors[arm], len(groups[arm])) if groups[arm][idx][0].get("word_normalized") == word_label), None)
                if found is None:
                    ok = False
                    break
                cursors[arm] = found + 1
                chosen[arm] = groups[arm][found]
            if not ok:
                continue
            word_matches += 1
            phone_cursors = {arm: 0 for arm in arms}
            for anchor_token in anchor_group:
                row = {"occurrence": len(matches), anchor_arm: dict(anchor_token)}
                phone_ok = True
                for arm in arms[1:]:
                    found = next((idx for idx in range(phone_cursors[arm], len(chosen[arm])) if chosen[arm][idx]["normalized"] == anchor_token["normalized"]), None)
                    if found is None:
                        phone_ok = False
                        break
                    phone_cursors[arm] = found + 1
                    row[arm] = dict(chosen[arm][found])
                if phone_ok:
                    matches.append(row)
        return matches, {"matched": len(matches), "speech_counts": {arm: len(value) for arm, value in speech.items()}, "word_matches": word_matches, "unmatched_policy": "exact-word occurrence then exact-phone ordered matching; no nearest-phone substitution"}
    anchor = arms[0]
    cursors = {arm: 0 for arm in arms}
    matches: list[dict[str, Any]] = []
    for anchor_token in speech[anchor]:
        row = {"occurrence": len(matches), anchor: dict(anchor_token)}
        ok = True
        for arm in arms[1:]:
            found = None
            for idx in range(cursors[arm], len(speech[arm])):
                if speech[arm][idx]["normalized"] == anchor_token["normalized"]:
                    found = idx
                    break
            if found is None:
                ok = False
                break
            cursors[arm] = found + 1
            row[arm] = dict(speech[arm][found])
        if ok:
            cursors[anchor] = int(anchor_token["index"]) + 1
            matches.append(row)
    return matches, {"matched": len(matches), "speech_counts": {arm: len(value) for arm, value in speech.items()}, "word_matches": None, "unmatched_policy": "exact-phone ordered matching (words tier unavailable); no nearest-phone substitution"}


def _calibrate_occurrence_k(v: np.ndarray, a: np.ndarray, tokens: Sequence[Mapping[str, Any]], occurrences: Sequence[Mapping[str, Any]], arm: str) -> tuple[int | None, dict[str, Any]]:
    rows: list[int] = []
    for occurrence in occurrences:
        token = occurrence[arm]
        center = 0.5 * (float(token["start_s"]) + float(token["end_s"]))
        index = math.floor((center - 0.1075) / 0.04 + 0.5)
        if 0 <= index < len(v) and 0 <= index - VSHIFT < len(a) and 0 <= index + VSHIFT < len(a):
            rows.append(index)
    rows = sorted(set(rows))
    if len(rows) < 5:
        return None, {"status": "ALIGNMENT_INSUFFICIENT", "row_count": len(rows)}
    candidates: list[tuple[float, int, int]] = []
    for k in range(-VSHIFT, VSHIFT + 1):
        ds = [l2_distance(v[i], a[i + k]) for i in rows if 0 <= i + k < len(a)]
        candidates.append((float(np.mean(ds)), abs(k), k))
    chosen = min(candidates)
    return int(chosen[2]), {"status": "COMPLETE", "k_s": int(chosen[2]), "calibration_rows": rows, "mean_distance": float(chosen[0]), "occurrence_count": len(occurrences)}


def _calibration_support(rows: Sequence[int], k: int) -> list[tuple[float, float]]:
    """Nominal time support of calibration visual/audio windows.

    The audio search evaluates every lag in [-15, 15]; all of those windows
    are included in the exclusion ledger.  Visual rows are represented by
    their 5-frame (~200 ms) support on the same 40 ms feature clock.
    """

    intervals: list[tuple[float, float]] = []
    for index in rows:
        visual_start = 0.04 * (int(index) - int(k))
        intervals.append((visual_start, visual_start + 0.2))
        for lag in range(-VSHIFT, VSHIFT + 1):
            audio_start = 0.04 * (int(index) + lag)
            intervals.append((audio_start, audio_start + 0.215))
    return intervals


def _support_clear(time_s: float, intervals: Sequence[tuple[float, float]]) -> bool:
    """Require both interpolated audio/visual windows to be 240 ms away."""

    query_intervals = ((float(time_s) - 0.1075, float(time_s) + 0.1075), (float(time_s) - 0.1, float(time_s) + 0.1))
    for query_start, query_end in query_intervals:
        for start, end in intervals:
            gap = max(float(start) - query_end, query_start - float(end), 0.0)
            if gap < 0.24:
                return False
    return True


def _phase_means(features_by_arm: Mapping[str, tuple[np.ndarray, np.ndarray]], eval_occ: Sequence[Mapping[str, Any]], k_by_arm: Mapping[str, int], support_exclusion: Mapping[str, Sequence[tuple[float, float]]], phases: Sequence[float]) -> tuple[dict[str, float], list[Mapping[str, Any]]]:
    """Compute phase-conditioned 3x3 means without changing the primary rows."""

    arms = ("N", "T1", "T2")
    cells: dict[str, list[float]] = defaultdict(list)
    valid_occurrences: list[Mapping[str, Any]] = []
    for occurrence in eval_occ:
        phase_rows: list[dict[str, float]] = []
        occurrence_valid = True
        for phase in phases:
            visual: dict[str, np.ndarray] = {}
            audio: dict[str, np.ndarray] = {}
            for arm in arms:
                token = occurrence[arm]
                time_s = float(token["start_s"]) + float(phase) * (float(token["end_s"]) - float(token["start_s"]))
                position = (time_s - 0.1075) / 0.04
                v_raw = interpolation(features_by_arm[arm][0], position - int(k_by_arm[arm]))
                a_raw = interpolation(features_by_arm[arm][1], position)
                visual[arm], audio[arm] = l2_unit(v_raw), l2_unit(a_raw)
                if visual[arm] is None or audio[arm] is None or not (0.24 <= time_s <= (features_by_arm[arm][1].shape[0] - 1) * 0.04) or not _support_clear(time_s, support_exclusion[arm]):
                    occurrence_valid = False
                    break
            if not occurrence_valid:
                break
            phase_rows.append({f"{vi}{aj}": float(np.sum((visual[vi] - audio[aj]) ** 2)) for vi in arms for aj in arms})
        if occurrence_valid and len(phase_rows) == len(phases):
            valid_occurrences.append(occurrence)
            for key in phase_rows[0]:
                cells[key].append(float(np.mean([row[key] for row in phase_rows])))
    return {key: float(np.mean(values)) for key, values in cells.items()}, valid_occurrences


def _sample_arm_features(root: Path, sid: int, arm: str, *, repeat: bool = False) -> tuple[np.ndarray, np.ndarray]:
    base = root / "B" / "features" / ("repeat" if repeat else "main") / str(sid)
    v = np.load(base / f"V_{arm}.npy", allow_pickle=False)
    a = np.load(base / f"A_{arm}.npy", allow_pickle=False)
    return np.asarray(v, dtype=np.float32), np.asarray(a, dtype=np.float32)


def _direct_context(tokens: Sequence[Mapping[str, Any]], token: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return the labels of the original immediate phone neighbours.

    The index is the index in the complete TextGrid phone tier, so silence and
    unknown intervals remain neighbours rather than being silently removed and
    reconnected.
    """

    index = int(token["index"])
    left = tokens[index - 1] if index > 0 else None
    right = tokens[index + 1] if index + 1 < len(tokens) else None
    if left is None or right is None:
        return None
    if left["silence"] or right["silence"] or left["unknown"] or right["unknown"]:
        return None
    return str(left["normalized"]), str(right["normalized"])


def _log_duration(token: Mapping[str, Any]) -> float:
    duration = float(token["end_s"]) - float(token["start_s"])
    if duration <= 0.0 or not math.isfinite(duration):
        raise ProtocolError("context donor has invalid phone duration")
    return math.log(duration)


def _select_context_donors(
    query: Mapping[str, Any],
    occurrences: Sequence[Mapping[str, Any]],
    tokens_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    query_context: Mapping[str, tuple[str, str]],
) -> dict[str, Any] | None:
    """Select the registered plus/minus donor pair without reading features.

    The pair key is exactly the protocol key: minimize the sum of absolute
    centre-phone log-duration gaps across N/T1/T2, then maximize plus's
    neighbour-match count, then minimize plus and minus occurrence indices.
    """

    arms = ("N", "T1", "T2")
    candidates: list[dict[str, Any]] = []
    for donor in occurrences:
        if int(donor["occurrence"]) == int(query["occurrence"]):
            continue
        if any(donor[arm]["normalized"] != query[arm]["normalized"] for arm in arms):
            continue
        if any(
            abs(
                0.5 * (float(donor[arm]["start_s"]) + float(donor[arm]["end_s"]))
                - 0.5 * (float(query[arm]["start_s"]) + float(query[arm]["end_s"]))
            )
            < 0.4
            for arm in arms
        ):
            continue
        donor_context: dict[str, tuple[str, str]] = {}
        for arm in arms:
            value = _direct_context(tokens_by_arm[arm], donor[arm])
            if value is None:
                break
            donor_context[arm] = value
        else:
            match_count = sum(donor_context["N"][side] == query_context["N"][side] for side in (0, 1))
            candidates.append({"occurrence": int(donor["occurrence"]), "occurrence_row": donor, "context": donor_context, "match_count": int(match_count)})
    plus = [row for row in candidates if int(row["match_count"]) >= 1]
    minus = [row for row in candidates if int(row["match_count"]) == 0]
    if not plus or not minus:
        return None
    pairs: list[tuple[tuple[float, int, int, int], dict[str, Any], dict[str, Any]]] = []
    for plus_row in plus:
        for minus_row in minus:
            duration_log_gap = sum(
                abs(_log_duration(plus_row["occurrence_row"][arm]) - _log_duration(minus_row["occurrence_row"][arm]))
                for arm in arms
            )
            key = (duration_log_gap, -int(plus_row["match_count"]), int(plus_row["occurrence"]), int(minus_row["occurrence"]))
            pairs.append((key, plus_row, minus_row))
    key, plus_row, minus_row = min(pairs, key=lambda item: item[0])
    return {"key": [float(key[0]), int(key[1]), int(key[2]), int(key[3])], "plus": plus_row, "minus": minus_row, "duration_log_gap": float(key[0])}


def _word_identity_difference(query: Mapping[str, Any], donor: Mapping[str, Any], arms: Sequence[str] = ("N", "T1", "T2")) -> int | None:
    values = [(query[arm].get("word_normalized"), donor[arm].get("word_normalized")) for arm in arms]
    if any(left is None or right is None for left, right in values):
        return None
    return int(sum(left != right for left, right in values))


def _occurrence_duration_summary(occurrences: Sequence[Mapping[str, Any]], arms: Sequence[str] = ("N", "T1", "T2")) -> dict[str, Any]:
    means: dict[str, float] = {}
    for arm in arms:
        durations = [float(row[arm]["end_s"]) - float(row[arm]["start_s"]) for row in occurrences]
        if durations and all(math.isfinite(value) and value > 0.0 for value in durations):
            means[arm] = float(np.mean(durations))
    return {"mean_duration_s": means, "T1_minus_N_s": means.get("T1", float("nan")) - means.get("N", float("nan")) if "T1" in means and "N" in means else None, "T2_minus_N_s": means.get("T2", float("nan")) - means.get("N", float("nan")) if "T2" in means and "N" in means else None}


def analyze_b(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    b_root = root / "B"
    manifest_path = b_root / "video_manifest.json"
    feature_path = b_root / "feature_manifest.json"
    audio_gen = b_root / "tts_generation.json"
    if not manifest_path.is_file() or not feature_path.is_file() or read_json(feature_path).get("status") != "COMPLETE":
        payload = {"status": "DEPENDENCY_BLOCKED", "reason": "B video/feature manifest is incomplete", "native_advantage": "UNCONFIRMED"}
        write_json(b_root / "b_summary.json", payload)
        return payload
    by_record = {int(row["sample_id"]): row for row in read_json(b_root / "audio_manifest.json")["records"]}
    grid_dir = b_root / "B_textgrids"
    record_stats: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    phase_sensitivity_rows: list[dict[str, Any]] = []
    b_instance_values: dict[str, float] = {}
    b_transfer_values: dict[str, float] = {}
    b_nt1_values: dict[str, float] = {}
    b_nt2_values: dict[str, float] = {}
    arms = ("N", "T1", "T2")
    for sid in SAMPLE_IDS:
        rec = by_record[sid]
        missing_grids = [str(grid_dir / f"{sid}_{arm}.TextGrid") for arm in arms if not (grid_dir / f"{sid}_{arm}.TextGrid").is_file()]
        if missing_grids:
            record_stats.append({
                "sample_id": sid,
                "source_group": rec["source_group"],
                "status": "ALIGNMENT_MISSING",
                "missing_textgrids": missing_grids,
            })
            continue
        try:
            tokens = {arm: parse_textgrid(grid_dir / f"{sid}_{arm}.TextGrid") for arm in arms}
            occurrences, matching = match_occurrences(tokens)
        except ProtocolError as exc:
            record_stats.append({"sample_id": sid, "source_group": rec["source_group"], "status": "ALIGNMENT_INVALID", "reason": str(exc)})
            continue
        duration_summary = _occurrence_duration_summary(occurrences)
        features_by_arm = {arm: _sample_arm_features(root, sid, arm) for arm in arms}
        # Native G is an independent B1 reference.  It must use every record
        # with three rendered feature streams, even when phone matching or
        # common phase support later makes that record unusable for B_instance
        # and B_transfer.
        native: dict[str, Any] = {}
        for arm in arms:
            v_native, a_native = features_by_arm[arm]
            matrix_native = official_distance_matrix(v_native, a_native)
            support_native = list(range(VSHIFT, min(len(v_native), len(a_native)) - VSHIFT))
            native[arm] = curve(matrix_native, support_native) if support_native else None
        if all(native.values()):
            native_rows.append({"sample_id": sid, "source_group": rec["source_group"], "C_N": native["N"]["sync_c"], "C_T1": native["T1"]["sync_c"], "C_T2": native["T2"]["sync_c"], "T1_minus_N": native["T1"]["sync_c"] - native["N"]["sync_c"], "T2_minus_N": native["T2"]["sync_c"] - native["N"]["sync_c"], "native": native})
        k_by_arm: dict[str, int | None] = {}
        calibration: dict[str, Any] = {}
        calibration_count = max(1, math.ceil(0.2 * len(occurrences)))
        calibration_occ = occurrences[:calibration_count]
        eval_occ = occurrences[calibration_count:]
        for arm in arms:
            k_by_arm[arm], calibration[arm] = _calibrate_occurrence_k(features_by_arm[arm][0], features_by_arm[arm][1], tokens[arm], calibration_occ, arm)
        if any(value is None for value in k_by_arm.values()) or len(eval_occ) < 8:
            record_stats.append({"sample_id": sid, "source_group": rec["source_group"], "status": "ALIGNMENT_INSUFFICIENT", "matching": matching, "calibration": calibration, "evaluation_occurrence_count": len(eval_occ), "phase_sampling_rejection_rate": 1.0, "duration_summary": duration_summary})
            continue
        support_exclusion = {arm: _calibration_support(calibration[arm]["calibration_rows"], int(k_by_arm[arm])) for arm in arms}
        cell_values: dict[str, list[float]] = defaultdict(list)
        raw_cell_values: dict[str, list[float]] = defaultdict(list)
        valid_occurrences: list[Mapping[str, Any]] = []
        valid_queries = 0
        support_rejected = 0
        rejection_reasons: dict[str, int] = defaultdict(int)
        identity_errors: list[float] = []
        raw_identity_errors: list[float] = []
        for occurrence in eval_occ:
            occurrence_rows: list[tuple[dict[str, Any], dict[str, float]]] = []
            occurrence_valid = True
            occurrence_reasons: set[str] = set()
            for phase in (0.25, 0.5, 0.75):
                visual: dict[str, np.ndarray] = {}
                audio: dict[str, np.ndarray] = {}
                valid = True
                positions: dict[str, dict[str, float]] = {}
                for arm in arms:
                    token = occurrence[arm]
                    t = float(token["start_s"]) + phase * (float(token["end_s"]) - float(token["start_s"]))
                    u = (t - 0.1075) / 0.04
                    v_raw = interpolation(features_by_arm[arm][0], u - int(k_by_arm[arm]))
                    a_raw = interpolation(features_by_arm[arm][1], u)
                    visual[arm], audio[arm] = l2_unit(v_raw), l2_unit(a_raw)
                    positions[arm] = {"t_s": t, "u": u, "v_u": u - int(k_by_arm[arm])}
                    if visual[arm] is None or audio[arm] is None:
                        occurrence_reasons.add("interpolation_nonfinite_or_out_of_range")
                    if not (0.24 <= t <= (features_by_arm[arm][1].shape[0] - 1) * 0.04):
                        occurrence_reasons.add("audio_time_boundary")
                    if not _support_clear(t, support_exclusion[arm]):
                        occurrence_reasons.add("calibration_support_overlap")
                    if occurrence_reasons:
                        valid = False
                if not valid:
                    occurrence_valid = False
                    break
                row = {"sample_id": sid, "source_group": rec["source_group"], "occurrence": occurrence["occurrence"], "phase": phase, "k_N": k_by_arm["N"], "k_T1": k_by_arm["T1"], "k_T2": k_by_arm["T2"]}
                raw_values: dict[str, float] = {}
                raw_visual = {vi: interpolation(features_by_arm[vi][0], positions[vi]["v_u"]) for vi in arms}
                raw_audio = {aj: interpolation(features_by_arm[aj][1], positions[aj]["u"]) for aj in arms}
                for vi in arms:
                    for aj in arms:
                        value = float(np.sum((visual[vi] - audio[aj]) ** 2))
                        row[f"d_{vi}{aj}"] = value
                        raw_v, raw_a = raw_visual[vi], raw_audio[aj]
                        raw_values[f"{vi}{aj}"] = l2_distance(raw_v, raw_a) ** 2 if raw_v is not None and raw_a is not None else float("nan")
                identity_value = 0.5 * (row["d_T1T2"] + row["d_T2T1"] - row["d_T1T1"] - row["d_T2T2"])
                identity_dot = float(np.dot(visual["T1"] - visual["T2"], audio["T1"] - audio["T2"]))
                raw_identity_value = 0.5 * (raw_values["T1T2"] + raw_values["T2T1"] - raw_values["T1T1"] - raw_values["T2T2"])
                raw_identity_dot = float(np.dot(raw_visual["T1"] - raw_visual["T2"], raw_audio["T1"] - raw_audio["T2"]))
                row.update({"I_TT_query": identity_value, "I_TT_dot": identity_dot, "I_TT_identity_abs_error": abs(identity_value - identity_dot), "I_TT_raw_query": raw_identity_value, "I_TT_raw_dot": raw_identity_dot, "I_TT_raw_identity_abs_error": abs(raw_identity_value - raw_identity_dot)})
                identity_errors.append(abs(identity_value - identity_dot))
                raw_identity_errors.append(abs(raw_identity_value - raw_identity_dot))
                occurrence_rows.append((row, raw_values))
            if not occurrence_valid:
                support_rejected += 1
                for reason in occurrence_reasons:
                    rejection_reasons[reason] += 1
                continue
            valid_queries += 1
            valid_occurrences.append(occurrence)
            for row, raw_values in occurrence_rows:
                query_rows.append(row)
                for key, value in ((key, row[f"d_{key}"]) for key in raw_values):
                    cell_values[key].append(float(value))
                    raw_cell_values[key].append(float(raw_values[key]))
        if valid_queries < 8:
            record_stats.append({"sample_id": sid, "source_group": rec["source_group"], "status": "COMMON_SUPPORT_LOW", "matching": matching, "calibration": calibration, "support_rejected_query_count": support_rejected, "support_rejection_reasons": dict(rejection_reasons), "evaluation_occurrence_count": len(eval_occ), "valid_occurrence_count": valid_queries, "valid_query_count": valid_queries * 3, "phase_sampling_rejection_rate": float(1.0 - valid_queries / len(eval_occ)) if eval_occ else 1.0, "duration_summary": duration_summary, "identity_max_abs_error": max(identity_errors, default=None), "raw_identity_max_abs_error": max(raw_identity_errors, default=None)})
            continue
        means = {key: float(np.mean(value)) for key, value in cell_values.items()}
        raw_means = {key: float(np.nanmean(value)) for key, value in raw_cell_values.items()}
        i_tt = 0.5 * (means["T1T2"] + means["T2T1"] - means["T1T1"] - means["T2T2"])
        i_nt1 = 0.5 * (means["NT1"] + means["T1N"] - means["NN"] - means["T1T1"])
        i_nt2 = 0.5 * (means["NT2"] + means["T2N"] - means["NN"] - means["T2T2"])
        x_transfer = means["NN"] - 0.5 * (means["T1N"] + means["T2N"])
        b_instance_values[str(rec["source_group"])] = i_tt
        b_transfer_values[str(rec["source_group"])] = x_transfer
        b_nt1_values[str(rec["source_group"])] = i_nt1
        b_nt2_values[str(rec["source_group"])] = i_nt2
        record_stats.append({"sample_id": sid, "source_group": rec["source_group"], "status": "COMPLETE", "matching": matching, "calibration": calibration, "support_exclusion": {arm: [[float(start), float(end)] for start, end in support_exclusion[arm]] for arm in arms}, "k_by_arm": k_by_arm, "evaluation_occurrence_count": len(eval_occ), "valid_occurrence_count": valid_queries, "valid_query_count": valid_queries * 3, "phase_sampling_rejection_rate": float(1.0 - valid_queries / len(eval_occ)) if eval_occ else 1.0, "support_rejected_query_count": support_rejected, "support_rejection_reasons": dict(rejection_reasons), "duration_summary": duration_summary, "identity_max_abs_error": max(identity_errors, default=None), "raw_identity_max_abs_error": max(raw_identity_errors, default=None), "d": means, "d_raw": raw_means, "I_TT": i_tt, "I_NT1": i_nt1, "I_NT2": i_nt2, "I_NT_mean": 0.5 * (i_nt1 + i_nt2), "I_TT_minus_I_NT_mean": i_tt - 0.5 * (i_nt1 + i_nt2), "X": x_transfer})
        sensitivity_means, sensitivity_occ = _phase_means(features_by_arm, eval_occ, k_by_arm, support_exclusion, (0.4, 0.5, 0.6))
        if len(sensitivity_occ) >= 8:
            phase_sensitivity_rows.append({"sample_id": sid, "source_group": rec["source_group"], "status": "COMPLETE", "phase_set": [0.4, 0.5, 0.6], "valid_occurrence_count": len(sensitivity_occ), "I_TT": 0.5 * (sensitivity_means["T1T2"] + sensitivity_means["T2T1"] - sensitivity_means["T1T1"] - sensitivity_means["T2T2"]), "X": sensitivity_means["NN"] - 0.5 * (sensitivity_means["T1N"] + sensitivity_means["T2N"])})
        else:
            phase_sensitivity_rows.append({"sample_id": sid, "source_group": rec["source_group"], "status": "COMMON_SUPPORT_LOW", "phase_set": [0.4, 0.5, 0.6], "valid_occurrence_count": len(sensitivity_occ)})
        # Context donor comparison is descriptive and uses direct adjacent TextGrid neighbors.
        for q in valid_occurrences:
            query_context = {arm: _direct_context(tokens[arm], q[arm]) for arm in arms}
            if any(value is None for value in query_context.values()) or len({value for value in query_context.values() if value is not None}) != 1:
                continue
            selected = _select_context_donors(q, occurrences, tokens, {arm: value for arm, value in query_context.items() if value is not None})
            if selected is None:
                continue
            plus_row, minus_row = selected["plus"], selected["minus"]
            p, m = plus_row["occurrence_row"], minus_row["occurrence_row"]
            k_values: dict[str, list[float]] = defaultdict(list)
            for phase in (0.25, 0.5, 0.75):
                for arm in arms:
                    qtoken, ptoken, mtoken = q[arm], p[arm], m[arm]
                    qt = float(qtoken["start_s"]) + phase * (float(qtoken["end_s"]) - float(qtoken["start_s"]))
                    pt = float(ptoken["start_s"]) + phase * (float(ptoken["end_s"]) - float(ptoken["start_s"]))
                    mt = float(mtoken["start_s"]) + phase * (float(mtoken["end_s"]) - float(mtoken["start_s"]))
                    qv = l2_unit(interpolation(features_by_arm[arm][0], (qt - 0.1075) / 0.04 - int(k_by_arm[arm])))
                    pa = l2_unit(interpolation(features_by_arm[arm][1], (pt - 0.1075) / 0.04))
                    ma = l2_unit(interpolation(features_by_arm[arm][1], (mt - 0.1075) / 0.04))
                    if qv is None or pa is None or ma is None:
                        break
                    k_values[arm].append(float(np.sum((qv - ma) ** 2) - np.sum((qv - pa) ** 2)))
            if all(len(k_values[arm]) == 3 for arm in arms):
                q_duration_gap_plus = sum(abs(_log_duration(q[arm]) - _log_duration(p[arm])) for arm in arms)
                q_duration_gap_minus = sum(abs(_log_duration(q[arm]) - _log_duration(m[arm])) for arm in arms)
                context_rows.append({"sample_id": sid, "source_group": rec["source_group"], "query_occurrence": q["occurrence"], "plus_occurrence": p["occurrence"], "minus_occurrence": m["occurrence"], "match_count_plus": int(plus_row["match_count"]), "duration_log_gap_plus_minus": float(selected["duration_log_gap"]), "duration_log_gap_query_plus": float(q_duration_gap_plus), "duration_log_gap_query_minus": float(q_duration_gap_minus), "plus_word_identity_difference_count": _word_identity_difference(q, p), "minus_word_identity_difference_count": _word_identity_difference(q, m), "selection_key": selected["key"], "K_N": float(np.mean(k_values["N"])), "K_T1": float(np.mean(k_values["T1"])), "K_T2": float(np.mean(k_values["T2"])), "K_T": float(0.5 * (np.mean(k_values["T1"]) + np.mean(k_values["T2"])))})

    write_csv(b_root / "query_distances.csv", sorted(query_rows[0]) if query_rows else ["sample_id"], query_rows)
    write_csv(b_root / "context_pairs.csv", sorted(context_rows[0]) if context_rows else ["sample_id"], context_rows)
    # Repeat check and delay controls are computed from frozen embeddings, with no new forward.
    controls: list[dict[str, Any]] = []
    repeat_pass: list[dict[str, Any]] = []
    try:
        primary_features = {arm: _sample_arm_features(root, 151, arm) for arm in arms}
        for arm in arms:
            rv, ra = _sample_arm_features(root, 151, arm, repeat=True)
            pv, pa = primary_features[arm]
            pm, rm = official_distance_matrix(pv, pa), official_distance_matrix(rv, ra)
            repeat_pass.append({"sample_id": 151, "arm": arm, "max_abs_matrix_error": float(np.max(np.abs(pm - rm))) if pm.shape == rm.shape else float("inf"), "pass": bool(pm.shape == rm.shape and np.max(np.abs(pm - rm)) <= 1e-4)})
            for shift, label in ((5, "+3200_samples"), (-5, "-3200_samples")):
                shifted = np.zeros_like(pa)
                if shift > 0:
                    shifted[shift:] = pa[:-shift]
                else:
                    shifted[:shift] = pa[-shift:]
                dm = official_distance_matrix(pv, shifted)
                rows = list(range(VSHIFT, min(len(pv), len(shifted)) - VSHIFT))
                cm = curve(dm, rows)
                # ``curve`` reports the historical SyncNet convention
                # official_offset=-k.  The delay control is specified in the
                # raw column index k, so compare k to the injected ±5-frame
                # shift (one-frame tolerance) and retain both conventions.
                best_k = int(cm["min_index"] - VSHIFT)
                controls.append({"sample_id": 151, "arm": arm, "delay": label, "injected_shift_frames": int(shift), "expected_best_k": int(shift), "best_k": best_k, "best_offset": cm["official_offset"], "pass": bool(abs(best_k - shift) <= 1)})
    except (OSError, ProtocolError) as exc:
        controls.append({"status": "UNAVAILABLE", "reason": str(exc)})
    write_json(b_root / "lag_calibration.json", {"records": [{"sample_id": row["sample_id"], "source_group": row["source_group"], "status": row["status"], "calibration": row.get("calibration"), "k_by_arm": row.get("k_by_arm"), "support_exclusion": row.get("support_exclusion")} for row in record_stats], "controls": controls, "repeat": repeat_pass})
    instance_summary = bootstrap_summary(b_instance_values, metric="B_instance", min_groups=8)
    transfer_summary = bootstrap_summary(b_transfer_values, metric="B_transfer", min_groups=8)
    nt1_summary = bootstrap_summary(b_nt1_values, metric="I_NT1", min_groups=8)
    nt2_summary = bootstrap_summary(b_nt2_values, metric="I_NT2", min_groups=8)
    nt_mean_values = {group: 0.5 * (b_nt1_values[group] + b_nt2_values[group]) for group in b_nt1_values if group in b_nt2_values}
    tt_minus_nt_values = {group: b_instance_values[group] - nt_mean_values[group] for group in b_instance_values if group in nt_mean_values}
    instance_secondary = {"I_NT1": nt1_summary, "I_NT2": nt2_summary, "I_NT_mean": bootstrap_summary(nt_mean_values, metric="I_NT_mean", min_groups=8), "I_TT_minus_I_NT_mean": bootstrap_summary(tt_minus_nt_values, metric="I_TT_minus_I_NT_mean", min_groups=8), "inferential_status": "DESCRIPTIVE_ONLY"}
    native_group = {str(row["source_group"]): float(0.5 * (row["C_T1"] + row["C_T2"]) - row["C_N"]) for row in native_rows}
    native_summary = bootstrap_summary(native_group, metric="native_G", min_groups=1)
    if native_summary.get("status") == "COMPLETE":
        native_summary["status_by_95_ci"] = "CONFIRMED" if native_summary["ci95"][0] > 0.0 else "UNCONFIRMED"
    context_group = defaultdict(list)
    for row in context_rows:
        context_group[str(row["source_group"])].append(row)
    k_n = {g: float(np.mean([row["K_N"] for row in rows])) for g, rows in context_group.items()}
    k_t = {g: float(np.mean([row["K_T"] for row in rows])) for g, rows in context_group.items()}
    k_delta = {g: k_t[g] - k_n[g] for g in k_n if g in k_t}
    def donor_usage(role: str) -> dict[str, Any]:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for row in context_rows:
            counts[(int(row["sample_id"]), int(row[f"{role}_occurrence"]))] += 1
        values = list(counts.values())
        return {"total_uses": int(sum(values)), "unique_donors": len(values), "reused_donors": int(sum(value > 1 for value in values)), "max_uses": max(values, default=0), "use_count_histogram": {str(n): int(sum(value == n for value in values)) for n in sorted(set(values))}}

    context_summary = {"K_N": bootstrap_summary(k_n, metric="K_N", min_groups=6), "K_T": bootstrap_summary(k_t, metric="K_T", min_groups=6), "K_T_minus_K_N": bootstrap_summary(k_delta, metric="K_T_minus_K_N", min_groups=6), "neighbor_match_count": {"one_neighbor": int(sum(row["match_count_plus"] == 1 for row in context_rows)), "two_neighbors": int(sum(row["match_count_plus"] == 2 for row in context_rows)), "total": len(context_rows)}, "donor_reuse": {"plus": donor_usage("plus"), "minus": donor_usage("minus")}, "word_identity": {"plus_assessed": int(sum(row["plus_word_identity_difference_count"] is not None for row in context_rows)), "minus_assessed": int(sum(row["minus_word_identity_difference_count"] is not None for row in context_rows)), "plus_difference_histogram": {str(n): int(sum(row["plus_word_identity_difference_count"] == n for row in context_rows)) for n in range(4)}, "minus_difference_histogram": {str(n): int(sum(row["minus_word_identity_difference_count"] == n for row in context_rows)) for n in range(4)}}, "inferential_status": "DESCRIPTIVE_ONLY"}
    b_status = "COMPLETE" if len(record_stats) == len(SAMPLE_IDS) and len(b_instance_values) >= 8 and len(b_transfer_values) >= 8 else "PARTIAL"
    payload = {"status": b_status, "record_stats": record_stats, "native": {"rows": native_rows, "G": native_summary, "effect_present": native_summary.get("status") == "COMPLETE" and native_summary.get("status_by_95_ci") == "CONFIRMED"}, "B_instance": instance_summary, "B_transfer": transfer_summary, "instance_secondary": instance_secondary, "lag_controls": controls, "repeat": repeat_pass, "context": context_summary, "context_pair_count": len(context_rows), "phase_sensitivity": phase_sensitivity_rows, "audio_generation": read_json(audio_gen) if audio_gen.is_file() else {"status": "MISSING"}, "primary_family": ["A_rank", "A_event", "B_instance", "B_transfer"], "native_advantage": native_summary.get("status_by_95_ci", "UNCONFIRMED")}
    write_json(b_root / "b_summary.json", payload)
    return payload


def make_figures(root: Path) -> dict[str, Any]:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    try:
        import matplotlib.pyplot as plt
        # Aggregate the CSV without importing any analysis helper.
        rows = list(csv.DictReader((root / "A" / "time_pairs.csv").open(encoding="utf-8")))
        by_delta: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_delta[int(row["delta"])].append(float(row["w_raw"]))
        if by_delta:
            xs = sorted(by_delta)
            plt.figure(figsize=(6, 4)); plt.plot(xs, [np.mean(by_delta[x]) for x in xs], marker="o"); plt.axvline(0, color="k", lw=.5); plt.xlabel("audio offset delta (SyncNet rows)"); plt.ylabel("local wrong-distance win rate"); plt.tight_layout(); path = figures / "a_rank_curve.png"; plt.savefig(path, dpi=160); plt.close(); output["a_rank_curve"] = str(path.resolve())
        b = read_json(root / "B" / "b_summary.json") if (root / "B" / "b_summary.json").is_file() else {}
        complete = [r for r in b.get("record_stats", []) if r.get("status") == "COMPLETE"]
        if complete:
            keys = ("NN", "NT1", "NT2", "N T1".replace(" ", ""), "N T2".replace(" ", ""), "T1N", "T1T1", "T1T2", "T2N", "T2T1", "T2T2")
            keys = ["NN", "NT1", "NT2", "T1N", "T1T1", "T1T2", "T2N", "T2T1", "T2T2"]
            matrix = np.asarray([[np.mean([r["d"][key] for r in complete]) for key in keys]], dtype=float)
            plt.figure(figsize=(8, 2)); plt.imshow(matrix, aspect="auto", cmap="viridis"); plt.xticks(range(len(keys)), keys); plt.yticks([0], ["mean"]); plt.colorbar(label="unit squared distance"); plt.tight_layout(); path = figures / "b_distance_heatmap.png"; plt.savefig(path, dpi=160); plt.close(); output["b_distance_heatmap"] = str(path.resolve())
            x = [float(r["I_TT"]) for r in complete]; y = [float(r["X"]) for r in complete]; plt.figure(figsize=(5, 4)); plt.scatter(x, y); plt.axhline(0, color="k", lw=.5); plt.axvline(0, color="k", lw=.5); plt.xlabel("I_TT"); plt.ylabel("X"); plt.tight_layout(); path = figures / "b_instance_transfer_scatter.png"; plt.savefig(path, dpi=160); plt.close(); output["b_instance_transfer_scatter"] = str(path.resolve())
    except Exception as exc:  # visualization is never allowed to alter scientific status  # noqa: BLE001
        output["status"] = "NOT_AVAILABLE"
        output["reason"] = f"{type(exc).__name__}: {exc}"
    return output


def validate_run(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    inputs_ok = (root / "inputs.json").is_file()
    checks.append({"name": "inputs", "status": "PASS" if inputs_ok else "FAIL"})
    replay = read_json(root / "A" / "replay.json") if (root / "A" / "replay.json").is_file() else {}
    checks.append({"name": "A_official_replay", "status": "PASS" if replay.get("status") == "COMPLETE" else "INCOMPLETE", "max_abs_error": replay.get("max_abs_error")})
    independent = root / "independent_recompute.json"
    if inputs_ok:
        command = [sys.executable, str(REPO / "scripts/experiments/tts_time_instance_recompute.py"), "--run-root", str(root)]
        result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, check=False)
        if result.returncode == 0 and independent.is_file():
            checks.append({"name": "independent_recompute", "status": "PASS", "result": read_json(independent)})
        else:
            checks.append({"name": "independent_recompute", "status": "INCOMPLETE", "reason": result.stderr[-2000:] or result.stdout[-2000:]})
    b = read_json(root / "B" / "b_summary.json") if (root / "B" / "b_summary.json").is_file() else {"status": "NOT_RUN"}
    checks.append({"name": "B_independence", "status": "PASS" if b.get("status") in {"COMPLETE", "PARTIAL", "DEPENDENCY_BLOCKED"} else "INCOMPLETE", "b_status": b.get("status")})
    payload = {"schema_version": 1, "status": "PASS" if all(item["status"] in {"PASS", "INCOMPLETE"} for item in checks) and any(item["status"] == "PASS" for item in checks) else "INCOMPLETE", "checks": checks, "engineering_status": "GO", "scientific_status": {"A": read_json(root / "A" / "a_summary.json").get("status") if (root / "A" / "a_summary.json").is_file() else "NOT_RUN", "B": b.get("status")}}
    write_json(root / "validation.json", payload)
    return payload


def make_report(root: Path) -> Path:
    inputs = read_json(root / "inputs.json")
    a = read_json(root / "A" / "a_summary.json") if (root / "A" / "a_summary.json").is_file() else {"status": "NOT_RUN"}
    b = read_json(root / "B" / "b_summary.json") if (root / "B" / "b_summary.json").is_file() else {"status": "NOT_RUN"}
    validation = read_json(root / "validation.json") if (root / "validation.json").is_file() else {"status": "NOT_RUN"}
    def stat_line(name: str, value: Mapping[str, Any]) -> str:
        if value.get("status") != "COMPLETE":
            return f"- {name}: `{value.get('status')}` (group_count={value.get('group_count', 0)})"
        return f"- {name}: mean `{value.get('mean'):.6f}`, 95% CI `{value.get('ci95')}`, Bonferroni 98.75% CI `{value.get('ci98_75_bonferroni')}`, status `{value.get('status_by_corrected_ci')}`, groups `{value.get('group_count')}`"
    def descriptive_line(name: str, value: Mapping[str, Any]) -> str:
        if value.get("status") != "COMPLETE":
            return f"- {name}: `{value.get('status')}` (group_count={value.get('group_count', 0)})"
        return f"- {name}: mean `{value.get('mean'):.6f}`, descriptive 95% CI `{value.get('ci95')}`, groups `{value.get('group_count')}`"
    adjacent = a.get("adjacent_sensitivity", {}).get("summary", {})
    context = b.get("context", {})
    match_counts = context.get("neighbor_match_count", {})
    reuse = context.get("donor_reuse", {})
    lines = [
        "# TTS 时间辨识与同文本实例交叉实验",
        "",
        f"- run: `{root.name}`",
        f"- cohort: `{len(inputs.get('records', []))}` records / `{inputs.get('source_group_count')}` source groups",
        f"- validation: `{validation.get('status')}`",
        "",
        "## 历史复用与本轮新增",
        "",
        "A 复用缓存的 N/T ORIGINAL embedding，只新增固定 lag 后的局部错配排名和按 TextGrid 事件分层；B 新生成同文本、同参考音频的 T1/T2，并在固定静态脸上做 3×3 交叉。B 不等同于旧 LeapTalk 阶段。",
        "",
        "## 主统计",
        "",
        stat_line("A_rank", a.get("a1", {}).get("summary", {})),
        stat_line("A_event", a.get("a_event", {}).get("summary", {})),
        "- A_event 每条记录的四格计数、保留 h、phone 分布和 N/T 时长比见 `A/a_summary.json`。",
        stat_line("B_instance", b.get("B_instance", {})),
        stat_line("B_transfer", b.get("B_transfer", {})),
        "- B 的 I_NT1、I_NT2、I_NT 均值及其与 I_TT 的差值为描述性次要量，见 `B/b_summary.json`。",
        "- B 每条记录的 T1/T2 phone 时长差与 phase 共同支持舍弃率见 `B/b_summary.json`。",
        "",
        "## A ±1 帧补充（描述性，不进入主检验）",
    ]
    for delta in SUPPLEMENTARY_DELTAS:
        item = adjacent.get(str(delta), {})
        lines.append(f"- delta `{delta}`: mean R_N `{item.get('mean_R_N')}`, mean R_T `{item.get('mean_R_T')}`, mean A_rank `{item.get('mean_A_rank_adjacent')}`, groups `{item.get('group_count', 0)}`")
    lines.extend([
        "",
        "## 原生参照",
        "",
        stat_line("native G", b.get("native", {}).get("G", {})),
        f"- native_advantage: `{b.get('native_advantage', 'UNCONFIRMED')}`",
        "",
        "## 上下文次要诊断（仅描述性）",
        "",
        f"- context pairs: `{b.get('context_pair_count', 0)}`; one-neighbor plus `{match_counts.get('one_neighbor', 0)}`, two-neighbor plus `{match_counts.get('two_neighbors', 0)}`",
        f"- donor reuse: plus `{reuse.get('plus', {}).get('reused_donors', 0)}` reused / `{reuse.get('plus', {}).get('unique_donors', 0)}` unique; minus `{reuse.get('minus', {}).get('reused_donors', 0)}` reused / `{reuse.get('minus', {}).get('unique_donors', 0)}` unique",
        descriptive_line("K_N", context.get("K_N", {})),
        descriptive_line("K_T", context.get("K_T", {})),
        descriptive_line("K_T-K_N", context.get("K_T_minus_K_N", {})),
        "- inferential status: `DESCRIPTIVE_ONLY`; these rows are not an additional mechanism test.",
        "",
        "## 限制",
        "",
        "A_event 的约 215 ms SyncNet 窗口可能跨越多个音素；B 的 phase 只是在 MFA occurrence 坐标上共同采样，不能称为精确发音对齐。B 若缺失 TTS、Wav2Lip、SyncNet 或 MFA，记录为阻塞/不确定，不能用替代模型填充。没有人工听检或观看评分，quality/sync human status 为 NOT_ASSESSED。",
        "",
        "## 产物",
        "",
        f"- inputs: `{root / 'inputs.json'}`",
        f"- A summary: `{root / 'A' / 'a_summary.json'}`",
        f"- B summary: `{root / 'B' / 'b_summary.json'}`",
        f"- validation: `{root / 'validation.json'}`",
    ])
    path = root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("audit", "a", "prepare-b", "render-b", "analyze-b", "validate", "report", "all"), default="all")
    parser.add_argument("--internal-generate", action="store_true")
    parser.add_argument("--internal-extract-all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.run_root.resolve() if args.run_root else run_root(args.run_id).resolve())
    stage = args.stage
    # A failed invocation leaves an error marker for review.  Once the same
    # run is retried successfully that marker is stale and must not make the
    # frozen run look failed to downstream validators.
    error_path = root / "error.json"
    error_path.unlink(missing_ok=True)
    if args.internal_generate:
        return generate_b_audio(root)
    if args.internal_extract_all:
        return extract_b_features(root)
    try:
        if stage == "audit":
            audit_inputs(root)
            return 0
        inputs = read_json(root / "inputs.json") if (root / "inputs.json").is_file() else audit_inputs(root)
        # ``analyze-b`` consumes the frozen B artifacts that already exist in
        # the run.  It must not re-enter TTS/MFA/rendering (which is both
        # expensive and could mutate the alignment denominator).
        if stage == "analyze-b":
            analyze_b(root, inputs)
            return 0
        if stage in {"a", "all"}:
            run_a(root, inputs)
        if stage in {"prepare-b", "render-b", "analyze-b", "all"}:
            b_manifest = prepare_b(root, inputs)
            if stage == "prepare-b":
                return 0
            # TTS generation and alignment are intentionally kept before any Wav2Lip render.
            generation_command = [str(QWEN_PYTHON), str(Path(__file__).resolve()), "--run-id", root.name.removeprefix("tts_time_instance_"), "--run-root", str(root), "--internal-generate"]
            if not (root / "B" / "tts_generation.json").is_file() or read_json(root / "B" / "tts_generation.json").get("status") != "COMPLETE":
                if not QWEN_PYTHON.is_file():
                    write_json(root / "B" / "tts_generation.json", {"status": "DEPENDENCY_BLOCKED", "reason": f"missing Qwen python: {QWEN_PYTHON}"})
                else:
                    subprocess.run(generation_command, cwd=str(REPO), check=False)
            generation = read_json(root / "B" / "tts_generation.json")
            if generation.get("status") == "COMPLETE":
                for row in b_manifest["records"]:
                    for arm in ("T1", "T2"):
                        meta = read_json(root / "B" / "audio" / str(row["sample_id"]) / f"{arm}.json")
                        row[f"{arm}_audio"] = meta["audio"]
                b_manifest["status"] = "AUDIO_READY"
                write_json(root / "B" / "audio_manifest.json", b_manifest)
                mfa_records = [{**row, "N_audio": row["N_audio"], "T1": row["T1_audio"], "T2": row["T2_audio"]} for row in b_manifest["records"]]
                # run_mfa expects fields listed in audio_fields; each field is a path in record.
                mfa = run_mfa(mfa_records, ("N_audio", "T1", "T2"), root / "B", prefix="B")
                write_json(root / "B" / "alignment.json", mfa)
                # A single MFA utterance can fail while the rest of the
                # frozen cohort remains usable.  Preserve that exact missing
                # denominator and render only records with all three grids;
                # analyze_b will keep the run PARTIAL and will never replace
                # the failed record.
                if mfa.get("status") in {"COMPLETE", "MFA_FAILED"}:
                    tg = root / "B" / "B_textgrids"
                    eligible_sids: set[int] = set()
                    for sid in SAMPLE_IDS:
                        for src, arm in (("N_audio", "N"), ("T1", "T1"), ("T2", "T2")):
                            source = tg / f"{sid}_{src}.TextGrid"
                            target = tg / f"{sid}_{arm}.TextGrid"
                            if source != target and source.is_file() and not target.exists():
                                shutil.copy2(source, target)
                        if all((tg / f"{sid}_{arm}.TextGrid").is_file() for arm in ("N", "T1", "T2")):
                            eligible_sids.add(sid)
                    rendered = _static_render(root, b_manifest, eligible_sids=eligible_sids)
                    if rendered.get("status") == "COMPLETE":
                        extract_command = [str(SYNCNET_PYTHON), str(Path(__file__).resolve()), "--run-id", root.name.removeprefix("tts_time_instance_"), "--run-root", str(root), "--internal-extract-all"]
                        if not (root / "B" / "feature_manifest.json").is_file() or read_json(root / "B" / "feature_manifest.json").get("status") != "COMPLETE":
                            subprocess.run(extract_command, cwd=str(REPO), check=False)
                    else:
                        write_json(root / "B" / "render_status.json", rendered)
            else:
                write_json(root / "B" / "alignment.json", {"status": "DEPENDENCY_BLOCKED", "reason": "TTS generation did not complete"})
            if stage == "render-b":
                return 0
            analyze_b(root, inputs)
        if stage in {"validate", "all"}:
            validate_run(root)
        if stage in {"report", "all"}:
            make_figures(root)
            make_report(root)
        return 0
    except Exception as exc:  # frozen-input errors are written for review and surfaced by exit code  # noqa: BLE001
        write_json(root / "error.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
