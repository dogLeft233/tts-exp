"""F0-contour exchange experiment for the local Wav2Lip/SyncNet pipeline.

The file intentionally keeps the protocol in one auditable runner.  The
scientific unit is a natural/TTS utterance pair; all audio is first decoded to
16 kHz mono and all subsequent transformations preserve the receiver's time
axis.  The pure audio functions in this module are usable without Wav2Lip or
SyncNet and are covered by ``tests/experiments/test_tts_f0_swap.py``.

The expensive stages are optional at runtime.  Missing pyworld, Wav2Lip or
SyncNet resources are recorded as explicit blocked statuses rather than being
replaced by a different pitch or timing operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import unicodedata
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly


# ``scripts/experiments/tts_f0_swap.py`` -> repo root.
REPO = Path(__file__).resolve().parents[2]
RUN_PREFIX = "tts_f0_swap"
PROTOCOL_ID = "tts_f0_swap_v1"
PROTOCOL_REVISION = "tts_f0_swap_v1"
REPAIR_PROTOCOL_ID = "tts_f0_swap_repair_v2"
SAMPLE_RATE = 16_000
FPS = 25
FRAME_PERIOD_MS = 5.0
FRAME_PERIOD_S = FRAME_PERIOD_MS / 1000.0
F0_FLOOR = 60.0
F0_CEIL = 600.0
# Harvest is called with a 60--600 Hz search range, but StoneMask refines
# voiced values after Harvest and can move an otherwise valid low-pitch frame
# a few hertz below the requested floor.  Keep that measured value as part of
# the frozen carrier instead of clipping it or rejecting the no-op ID arm.
# The tolerance is fixed before the formal rerun and is only a post-StoneMask
# validation envelope; it does not change the analysis/synthesis parameters.
F0_POST_STONEMASK_FLOOR = 50.0
MIN_DURATION_S = 3.0
MAX_DURATION_S = 12.0
MAX_CLIPPING_RATIO = 0.001
MAX_ALIGNMENT_ERROR_S = 0.010
MIN_AUDIO_SAMPLES = SAMPLE_RATE
SAFE_PEAK = 0.98
PHONE_INTERPOLATION_TOLERANCE_S = 0.010
MIN_EFFECTIVE_COVERAGE = 0.40
MIN_EFFECTIVE_PHONES = 5
MIN_VOICED_SECONDS = 1.0
MIN_CONTOUR_RMS_ST = 0.5
MIN_DOSE_FRACTION = 0.75
MIN_FORMAL_SPEAKERS = 6
MIN_FORMAL_PAIRS = 18
TARGET_FORMAL_SPEAKERS = 8
PAIRS_PER_SPEAKER = 3
PILOT_COUNT = 4
SEED = 42
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_DRAWS = 20_000
BONFERRONI_ALPHA = 0.05 / 4.0
VSHIFT = 15
MIN_SCORE_WINDOWS = 50
DELAY_SAMPLES = 3200
DELAY_FRAMES = 5
REPEAT_CURVE_BOUND = 1e-4
REFERENCE_EFFECT = 0.100

MANIFEST = REPO / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
RESOURCE_ROOT = REPO / "results/rhythm_style_500/aishell1_test_400"
PORTRAIT_INPUTS = REPO / "runs/tts_time_instance_20260917_v1/inputs.json"
PORTRAIT_AUDIO_MANIFEST = REPO / "runs/tts_time_instance_20260917_v1/B/audio_manifest.json"
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_PYTHON = Path(os.environ.get("WAV2LIP_PYTHON", "/home/wjj/.venvs/wav2lip/bin/python"))
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path(os.environ.get("SYNCNET_PYTHON", "/home/wjj/.venvs/syncnet/bin/python"))
RENDER_WORKER = REPO / "scripts/experiments/static_image_bridge/render_worker.py"
SCORE_WORKER = REPO / "scripts/experiments/static_image_bridge/score_worker.py"
FFMPEG = Path(os.environ.get("FFMPEG", "/home/wjj/miniconda3/bin/ffmpeg"))
SPEC_PATH = REPO / "basic-memory/Research/TTS 音高起伏双向交换的 Wav2Lip 因果诊断 Spec.md"
REPAIR_SPEC_PATH = REPO / "basic-memory/Research/TTS 音高起伏交换的 WORLD 与测量链修复 Spec.md"

SILENCE_LABELS = {"", "sil", "sp", "<eps>"}
UNKNOWN_LABELS = {"spn", "unk", "<unk>", "unknown", "oov", "<oov>"}


class ProtocolError(RuntimeError):
    """Raised when an auditable experimental contract is violated."""


class AudioBackendUnavailable(ProtocolError):
    """Raised when pyworld is not available in the selected environment."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _implementation_fingerprint() -> dict[str, Any]:
    """Return the code/spec hashes used to invalidate resumable stages."""

    return {
        "runner_sha256": file_sha256(Path(__file__)),
        "spec_sha256": file_sha256(SPEC_PATH) if SPEC_PATH.is_file() else None,
    }


def _fingerprint_matches(payload: Mapping[str, Any]) -> bool:
    current = _implementation_fingerprint()
    return payload.get("runner_sha256") == current["runner_sha256"] and payload.get("spec_sha256") == current["spec_sha256"]


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: Path, value: Any, *, self_hash: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value) if isinstance(value, Mapping) else value
    if self_hash and isinstance(payload, dict):
        payload.pop("artifact_sha256", None)
        payload["artifact_sha256"] = canonical_json_sha256(payload)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_json(path: Path, *, self_hash: bool = False) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    if self_hash:
        recorded = payload.get("artifact_sha256")
        body = dict(payload)
        body.pop("artifact_sha256", None)
        if not isinstance(recorded, str) or recorded != canonical_json_sha256(body):
            raise ProtocolError(f"self-hash mismatch: {path}")
    return payload


def run_id_valid(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("run-id may contain only letters, numbers, underscores, and hyphens")
    return value


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"{RUN_PREFIX}_{run_id_valid(run_id)}"


class RunPaths:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def protocol(self) -> Path: return self.root / "protocol.json"
    @property
    def inputs(self) -> Path: return self.root / "inputs.json"
    @property
    def parameters_dir(self) -> Path: return self.root / "parameters"
    @property
    def audio_dir(self) -> Path: return self.root / "audio"
    @property
    def audio_qc(self) -> Path: return self.root / "audio_qc.csv"
    @property
    def pilot_qc(self) -> Path: return self.root / "pilot_qc.json"
    @property
    def pilot_quality(self) -> Path: return self.root / "pilot_quality_manifest.json"
    @property
    def videos_dir(self) -> Path: return self.root / "videos"
    @property
    def videos_manifest(self) -> Path: return self.root / "videos_manifest.json"
    @property
    def features_dir(self) -> Path: return self.root / "features"
    @property
    def features_manifest(self) -> Path: return self.root / "features_manifest.json"
    @property
    def distances_dir(self) -> Path: return self.root / "distances"
    @property
    def scores_csv(self) -> Path: return self.root / "scores.csv"
    @property
    def score_manifest(self) -> Path: return self.root / "scores_manifest.json"
    @property
    def analysis(self) -> Path: return self.root / "analysis.json"
    @property
    def bootstrap_indices(self) -> Path: return self.root / "bootstrap_indices.npy"
    @property
    def validation(self) -> Path: return self.root / "validation.json"
    @property
    def report(self) -> Path: return self.root / "report.md"
    @property
    def plots_dir(self) -> Path: return self.root / "plots"
    @property
    def logs_dir(self) -> Path: return self.root / "logs"
    @property
    def controls_dir(self) -> Path: return self.root / "controls"


def _resolve_asset(value: str | Path, *, expected_sha256: str | None = None) -> tuple[Path, str]:
    """Resolve only the known repository prefix from old manifests."""

    raw = Path(str(value)).expanduser()
    candidates: list[tuple[Path, str]] = [(raw, "as_recorded")]
    parts = raw.parts
    if raw.is_absolute() and "tts-exp" in parts:
        index = len(parts) - 1 - parts[::-1].index("tts-exp")
        candidates.append((REPO.joinpath(*parts[index + 1:]), "tts_exp_prefix_relocated"))
    elif not raw.is_absolute():
        candidates.append((REPO / raw, "repo_relative"))
    candidates.append((RESOURCE_ROOT / raw.name, "resource_root_basename"))
    seen: set[Path] = set()
    for candidate, reason in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if expected_sha256 is None or file_sha256(resolved) == expected_sha256:
            return resolved, reason
    raise ProtocolError(f"missing or hash-mismatched asset: {value}")


def decode_audio(path: str | Path, target_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode mono audio and deterministically resample without time warping."""

    source = Path(path)
    try:
        import soundfile as sf

        values, rate = sf.read(str(source), dtype="float64", always_2d=False)
    except Exception as exc:  # pragma: no cover - backend-specific decoder errors
        raise ProtocolError(f"cannot decode audio {source}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ProtocolError(f"audio is not finite mono data: {source}")
    rate = int(rate)
    original_count = int(values.size)
    if rate != target_rate:
        gcd = math.gcd(rate, target_rate)
        values = resample_poly(values, target_rate // gcd, rate // gcd).astype(np.float64, copy=False)
        target_count = int(round(original_count * target_rate / rate))
        if values.size > target_count:
            values = values[:target_count]
        elif values.size < target_count:
            values = np.pad(values, (0, target_count - values.size))
    if values.size < MIN_AUDIO_SAMPLES:
        raise ProtocolError(f"audio is shorter than one second: {source}")
    peak = float(np.max(np.abs(values), initial=0.0))
    if peak > 1.0 + 1e-7:
        raise ProtocolError(f"decoded samples exceed [-1,1]: {source} peak={peak}")
    clipping_ratio = float(np.mean(np.abs(values) >= 0.9999))
    return np.ascontiguousarray(values), {
        "path": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "source_sample_rate": rate,
        "source_sample_count": original_count,
        "sample_rate": target_rate,
        "sample_count": int(values.size),
        "duration_s": float(values.size / target_rate),
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "resampled": bool(rate != target_rate),
        "resample_method": "scipy.signal.resample_poly" if rate != target_rate else "none",
    }


def pcm16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < MIN_AUDIO_SAMPLES or not np.isfinite(values).all():
        raise ProtocolError("PCM16 input is empty or non-finite")
    if float(np.max(np.abs(values), initial=0.0)) >= 1.0:
        raise ProtocolError("clipping is forbidden before PCM conversion")
    return np.rint(values * 32768.0).clip(-32768, 32767).astype("<i2")


def write_pcm16(path: Path, values: np.ndarray) -> dict[str, Any]:
    pcm = pcm16(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "pcm_sha256": bytes_sha256(pcm.tobytes()),
        "sample_count": int(pcm.size),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
        "rms": rms(values),
        "peak": float(np.max(np.abs(values), initial=0.0)),
    }


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float64))) if array.size else 0.0


def rms_db_delta(left: np.ndarray, right: np.ndarray) -> float:
    l, r = rms(left), rms(right)
    if l <= 0.0 or r <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(l / r))


def _tier_section(raw: str, tier_name: str) -> str:
    marker = re.search(rf'name\s*=\s*"{re.escape(tier_name)}"', raw, flags=re.IGNORECASE)
    if marker is None:
        return ""
    tail = raw[marker.end():]
    next_tier = re.search(r'item\s*\[\s*\d+\s*\]\s*:', tail, flags=re.IGNORECASE)
    return tail[: next_tier.start()] if next_tier else tail


_INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:\s*xmin\s*=\s*([0-9.eE+-]+).*?"
    r"xmax\s*=\s*([0-9.eE+-]+).*?text\s*=\s*\"(.*?)\"",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_label(label: str) -> str:
    return unicodedata.normalize("NFC", " ".join(str(label).strip().split()))


def parse_textgrid(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")

    def parse_tier(name: str) -> list[dict[str, Any]]:
        intervals: list[dict[str, Any]] = []
        for start, end, label in _INTERVAL_RE.findall(_tier_section(raw, name)):
            a, b = float(start), float(end)
            if not math.isfinite(a) or not math.isfinite(b) or b <= a:
                raise ProtocolError(f"invalid TextGrid interval {path}: {a}..{b}")
            normalized = normalize_label(label)
            lower = normalized.lower()
            intervals.append({
                "index": len(intervals), "label": normalized, "normalized": normalized,
                "start_s": a, "end_s": b, "silence": lower in SILENCE_LABELS,
                "unknown": lower in UNKNOWN_LABELS,
            })
        for prev, current in zip(intervals, intervals[1:]):
            if current["start_s"] < prev["end_s"] - 1e-9:
                raise ProtocolError(f"non-monotonic TextGrid tier: {path}")
        return intervals

    phones = parse_tier("phones")
    words = parse_tier("words")
    if not phones or not any(not row["silence"] and not row["unknown"] for row in phones):
        raise ProtocolError(f"phones tier has no usable speech: {path}")
    for phone in phones:
        midpoint = 0.5 * (phone["start_s"] + phone["end_s"])
        owner = next((word for word in words if word["start_s"] <= midpoint < word["end_s"] and not word["silence"] and word["label"]), None)
        phone["word_index"] = int(owner["index"]) if owner else None
        phone["word_label"] = owner["label"] if owner else None
    return {"path": str(Path(path).resolve()), "phones": phones, "words": words}


def speech_phones(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in grid["phones"] if not row["silence"] and not row["unknown"]]


def phone_sequence(grid: Mapping[str, Any]) -> list[str]:
    return [str(row["normalized"]) for row in speech_phones(grid)]


def _finite_phone_bounds(grid: Mapping[str, Any], duration_s: float) -> bool:
    for row in grid["phones"]:
        if row["start_s"] < -MAX_ALIGNMENT_ERROR_S or row["end_s"] > duration_s + MAX_ALIGNMENT_ERROR_S:
            return False
    return True


class WorldParams:
    def __init__(self, f0: np.ndarray, time: np.ndarray, sp: np.ndarray, ap: np.ndarray, sample_count: int, sample_rate: int = SAMPLE_RATE, frame_period_ms: float = FRAME_PERIOD_MS, f0_floor: float = F0_FLOOR, f0_ceil: float = F0_CEIL):
        self.f0, self.time, self.sp, self.ap = f0, time, sp, ap
        self.sample_count, self.sample_rate = int(sample_count), int(sample_rate)
        self.frame_period_ms, self.f0_floor, self.f0_ceil = float(frame_period_ms), float(f0_floor), float(f0_ceil)

    def arrays(self) -> dict[str, np.ndarray]:
        return {"f0": np.asarray(self.f0), "time": np.asarray(self.time), "sp": np.asarray(self.sp), "ap": np.asarray(self.ap)}


def _pyworld() -> Any:
    try:
        import pyworld  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise AudioBackendUnavailable("pyworld is unavailable; install it in a dedicated F0 environment") from exc
    return pyworld


def _audio_backend_info() -> dict[str, Any]:
    try:
        pw = _pyworld()
        return {"name": "pyworld", "version": str(getattr(pw, "__version__", "unknown")), "independent_measurement": "dio->stonemask"}
    except AudioBackendUnavailable as exc:
        return {"name": "pyworld", "status": "AUDIO_BACKEND_UNAVAILABLE", "error": str(exc), "independent_measurement": "dio->stonemask"}


def world_analyze(values: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> WorldParams:
    pw = _pyworld()
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if values.size < MIN_AUDIO_SAMPLES or not np.isfinite(values).all():
        raise ProtocolError("WORLD input is invalid")
    f0, time = pw.harvest(values, sample_rate, frame_period=FRAME_PERIOD_MS, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL)
    f0 = pw.stonemask(values, f0, time, sample_rate)
    sp = pw.cheaptrick(values, f0, time, sample_rate)
    ap = pw.d4c(values, f0, time, sample_rate)
    f0 = np.ascontiguousarray(np.asarray(f0, dtype=np.float64))
    time = np.ascontiguousarray(np.asarray(time, dtype=np.float64))
    sp = np.ascontiguousarray(np.asarray(sp, dtype=np.float64))
    ap = np.ascontiguousarray(np.asarray(ap, dtype=np.float64))
    if f0.ndim != 1 or time.shape != f0.shape or sp.ndim != 2 or ap.shape != sp.shape or sp.shape[0] != f0.size:
        raise ProtocolError("WORLD returned inconsistent F0/sp/ap arrays")
    if not all(np.isfinite(array).all() for array in (f0, time, sp, ap)):
        raise ProtocolError("WORLD returned non-finite arrays")
    return WorldParams(f0=f0, time=time, sp=sp, ap=ap, sample_count=int(values.size))


def world_synthesize(params: WorldParams, f0: np.ndarray) -> np.ndarray:
    pw = _pyworld()
    target = np.ascontiguousarray(np.asarray(f0, dtype=np.float64).reshape(-1))
    if target.shape != params.f0.shape or not np.isfinite(target).all():
        raise ProtocolError("WORLD target F0 shape/value is invalid")
    result = np.asarray(pw.synthesize(target, params.sp, params.ap, params.sample_rate, frame_period=FRAME_PERIOD_MS), dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ProtocolError("WORLD synthesis returned invalid waveform")
    return result


def exact_length(values: np.ndarray, target_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size > target_count:
        return array[:target_count].copy(), {"action": "right_crop", "raw_count": int(array.size), "target_count": target_count, "adjustment_samples": int(array.size - target_count)}
    if array.size < target_count:
        return np.pad(array, (0, target_count - array.size)), {"action": "right_zero_pad", "raw_count": int(array.size), "target_count": target_count, "adjustment_samples": int(target_count - array.size)}
    return array.copy(), {"action": "none", "raw_count": int(array.size), "target_count": target_count, "adjustment_samples": 0}


def normalize_rms(values: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    current = rms(values)
    if current <= 0.0 or target_rms <= 0.0:
        raise ProtocolError("RMS matching requires non-zero values")
    scale = float(target_rms / current)
    return np.asarray(values, dtype=np.float64) * scale, scale


def _contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return [chunk for chunk in np.split(indices, breaks) if chunk.size]


class F0Mapping:
    def __init__(self, receiver_indices: np.ndarray, donor_left: np.ndarray, donor_right: np.ndarray, donor_alpha: np.ndarray, donor_time: np.ndarray, phone_index: np.ndarray, weights: np.ndarray, valid: np.ndarray, invalid_reasons: tuple[str, ...]):
        self.receiver_indices, self.donor_left, self.donor_right = receiver_indices, donor_left, donor_right
        self.donor_alpha, self.donor_time, self.phone_index = donor_alpha, donor_time, phone_index
        self.weights, self.valid, self.invalid_reasons = weights, valid, invalid_reasons

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "receiver_indices": self.receiver_indices, "donor_left": self.donor_left,
            "donor_right": self.donor_right, "donor_alpha": self.donor_alpha,
            "donor_time": self.donor_time, "phone_index": self.phone_index,
            "weights": self.weights, "valid": self.valid.astype(np.uint8),
        }


def _match_phone_occurrences(receiver_grid: Mapping[str, Any], donor_grid: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    receiver = speech_phones(receiver_grid)
    donor = speech_phones(donor_grid)
    if [row["normalized"] for row in receiver] != [row["normalized"] for row in donor]:
        raise ProtocolError("natural/TTS speech phone sequences differ; no LCS fallback is allowed")
    return list(zip(receiver, donor, strict=True))


def build_f0_mapping(receiver: WorldParams, donor: WorldParams, receiver_grid: Mapping[str, Any], donor_grid: Mapping[str, Any]) -> F0Mapping:
    pairs = _match_phone_occurrences(receiver_grid, donor_grid)
    n = receiver.f0.size
    left = np.full(n, -1, dtype=np.int64)
    right = np.full(n, -1, dtype=np.int64)
    alpha = np.zeros(n, dtype=np.float64)
    donor_time = np.full(n, np.nan, dtype=np.float64)
    phone_index = np.full(n, -1, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    weights = np.zeros(n, dtype=np.float64)
    # ``mapped`` is kept separate from ``valid`` while building the taper.
    # The two endpoint frames of a long run have a valid donor interpolation,
    # but their registered taper weight is exactly zero and therefore they are
    # outside M.  Keeping the raw run avoids accidentally moving the taper
    # inward when an endpoint is removed.
    mapped = np.zeros(n, dtype=bool)
    reasons = ["outside_phone"] * n
    for phone_number, (r_phone, d_phone) in enumerate(pairs):
        r_start, r_end = float(r_phone["start_s"]), float(r_phone["end_s"])
        d_start, d_end = float(d_phone["start_s"]), float(d_phone["end_s"])
        if r_end <= r_start or d_end <= d_start:
            continue
        r_indices = np.flatnonzero((receiver.time >= r_start) & (receiver.time < r_end))
        for index in r_indices:
            phone_index[index] = int(r_phone["index"])
            if receiver.f0[index] <= 0.0:
                reasons[index] = "receiver_unvoiced"
                continue
            u = float((receiver.time[index] - r_start) / (r_end - r_start))
            q = d_start + u * (d_end - d_start)
            if q < d_start or q >= d_end:
                reasons[index] = "donor_time_outside_phone"
                continue
            pos = int(np.searchsorted(donor.time, q, side="left"))
            exact = pos < donor.time.size and abs(float(donor.time[pos]) - q) <= 1e-8
            if exact:
                if donor.f0[pos] <= 0.0:
                    reasons[index] = "donor_unvoiced"
                    continue
                lidx = ridx = pos
                a = 0.0
            else:
                lidx, ridx = pos - 1, pos
                if lidx < 0 or ridx >= donor.time.size:
                    reasons[index] = "donor_time_edge"
                    continue
                if ridx != lidx + 1 or donor.f0[lidx] <= 0.0 or donor.f0[ridx] <= 0.0:
                    reasons[index] = "donor_not_adjacent_voiced"
                    continue
                if donor.time[lidx] < d_start or donor.time[ridx] >= d_end:
                    reasons[index] = "donor_cross_phone_boundary"
                    continue
                if q - donor.time[lidx] > PHONE_INTERPOLATION_TOLERANCE_S or donor.time[ridx] - q > PHONE_INTERPOLATION_TOLERANCE_S:
                    reasons[index] = "donor_gap_over_10ms"
                    continue
                gap = float(donor.time[ridx] - donor.time[lidx])
                if gap <= 0.0:
                    reasons[index] = "donor_duplicate_time"
                    continue
                a = float((q - donor.time[lidx]) / gap)
            left[index], right[index], alpha[index] = lidx, ridx, a
            donor_time[index] = q
            mapped[index] = True
            reasons[index] = "valid"

        # Fade each phone's valid runs separately; no cross-phone smoothing.
        phone_valid = np.asarray([i for i in r_indices if mapped[i]], dtype=np.int64)
        for run in _contiguous_runs(phone_valid):
            if run.size < 5:
                for i in run:
                    mapped[i] = False
                    reasons[i] = "short_valid_segment"
                continue
            length = int(run.size)
            for j, i in enumerate(run):
                weight = min(1.0, j / 4.0, (length - 1 - j) / 4.0)
                weights[i] = weight
                if weight <= 0.0:
                    reasons[i] = "fade_endpoint"
                else:
                    valid[i] = True
    return F0Mapping(
        receiver_indices=np.arange(n, dtype=np.int64), donor_left=left, donor_right=right,
        donor_alpha=alpha, donor_time=donor_time, phone_index=phone_index,
        weights=weights, valid=(valid & (weights > 0.0)), invalid_reasons=tuple(reasons),
    )


def _mapping_donor_semitones(params: WorldParams, mapping: F0Mapping) -> np.ndarray:
    # The mapped array lives on the receiver frame grid.  Donor and receiver
    # utterances generally have different frame counts, so allocating from
    # ``params.f0.shape`` makes the receiver mask mis-index the donor result.
    out = np.full(mapping.valid.shape, np.nan, dtype=np.float64)
    mask = mapping.valid
    if np.any(mask):
        li, ri, a = mapping.donor_left[mask], mapping.donor_right[mask], mapping.donor_alpha[mask]
        lf, rf = params.f0[li], params.f0[ri]
        if np.any(lf <= 0.0) or np.any(rf <= 0.0):
            raise ProtocolError("mapping contains unvoiced donor frames")
        out[mask] = (1.0 - a) * (12.0 * np.log2(lf)) + a * (12.0 * np.log2(rf))
    return out


def f0_interventions(receiver: WorldParams, donor: WorldParams, mapping: F0Mapping) -> dict[str, Any]:
    """Build the ID/LEVEL/CONTOUR targets on the receiver grid.

    The repair protocol deliberately keeps the ordinary mean over *all*
    receiver voiced frames fixed.  The weighted means are used only to
    define the donor and receiver levels; the taper is then part of the
    transferred signal.  In particular, do not subtract a second ``w**2``
    correction here: that was the bug in the v1 runner and changes the
    scientific intervention.
    """
    original = np.asarray(receiver.f0, dtype=np.float64).copy()
    if np.any((original > 0.0) & ((original < F0_POST_STONEMASK_FLOOR) | (original > F0_CEIL))):
        raise ProtocolError("receiver F0 is outside the frozen WORLD range after StoneMask tolerance")
    donor_st = _mapping_donor_semitones(donor, mapping)
    valid = np.asarray(mapping.valid, dtype=bool) & (np.asarray(mapping.weights, dtype=np.float64) > 0.0)
    weights = np.asarray(mapping.weights, dtype=np.float64)
    receiver_st = np.zeros_like(original)
    positive = original > 0.0
    receiver_st[positive] = 12.0 * np.log2(original[positive])
    denominator = float(np.sum(weights[valid]))
    if denominator <= 0.0:
        raise ProtocolError("F0 mapping has no weighted support")
    mu_receiver = float(np.sum(weights[valid] * receiver_st[valid]) / denominator)
    mu_donor = float(np.sum(weights[valid] * donor_st[valid]) / denominator)
    level_delta = np.zeros_like(receiver_st)
    contour_delta = np.zeros_like(receiver_st)
    level_delta[valid] = weights[valid] * (mu_donor - mu_receiver)
    centered_delta = (donor_st[valid] - mu_donor) - (receiver_st[valid] - mu_receiver)
    contour_delta[valid] = weights[valid] * centered_delta
    level_st = receiver_st + level_delta
    contour_st = receiver_st + contour_delta

    # Copy first, then assign only M.  This makes the no-op contract outside
    # the mapped voiced support exact at the array level.
    level = original.copy()
    contour = original.copy()
    level[valid] = np.power(2.0, level_st[valid] / 12.0)
    contour[valid] = np.power(2.0, contour_st[valid] / 12.0)
    for name, values in (("ID", original), ("LEVEL", level), ("CONTOUR", contour)):
        if np.any((values > 0.0) & ((values < F0_POST_STONEMASK_FLOOR - 1e-8) | (values > F0_CEIL + 1e-8))):
            raise ProtocolError(f"{name} target F0 leaves the frozen WORLD range after StoneMask tolerance")

    voiced_delta = contour_st[positive] - receiver_st[positive]
    ordinary_identity_error = float(np.mean(voiced_delta)) if voiced_delta.size else float("nan")
    weighted_identity_error = float(np.sum(weights[valid] * contour_delta[valid]) / denominator)
    support_s = valid & (weights >= 0.5)

    def _dose(delta: np.ndarray) -> float | None:
        if not np.any(support_s):
            return None
        return float(np.sqrt(np.mean(np.square(delta[support_s]), dtype=np.float64)))

    return {
        "f0": {"ID": original, "LEVEL": level, "CONTOUR": contour},
        "target_semitones": {"ID": receiver_st, "LEVEL": level_st, "CONTOUR": contour_st, "DONOR": donor_st},
        "target_deltas": {"LEVEL": level_delta, "CONTOUR": contour_delta},
        "mu_receiver": mu_receiver, "mu_donor": mu_donor,
        "weighted_support": denominator,
        "ordinary_identity_error": ordinary_identity_error,
        # Keep the old key as an alias for callers that only understand the
        # v1 result shape; its meaning is now the required ordinary mean.
        "identity_error": ordinary_identity_error,
        "weighted_identity_error": weighted_identity_error,
        "contour_identity_correction_st": 0.0,
        "effective_coverage": float(np.sum(weights) / max(1, np.sum(positive))),
        "effective_phone_count": int(len(set(int(x) for x in mapping.phone_index[valid]))),
        "voiced_frame_count": int(np.sum(positive)),
        "support_frame_count": int(np.sum(support_s)),
        "dose_contour_rms_st": _dose(contour_delta),
        "dose_level_rms_st": _dose(level_delta),
    }


def _common_safe_scale(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    peak = max(float(np.max(np.abs(np.asarray(value)), initial=0.0)) for value in arrays.values())
    scale = min(1.0, SAFE_PEAK / peak) if peak > 0.0 else 1.0
    return {key: np.asarray(value, dtype=np.float64) * scale for key, value in arrays.items()}, float(scale)


def envelope_qc(candidate: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
    frame = round(0.020 * SAMPLE_RATE)
    hop = round(0.010 * SAMPLE_RATE)
    count = max(0, 1 + (min(len(candidate), len(identity)) - frame) // hop)
    if count <= 0:
        return {"status": "ENVELOPE_CONFOUND", "frame_count": 0, "median_db": None, "p90_db": None}
    cand_values, id_values = [], []
    for i in range(count):
        start = i * hop
        cand_values.append(rms(candidate[start:start + frame]))
        id_values.append(rms(identity[start:start + frame]))
    id_rms = np.asarray(id_values, dtype=np.float64)
    candidate_rms = np.asarray(cand_values, dtype=np.float64)
    threshold = float(np.max(id_rms) * 10.0 ** (-40.0 / 20.0))
    mask = id_rms > threshold
    delta = np.abs(20.0 * np.log10(np.maximum(candidate_rms[mask], 1e-12) / np.maximum(id_rms[mask], 1e-12))) if np.any(mask) else np.asarray([], dtype=np.float64)
    median = float(np.median(delta)) if delta.size else None
    p90 = float(np.quantile(delta, 0.9, method="linear")) if delta.size else None
    return {"status": "PASS" if delta.size and median <= 1.0 and p90 <= 3.0 else "ENVELOPE_CONFOUND", "frame_count": int(delta.size), "median_db": median, "p90_db": p90}


def _measurement_array(value: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize a saved measurement to ``(f0, time)`` without interpolation."""

    if isinstance(value, Mapping):
        f0 = np.asarray(value.get("f0"), dtype=np.float64)
        time_value = value.get("time")
        time = None if time_value is None else np.asarray(time_value, dtype=np.float64)
    else:
        f0 = np.asarray(value, dtype=np.float64)
        time = None
    return f0, time


def _f0_error_stats(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"count": 0, "median_st": None, "p90_st": None}
    delta = np.abs(12.0 * np.log2(np.maximum(left[mask], 1e-12)) - 12.0 * np.log2(np.maximum(right[mask], 1e-12)))
    return {"count": int(delta.size), "median_st": float(np.median(delta)), "p90_st": float(np.quantile(delta, 0.90, method="linear"))}


def target_qc(
    receiver: WorldParams,
    interventions: Mapping[str, Any],
    mapping: F0Mapping,
    *,
    measured: Mapping[str, np.ndarray] | None = None,
    measurements: Mapping[str, Any] | None = None,
    arrays: Mapping[str, np.ndarray] | None = None,
    measurement_mode: str = "dio_raw_gate",
) -> dict[str, Any]:
    """Check the v2 formula and the final, on-disk measurement contract.

    ``measured`` is retained as a small v1 compatibility shim for unit tests;
    repair runs pass ``measurements`` with both H (Harvest) and D (DIO)
    trajectories and their time grids.
    """

    result: dict[str, Any] = {
        "status": "PASS",
        "reasons": [],
        "effective_coverage": float(interventions["effective_coverage"]),
        "effective_phone_count": int(interventions["effective_phone_count"]),
        "voiced_frame_count": int(interventions["voiced_frame_count"]),
        "support_frame_count": int(interventions.get("support_frame_count", 0)),
        "dose_contour_rms_st": interventions.get("dose_contour_rms_st"),
        "dose_level_rms_st": interventions.get("dose_level_rms_st"),
        "ordinary_identity_error": interventions.get("ordinary_identity_error", interventions.get("identity_error")),
        "weighted_identity_error": interventions.get("weighted_identity_error"),
        "identity_error": interventions.get("ordinary_identity_error", interventions.get("identity_error")),
        "measurement_algorithm": "dio->stonemask" if (measured is not None or measurements is not None) else None,
        "measurement_mode": measurement_mode,
    }
    reasons: list[str] = []
    if (
        result["effective_coverage"] < MIN_EFFECTIVE_COVERAGE
        or result["effective_phone_count"] < MIN_EFFECTIVE_PHONES
        or result["voiced_frame_count"] * FRAME_PERIOD_S < MIN_VOICED_SECONDS
    ):
        reasons.append("INSUFFICIENT_SUPPORT")
    ordinary_error = result["ordinary_identity_error"]
    if ordinary_error is None or not math.isfinite(float(ordinary_error)) or abs(float(ordinary_error)) > 1e-8:
        reasons.append("CONTOUR_MEAN_IDENTITY_FAIL")
    original_f0 = np.asarray(receiver.f0, dtype=np.float64)
    original_mask = original_f0 > 0.0
    for arm in ("ID", "LEVEL", "CONTOUR"):
        target = np.asarray(interventions.get("f0", {}).get(arm, np.full(original_f0.shape, np.nan)), dtype=np.float64)
        if target.shape != original_f0.shape or not np.isfinite(target).all() or np.any(target < 0.0):
            reasons.append(f"{arm}_TARGET_INVALID")
            continue
        if not np.array_equal(target > 0.0, original_mask):
            reasons.append(f"{arm}_TARGET_VOICED_MASK_CHANGED")
        if not np.array_equal(target[~np.asarray(mapping.valid, dtype=bool)], original_f0[~np.asarray(mapping.valid, dtype=bool)]):
            reasons.append(f"{arm}_M_OUTSIDE_CHANGED")
    contour_dose = result["dose_contour_rms_st"]
    result["dose_status"] = "UNMEASURABLE" if contour_dose is None else "LOW_DOSE" if contour_dose < MIN_CONTOUR_RMS_ST else "IDENTIFIABLE_DOSE"
    if contour_dose is None or result["dose_level_rms_st"] is None:
        reasons.append("EMPTY_DOSE_SUPPORT")

    if arrays is not None:
        checked: dict[str, np.ndarray] = {}
        for arm, values in arrays.items():
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            checked[arm] = values
            if values.shape != (receiver.sample_count,) or not np.isfinite(values).all():
                reasons.append(f"{arm}_INVALID_WAVEFORM")
            elif np.max(np.abs(values), initial=0.0) >= 1.0:
                reasons.append(f"{arm}_CLIPPING")
        if "RAW" in checked and "ID" in checked:
            for arm in ("ID", "LEVEL", "CONTOUR"):
                if arm in checked and abs(rms_db_delta(checked[arm], checked["RAW"])) > 0.1 + 1e-9:
                    reasons.append("RMS_MISMATCH_RAW")
                if arm in checked and abs(rms_db_delta(checked[arm], checked["ID"])) > 0.1 + 1e-9:
                    reasons.append("RMS_MISMATCH_ID")
        for arm in ("LEVEL", "CONTOUR"):
            if arm in checked and "ID" in checked:
                env = envelope_qc(checked[arm], checked["ID"])
                result[f"envelope_{arm}"] = env
                if env["status"] != "PASS":
                    reasons.append("ENVELOPE_CONFOUND")

    normalized: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    if measurements is None and measured is not None:
        measurements = {f"D_{arm}": value for arm, value in measured.items()}
    if measurements is not None:
        for name, value in measurements.items():
            try:
                f0, time = _measurement_array(value)
            except (TypeError, ValueError):
                reasons.append(f"{name}_MEASURE_INVALID")
                continue
            if f0.shape != receiver.f0.shape or not np.isfinite(f0).all() or np.any(f0 < 0.0):
                reasons.append(f"{name}_MEASURE_INVALID")
                continue
            if time is not None:
                if time.shape != receiver.time.shape or not np.isfinite(time).all() or not np.allclose(time, receiver.time, atol=1e-8, rtol=0.0):
                    result[f"{name}_time_grid_status"] = "TIME_GRID_MISMATCH"
                    reasons.append("TIME_GRID_MISMATCH")
                    continue
            normalized[name] = (f0, time)

        # Require every output measurement in a repair run.  Missing values
        # remain failures; they are never silently converted to zero arrays.
        required = ("H_RAW", "H_ID", "H_LEVEL", "H_CONTOUR", "D_RAW", "D_ID", "D_LEVEL", "D_CONTOUR") if measurements is not measured else ("D_ID", "D_LEVEL", "D_CONTOUR")
        for name in required:
            if name not in normalized:
                reasons.append(f"{name}_MEASURE_MISSING")

        if measurement_mode not in {"dio_raw_gate", "paired_harvest"}:
            reasons.append(f"UNKNOWN_MEASUREMENT_MODE:{measurement_mode}")
        reference_name = "D_ID" if measurement_mode == "dio_raw_gate" else "H_ID"
        reference_source_name = "H_source"
        source_measurement = normalized.get(reference_source_name, (receiver.f0, receiver.time))[0]
        reference = normalized.get(reference_name, (None, None))[0]
        if reference is not None:
            reference_mask = reference > 0.0
            source_mask = np.asarray(source_measurement, dtype=np.float64) > 0.0
            mismatch = float(np.mean(reference_mask != source_mask))
            result["ID_mask_mismatch_vs_H_source"] = mismatch
            result["ID_coverage_vs_H_source"] = float(np.sum(reference_mask & source_mask) / np.sum(source_mask)) if np.any(source_mask) else None
            id_stats = _f0_error_stats(reference, source_measurement, reference_mask & source_mask)
            result["ID_f0_abs_error_median_st"] = id_stats["median_st"]
            result["ID_f0_abs_error_p90_st"] = id_stats["p90_st"]
            # The historical v2 gate treated DIO's voiced mask as the
            # reference.  In paired_harvest mode the carrier's own Harvest
            # measurement is the pre-declared baseline; mask disagreement is
            # retained as a diagnostic, while overlap and F0 error remain hard
            # checks.  This prevents a detector disagreement on RAW from
            # being mistaken for a failed intervention.
            if measurement_mode == "dio_raw_gate" and mismatch > 0.10:
                reasons.append("ID_VOICED_MASK_MISMATCH")
            if result["ID_coverage_vs_H_source"] is None or result["ID_coverage_vs_H_source"] < 0.80:
                reasons.append("ID_ORIGINAL_VOICED_COVERAGE")
            if id_stats["median_st"] is None or id_stats["median_st"] > 1.0 or id_stats["p90_st"] > 3.0:
                reasons.append("ID_F0_ERROR")

        if reference is not None:
            id_mask = reference > 0.0
            target_support = np.asarray(mapping.valid, dtype=bool) & (np.asarray(mapping.weights) >= 0.5)
            source_mask = np.asarray(source_measurement, dtype=np.float64) > 0.0
            for arm in ("LEVEL", "CONTOUR"):
                candidate_name = f"D_{arm}" if measurement_mode == "dio_raw_gate" else f"H_{arm}"
                candidate = normalized.get(candidate_name, (None, None))[0]
                if candidate is None:
                    continue
                candidate_mask = candidate > 0.0
                mask_mismatch = float(np.mean(candidate_mask != id_mask))
                result[f"{arm}_mask_mismatch_vs_D_ID"] = mask_mismatch
                if measurement_mode == "dio_raw_gate" and mask_mismatch > 0.05:
                    reasons.append(f"{arm}_VOICED_MASK_MISMATCH")
                # The denominator is fixed before looking at the candidate:
                # strong target support intersected with the measured ID
                # baseline.  Candidate-specific voiced frames only affect the
                # numerator, so lost frames cannot be silently removed.
                fixed_support = target_support & id_mask & source_mask
                compared = fixed_support & candidate_mask
                support_count = int(np.sum(fixed_support))
                result[f"{arm}_target_support_coverage"] = float(np.sum(compared) / support_count) if support_count else None
                if not support_count or result[f"{arm}_target_support_coverage"] < 0.80:
                    reasons.append(f"{arm}_F0_MEASURE_COVERAGE")
                if np.any(compared):
                    measured_delta = 12.0 * np.log2(np.maximum(candidate[compared], 1e-12)) - 12.0 * np.log2(np.maximum(reference[compared], 1e-12))
                    target_delta = np.asarray(interventions["target_semitones"][arm])[compared] - np.asarray(interventions["target_semitones"]["ID"])[compared]
                    errors = np.abs(measured_delta - target_delta)
                    median = float(np.median(errors)); p90 = float(np.quantile(errors, 0.90, method="linear"))
                    result[f"{arm}_target_delta_abs_error_median_st"] = median
                    result[f"{arm}_target_delta_abs_error_p90_st"] = p90
                    if median > 1.0 or p90 > 3.0:
                        reasons.append(f"{arm}_F0_ERROR")
                else:
                    result[f"{arm}_target_delta_abs_error_median_st"] = None
                    result[f"{arm}_target_delta_abs_error_p90_st"] = None
                    reasons.append(f"{arm}_F0_ERROR")

    result["reasons"] = sorted(set(reasons))
    result["status"] = "PASS" if not reasons else "QC_FAIL"
    return result


def _manifest_rows(path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise ProtocolError(f"manifest has no records list: {path}")
    rows = [dict(row) for row in payload["records"] if isinstance(row, Mapping)]
    if not rows:
        raise ProtocolError(f"manifest records are empty: {path}")
    return rows


def _asset_from_row(row: Mapping[str, Any], field: str, *, fallback: Path) -> tuple[Path, str]:
    value = row.get(field) or row.get("filepath") or row.get("audio_path")
    if value is None:
        raise ProtocolError(f"manifest row has no {field}: {row.get('paired_key')}")
    # The manifest is allowed to contain the stale /mnt/e prefix, but only the
    # known repository prefix may be relocated.  If that fails, use the fixed
    # resource-root basename as a last explicit fallback.
    try:
        return _resolve_asset(str(value))
    except ProtocolError:
        candidate = fallback / Path(str(value)).name
        if candidate.is_file():
            return candidate.resolve(), "resource_root_basename"
        raise


def _row_grid_path(row: Mapping[str, Any], condition: str) -> tuple[Path, str]:
    value = row.get("textgrid_path")
    if value is None:
        candidate = RESOURCE_ROOT / "mfa" / condition / f"{int(row['sample_id']):04d}.TextGrid"
        if candidate.is_file():
            return candidate.resolve(), "resource_root_mfa"
        raise ProtocolError(f"manifest row has no TextGrid path: {row.get('paired_key')}")
    try:
        return _resolve_asset(str(value))
    except ProtocolError:
        candidate = RESOURCE_ROOT / "mfa" / condition / f"{int(row['sample_id']):04d}.TextGrid"
        if candidate.is_file():
            return candidate.resolve(), "resource_root_mfa"
        raise


def _pair_sort_key(paired_key: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_ID}|{paired_key}".encode("utf-8")).hexdigest()


def audit_and_freeze(paths: RunPaths, *, manifest_path: Path = MANIFEST, smoke: bool = False) -> dict[str, Any]:
    """Audit natural/TTS rows and freeze pilot/formal selection before scores."""

    raw_rows = _manifest_rows(manifest_path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in raw_rows:
        if str(row.get("variant", "raw")) != "raw":
            continue
        condition = str(row.get("condition", ""))
        if condition not in {"natural", "tts"}:
            continue
        key = (str(row.get("paired_key", "")), condition)
        if not key[0] or key in by_key:
            continue
        by_key[key] = row
    pair_candidates: list[dict[str, Any]] = []
    keys = sorted({key for key, condition in by_key if condition == "natural"})
    for paired_key in keys:
        nrow = by_key.get((paired_key, "natural"))
        trow = by_key.get((paired_key, "tts"))
        reasons: list[str] = []
        record: dict[str, Any] = {"paired_key": paired_key, "sample_id": nrow.get("sample_id") if nrow else None}
        if nrow is None or trow is None:
            reasons.append("MISSING_NATURAL_OR_TTS")
        else:
            if str(trow.get("tts_provider")) != "faster_qwen3":
                reasons.append("WRONG_TTS_PROVIDER")
            if str(nrow.get("source_utterance_id")) != str(trow.get("source_utterance_id")):
                reasons.append("SOURCE_UTTERANCE_MISMATCH")
            if normalize_label(str(nrow.get("transcript", ""))) != normalize_label(str(trow.get("transcript", ""))):
                reasons.append("TRANSCRIPT_MISMATCH")
            if str(nrow.get("speaker_id")) != str(trow.get("speaker_id")):
                reasons.append("SPEAKER_MISMATCH")
            try:
                npath, nresolve = _asset_from_row(nrow, "audio_path", fallback=RESOURCE_ROOT / "natural")
                tpath, tresolve = _asset_from_row(trow, "audio_path", fallback=RESOURCE_ROOT / "tts")
                _nvalues, nmeta = decode_audio(npath)
                _tvalues, tmeta = decode_audio(tpath)
                record.update({"natural_audio": str(npath), "tts_audio": str(tpath), "natural_audio_resolve": nresolve, "tts_audio_resolve": tresolve, "natural_meta": nmeta, "tts_meta": tmeta, "natural_container_sha256": file_sha256(npath), "tts_container_sha256": file_sha256(tpath)})
                for label, meta in (("natural", nmeta), ("tts", tmeta)):
                    if not (MIN_DURATION_S <= float(meta["duration_s"]) <= MAX_DURATION_S):
                        reasons.append(f"{label.upper()}_DURATION_OUT_OF_RANGE")
                    if float(meta["clipping_ratio"]) > MAX_CLIPPING_RATIO:
                        reasons.append(f"{label.upper()}_CLIPPING")
                ngrid_path, ngresolve = _row_grid_path(nrow, "natural")
                tgrid_path, tgresolve = _row_grid_path(trow, "tts")
                ngrid, tgrid = parse_textgrid(ngrid_path), parse_textgrid(tgrid_path)
                record.update({"natural_textgrid": str(ngrid_path), "tts_textgrid": str(tgrid_path), "natural_textgrid_sha256": file_sha256(ngrid_path), "tts_textgrid_sha256": file_sha256(tgrid_path), "natural_textgrid_resolve": ngresolve, "tts_textgrid_resolve": tgresolve, "natural_phones": ngrid["phones"], "tts_phones": tgrid["phones"]})
                if not _finite_phone_bounds(ngrid, float(nmeta["duration_s"])) or not _finite_phone_bounds(tgrid, float(tmeta["duration_s"])):
                    reasons.append("TEXTGRID_OUT_OF_AUDIO_BOUNDS")
                unknown = [x["label"] for x in ngrid["phones"] + tgrid["phones"] if x["unknown"]]
                if unknown:
                    reasons.append("UNKNOWN_PHONE_LABEL")
                if phone_sequence(ngrid) != phone_sequence(tgrid):
                    reasons.append("PHONE_SEQUENCE_MISMATCH")
                record.update({"phone_count": len(phone_sequence(ngrid)), "unknown_phone_labels": sorted(set(unknown)), "transcript": str(nrow.get("transcript", "")), "speaker_id": str(nrow.get("speaker_id", "")), "split": str(nrow.get("split", "")), "source_utterance_id": str(nrow.get("source_utterance_id", ""))})
            except (OSError, ValueError, ProtocolError) as exc:
                reasons.append(f"INPUT_ERROR:{type(exc).__name__}:{exc}")
        if reasons:
            failures.append({**record, "reasons": sorted(set(reasons))})
        else:
            record["pair_status"] = "eligible"
            record["sort_key"] = _pair_sort_key(paired_key)
            pair_candidates.append(record)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pair_candidates:
        by_speaker[str(record["speaker_id"])].append(record)
    for _speaker, speaker_rows in by_speaker.items():
        speaker_rows.sort(key=lambda row: str(row["sort_key"]))
    speakers = sorted(by_speaker)
    pilot: list[dict[str, Any]] = []
    for speaker in speakers[:PILOT_COUNT]:
        if by_speaker[speaker]:
            pilot.append(by_speaker[speaker][0])
    pilot_keys = {str(row["paired_key"]) for row in pilot}
    formal_speakers = [speaker for speaker in speakers if len([row for row in by_speaker[speaker] if row["paired_key"] not in pilot_keys]) >= PAIRS_PER_SPEAKER]
    formal_speakers = formal_speakers[:TARGET_FORMAL_SPEAKERS]
    formal: list[dict[str, Any]] = []
    for speaker in formal_speakers:
        formal.extend([row for row in by_speaker[speaker] if row["paired_key"] not in pilot_keys][:PAIRS_PER_SPEAKER])
    if smoke:
        formal = formal[:1]
    support_status = "PASS" if len(formal_speakers) >= MIN_FORMAL_SPEAKERS and len(formal) >= MIN_FORMAL_PAIRS else "INSUFFICIENT_SUPPORT"
    portrait = _freeze_portrait()
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_revision": PROTOCOL_REVISION,
        "status": "complete", "engineering_status": "AUDITED", "support_status": support_status,
        "manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha256(manifest_path),
        "resource_root": str(RESOURCE_ROOT.resolve()), "seed": SEED, "frozen_selection_rule": "speaker ascending; sha256(protocol|paired_key) within speaker",
        "pilot_pair_count": len(pilot), "formal_pair_count": len(formal), "pilot_speakers": sorted({str(row.get("speaker_id")) for row in pilot}), "formal_speakers": formal_speakers,
        "pilot_keys": [str(row["paired_key"]) for row in pilot], "formal_keys": [str(row["paired_key"]) for row in formal],
        "portrait": portrait, "candidate_count": len(pair_candidates), "excluded_count": len(failures),
        "records": [{**row, "role": "pilot"} for row in pilot] + [{**row, "role": "formal"} for row in formal],
        "excluded": failures,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    if paths.inputs.is_file():
        existing = verify_json(paths.inputs, self_hash=True)
        body = dict(existing); body.pop("artifact_sha256", None)
        if body != payload:
            raise ProtocolError("existing inputs.json differs from frozen audit; use a new run-id")
        return existing
    write_json(paths.inputs, payload, self_hash=True)
    return verify_json(paths.inputs, self_hash=True)


def _freeze_portrait() -> dict[str, Any]:
    try:
        inputs = read_json(PORTRAIT_INPUTS)
        row = next(row for row in inputs.get("records", []) if int(row.get("sample_id", -1)) == 151)
        portrait, resolve = _resolve_asset(str(row["portrait"]))
        portrait_hash = file_sha256(portrait)
        import cv2

        image = cv2.imread(str(portrait), cv2.IMREAD_COLOR)
        if image is None:
            raise ProtocolError(f"cannot decode portrait: {portrait}")
        rgb_hash = bytes_sha256(np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).tobytes())
        box: list[int] | None = None
        if PORTRAIT_AUDIO_MANIFEST.is_file():
            audio_manifest = read_json(PORTRAIT_AUDIO_MANIFEST)
            audio_row = next(item for item in audio_manifest.get("records", []) if int(item.get("sample_id", -1)) == 151)
            box = [int(x) for x in audio_row.get("box_xyxy", [])]
        if box is None or len(box) != 4:
            raise ProtocolError("portrait box for sample 151 is missing")
        return {"sample_id": 151, "path": str(portrait), "sha256": portrait_hash, "rgb_sha256": rgb_hash, "resolve": resolve, "box_xyxy": box, "width": int(image.shape[1]), "height": int(image.shape[0])}
    except Exception as exc:
        return {"status": "INPUT_UNAVAILABLE", "reason": str(exc)}


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"cannot decode portrait: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _load_pair_arrays(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    natural, nmeta = decode_audio(str(record["natural_audio"]))
    tts, tmeta = decode_audio(str(record["tts_audio"]))
    ngrid = {"phones": record["natural_phones"], "words": []}
    tgrid = {"phones": record["tts_phones"], "words": []}
    return natural, tts, nmeta | {"grid": ngrid}, tmeta | {"grid": tgrid}


def _materialize_arm(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.is_file():
        existing, _meta = decode_audio(path)
        # A resumed run reads PCM16 back as float64, so comparing decoded
        # floats byte-for-byte would reject its own quantized artifact.  Bind
        # the cache to the exact PCM16 contract instead.
        if existing.shape != values.shape or not np.array_equal(pcm16(existing), pcm16(values)):
            raise ProtocolError(f"existing audio arm differs from frozen construction: {path}")
        # ``decode_audio`` reports source hash but not PCM hash; compute it
        # from the exact PCM16 bytes used for the contract.
        pcm = pcm16(existing)
        return {"path": str(path.resolve()), "container_sha256": file_sha256(path), "pcm_sha256": bytes_sha256(pcm.tobytes()), "sample_count": int(existing.size), "rms": rms(existing), "peak": float(np.max(np.abs(existing), initial=0.0)), "resumed": True}
    return write_pcm16(path, values)


def _validate_measurement_grid(reference: WorldParams, time: np.ndarray, *, label: str) -> None:
    time = np.asarray(time, dtype=np.float64)
    if time.shape != reference.time.shape or not np.isfinite(time).all() or not np.allclose(time, reference.time, atol=1e-8, rtol=0.0):
        raise ProtocolError(f"TIME_GRID_MISMATCH:{label}: expected={reference.time.shape} got={time.shape}")


def _independent_f0_measure(params: WorldParams, values: np.ndarray) -> dict[str, np.ndarray]:
    """Re-extract F0 through DIO→stonemask, never interpolating grids."""

    pw = _pyworld()
    waveform = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if waveform.size < MIN_AUDIO_SAMPLES or not np.isfinite(waveform).all():
        raise ProtocolError("independent F0 input is invalid")
    f0, time = pw.dio(
        waveform,
        params.sample_rate,
        frame_period=params.frame_period_ms,
        f0_floor=params.f0_floor,
        f0_ceil=params.f0_ceil,
    )
    f0 = np.asarray(pw.stonemask(waveform, f0, time, params.sample_rate), dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if f0.ndim != 1 or time.shape != f0.shape or f0.size == 0 or not np.isfinite(f0).all() or not np.isfinite(time).all():
        raise ProtocolError("independent DIO F0 extraction returned invalid arrays")
    _validate_measurement_grid(params, time, label="DIO")
    return {"f0": np.ascontiguousarray(f0), "time": np.ascontiguousarray(time), "method": "dio->stonemask"}


def _harvest_f0_measure(params: WorldParams, values: np.ndarray) -> dict[str, np.ndarray]:
    """Re-extract Harvest→StoneMask on the final WAV without reusing targets."""

    measured = world_analyze(values, sample_rate=params.sample_rate)
    _validate_measurement_grid(params, measured.time, label="Harvest")
    return {"f0": np.ascontiguousarray(measured.f0), "time": np.ascontiguousarray(measured.time), "method": "harvest->stonemask"}


def _trajectory(value: Any, fallback_time: np.ndarray | None = None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        f0 = np.asarray(value.get("f0"), dtype=np.float64)
        time = np.asarray(value.get("time"), dtype=np.float64) if value.get("time") is not None else fallback_time
        return {"f0": f0, "time": time, "method": value.get("method")}
    return {"f0": np.asarray(value, dtype=np.float64), "time": fallback_time, "method": None}


def _trajectory_comparison(name: str, left: Any, right: Any) -> dict[str, Any]:
    """Compare two measured trajectories while preserving every denominator."""

    a, b = _trajectory(left), _trajectory(right)
    af, bf = a["f0"], b["f0"]
    result: dict[str, Any] = {"comparison": name, "status": "PASS"}
    if af.ndim != 1 or bf.shape != af.shape or not np.isfinite(af).all() or not np.isfinite(bf).all() or np.any(af < 0.0) or np.any(bf < 0.0):
        return result | {"status": "MEASURE_INVALID", "total_frames": int(af.size), "a_voiced": None, "b_voiced": None, "intersection": None, "coverage": None, "mask_mismatch": None, "f0_abs_error_median_st": None, "f0_abs_error_p90_st": None}
    a_mask, b_mask = af > 0.0, bf > 0.0
    both = a_mask & b_mask
    a_count, b_count, intersection = int(np.sum(a_mask)), int(np.sum(b_mask)), int(np.sum(both))
    result.update({"total_frames": int(af.size), "a_voiced": a_count, "b_voiced": b_count, "intersection": intersection, "lost_voiced": int(np.sum(a_mask & ~b_mask)), "gained_voiced": int(np.sum(~a_mask & b_mask)), "coverage": float(intersection / a_count) if a_count else None, "mask_mismatch": float(np.mean(a_mask != b_mask))})
    if intersection:
        stats = _f0_error_stats(af, bf, both)
        result.update({"f0_abs_error_median_st": stats["median_st"], "f0_abs_error_p90_st": stats["p90_st"]})
    else:
        result.update({"f0_abs_error_median_st": None, "f0_abs_error_p90_st": None})
    return result


def _four_cell_counts(reference: np.ndarray, raw: np.ndarray, identity: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64) > 0.0
    raw_mask = np.asarray(raw, dtype=np.float64) > 0.0
    id_mask = np.asarray(identity, dtype=np.float64) > 0.0
    if ref.shape != raw_mask.shape or ref.shape != id_mask.shape:
        return {"status": "MEASURE_INVALID"}
    return {
        "status": "PASS",
        "reference": "H_RAW",
        "both_raw_and_id_voiced": int(np.sum(ref & raw_mask & id_mask)),
        "only_raw_voiced": int(np.sum(ref & raw_mask & ~id_mask)),
        "only_id_voiced": int(np.sum(ref & ~raw_mask & id_mask)),
        "both_unvoiced": int(np.sum(ref & ~raw_mask & ~id_mask)),
        "reference_voiced": int(np.sum(ref)),
    }


def build_audio_for_pair(
    record: Mapping[str, Any],
    paths: RunPaths,
    *,
    require_measurement: bool = True,
    measurement_mode: str = "dio_raw_gate",
    protocol_id: str = REPAIR_PROTOCOL_ID,
) -> dict[str, Any]:
    """Construct the eight arms and QC the actual PCM16 files on disk."""

    sid = str(record["paired_key"])
    natural, tts, _nmeta, _tmeta = _load_pair_arrays(record)
    nparams, tparams = world_analyze(natural), world_analyze(tts)
    ngrid, tgrid = {"phones": record["natural_phones"], "words": []}, {"phones": record["tts_phones"], "words": []}
    n_to_t = build_f0_mapping(nparams, tparams, ngrid, tgrid)
    t_to_n = build_f0_mapping(tparams, nparams, tgrid, ngrid)
    n_intervention, t_intervention = f0_interventions(nparams, tparams, n_to_t), f0_interventions(tparams, nparams, t_to_n)
    receiver_data: dict[str, dict[str, Any]] = {}
    for receiver_name, raw, params, intervention, mapping in (("N", natural, nparams, n_intervention, n_to_t), ("T", tts, tparams, t_intervention, t_to_n)):
        synth: dict[str, np.ndarray] = {}
        rms_scales: dict[str, float] = {}
        length_adjustments: dict[str, dict[str, Any]] = {}
        for arm in ("ID", "LEVEL", "CONTOUR"):
            reconstructed = world_synthesize(params, intervention["f0"][arm])
            reconstructed, length_meta = exact_length(reconstructed, raw.size)
            if abs(int(length_meta["adjustment_samples"])) > int(round(FRAME_PERIOD_S * SAMPLE_RATE)) + 1:
                raise ProtocolError(f"{receiver_name}_{arm} WORLD length correction exceeds 81 samples")
            reconstructed, scale = normalize_rms(reconstructed, rms(raw))
            synth[arm] = reconstructed
            rms_scales[arm] = scale
            length_adjustments[arm] = length_meta
        scaled, common_scale = _common_safe_scale({"RAW": raw, **synth})
        arms_values = {f"{receiver_name}_RAW": scaled["RAW"], f"{receiver_name}_ID": scaled["ID"], f"{receiver_name}_LEVEL": scaled["LEVEL"], f"{receiver_name}_CONTOUR": scaled["CONTOUR"]}
        receiver_data[receiver_name] = {"params": params, "intervention": intervention, "mapping": mapping, "arrays": arms_values, "target_f0s": intervention["f0"], "rms_scales": rms_scales, "length_adjustments": length_adjustments, "common_safe_scale": common_scale}

    paths.parameters_dir.mkdir(parents=True, exist_ok=True)
    parameter_path = paths.parameters_dir / f"{sid}.npz"
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    output_paths: dict[str, dict[str, Any]] = {}
    for receiver_name, data in receiver_data.items():
        output_paths[receiver_name] = {}
        final_arrays: dict[str, np.ndarray] = {}
        for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
            name = f"{receiver_name}_{arm}"
            meta = _materialize_arm(paths.audio_dir / sid / f"{name}.wav", data["arrays"][name])
            output_paths[receiver_name][arm] = meta
            final_arrays[arm], _ = decode_audio(meta["path"])

        params = data["params"]
        measurements: dict[str, Any] = {
            "H_source": {"f0": params.f0, "time": params.time, "method": "harvest->stonemask"},
        }
        measurement_errors: dict[str, str] = {}
        if require_measurement:
            for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                try:
                    measurements[f"H_{arm}"] = _harvest_f0_measure(params, final_arrays[arm])
                except Exception as exc:  # preserve all missing-arm evidence
                    measurement_errors[f"H_{arm}"] = f"{type(exc).__name__}:{exc}"
                try:
                    measurements[f"D_{arm}"] = _independent_f0_measure(params, final_arrays[arm])
                except Exception as exc:
                    measurement_errors[f"D_{arm}"] = f"{type(exc).__name__}:{exc}"
        qc_measurements = {key: value for key, value in measurements.items() if key.startswith("H_") or key.startswith("D_")}
        qc = target_qc(
            params,
            data["intervention"],
            data["mapping"],
            measurements=qc_measurements if require_measurement else None,
            arrays=final_arrays,
            measurement_mode=measurement_mode,
        )
        if measurement_errors:
            qc["measurement_status"] = "MEASUREMENT_INCOMPLETE"
            qc["measurement_errors"] = measurement_errors
        else:
            qc["measurement_status"] = "MEASURED" if require_measurement else "NOT_REQUESTED"
        # Fixed diagnostic matrix uses only measured trajectories.  A missing
        # trajectory produces a row with explicit missing status.
        for left, right in (("H_source", "H_RAW"), ("H_RAW", "D_RAW"), ("D_RAW", "D_ID"), ("H_RAW", "H_ID"), ("H_source", "D_ID"), ("D_ID", "D_LEVEL"), ("D_ID", "D_CONTOUR")):
            comparison = _trajectory_comparison(f"{left}->{right}", measurements.get(left, {"f0": np.asarray([], dtype=np.float64)}), measurements.get(right, {"f0": np.asarray([], dtype=np.float64)}))
            comparison.update({"paired_key": sid, "receiver": receiver_name})
            diagnostics.append(comparison)
        if all(name in measurements for name in ("H_RAW", "D_RAW", "D_ID")):
            four = _four_cell_counts(measurements["H_RAW"]["f0"], measurements["D_RAW"]["f0"], measurements["D_ID"]["f0"])
        else:
            four = {"status": "MEASURE_MISSING"}
        data["measurements"] = measurements
        data["measurement_errors"] = measurement_errors
        data["four_cells"] = four
        data["final_arrays"] = final_arrays
        qc_row = {"paired_key": sid, "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "receiver": receiver_name, **{key: value for key, value in qc.items() if not isinstance(value, (dict, list))}}
        qc_row["reasons"] = ";".join(qc.get("reasons", []))
        rows.append(qc_row)

    def _stored_measurement(data: Mapping[str, Any], name: str, component: str) -> np.ndarray:
        item = data.get("measurements", {}).get(name)
        if isinstance(item, Mapping) and component in item:
            return np.asarray(item[component], dtype=np.float64)
        return np.full(data["params"].f0.shape, np.nan, dtype=np.float64)

    save_arrays: dict[str, np.ndarray] = {}
    for receiver_name, data in receiver_data.items():
        prefix = receiver_name
        params = data["params"]
        mapping = data["mapping"]
        save_arrays.update({f"{prefix}_f0": params.f0, f"{prefix}_time": params.time, f"{prefix}_sp": params.sp, f"{prefix}_ap": params.ap})
        for arm in ("ID", "LEVEL", "CONTOUR"):
            save_arrays[f"{prefix}_target_{arm}"] = data["target_f0s"][arm]
        for name in ("H_source", "H_RAW", "H_ID", "H_LEVEL", "H_CONTOUR", "D_RAW", "D_ID", "D_LEVEL", "D_CONTOUR"):
            save_arrays[f"{prefix}_{name}_f0"] = _stored_measurement(data, name, "f0")
            save_arrays[f"{prefix}_{name}_time"] = _stored_measurement(data, name, "time")
        save_arrays.update({f"{prefix}_weight": mapping.weights, f"{prefix}_valid": mapping.valid.astype(np.uint8), f"{prefix}_donor_time": mapping.donor_time, f"{prefix}_phone_index": mapping.phone_index, f"{prefix}_donor_left": mapping.donor_left, f"{prefix}_donor_right": mapping.donor_right, f"{prefix}_donor_alpha": mapping.donor_alpha})
    np.savez_compressed(parameter_path, **save_arrays)
    parameter_meta = {
        "protocol_id": protocol_id,
        "paired_key": sid,
        "sample_id": record.get("sample_id"),
        "parameter_sha256": file_sha256(parameter_path),
        "world": {"frame_period_ms": FRAME_PERIOD_MS, "f0_floor": F0_FLOOR, "f0_ceil": F0_CEIL, "post_stonemask_floor": F0_POST_STONEMASK_FLOOR, "analysis": "harvest->stonemask; cheaptrick; d4c", "independent_measurement": "dio->stonemask", "formula": "ordinary_mean_preserving_v2", "measurement_mode": measurement_mode},
        "receivers": {
            name: {
                "mu_receiver": data["intervention"]["mu_receiver"], "mu_donor": data["intervention"]["mu_donor"], "weighted_support": data["intervention"]["weighted_support"], "effective_coverage": data["intervention"]["effective_coverage"], "effective_phone_count": data["intervention"]["effective_phone_count"], "dose_contour_rms_st": data["intervention"]["dose_contour_rms_st"], "dose_level_rms_st": data["intervention"]["dose_level_rms_st"], "ordinary_identity_error": data["intervention"]["ordinary_identity_error"], "weighted_identity_error": data["intervention"]["weighted_identity_error"], "mapping_reasons": dict((reason, data["mapping"].invalid_reasons.count(reason)) for reason in sorted(set(data["mapping"].invalid_reasons))), "mapping_invalid_reasons": list(data["mapping"].invalid_reasons), "length_adjustments": data["length_adjustments"], "common_safe_scale": data["common_safe_scale"], "rms_scales": data["rms_scales"], "measurement_errors": data["measurement_errors"], "four_cells": data["four_cells"], "qc": next(row for row in rows if row["receiver"] == name),
            }
            for name, data in receiver_data.items()
        },
    }
    write_json(parameter_path.with_suffix(".json"), parameter_meta, self_hash=True)
    return {"protocol_id": protocol_id, "paired_key": sid, "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "audio": output_paths, "parameter_path": str(parameter_path.resolve()), "parameter_sha256": file_sha256(parameter_path), "qc": rows, "diagnostics": diagnostics, "four_cells": {name: data["four_cells"] for name, data in receiver_data.items()}, "world": {"N": nparams, "T": tparams}}


def write_qc_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if "paired_key" in fields:
        fields.remove("paired_key"); fields.insert(0, "paired_key")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audio_stage(
    paths: RunPaths,
    inputs: Mapping[str, Any],
    *,
    pilot_only: bool = False,
    measurement_mode: str = "dio_raw_gate",
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    wanted_role = "pilot" if pilot_only else "formal"
    records = [row for row in inputs.get("records", []) if str(row.get("role")) == wanted_role]
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        try:
            if measurement_mode == "dio_raw_gate" and protocol_id == PROTOCOL_ID:
                # Keep the historical two-argument call path for lightweight
                # callers/tests that replace the builder with a stub.
                result = build_audio_for_pair(record, paths)
            else:
                result = build_audio_for_pair(record, paths, measurement_mode=measurement_mode, protocol_id=protocol_id)
            manifests.append(result)
            rows.extend(result["qc"])
            print(f"AUDIO {index}/{len(records)} {record['paired_key']}", flush=True)
        except AudioBackendUnavailable as exc:
            failures.append({"paired_key": record.get("paired_key"), "stage": "audio", "status": "AUDIO_BACKEND_UNAVAILABLE", "error": str(exc)})
            rows.extend({"paired_key": record.get("paired_key"), "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "receiver": receiver, "status": "AUDIO_BACKEND_UNAVAILABLE", "measurement_status": "AUDIO_BACKEND_UNAVAILABLE", "reasons": str(exc)} for receiver in ("N", "T"))
        except Exception as exc:  # noqa: BLE001 - retain per-pair failure ledger
            failures.append({"paired_key": record.get("paired_key"), "stage": "audio", "status": "QC_FAIL", "error": str(exc)})
            rows.extend({"paired_key": record.get("paired_key"), "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "receiver": receiver, "status": "QC_FAIL", "measurement_status": "NOT_AVAILABLE", "reasons": f"{type(exc).__name__}:{exc}"} for receiver in ("N", "T"))
    write_qc_csv(paths.audio_qc, rows)
    quality_manifest_sha256 = None
    if pilot_only:
        quality_rows: list[dict[str, Any]] = []
        for result in sorted(manifests, key=lambda item: str(item.get("paired_key"))):
            for receiver in ("N", "T"):
                for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                    audio_row = result.get("audio", {}).get(receiver, {}).get(arm, {})
                    quality_rows.append({"blind_id": f"clip_{len(quality_rows) + 1:03d}", "paired_key": result.get("paired_key"), "receiver": receiver, "arm": arm, "audio": audio_row.get("path"), "audio_sha256": audio_row.get("container_sha256")})
        quality_payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "QUALITY_NOT_ASSESSED", "purpose": "pilot blind listening table; no human listening was performed by the automatic runner", "rows": quality_rows}
        write_json(paths.pilot_quality, quality_payload, self_hash=True)
        quality_manifest_sha256 = file_sha256(paths.pilot_quality)
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "complete" if not failures else "incomplete", "pilot_only": pilot_only, "quality_status": "QUALITY_NOT_ASSESSED", "quality_manifest": str(paths.pilot_quality.resolve()) if pilot_only else None, "quality_manifest_sha256": quality_manifest_sha256, "pair_count": len(manifests), "expected_pair_count": len(records), "expected_keys": [str(record["paired_key"]) for record in records] if pilot_only else None, "manifests": [{key: value for key, value in result.items() if key != "world"} for result in manifests], "qc_rows": len(rows), "failures": failures, "audio_qc_sha256": file_sha256(paths.audio_qc) if paths.audio_qc.is_file() else None, "backend": _audio_backend_info()}
    target = paths.pilot_qc if pilot_only else paths.root / "audio_manifest.json"
    write_json(target, payload, self_hash=True)
    return verify_json(target, self_hash=True)


def _load_audio_manifest(paths: RunPaths) -> dict[str, Any]:
    path = paths.root / "audio_manifest.json"
    if not path.is_file():
        raise ProtocolError("audio_manifest.json is missing")
    return verify_json(path, self_hash=True)


def _audio_arm_index(audio_manifest: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for result in audio_manifest.get("manifests", []):
        key = str(result.get("paired_key"))
        audio = result.get("audio", {})
        for receiver in ("N", "T"):
            for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                row = audio.get(receiver, {}).get(arm)
                if isinstance(row, Mapping):
                    index[(key, f"{receiver}_{arm}")] = row
    return index


def _portrait_rgb_sha256(path: Path) -> str:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"cannot read portrait: {path}")
    return bytes_sha256(np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).tobytes())


def _video_info(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open generated video: {path}")
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if count < 25 or not math.isfinite(fps) or abs(fps - FPS) > 0.01:
        raise ProtocolError(f"generated video has invalid support: {path} frames={count} fps={fps}")
    return {"frame_count": count, "fps": fps, "width": width, "height": height}


def _render_one(paths: RunPaths, record: Mapping[str, Any], audio_index: Mapping[tuple[str, str], Mapping[str, Any]], arm: str, *, device: str = "cuda", force: bool = False, output_override: Path | None = None) -> dict[str, Any]:
    portrait_meta = record.get("_portrait") or {}
    portrait = Path(str(portrait_meta.get("path", ""))).resolve()
    if not portrait.is_file():
        raise ProtocolError(f"portrait is missing: {portrait}")
    box = [int(x) for x in portrait_meta.get("box_xyxy", [])]
    if len(box) != 4 or not (0 <= box[0] < box[2] <= int(portrait_meta.get("width", 0)) and 0 <= box[1] < box[3] <= int(portrait_meta.get("height", 0))):
        raise ProtocolError("frozen portrait box is invalid")
    audio_row = audio_index.get((str(record["paired_key"]), arm))
    if audio_row is None:
        raise ProtocolError(f"audio arm missing: {record['paired_key']}/{arm}")
    audio = Path(str(audio_row["path"])).resolve()
    output = output_override or (paths.videos_dir / arm / f"{record['paired_key']}.mkv")
    sidecar = output.with_suffix(".json")
    if output.is_file() and sidecar.is_file() and not force:
        old = verify_json(sidecar, self_hash=True)
        if old.get("audio_sha256") != audio_row.get("container_sha256") or old.get("output_sha256") != file_sha256(output):
            raise ProtocolError(f"resumed video identity changed: {record['paired_key']}/{arm}")
        if not _fingerprint_matches(old):
            raise ProtocolError(f"stale video cache after runner/spec change: {record['paired_key']}/{arm}")
        if old.get("image_sha256") != file_sha256(portrait) or old.get("box_xyxy") != box or old.get("checkpoint_sha256") != file_sha256(WAV2LIP_CHECKPOINT) or old.get("device") != device:
            raise ProtocolError(f"stale video cache after image/box/checkpoint change: {record['paired_key']}/{arm}")
        _video_info(output)
        return old
    if output.exists() or sidecar.exists():
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    worker_result = paths.videos_dir / "work" / str(record["paired_key"]) / f"{output.stem}.worker.json"
    log = paths.logs_dir / "render" / str(record["paired_key"]) / f"{arm}.log"
    command = [str(WAV2LIP_PYTHON), str(RENDER_WORKER), "--image", str(portrait), "--image-rgb-sha256", _portrait_rgb_sha256(portrait), "--audio", str(audio), "--box", *[str(x) for x in box], "--checkpoint", str(WAV2LIP_CHECKPOINT), "--ffmpeg", str(FFMPEG), "--outfile", str(output), "--result", str(worker_result), "--batch-size", "4", "--seed", str(SEED), "--device", device]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, check=False, env={**os.environ, "PYTHONHASHSEED": str(SEED)})
    if result.returncode != 0:
        raise ProtocolError(f"Wav2Lip render failed ({result.returncode}); see {log}")
    if not worker_result.is_file() or not output.is_file():
        raise ProtocolError(f"Wav2Lip worker did not produce output: {worker_result}")
    worker = verify_json(worker_result)
    if worker.get("status") != "complete" or worker.get("device") != device or worker.get("source_frame_indices") != [0] * int(worker.get("frames_rendered", 0)):
        raise ProtocolError(f"Wav2Lip worker violated static-frame contract: {record['paired_key']}/{arm}")
    info = _video_info(output)
    payload = {
        "schema_version": 1, "status": "complete", **_implementation_fingerprint(), "paired_key": str(record["paired_key"]), "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "video_arm": arm, "audio_arm": arm, "image": str(portrait), "image_sha256": file_sha256(portrait), "image_rgb_sha256": _portrait_rgb_sha256(portrait), "box_xyxy": box, "audio": str(audio), "audio_sha256": audio_row.get("container_sha256"), "audio_pcm_sha256": audio_row.get("pcm_sha256"), "output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": info["frame_count"], "fps": info["fps"], "checkpoint_sha256": file_sha256(WAV2LIP_CHECKPOINT), "device": worker.get("device"), "seed": SEED, "worker_result": str(worker_result.resolve()), "worker_result_sha256": file_sha256(worker_result), "command": command, "log": str(log.resolve()),
    }
    write_json(sidecar, payload, self_hash=True)
    return verify_json(sidecar, self_hash=True)


def run_render_stage(paths: RunPaths, inputs: Mapping[str, Any], audio_manifest: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    if inputs.get("portrait", {}).get("status") == "INPUT_UNAVAILABLE":
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "incomplete", "engineering_status": "INPUT_UNAVAILABLE", "failures": [{"stage": "render", "error": inputs["portrait"].get("reason")}], "videos": []}
        write_json(paths.videos_manifest, payload, self_hash=True)
        return verify_json(paths.videos_manifest, self_hash=True)
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise ProtocolError("CUDA requested but unavailable")
        except ImportError as exc:
            raise ProtocolError(f"torch unavailable for CUDA render: {exc}") from exc
    audio_index = _audio_arm_index(audio_manifest)
    records = [row for row in inputs.get("records", []) if row.get("role") == "formal"]
    # Audio metadata is already frozen; candidate arms are rendered only when
    # both receiver QC rows pass.  RAW/ID are still required for every formal
    # pair and therefore provide a clear background denominator.
    qc_rows = []
    if paths.audio_qc.is_file():
        with paths.audio_qc.open(encoding="utf-8", newline="") as handle:
            qc_rows = list(csv.DictReader(handle))
    qc_by_key_receiver = {(str(row.get("paired_key")), str(row.get("receiver"))): row for row in qc_rows}
    videos: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        record = dict(record)
        record["_portrait"] = inputs["portrait"]
        for receiver in ("N", "T"):
            qc = qc_by_key_receiver.get((str(record["paired_key"]), receiver), {})
            qc_pass = str(qc.get("status", "")) == "PASS"
            arms = [f"{receiver}_RAW", f"{receiver}_ID"]
            if qc_pass:
                arms.extend([f"{receiver}_LEVEL", f"{receiver}_CONTOUR"])
            for arm in arms:
                try:
                    videos.append(_render_one(paths, record, audio_index, arm, device=device))
                except Exception as exc:  # noqa: BLE001 - preserve missing cells
                    failures.append({"paired_key": record.get("paired_key"), "video_arm": arm, "stage": "render", "error": str(exc)})
        print(f"VIDEO {record['paired_key']} count={sum(v['paired_key'] == record['paired_key'] for v in videos)}", flush=True)
    expected_min = len(records) * 4
    status = "complete" if not failures and len(videos) >= expected_min else "incomplete"
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": status, "engineering_status": "PASS" if status == "complete" else "INCOMPLETE", "record_count": len(records), "video_count": len(videos), "expected_minimum_videos": expected_min, "videos": videos, "failures": failures, "render_device_request": device, "checkpoint_sha256": file_sha256(WAV2LIP_CHECKPOINT) if WAV2LIP_CHECKPOINT.is_file() else None}
    write_json(paths.videos_manifest, payload, self_hash=True)
    return verify_json(paths.videos_manifest, self_hash=True)


def _crop_score_video(path: Path, box: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    x1, y1, x2, y2 = [int(x) for x in box]
    side = x2 - x1
    if side <= 0 or y2 - y1 != side:
        raise ProtocolError("SyncNet score box must be a positive square")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open generated video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            if sx2 > sx1 and sy2 > sy1:
                canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
            frames.append(cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LINEAR))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if len(frames) < 25 or abs(fps - FPS) > 0.01:
        raise ProtocolError(f"video has insufficient SyncNet support: {path} frames={len(frames)} fps={fps}")
    return np.stack(frames, axis=0), {"frame_count": len(frames), "fps": fps, "box": [x1, y1, x2, y2]}


def _load_syncnet(device: str) -> tuple[Any, Any, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError(f"torch unavailable for SyncNet: {exc}") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable for SyncNet")
    if str(SYNCNET_ROOT) not in sys.path:
        sys.path.insert(0, str(SYNCNET_ROOT))
    try:
        from SyncNetInstance import SyncNetInstance
    except Exception as exc:  # pragma: no cover
        raise ProtocolError(f"cannot import SyncNetInstance: {exc}") from exc
    scorer = SyncNetInstance(device=device)
    if not SYNCNET_MODEL.is_file():
        raise ProtocolError(f"SyncNet checkpoint is missing: {SYNCNET_MODEL}")
    scorer.loadParameters(str(SYNCNET_MODEL))
    scorer.eval()
    return scorer, torch, device


def extract_visual_embedding(video: np.ndarray, scorer: Any, torch: Any, *, device: str) -> np.ndarray:
    count = int(video.shape[0] - 4)
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, 20):
            end = min(count, start + 20)
            sequences = np.stack([video[i:i + 5] for i in range(start, end)], axis=0)
            tensor = torch.from_numpy(np.transpose(sequences, (0, 4, 1, 2, 3))).float().to(device)
            values.append(scorer.__S__.forward_lip(tensor).detach().cpu().numpy().astype(np.float32))
    embedding = np.concatenate(values, axis=0)
    if embedding.ndim != 2 or embedding.shape[1] != 1024 or not np.isfinite(embedding).all():
        raise ProtocolError(f"SyncNet visual embedding has invalid shape: {embedding.shape}")
    return embedding


def extract_audio_embedding(path: Path, scorer: Any, torch: Any, *, device: str) -> tuple[np.ndarray, int]:
    import python_speech_features
    from scipy.io import wavfile

    rate, values = wavfile.read(str(path))
    values = np.asarray(values)
    if int(rate) != SAMPLE_RATE or values.ndim != 1 or values.dtype.kind not in "iu":
        raise ProtocolError(f"SyncNet audio must be mono 16 kHz PCM: {path}")
    mfcc = np.asarray(list(zip(*python_speech_features.mfcc(values, int(rate)))), dtype=np.float32)
    if mfcc.ndim != 2 or mfcc.shape[0] != 13:
        raise ProtocolError(f"unexpected SyncNet MFCC shape: {mfcc.shape}")
    count = int((mfcc.shape[1] - 20) // 4 + 1)
    if count < 1:
        raise ProtocolError(f"audio has no SyncNet windows: {path}")
    values_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, 20):
            end = min(count, start + 20)
            chunks = np.asarray([mfcc[:, i * 4:i * 4 + 20] for i in range(start, end)], dtype=np.float32)
            tensor = torch.from_numpy(chunks[:, None]).float().to(device)
            values_out.append(scorer.__S__.forward_aud(tensor).detach().cpu().numpy().astype(np.float32))
    embedding = np.concatenate(values_out, axis=0)
    if embedding.ndim != 2 or embedding.shape[1] != 1024 or not np.isfinite(embedding).all():
        raise ProtocolError(f"SyncNet audio embedding has invalid shape: {embedding.shape}")
    return embedding, int(values.size)


def syncnet_distance_matrix(visual: np.ndarray, audio: np.ndarray, *, vshift: int = VSHIFT) -> np.ndarray:
    """Match torch pairwise_distance(eps=1e-6) in float32."""

    visual = np.asarray(visual, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != 1024 or audio.shape[1] != 1024:
        raise ProtocolError("invalid SyncNet embedding shape")
    rows = min(visual.shape[0], audio.shape[0])
    result = np.full((rows, 2 * vshift + 1), np.nan, dtype=np.float64)
    padded = np.pad(audio, ((vshift, vshift), (0, 0)), mode="constant")
    for t in range(rows):
        delta = visual[t:t + 1] - padded[t:t + 2 * vshift + 1]
        # PyTorch's pairwise_distance adds eps to every coordinate before its
        # norm.  The explicit float32 operation preserves the official curve.
        result[t] = np.sqrt(np.sum((delta + np.float32(1e-6)) ** 2, axis=1, dtype=np.float32), dtype=np.float32)
    return result


def common_window_indices(visual: Mapping[str, np.ndarray], audio: Mapping[str, np.ndarray], *, vshift: int = VSHIFT) -> np.ndarray:
    if not visual or not audio:
        return np.asarray([], dtype=np.int64)
    limit = min(min(len(value) for value in visual.values()), min(len(value) for value in audio.values()))
    indices = np.arange(vshift, limit - vshift, dtype=np.int64)
    return indices


def curve_metrics(matrix: np.ndarray, support: Sequence[int], *, k0: int | None = None, vshift: int = VSHIFT) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    support = np.asarray(list(support), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 * vshift + 1 or support.size < 1 or np.any(support < 0) or np.any(support >= matrix.shape[0]):
        raise ProtocolError("invalid SyncNet curve support")
    values = matrix[support]
    if not np.isfinite(values).all():
        raise ProtocolError("SyncNet curve support contains non-finite distances")
    curve = values.mean(axis=0, dtype=np.float64)
    min_index = int(np.flatnonzero(curve == np.min(curve))[0])
    lags = np.arange(-vshift, vshift + 1, dtype=np.int64)
    result: dict[str, Any] = {"curve": curve.tolist(), "lags": lags.tolist(), "support_count": int(support.size), "support_rows": support.tolist(), "D": float(curve[min_index]), "B": float(np.median(curve)), "C": float(np.median(curve) - curve[min_index]), "k_star": int(lags[min_index]), "d_k0": None, "k0": k0}
    if k0 is not None:
        if k0 < -vshift or k0 > vshift:
            raise ProtocolError(f"k0 out of range: {k0}")
        result["d_k0"] = float(curve[vshift + int(k0)])
    return result


CELL_SUFFIXES = {
    "own": (("RAW", "RAW"), ("ID", "ID"), ("LEVEL", "LEVEL"), ("CONTOUR", "CONTOUR")),
    "fixed_audio": (("LEVEL", "ID"), ("CONTOUR", "ID")),
    "fixed_video": (("ID", "LEVEL"), ("ID", "CONTOUR")),
    "raw_replacement": (("ID", "RAW"), ("CONTOUR", "RAW")),
}


def expected_score_cells() -> tuple[tuple[str, str], ...]:
    cells: list[tuple[str, str]] = []
    for group in CELL_SUFFIXES.values():
        cells.extend(group)
    return tuple(cells)


def run_score_stage(paths: RunPaths, inputs: Mapping[str, Any], videos_manifest: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    if not SYNCNET_MODEL.is_file():
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "incomplete", "engineering_status": "INPUT_UNAVAILABLE", "failures": [{"stage": "score", "error": f"missing SyncNet model: {SYNCNET_MODEL}"}], "rows": []}
        write_json(paths.score_manifest, payload, self_hash=True)
        return verify_json(paths.score_manifest, self_hash=True)
    audio_manifest = _load_audio_manifest(paths)
    audio_index = _audio_arm_index(audio_manifest)
    feature_cache_valid = False
    cached_feature_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    if paths.features_manifest.is_file():
        try:
            cached_features = verify_json(paths.features_manifest, self_hash=True)
            feature_cache_valid = _fingerprint_matches(cached_features) and cached_features.get("model_sha256") == file_sha256(SYNCNET_MODEL)
            raw_feature_rows = cached_features.get("rows", [])
            if isinstance(raw_feature_rows, list):
                cached_feature_rows = {(str(row.get("paired_key")), str(row.get("receiver")), str(row.get("arm"))): row for row in raw_feature_rows if isinstance(row, Mapping)}
            else:
                feature_cache_valid = False
        except (OSError, ProtocolError, ValueError):
            feature_cache_valid = False
    video_index = {(str(row.get("paired_key")), str(row.get("video_arm"))): row for row in videos_manifest.get("videos", [])}
    qc_by_key_receiver: dict[tuple[str, str], Mapping[str, Any]] = {}
    if paths.audio_qc.is_file():
        with paths.audio_qc.open(encoding="utf-8", newline="") as handle:
            qc_by_key_receiver = {(str(row.get("paired_key")), str(row.get("receiver"))): row for row in csv.DictReader(handle)}
    formal_records = [row for row in inputs.get("records", []) if row.get("role") == "formal"]
    score_records: list[Mapping[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    for record in formal_records:
        key = str(record["paired_key"])
        reasons: list[str] = []
        for receiver in ("N", "T"):
            qc = qc_by_key_receiver.get((key, receiver))
            if qc is None or str(qc.get("status")) != "PASS":
                reasons.append(f"{receiver}_AUDIO_QC_NOT_PASS")
            for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                if (key, f"{receiver}_{arm}") not in audio_index:
                    reasons.append(f"MISSING_AUDIO_{receiver}_{arm}")
        if reasons:
            excluded_pairs.append({"paired_key": key, "speaker_id": record.get("speaker_id"), "reasons": sorted(set(reasons))})
        else:
            score_records.append(record)
    try:
        scorer, torch, device = _load_syncnet(device)
    except Exception as exc:
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "incomplete", "engineering_status": "INPUT_UNAVAILABLE", "failures": [{"stage": "score", "error": str(exc)}], "rows": []}
        write_json(paths.score_manifest, payload, self_hash=True)
        return verify_json(paths.score_manifest, self_hash=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for record in score_records:
        key = str(record["paired_key"])
        for receiver in ("N", "T"):
            arms = ["RAW", "ID", "LEVEL", "CONTOUR"]
            videos: dict[str, np.ndarray] = {}
            audios: dict[str, np.ndarray] = {}
            try:
                for arm in arms:
                    video_row = video_index.get((key, f"{receiver}_{arm}"))
                    audio_row = audio_index.get((key, f"{receiver}_{arm}"))
                    if video_row is None or audio_row is None:
                        raise ProtocolError(f"missing video/audio arm {key}/{receiver}_{arm}")
                    vpath, apath = Path(str(video_row["output"])), Path(str(audio_row["path"]))
                    feature_dir = paths.features_dir / key
                    vfeature = feature_dir / f"V_{receiver}_{arm}.npy"
                    afeature = feature_dir / f"A_{receiver}_{arm}.npy"
                    cached_row = cached_feature_rows.get((key, receiver, arm), {})
                    video_cache_ok = feature_cache_valid and vfeature.is_file() and cached_row.get("visual_sha256") == file_sha256(vfeature)
                    audio_cache_ok = feature_cache_valid and afeature.is_file() and cached_row.get("audio_sha256") == file_sha256(afeature)
                    if video_cache_ok:
                        vembed = np.asarray(np.load(vfeature, allow_pickle=False), dtype=np.float32)
                    else:
                        cropped, _ = _crop_score_video(vpath, inputs["portrait"]["box_xyxy"])
                        vembed = extract_visual_embedding(cropped, scorer, torch, device=device)
                        vfeature.parent.mkdir(parents=True, exist_ok=True); np.save(vfeature, vembed, allow_pickle=False)
                    if audio_cache_ok:
                        aembed = np.asarray(np.load(afeature, allow_pickle=False), dtype=np.float32)
                    else:
                        aembed, _audio_samples = extract_audio_embedding(apath, scorer, torch, device=device)
                        afeature.parent.mkdir(parents=True, exist_ok=True); np.save(afeature, aembed, allow_pickle=False)
                    if vembed.ndim != 2 or aembed.ndim != 2 or vembed.shape[1] != 1024 or aembed.shape[1] != 1024 or not np.isfinite(vembed).all() or not np.isfinite(aembed).all():
                        raise ProtocolError(f"invalid cached features for {key}/{receiver}_{arm}")
                    videos[arm], audios[arm] = vembed, aembed
                    feature_rows.append({"paired_key": key, "receiver": receiver, "arm": arm, "visual": str(vfeature.resolve()), "visual_sha256": file_sha256(vfeature), "visual_count": int(len(vembed)), "audio": str(afeature.resolve()), "audio_sha256": file_sha256(afeature), "audio_count": int(len(aembed)), "model_sha256": file_sha256(SYNCNET_MODEL), "device": device})
                support = common_window_indices(videos, audios)
                if support.size < MIN_SCORE_WINDOWS:
                    raise ProtocolError(f"common SyncNet support {support.size} < {MIN_SCORE_WINDOWS}")
                matrices: dict[tuple[str, str], np.ndarray] = {}
                for var in arms:
                    for aar in arms:
                        matrix = syncnet_distance_matrix(videos[var], audios[aar])
                        matrices[(var, aar)] = matrix
                        matrix_path = paths.distances_dir / key / f"{receiver}_{var}__{receiver}_{aar}.npy"
                        matrix_path.parent.mkdir(parents=True, exist_ok=True); np.save(matrix_path, matrix, allow_pickle=False)
                id_id = curve_metrics(matrices[("ID", "ID")], support)
                k0 = int(id_id["k_star"])
                for category, suffixes in CELL_SUFFIXES.items():
                    for var, aar in suffixes:
                        metrics = curve_metrics(matrices[(var, aar)], support, k0=k0)
                        matrix_path = paths.distances_dir / key / f"{receiver}_{var}__{receiver}_{aar}.npy"
                        rows.append({"schema_version": 1, "status": "complete", "paired_key": key, "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "receiver": receiver, "category": category, "video_arm": f"{receiver}_{var}", "audio_arm": f"{receiver}_{aar}", "matrix": str(matrix_path.resolve()), "matrix_sha256": file_sha256(matrix_path), "id_id_k0": k0, "fixed_audio_track": f"{receiver}_ID" if category == "fixed_audio" else None, "fixed_video_arm": f"{receiver}_ID" if category in {"fixed_video", "raw_replacement"} else None, "device": device, "syncnet_model_sha256": file_sha256(SYNCNET_MODEL), **metrics})
                print(f"SCORE {key} {receiver} cells=10 support={support.size}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append({"paired_key": key, "receiver": receiver, "stage": "score", "error": str(exc)})
    expected = len(score_records) * 2 * 10
    # A short utterance can have fewer than the preregistered 50 common
    # SyncNet windows even after audio and rendering pass.  That is a declared
    # scientific-support exclusion, not a scorer crash: preserve its failure
    # ledger and let analysis report INSUFFICIENT_SUPPORT.  Any other runtime
    # failure keeps the stage incomplete so it cannot be mistaken for a valid
    # experiment.
    support_only_failures = bool(failures) and all("common SyncNet support" in str(item.get("error", "")) for item in failures)
    excluded_support_only = bool(excluded_pairs) and all(
        all(
            str(reason).endswith("_AUDIO_QC_NOT_PASS")
            or str(reason).startswith("MISSING_AUDIO_")
            for reason in item.get("reasons", [])
        )
        for item in excluded_pairs
    )
    support_limited = support_only_failures or excluded_support_only
    stage_complete = bool(score_records or support_limited) and (not failures or support_only_failures)
    status = "complete" if stage_complete else "incomplete"
    engineering_status = (
        "PASS"
        if status == "complete" and len(rows) == expected and not excluded_pairs and not failures
        else "INSUFFICIENT_SUPPORT"
        if status == "complete" and support_limited
        else "INCOMPLETE"
    )
    feature_payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "complete" if feature_rows else "incomplete", "rows": feature_rows, "model_sha256": file_sha256(SYNCNET_MODEL), "device": device}
    write_json(paths.features_manifest, feature_payload, self_hash=True)
    # Only CSV display fields are rounded.  ``curve`` and matrix files retain
    # full precision for recomputation.
    csv_rows = []
    for row in rows:
        copy = dict(row)
        for field in ("C", "D", "B", "d_k0"):
            if copy.get(field) is not None:
                copy[field] = round(float(copy[field]), 3)
        csv_rows.append(copy)
    paths.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in csv_rows for key in row})
    with paths.scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(csv_rows)
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": status, "engineering_status": engineering_status, "support_limited": support_limited, "formal_record_count": len(formal_records), "scored_record_count": len(score_records), "excluded_pairs": excluded_pairs, "expected_rows": expected, "row_count": len(rows), "feature_rows": feature_rows, "features_manifest_sha256": file_sha256(paths.features_manifest), "failures": failures, "rows": rows, "score_csv_sha256": file_sha256(paths.scores_csv) if paths.scores_csv.is_file() else None, "syncnet_model_sha256": file_sha256(SYNCNET_MODEL)}
    write_json(paths.score_manifest, payload, self_hash=True)
    return verify_json(paths.score_manifest, self_hash=True)


def delayed_audio(values: np.ndarray, delay_samples: int = DELAY_SAMPLES) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= abs(int(delay_samples)):
        raise ProtocolError("audio is too short for the fixed delay control")
    delay = int(delay_samples)
    if delay > 0:
        return np.concatenate((np.zeros(delay, dtype=np.float64), values[:-delay]))
    if delay < 0:
        advance = -delay
        return np.concatenate((values[advance:], np.zeros(advance, dtype=np.float64)))
    return values.copy()


def run_controls(paths: RunPaths, inputs: Mapping[str, Any], audio_manifest: Mapping[str, Any], videos_manifest: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    records = [row for row in inputs.get("records", []) if row.get("role") == "formal"]
    audio_index = _audio_arm_index(audio_manifest)
    video_index = {(str(row.get("paired_key")), str(row.get("video_arm"))): row for row in videos_manifest.get("videos", [])}
    qc_by_key_receiver: dict[tuple[str, str], Mapping[str, Any]] = {}
    if paths.audio_qc.is_file():
        with paths.audio_qc.open(encoding="utf-8", newline="") as handle:
            qc_by_key_receiver = {(str(row.get("paired_key")), str(row.get("receiver"))): row for row in csv.DictReader(handle)}
    first = next(
        (
            candidate
            for candidate in records
            if all(str(qc_by_key_receiver.get((str(candidate["paired_key"]), receiver), {}).get("status")) == "PASS" for receiver in ("N", "T"))
            and all((str(candidate["paired_key"]), f"{receiver}_{arm}") in audio_index for receiver in ("N", "T") for arm in ("RAW", "ID", "LEVEL", "CONTOUR"))
            and all((str(candidate["paired_key"]), f"{receiver}_ID") in video_index for receiver in ("N", "T"))
        ),
        None,
    )
    if not records:
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "NOT_RUN", "reason": "no formal records"}
        write_json(paths.root / "controls.json", payload, self_hash=True)
        return verify_json(paths.root / "controls.json", self_hash=True)
    if first is None:
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "NOT_RUN", "reason": "no scoreable formal pair"}
        write_json(paths.root / "controls.json", payload, self_hash=True)
        return verify_json(paths.root / "controls.json", self_hash=True)
    record = dict(first); record["_portrait"] = inputs.get("portrait", {})
    controls: list[dict[str, Any]] = []
    try:
        scorer, torch, device = _load_syncnet(device)
        duplicate_visuals: dict[str, np.ndarray] = {}
        for receiver in ("N", "T"):
            arm = f"{receiver}_ID"
            output = paths.controls_dir / f"{arm}_repeat.mkv"
            duplicate = _render_one(paths, record, audio_index, arm, device=device, force=True, output_override=output)
            cropped, _ = _crop_score_video(output, inputs["portrait"]["box_xyxy"])
            duplicate_visuals[receiver] = extract_visual_embedding(cropped, scorer, torch, device=device)
            original_video = video_index.get((str(first["paired_key"]), arm))
            if original_video is None:
                raise ProtocolError(f"control baseline video missing: {arm}")
            original_cropped, _ = _crop_score_video(Path(str(original_video["output"])), inputs["portrait"]["box_xyxy"])
            original_visual = extract_visual_embedding(original_cropped, scorer, torch, device=device)
            audio_path = Path(str(audio_index[(str(first["paired_key"]), arm)]["path"]))
            audio_embed, _ = extract_audio_embedding(audio_path, scorer, torch, device=device)
            support = common_window_indices({"original": original_visual, "duplicate": duplicate_visuals[receiver]}, {"audio": audio_embed})
            if support.size < MIN_SCORE_WINDOWS:
                raise ProtocolError(f"control support is too short: {support.size}")
            curve_original = curve_metrics(syncnet_distance_matrix(original_visual, audio_embed), support)
            curve_duplicate = curve_metrics(syncnet_distance_matrix(duplicate_visuals[receiver], audio_embed), support)
            max_diff = float(np.max(np.abs(np.asarray(curve_original["curve"]) - np.asarray(curve_duplicate["curve"]))))
            controls.append({"name": f"{arm}_DUPLICATE", "status": "PASS" if max_diff <= REPEAT_CURVE_BOUND else "FAIL", "curve_max_abs_error": max_diff, "original": curve_original, "duplicate": curve_duplicate, "support_count": int(support.size), "video": duplicate})
            if abs(int(curve_original["k_star"])) == VSHIFT:
                controls.append({"name": f"{arm}_DELAY_200MS", "status": "UNINTERPRETABLE_EDGE", "reason": "baseline lag is on search boundary"})
                continue
            original_values, _ = decode_audio(audio_path)
            for sign, expected_delta, label in ((1, DELAY_FRAMES, "plus"), (-1, -DELAY_FRAMES, "minus")):
                signed_delay = sign * DELAY_SAMPLES
                delay_values = delayed_audio(original_values, signed_delay)
                delay_path = paths.controls_dir / f"{arm}_delay_{label}_200ms.wav"
                _materialize_arm(delay_path, delay_values)
                delay_embed, _ = extract_audio_embedding(delay_path, scorer, torch, device=device)
                delayed_matrix = syncnet_distance_matrix(original_visual, delay_embed)
                delayed_metrics = curve_metrics(delayed_matrix, support)
                expected = int(curve_original["k_star"]) + expected_delta
                status = "PASS" if abs(int(delayed_metrics["k_star"]) - expected) <= 1 else "FAIL"
                controls.append({"name": f"{arm}_DELAY_{label.upper()}_200MS", "status": status, "delay_samples": signed_delay, "delay_frames": expected_delta, "baseline_k_star": curve_original["k_star"], "delayed_k_star": delayed_metrics["k_star"], "expected_delayed_k_star": expected, "support_count": int(support.size), "audio": str(delay_path.resolve())})
    except Exception as exc:  # noqa: BLE001
        controls.append({"name": "CONTROL_ERROR", "status": "FAIL", "error": str(exc)})
    status = "PASS" if controls and all(item.get("status") == "PASS" for item in controls) else "CONTROL_FAIL"
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": status, "controls": controls, "paired_key": first.get("paired_key"), "seed": SEED}
    write_json(paths.root / "controls.json", payload, self_hash=True)
    return verify_json(paths.root / "controls.json", self_hash=True)


def _bootstrap_indices(group_count: int, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    if group_count < 1:
        raise ProtocolError("bootstrap requires at least one speaker")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, group_count, size=(draws, group_count), dtype=np.int64)


def _quantile(values: np.ndarray, p: float) -> float:
    try:
        return float(np.quantile(values, p, method="linear"))
    except TypeError:  # pragma: no cover - old NumPy
        return float(np.quantile(values, p, interpolation="linear"))


def speaker_bootstrap(values: Sequence[float], speakers: Sequence[str], *, indices: np.ndarray | None = None) -> dict[str, Any]:
    if len(values) != len(speakers) or len(values) == 0:
        return {"status": "NOT_ESTIMABLE", "reason": "empty or mismatched values"}
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for value, speaker in zip(values, speakers, strict=True):
        value = float(value)
        if not math.isfinite(value):
            raise ProtocolError("non-finite statistic in speaker bootstrap")
        by_speaker[str(speaker)].append(value)
    labels = sorted(by_speaker)
    group_values = np.asarray([np.mean(by_speaker[label], dtype=np.float64) for label in labels], dtype=np.float64)
    if indices is None:
        indices = _bootstrap_indices(len(labels))
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != len(labels) or np.any(indices < 0) or np.any(indices >= len(labels)):
        raise ProtocolError("speaker bootstrap index matrix is invalid")
    samples = group_values[indices].mean(axis=1, dtype=np.float64)
    lower_bc, upper_bc = _quantile(samples, BONFERRONI_ALPHA / 2.0), _quantile(samples, 1.0 - BONFERRONI_ALPHA / 2.0)
    return {"status": "COMPLETE", "mean": float(group_values.mean()), "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)], "ci98_75_bonferroni": [lower_bc, upper_bc], "speaker_count": len(labels), "speaker_labels": labels, "speaker_values": {label: float(value) for label, value in zip(labels, group_values, strict=True)}, "pair_count": len(values), "positive_speaker_count": int(np.sum(group_values > 0.0)), "draws": int(indices.shape[0]), "seed": BOOTSTRAP_SEED, "rng": "numpy.random.Generator(PCG64)", "quantile_method": "linear", "status_by_corrected_ci": "POSITIVE" if lower_bc > 0.0 else "NEGATIVE" if upper_bc < 0.0 else "INCONCLUSIVE"}


def _score_lookup(score_manifest: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = score_manifest.get("rows", [])
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("paired_key")), str(row.get("receiver")), str(row.get("category")))
        if key in result:
            raise ProtocolError(f"duplicate score row: {key}")
        result[key] = row
    return result


def _metric(lookup: Mapping[tuple[str, str, str], Mapping[str, Any]], key: str, receiver: str, category: str) -> float:
    row = lookup.get((key, receiver, category))
    if row is None or not math.isfinite(float(row.get("C", float("nan")))):
        raise ProtocolError(f"missing score metric: {key}/{receiver}/{category}")
    return float(row["C"])


def analyze_scores(paths: RunPaths, inputs: Mapping[str, Any], score_manifest: Mapping[str, Any], audio_manifest: Mapping[str, Any], *, controls: Mapping[str, Any] | None = None) -> dict[str, Any]:
    formal = [row for row in inputs.get("records", []) if row.get("role") == "formal"]
    score_rows: list[dict[str, Any]] = []
    for record in formal:
        key, speaker = str(record["paired_key"]), str(record.get("speaker_id", ""))
        try:
            # ``own`` has four rows; use explicit arm lookup to avoid relying
            # on CSV row order.
            own = [row for row in score_manifest.get("rows", []) if str(row.get("paired_key")) == key and str(row.get("receiver")) == "N" and str(row.get("category")) == "own"]
            own_t = [row for row in score_manifest.get("rows", []) if str(row.get("paired_key")) == key and str(row.get("receiver")) == "T" and str(row.get("category")) == "own"]
            by_arms_n = {str(row.get("video_arm", "")).split("_", 1)[1]: float(row["C"]) for row in own}
            by_arms_t = {str(row.get("video_arm", "")).split("_", 1)[1]: float(row["C"]) for row in own_t}
            n_raw, n_id, n_level, n_contour = [by_arms_n[arm] for arm in ("RAW", "ID", "LEVEL", "CONTOUR")]
            t_raw, t_id, t_level, t_contour = [by_arms_t[arm] for arm in ("RAW", "ID", "LEVEL", "CONTOUR")]
            # Fixed-audio and fixed-video categories each contain two cells,
            # indexed by their video/audio suffix.
            fixed_audio_n = {str(row["video_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "N" and row.get("category") == "fixed_audio"}
            fixed_audio_t = {str(row["video_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "T" and row.get("category") == "fixed_audio"}
            fixed_video_n = {str(row["audio_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "N" and row.get("category") == "fixed_video"}
            fixed_video_t = {str(row["audio_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "T" and row.get("category") == "fixed_video"}
            # In raw_replacement, the video arm distinguishes ID vs
            # CONTOUR while the audio arm is RAW for both cells.  Indexing by
            # audio_arm collapses the two rows and makes every otherwise
            # complete record look as if CONTOUR were missing.
            raw_replace_n = {str(row["video_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "N" and row.get("category") == "raw_replacement"}
            raw_replace_t = {str(row["video_arm"]).split("_", 1)[1]: float(row["C"]) for row in score_manifest["rows"] if str(row.get("paired_key")) == key and row.get("receiver") == "T" and row.get("category") == "raw_replacement"}
            values = {"g_N": fixed_audio_n["CONTOUR"] - n_id, "l_T": t_id - fixed_audio_t["CONTOUR"], "h_N": fixed_audio_n["CONTOUR"] - fixed_audio_n["LEVEL"], "h_T": fixed_audio_t["LEVEL"] - fixed_audio_t["CONTOUR"], "G_raw": t_raw - n_raw, "G_id": t_id - n_id, "own_gain_N": n_contour - n_id, "own_gain_T": t_contour - t_id, "evaluator_N": fixed_video_n["CONTOUR"] - n_id, "evaluator_T": fixed_video_t["CONTOUR"] - t_id, "raw_replace_N": raw_replace_n["CONTOUR"] - raw_replace_n["ID"], "raw_replace_T": raw_replace_t["CONTOUR"] - raw_replace_t["ID"], "interaction_N": n_contour - fixed_audio_n["CONTOUR"] - fixed_video_n["CONTOUR"] + n_id, "interaction_T": t_contour - fixed_audio_t["CONTOUR"] - fixed_video_t["CONTOUR"] + t_id}
            score_rows.append({"paired_key": key, "speaker_id": speaker, **values})
        except Exception as exc:  # noqa: BLE001
            score_rows.append({"paired_key": key, "speaker_id": speaker, "missing_reason": str(exc)})
    complete = [row for row in score_rows if "g_N" in row]
    by_speaker_count = defaultdict(int)
    for row in complete:
        by_speaker_count[str(row["speaker_id"])] += 1
    main = [row for row in complete if by_speaker_count[str(row["speaker_id"])] >= 2]
    included_speakers = sorted({str(row["speaker_id"]) for row in main})
    indices = _bootstrap_indices(len(included_speakers)) if included_speakers else np.empty((0, 0), dtype=np.int64)
    if included_speakers:
        paths.bootstrap_indices.parent.mkdir(parents=True, exist_ok=True); np.save(paths.bootstrap_indices, indices, allow_pickle=False)
    stats: dict[str, Any] = {}
    for metric in ("g_N", "l_T", "h_N", "h_T", "G_raw", "G_id", "G_id_minus_G_raw", "own_gain_N", "own_gain_T", "evaluator_N", "evaluator_T", "raw_replace_N", "raw_replace_T", "interaction_N", "interaction_T"):
        if metric == "G_id_minus_G_raw":
            values = [float(row["G_id"] - row["G_raw"]) for row in main]
        else:
            values = [float(row[metric]) for row in main if metric in row]
        speakers = [str(row["speaker_id"]) for row in main if metric == "G_id_minus_G_raw" or metric in row]
        stats[metric] = speaker_bootstrap(values, speakers, indices=indices if included_speakers else None)
    bg_gate = bool(stats["G_raw"].get("status") == "COMPLETE" and stats["G_id"].get("status") == "COMPLETE" and stats["G_raw"]["ci95"][0] > 0.0 and stats["G_id"]["ci95"][0] > 0.0)
    def corrected_positive(metric: str) -> bool:
        return stats[metric].get("status") == "COMPLETE" and stats[metric]["ci98_75_bonferroni"][0] > 0.0
    if len(main) < MIN_FORMAL_PAIRS or len(included_speakers) < MIN_FORMAL_SPEAKERS:
        scientific_status = "INSUFFICIENT_SUPPORT"
    elif corrected_positive("g_N") and corrected_positive("l_T"):
        scientific_status = "BIDIRECTIONAL_GENERATOR_SCORE_EFFECT"
    elif corrected_positive("g_N"):
        scientific_status = "NATURAL_DIRECTION_ONLY"
    elif corrected_positive("l_T"):
        scientific_status = "TTS_DIRECTION_ONLY"
    else:
        scientific_status = "INCONCLUSIVE"
    if scientific_status == "BIDIRECTIONAL_GENERATOR_SCORE_EFFECT" and bg_gate:
        native_status = "NATIVE_ADVANTAGE_RETAINED"
    else:
        native_status = "NATIVE_ADVANTAGE_UNCONFIRMED"
    if corrected_positive("h_N") and corrected_positive("h_T"):
        scientific_status += "+CONTOUR_OVER_LEVEL"
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "complete", "engineering_status": "PASS" if score_manifest.get("status") == "complete" else "INCOMPLETE", "manipulation_status": "FROM_AUDIO_QC", "native_status": native_status, "scientific_status": scientific_status, "record_count": len(score_rows), "complete_pair_count": len(complete), "main_pair_count": len(main), "included_speakers": included_speakers, "background_gate_pass": bg_gate, "statistics": stats, "records": score_rows, "controls": controls or {"status": "NOT_RUN"}, "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "speaker cluster; equal speaker weighting", "indices": str(paths.bootstrap_indices.resolve()) if paths.bootstrap_indices.is_file() else None, "indices_sha256": file_sha256(paths.bootstrap_indices) if paths.bootstrap_indices.is_file() else None}}
    write_json(paths.analysis, payload, self_hash=True)
    return verify_json(paths.analysis, self_hash=True)


def pilot_gate(pilot_manifest: Mapping[str, Any], *, expected_keys: Sequence[str] | None = None) -> dict[str, Any]:
    pair_status: dict[str, bool] = {}
    for result in pilot_manifest.get("manifests", []):
        key = str(result.get("paired_key"))
        rows = {str(row.get("receiver")): row for row in result.get("qc", [])}
        pair_status[key] = bool(rows.get("N", {}).get("status") == "PASS" and rows.get("T", {}).get("status") == "PASS")
    frozen_keys = [str(key) for key in expected_keys] if expected_keys is not None else [str(key) for key in pilot_manifest.get("expected_keys", [])]
    if not frozen_keys:
        expected_count = int(pilot_manifest.get("expected_pair_count", PILOT_COUNT))
        frozen_keys = list(pair_status)[:expected_count]
    for key in frozen_keys:
        pair_status.setdefault(key, False)
    passed = int(sum(pair_status.values()))
    required = max(0, len(frozen_keys) - 1)
    status = "PASS" if passed >= required else "MANIPULATION_NOT_VALIDATED"
    return {"status": status, "pair_status": pair_status, "passed_pairs": passed, "required_pairs": required, "pilot_pair_count": len(frozen_keys), "decision": "formal_audio_allowed" if status == "PASS" else "formal_audio_blocked"}


def validate_run(paths: RunPaths) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not paths.inputs.is_file():
        errors.append("inputs.json missing")
        payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "FAIL", "errors": errors, "warnings": warnings}
        write_json(paths.validation, payload, self_hash=True)
        return verify_json(paths.validation, self_hash=True)
    try:
        inputs = verify_json(paths.inputs, self_hash=True)
        if inputs.get("protocol_id") != PROTOCOL_ID:
            errors.append("protocol id mismatch")
        pilot_keys, formal_keys = set(inputs.get("pilot_keys", [])), set(inputs.get("formal_keys", []))
        if pilot_keys & formal_keys:
            errors.append("pilot/formal overlap")
        if len(inputs.get("formal_speakers", [])) < MIN_FORMAL_SPEAKERS:
            warnings.append("formal speaker support is below six")
        if paths.pilot_qc.is_file():
            pilot = verify_json(paths.pilot_qc, self_hash=True)
            if not _fingerprint_matches(pilot):
                errors.append("pilot QC fingerprint is stale")
            gate = pilot_gate(pilot, expected_keys=[str(row["paired_key"]) for row in inputs.get("records", []) if row.get("role") == "pilot"])
            if pilot.get("gate", {}).get("status") != gate.get("status"):
                errors.append("pilot gate does not match frozen QC")
        if paths.pilot_quality.is_file():
            verify_json(paths.pilot_quality, self_hash=True)
        if (paths.root / "audio_manifest.json").is_file():
            audio = _load_audio_manifest(paths)
            if not _fingerprint_matches(audio):
                errors.append("audio manifest fingerprint is stale")
            for result in audio.get("manifests", []):
                for receiver in ("N", "T"):
                    for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                        row = result.get("audio", {}).get(receiver, {}).get(arm)
                        if not isinstance(row, Mapping) or not Path(str(row.get("path", ""))).is_file():
                            errors.append(f"missing audio arm {result.get('paired_key')}/{receiver}_{arm}")
        if paths.score_manifest.is_file():
            score = verify_json(paths.score_manifest, self_hash=True)
            if not _fingerprint_matches(score):
                errors.append("score manifest fingerprint is stale")
            for row in score.get("rows", []):
                matrix = Path(str(row.get("matrix", "")))
                if not matrix.is_file() or file_sha256(matrix) != row.get("matrix_sha256"):
                    errors.append(f"score matrix hash mismatch {row.get('paired_key')}/{row.get('receiver')}")
        if paths.analysis.is_file():
            analysis = verify_json(paths.analysis, self_hash=True)
            if not _fingerprint_matches(analysis):
                errors.append("analysis fingerprint is stale")
            if analysis.get("status") != "complete":
                errors.append("analysis status is not complete")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"validation exception: {type(exc).__name__}: {exc}")
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": warnings, "checked_at": datetime.now(timezone.utc).isoformat()}
    write_json(paths.validation, payload, self_hash=True)
    return verify_json(paths.validation, self_hash=True)


def write_report(paths: RunPaths) -> dict[str, Any]:
    lines = [f"# {PROTOCOL_ID}", "", "本报告由冻结的 F0 双向交换协议自动生成。"]
    inputs = verify_json(paths.inputs, self_hash=True) if paths.inputs.is_file() else {}
    lines.extend(["", f"- 候选 pairs：{inputs.get('candidate_count', '未审计')}", f"- Pilot：{inputs.get('pilot_pair_count', 0)}；正式：{inputs.get('formal_pair_count', 0)}", f"- 形式支持：{inputs.get('support_status', '未知')}"])
    if paths.pilot_qc.is_file():
        pilot = verify_json(paths.pilot_qc, self_hash=True); gate = pilot_gate(pilot); lines.extend([f"- Pilot 数值门：{gate['status']}（{gate['passed_pairs']}/{gate['pilot_pair_count']} 对通过）", f"- 听检：{pilot.get('quality_status', 'QUALITY_NOT_ASSESSED')}"])
    if (paths.root / "audio_manifest.json").is_file():
        audio = _load_audio_manifest(paths); lines.append(f"- 音频阶段：{audio.get('status')}，完成 {audio.get('pair_count', 0)}/{audio.get('expected_pair_count', 0)} 对")
    if paths.videos_manifest.is_file():
        videos = verify_json(paths.videos_manifest, self_hash=True); lines.append(f"- 视频阶段：{videos.get('engineering_status', videos.get('status'))}，{videos.get('video_count', 0)} 个视频")
    if paths.score_manifest.is_file():
        scores = verify_json(paths.score_manifest, self_hash=True); lines.append(f"- 评分阶段：{scores.get('engineering_status', scores.get('status'))}，{scores.get('row_count', 0)}/{scores.get('expected_rows', 0)} cells")
    if paths.analysis.is_file():
        analysis = verify_json(paths.analysis, self_hash=True)
        lines.extend(["", "## 判读", "", f"- native_status：{analysis.get('native_status')}", f"- scientific_status：{analysis.get('scientific_status')}", f"- 完整 pair：{analysis.get('main_pair_count', 0)}；speaker：{len(analysis.get('included_speakers', []))}"])
        for name in ("g_N", "l_T", "h_N", "h_T", "G_raw", "G_id"):
            stat = analysis.get("statistics", {}).get(name, {})
            if stat.get("status") == "COMPLETE":
                lines.append(f"- {name}: mean={stat['mean']:+.4f}, 95% CI=[{stat['ci95'][0]:+.4f}, {stat['ci95'][1]:+.4f}], Bonferroni 98.75%=[{stat['ci98_75_bonferroni'][0]:+.4f}, {stat['ci98_75_bonferroni'][1]:+.4f}]")
    if paths.validation.is_file():
        validation = verify_json(paths.validation, self_hash=True); lines.extend(["", f"- validation：{validation.get('status')}", f"- errors：{len(validation.get('errors', []))}"])
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(paths.report.resolve()), "sha256": file_sha256(paths.report), "status": "complete"}


def _paths(run_id: str) -> RunPaths:
    root = run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root)


def _current_manifest(path: Path, *, label: str) -> dict[str, Any]:
    payload = verify_json(path, self_hash=True)
    if not _fingerprint_matches(payload):
        raise ProtocolError(f"stale {label} after runner/spec change: {path}")
    return payload


def run_stage(run_id: str, stage: str, *, device: str = "cuda", manifest_path: Path = MANIFEST, smoke: bool = False) -> dict[str, Any]:
    paths = _paths(run_id)
    if stage == "audit":
        inputs = audit_and_freeze(paths, manifest_path=manifest_path, smoke=smoke)
        protocol = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_revision": PROTOCOL_REVISION, "status": "complete", "inputs_sha256": file_sha256(paths.inputs), "spec": str(SPEC_PATH.resolve()), "spec_sha256": file_sha256(SPEC_PATH) if SPEC_PATH.is_file() else None, "runner": str(Path(__file__).resolve()), "runner_sha256": file_sha256(Path(__file__)), "runtime": {"python": platform.python_version(), "git": _git_commit()}}
        write_json(paths.protocol, protocol, self_hash=True)
        return inputs
    if not paths.inputs.is_file():
        audit_and_freeze(paths, manifest_path=manifest_path, smoke=smoke)
    inputs = verify_json(paths.inputs, self_hash=True)
    if stage == "pilot":
        pilot = run_audio_stage(paths, inputs, pilot_only=True)
        gate = pilot_gate(pilot, expected_keys=[str(row["paired_key"]) for row in inputs.get("records", []) if row.get("role") == "pilot"])
        pilot["gate"] = gate
        write_json(paths.pilot_qc, pilot, self_hash=True)
        return verify_json(paths.pilot_qc, self_hash=True)
    if stage == "audio":
        if not paths.pilot_qc.is_file():
            run_stage(run_id, "pilot", device=device, manifest_path=manifest_path, smoke=smoke)
        pilot = verify_json(paths.pilot_qc, self_hash=True)
        if not _fingerprint_matches(pilot):
            # Formula/spec changes invalidate pilot QC; regenerate it before
            # deciding whether formal audio is allowed.
            pilot = run_stage(run_id, "pilot", device=device, manifest_path=manifest_path, smoke=smoke)
        gate = pilot.get("gate") or pilot_gate(pilot)
        if gate.get("status") != "PASS":
            payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, **_implementation_fingerprint(), "status": "blocked", "engineering_status": "MANIPULATION_NOT_VALIDATED", "expected_pair_count": len([r for r in inputs.get("records", []) if r.get("role") == "formal"]), "pair_count": 0, "failures": [{"stage": "audio", "status": "MANIPULATION_NOT_VALIDATED", "gate": gate},], "manifests": []}
            write_json(paths.root / "audio_manifest.json", payload, self_hash=True)
            return verify_json(paths.root / "audio_manifest.json", self_hash=True)
        return run_audio_stage(paths, inputs, pilot_only=False)
    if stage == "render":
        audio = _current_manifest(paths.root / "audio_manifest.json", label="audio manifest")
        return run_render_stage(paths, inputs, audio, device=device)
    if stage == "score":
        videos = _current_manifest(paths.videos_manifest, label="video manifest")
        audio = _current_manifest(paths.root / "audio_manifest.json", label="audio manifest")
        controls = run_controls(paths, inputs, audio, videos, device=device)
        result = run_score_stage(paths, inputs, videos, device=device)
        result["controls"] = controls
        write_json(paths.score_manifest, result, self_hash=True)
        return verify_json(paths.score_manifest, self_hash=True)
    if stage == "analyze":
        score = _current_manifest(paths.score_manifest, label="score manifest")
        analysis = analyze_scores(paths, inputs, score, _load_audio_manifest(paths), controls=verify_json(paths.root / "controls.json", self_hash=True) if (paths.root / "controls.json").is_file() else None)
        return analysis
    if stage == "validate":
        return validate_run(paths)
    if stage == "report":
        return write_report(paths)
    raise ValueError(f"unknown stage: {stage}")


def run_all(run_id: str, *, device: str = "cuda", manifest_path: Path = MANIFEST, smoke: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("audit", "pilot", "audio", "render", "score", "analyze", "validate", "report"):
        try:
            result[stage] = run_stage(run_id, stage, device=device, manifest_path=manifest_path, smoke=smoke)
        except Exception as exc:  # noqa: BLE001 - always leave a readable report
            result[stage] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            paths = _paths(run_id)
            write_json(paths.root / f"{stage}_error.json", result[stage], self_hash=True)
            if stage in {"audit", "pilot", "audio"}:
                break
    try:
        result.setdefault("validate", run_stage(run_id, "validate", device=device, manifest_path=manifest_path, smoke=smoke))
        result.setdefault("report", run_stage(run_id, "report", device=device, manifest_path=manifest_path, smoke=smoke))
    except Exception as exc:  # pragma: no cover
        result["report"] = {"status": "error", "error": str(exc)}
    return result


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=False, timeout=10)
        return completed.stdout.strip() or None
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTS F0 contour exchange through local Wav2Lip/SyncNet")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("audit", "pilot", "audio", "render", "score", "analyze", "validate", "report", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run_all(args.run_id, device=args.device, manifest_path=args.manifest, smoke=args.smoke) if args.stage == "all" else run_stage(args.run_id, args.stage, device=args.device, manifest_path=args.manifest, smoke=args.smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.stage in {"validate"}:
        return 0 if result.get("status") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
