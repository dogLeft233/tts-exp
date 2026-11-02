"""CPU-only Fit40 snapshot selection and paired SPSA ask/tell primitives.

This module consumes already-produced official Stage-A SyncNet arrays.  It does
not render, score, train, call an API, or infer source-clock keys from audio.
The caller must bind every row and array to the frozen producer artifacts
before constructing :class:`FitScoreRow` values.

Each official array is ``[query, lag]`` with 31 lag columns and the producer's
query frame, rational timestamp, track frame, and valid-query mask.  A family
comparison computes one common valid-key intersection across every method in
that family.  Incomplete families are never ranked or used to seed SPSA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "fit40_cross_tfg_spsa_selection_v1"
RENDERER_SEED = 20261003
SPSA_RNG_SEED = 20261009
LAG_OFFSETS = tuple(range(-15, 16))
ZERO_LAG_COLUMN = 15
TAU = 0.050
EQUAL_MODEL_CBEST_RMS_MIN = 0.050
# Numerical conformance only: a prior independent Stage-A NPZ readback found
# at most 6.68e-6 absolute NumPy-vs-Torch curve difference. This tolerance is
# not applied to admission gates and does not imply bitwise equality.
METRIC_PARITY_ATOL = 1.0e-5
METRIC_PARITY_RTOL = 0.0
METRIC_BACKENDS = ("torch_cpu", "numpy_diagnostic")
RMS_REFERENCE_METHOD_IDS = frozenset({"rms", "rms_fixed"})
THETA_DIMENSION = 9
SPSA_ITERATIONS = 20
SPSA_BATCH_SPEAKERS = 4
SPSA_CLIPS_PER_SPEAKER = 2

QueryKey = tuple[int, int, int, int]
ContextKey = tuple[str, str, str]  # sample_id, speaker_id, model_id


class FitSelectionError(ValueError):
    """Raised for malformed input contracts, never for a scientific result."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical_input_pins_sha256(pins: Mapping[str, Mapping[str, Any]]) -> str:
    """Validate and hash explicit byte-pinned inputs; no path is opened here."""
    if not isinstance(pins, Mapping) or not pins:
        raise FitSelectionError("input pins must be a non-empty mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for name, pin in sorted(pins.items()):
        if not isinstance(name, str) or not name or not isinstance(pin, Mapping):
            raise FitSelectionError("input pin entries require a non-empty name and mapping")
        path = pin.get("path")
        size = pin.get("bytes")
        digest = pin.get("sha256")
        if not isinstance(path, str) or not path:
            raise FitSelectionError(f"input pin {name} has no path")
        if type(size) is not int or size < 0 or not _is_sha256(digest):
            raise FitSelectionError(f"input pin {name} has invalid byte count or SHA-256")
        entry: dict[str, Any] = {"path": path, "bytes": size, "sha256": digest}
        self_sha = pin.get("self_sha256")
        if self_sha is not None:
            if not _is_sha256(self_sha):
                raise FitSelectionError(f"input pin {name} has invalid self SHA-256")
            entry["self_sha256"] = self_sha
        normalized[name] = entry
    return _sha256(_canonical(normalized))


def _normalize_query_key(frame: Any, num: Any, den: Any, track: Any) -> QueryKey:
    if any(type(v) is not int for v in (frame, num, den, track)) or den <= 0:
        raise FitSelectionError("official query key fields must be integers with positive denominator")
    stamp = Fraction(num, den)
    return (frame, stamp.numerator, stamp.denominator, track)


def full_fit_candidate_binding_sha256(
    *, method_id: str, artifact_sha256: str, input_pins_sha256: str,
    source_step: int, theta: Sequence[float], point_kind: str,
) -> str:
    """Canonical binding for one declared full-Fit selection point."""
    values = _validate_theta(theta)
    if not method_id or not _is_sha256(artifact_sha256) or not _is_sha256(input_pins_sha256):
        raise FitSelectionError("candidate binding requires method, artifact, and input pins")
    if point_kind not in {"snapshot", "spsa_center"} or type(source_step) is not int or source_step < 0:
        raise FitSelectionError("candidate binding has an invalid point kind or step")
    payload = {
        "schema": "fit40_full_candidate_binding_v1", "method_id": method_id,
        "artifact_sha256": artifact_sha256, "input_pins_sha256": input_pins_sha256,
        "point_kind": point_kind, "source_step": source_step,
        "theta_sha256": _sha256(_canonical(list(values))), "renderer_seed": RENDERER_SEED,
    }
    return _sha256(_canonical(payload))


def _query_sort_key(key: QueryKey) -> tuple[int, Fraction, int]:
    return (key[0], Fraction(key[1], key[2]), key[3])


@dataclass(frozen=True)
class FitRoster:
    """Caller-bound Fit cohort and exact model identities."""

    sample_to_speaker: Mapping[str, str]
    model_ids: tuple[str, str, str]
    renderer_seed: int = RENDERER_SEED

    def __post_init__(self) -> None:
        mapping = dict(self.sample_to_speaker)
        object.__setattr__(self, "sample_to_speaker", MappingProxyType(mapping))
        object.__setattr__(self, "model_ids", tuple(self.model_ids))
        if self.renderer_seed != RENDERER_SEED:
            raise FitSelectionError(f"renderer seed must remain {RENDERER_SEED}")
        if len(mapping) != 40 or any(not isinstance(k, str) or not k for k in mapping):
            raise FitSelectionError("full Fit roster must contain exactly 40 named clips")
        if any(not isinstance(v, str) or not v for v in mapping.values()):
            raise FitSelectionError("every Fit clip must map to a named speaker")
        speakers: dict[str, int] = {}
        for speaker in mapping.values():
            speakers[speaker] = speakers.get(speaker, 0) + 1
        if len(speakers) != 20 or set(speakers.values()) != {2}:
            raise FitSelectionError("Fit roster must contain 20 speakers with exactly two clips each")
        if len(self.model_ids) != 3 or len(set(self.model_ids)) != 3 or any(not x for x in self.model_ids):
            raise FitSelectionError("the caller must pin exactly three distinct model IDs")

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sample_to_speaker.values())))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.sample_to_speaker))

    def batch_sample_ids(self, iteration: int) -> tuple[str, ...]:
        """Four consecutive speakers in a deterministic cyclic 20-speaker order."""
        if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
            raise FitSelectionError("SPSA iteration is outside the predeclared 0..19 range")
        ordered = self.speaker_ids
        start = (iteration * SPSA_BATCH_SPEAKERS) % len(ordered)
        speakers = tuple(ordered[(start + j) % len(ordered)] for j in range(SPSA_BATCH_SPEAKERS))
        selected = [s for s in self.sample_ids if self.sample_to_speaker[s] in speakers]
        if len(selected) != SPSA_BATCH_SPEAKERS * SPSA_CLIPS_PER_SPEAKER:
            raise FitSelectionError("SPSA batch does not contain four speakers and eight clips")
        return tuple(selected)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    artifact_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if not self.method_id or not _is_sha256(self.artifact_sha256) or not _is_sha256(self.binding_sha256):
            raise FitSelectionError("method specs require an ID and exact artifact/binding SHA-256 values")


def _reference_role_ids(method_ids: Sequence[str]) -> dict[str, str]:
    """Bind the semantic RMS role to its unmodified producer method ID."""
    identities = [method_id for method_id in method_ids if method_id == "identity"]
    rms_ids = [method_id for method_id in method_ids if method_id in RMS_REFERENCE_METHOD_IDS]
    if len(method_ids) != 2 or len(identities) != 1 or len(rms_ids) != 1:
        raise FitSelectionError(
            "reference methods must contain raw identity and exactly one raw RMS ID (rms or rms_fixed)"
        )
    return {"identity": identities[0], "rms": rms_ids[0]}


@dataclass(frozen=True)
class SnapshotSpec(MethodSpec):
    source_step: int
    theta: tuple[float, ...]
    point_kind: str = "snapshot"

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.source_step) is not int or self.source_step < 0:
            raise FitSelectionError("snapshot ordering step must be a non-negative integer")
        if self.point_kind not in {"snapshot", "spsa_center"}:
            raise FitSelectionError("point_kind must be snapshot or spsa_center")
        object.__setattr__(self, "theta", _validate_theta(self.theta))


def _validate_theta(theta: Sequence[float]) -> tuple[float, ...]:
    if len(theta) != THETA_DIMENSION:
        raise FitSelectionError("the frozen head has exactly nine theta values")
    result = tuple(float(x) for x in theta)
    if not all(math.isfinite(x) for x in result):
        raise FitSelectionError("theta values must all be finite")
    return result


@dataclass(frozen=True)
class FitScoreRow:
    """One official native-media/original-audio SyncNet result row.

    ``distance_by_query_lag`` is the exact official Stage-A layout [N,31].
    The W2L post-projection diagnostic [31,27] is intentionally incompatible.
    ``source_media_kind`` distinguishes camera/video rows from the frozen static
    portrait W2L rows; only the latter may omit ``source_video_sha256``.
    """

    row_id: str
    split: str
    sample_id: str
    speaker_id: str
    model_id: str
    face_id: str
    seed: int
    method_id: str
    status: str
    natural_pcm_sha256: str
    source_video_sha256: str | None
    source_face_sha256: str
    artifact_sha256: str
    binding_sha256: str
    input_pins_sha256: str
    query_keys: tuple[QueryKey, ...]
    valid_query_mask: tuple[bool, ...]
    distance_by_query_lag: np.ndarray | None
    lag_offsets_frames: tuple[int, ...] = LAG_OFFSETS
    failure_reason: str | None = None
    anchor_column: int = ZERO_LAG_COLUMN
    source_media_kind: str = "video"

    def __post_init__(self) -> None:
        if not all((self.row_id, self.split, self.sample_id, self.speaker_id, self.model_id, self.face_id, self.method_id)):
            raise FitSelectionError("score row identity fields must be non-empty")
        if self.seed != RENDERER_SEED:
            raise FitSelectionError(f"score row seed must remain {RENDERER_SEED}")
        if type(self.anchor_column) is not int or self.anchor_column != ZERO_LAG_COLUMN:
            raise FitSelectionError("score row must bind the frozen zero-lag anchor column 15")
        if self.source_media_kind not in ("video", "static_portrait"):
            raise FitSelectionError("source_media_kind must be video or static_portrait")
        if self.source_media_kind == "static_portrait" and self.source_video_sha256 is not None:
            raise FitSelectionError("static_portrait rows must not claim a source video SHA-256")
        if self.source_media_kind == "video" and not _is_sha256(self.source_video_sha256):
            raise FitSelectionError("video rows require a real source video SHA-256")
        for name, value in (("natural PCM", self.natural_pcm_sha256),
                            ("source face", self.source_face_sha256), ("artifact", self.artifact_sha256),
                            ("binding", self.binding_sha256), ("input pins", self.input_pins_sha256)):
            if not _is_sha256(value):
                raise FitSelectionError(f"score row has invalid {name} SHA-256")
        if self.status != "PASS":
            if self.distance_by_query_lag is not None:
                raise FitSelectionError("failed rows must not carry a score matrix")
            if self.query_keys or self.valid_query_mask:
                raise FitSelectionError("failed rows must not claim scored query support")
            return
        if len(self.query_keys) == 0 or len(self.query_keys) != len(self.valid_query_mask):
            raise FitSelectionError("passing rows require matching non-empty query keys and masks")
        normalized = tuple(_normalize_query_key(*key) for key in self.query_keys)
        if len(set(normalized)) != len(normalized):
            raise FitSelectionError("official row query keys must be unique")
        if any(type(value) is not bool for value in self.valid_query_mask):
            raise FitSelectionError("official valid_query_mask must contain booleans")
        if tuple(self.lag_offsets_frames) != LAG_OFFSETS:
            raise FitSelectionError("official lag offsets must be exactly -15..15")
        matrix = np.asarray(self.distance_by_query_lag)
        if matrix.ndim != 2 or matrix.shape != (len(self.query_keys), 31):
            raise FitSelectionError("official distance matrix must have shape [query_count,31]")
        if matrix.dtype != np.dtype("<f4") or not np.isfinite(matrix).all():
            raise FitSelectionError("official distance matrix must be finite little-endian float32")
        matrix = np.array(matrix, dtype="<f4", order="C", copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "query_keys", normalized)
        object.__setattr__(self, "valid_query_mask", tuple(self.valid_query_mask))
        object.__setattr__(self, "distance_by_query_lag", matrix)


def official_stagea_score_row(
    record: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    artifact_sha256: str,
    binding_sha256: str,
    input_pins_sha256: str,
    expected_anchor_column: int,
) -> FitScoreRow:
    """Adapt one frozen Stage-A record plus its downloaded raw NPZ arrays.

    The caller still owns whole-file/self-hash and manifest verification.  This
    function checks the producer schema, row identity, original-PCM contract,
    exact array names/shapes/hashes, the plan-bound anchor column, and
    source-clock key fields. The caller must derive ``expected_anchor_column``
    from its already byte-verified Stage-A plan.
    """
    if record.get("schema_version") != "cross_tfg_audio_head_stagea_v1":
        raise FitSelectionError("score record is not from the frozen Stage-A producer")
    status = record.get("status")
    required = ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id")
    if any(not isinstance(record.get(k), str) or not record[k] for k in required if k != "seed"):
        raise FitSelectionError("Stage-A record lacks a required identity field")
    if type(record.get("seed")) is not int or record["seed"] != RENDERER_SEED:
        raise FitSelectionError("Stage-A record has a different renderer seed")
    if record.get("split") != "fit":
        raise FitSelectionError("Fit-only selection accepts only rows from the frozen fit split")
    if type(expected_anchor_column) is not int or expected_anchor_column != ZERO_LAG_COLUMN:
        raise FitSelectionError("the bound Stage-A plan anchor must be the declared zero-lag column 15")
    source_media_kind = record.get("source_media_kind", "video")
    if source_media_kind not in ("video", "static_portrait"):
        raise FitSelectionError("Stage-A record has an unsupported source_media_kind")
    natural = record.get("natural_pcm_sha256")
    if not _is_sha256(natural) or record.get("score_audio_sha256") != natural:
        raise FitSelectionError("official scoring must use the exact original natural PCM")
    if status != "PASS":
        if arrays:
            raise FitSelectionError("failed Stage-A row must not be paired with score arrays")
        return FitScoreRow(
            row_id=record["row_id"], split=record["split"], sample_id=record["sample_id"],
            speaker_id=record["speaker_id"], model_id=record["model_id"], face_id=record["face_id"],
            seed=record["seed"], method_id=record["method_id"], status=status,
            natural_pcm_sha256=natural, source_video_sha256=record.get("source_video_sha256"),
            source_face_sha256=record.get("source_face_sha256"), artifact_sha256=artifact_sha256,
            binding_sha256=binding_sha256, input_pins_sha256=input_pins_sha256,
            query_keys=(), valid_query_mask=(), distance_by_query_lag=None,
            failure_reason=record.get("failure_reason"), anchor_column=expected_anchor_column,
            source_media_kind=source_media_kind,
        )
    arrays_required = {
        "query_frame_index", "query_timestamp_num", "query_timestamp_den", "track_frame_index",
        "valid_query_mask", "distance_by_query_lag", "lag_offsets_frames",
    }
    if set(arrays) != arrays_required:
        raise FitSelectionError(f"official NPZ fields differ from Stage-A contract: {sorted(set(arrays) ^ arrays_required)}")
    score = record.get("score")
    if not isinstance(score, Mapping):
        raise FitSelectionError("passing Stage-A row lacks its score record")
    raw_ref = score.get("raw_matrix_npz")
    if not isinstance(raw_ref, Mapping) or not isinstance(raw_ref.get("path"), str) or not raw_ref["path"]:
        raise FitSelectionError("passing Stage-A row lacks its raw NPZ file reference")
    if not _is_sha256(raw_ref.get("file_sha256")):
        raise FitSelectionError("passing Stage-A row lacks a whole-NPZ SHA-256")
    array_meta = raw_ref.get("arrays")
    if not isinstance(array_meta, Mapping) or set(array_meta) != arrays_required:
        raise FitSelectionError("Stage-A raw NPZ array metadata differs from the official array set")
    matrix = np.asarray(arrays["distance_by_query_lag"])
    if matrix.ndim != 2 or matrix.shape[1] != 31:
        raise FitSelectionError("Stage-A NPZ matrix must be [N,31], not the W2L proxy layout")
    count = matrix.shape[0]
    fields: dict[str, np.ndarray] = {}
    dtypes = {
        "query_frame_index": np.dtype("<i4"), "query_timestamp_num": np.dtype("<i8"),
        "query_timestamp_den": np.dtype("<i8"), "track_frame_index": np.dtype("<i4"),
        "valid_query_mask": np.dtype("bool"), "lag_offsets_frames": np.dtype("<i4"),
    }
    for key, expected_dtype in dtypes.items():
        value = np.asarray(arrays[key])
        if value.ndim != 1 or value.dtype != expected_dtype:
            raise FitSelectionError(f"Stage-A NPZ {key} has the wrong rank or exact dtype")
        fields[key] = value
    for key, value in arrays.items():
        meta = array_meta.get(key)
        array = np.asarray(value)
        if not isinstance(meta, Mapping):
            raise FitSelectionError(f"Stage-A raw NPZ metadata is absent for {key}")
        actual_array_sha = _sha256(np.ascontiguousarray(array).tobytes(order="C"))
        if (meta.get("dtype") != array.dtype.str or meta.get("shape") != list(array.shape)
                or meta.get("array_sha256") != actual_array_sha):
            raise FitSelectionError(f"Stage-A raw NPZ array metadata does not bind supplied {key} bytes")
    for key in ("query_frame_index", "query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if len(fields[key]) != count:
            raise FitSelectionError(f"Stage-A NPZ {key} length differs from query count")
    if fields["lag_offsets_frames"].tolist() != list(LAG_OFFSETS):
        raise FitSelectionError("Stage-A NPZ lag axis differs from -15..15")
    keys = tuple(
        _normalize_query_key(int(f), int(n), int(d), int(t))
        for f, n, d, t in zip(fields["query_frame_index"], fields["query_timestamp_num"],
                               fields["query_timestamp_den"], fields["track_frame_index"])
    )
    support_by_support = score.get("support_by_support")
    metrics_by_support = score.get("metrics_by_support")
    full_support = support_by_support.get("full") if isinstance(support_by_support, Mapping) else None
    full_metrics = metrics_by_support.get("full") if isinstance(metrics_by_support, Mapping) else None
    if not isinstance(full_support, Mapping) or not isinstance(full_metrics, Mapping):
        raise FitSelectionError("passing Stage-A row lacks its producer full-support summary")
    common_query_count = full_support.get("query_count")
    if (type(common_query_count) is not int or common_query_count < 0
            or common_query_count > int(np.count_nonzero(fields["valid_query_mask"]))):
        raise FitSelectionError("Stage-A full common-support count exceeds this row's valid raw queries")
    if common_query_count:
        if (full_metrics.get("anchor_column") != expected_anchor_column
                or full_metrics.get("zero_lag_column") != ZERO_LAG_COLUMN):
            raise FitSelectionError("Stage-A producer anchor/zero-lag columns differ from the bound plan")
    elif (full_metrics.get("status") != "EMPTY" or full_metrics.get("anchor_column") is not None
          or full_metrics.get("zero_lag_column") is not None):
        raise FitSelectionError("empty Stage-A full support must retain the producer's EMPTY metric record")
    return FitScoreRow(
        row_id=record["row_id"], split=record["split"], sample_id=record["sample_id"],
        speaker_id=record["speaker_id"], model_id=record["model_id"], face_id=record["face_id"],
        seed=record["seed"], method_id=record["method_id"], status=status,
        natural_pcm_sha256=natural, source_video_sha256=record.get("source_video_sha256"),
        source_face_sha256=record.get("source_face_sha256"), artifact_sha256=artifact_sha256,
        binding_sha256=binding_sha256, input_pins_sha256=input_pins_sha256,
        query_keys=keys, valid_query_mask=tuple(bool(x) for x in fields["valid_query_mask"]),
        distance_by_query_lag=matrix if status == "PASS" else None,
        lag_offsets_frames=tuple(int(x) for x in fields["lag_offsets_frames"]),
        failure_reason=record.get("failure_reason"), anchor_column=expected_anchor_column,
        source_media_kind=source_media_kind,
    )


@dataclass(frozen=True)
class MetricSummary:
    b_median: float
    dmin: float
    d0: float
    cbest: float
    canchor: float
    valid_query_count: int
    support_sha256: str


@dataclass(frozen=True)
class MethodEffects:
    """Equal-speaker effects by model and reference method."""

    deltas: Mapping[str, Mapping[str, Mapping[str, float]]]
    # deltas[model_id][reference_id][metric] where D0 is reference-candidate.
    cbest_vs_rms_by_model: Mapping[str, float]
    equal_model: Mapping[str, Mapping[str, float]]
    worst_model_cbest_vs_rms: float | None
    support_sha256: str
    support_counts: Mapping[str, int]
    complete: bool
    status: str
    errors: tuple[str, ...] = ()
    identity_reference_method_id: str = "identity"
    rms_reference_method_id: str = "rms"


@dataclass(frozen=True)
class _FamilyData:
    by_method_context: Mapping[str, Mapping[ContextKey, FitScoreRow]]
    summaries: Mapping[str, Mapping[ContextKey, MetricSummary]]
    support_by_context: Mapping[ContextKey, tuple[QueryKey, ...]]
    family_support_sha256: str
    errors: tuple[str, ...]


def _matrix_sha(row: FitScoreRow) -> str:
    if row.distance_by_query_lag is None:
        return ""
    return _sha256(np.ascontiguousarray(row.distance_by_query_lag, dtype="<f4").tobytes(order="C"))


def _support_sha(method_ids: Sequence[str], keys_by_context: Mapping[ContextKey, Sequence[QueryKey]]) -> str:
    payload = {
        "schema": "fit40_common_source_clock_support_v1",
        "methods": list(method_ids),
        "contexts": [
            {"context": list(context), "keys": [list(k) for k in keys_by_context[context]]}
            for context in sorted(keys_by_context)
        ],
    }
    return _sha256(_canonical(payload))


def _metric_on_keys(row: FitScoreRow, keys: Sequence[QueryKey], *, metric_backend: str) -> MetricSummary:
    if row.distance_by_query_lag is None:
        raise FitSelectionError("cannot summarize a failed score row")
    if metric_backend not in METRIC_BACKENDS:
        raise FitSelectionError(f"unknown metric backend {metric_backend!r}")
    index = {key: i for i, key in enumerate(row.query_keys)}
    selected = np.ascontiguousarray(row.distance_by_query_lag[[index[k] for k in keys]], dtype=np.float32)
    if metric_backend == "torch_cpu":
        try:
            import torch
        except ImportError as exc:
            raise FitSelectionError("canonical torch_cpu metric backend is unavailable; no NumPy fallback is allowed") from exc
        torch.set_num_threads(1)
        curve_tensor = torch.mean(torch.from_numpy(np.ascontiguousarray(selected.T)), dim=1)
        b_tensor = torch.median(curve_tensor)
        dmin_tensor = torch.min(curve_tensor, dim=0).values
        d0_tensor = curve_tensor[ZERO_LAG_COLUMN]
        b = np.float32(b_tensor.item())
        dmin = np.float32(dmin_tensor.item())
        d0 = np.float32(d0_tensor.item())
        curve = curve_tensor.detach().numpy().astype(np.float32, copy=False)
    else:
        # Engineering-only NumPy reduction; explicit diagnostic backend callers
        # must rerun final admission with canonical Stage-A torch_cpu metrics.
        curve = np.mean(selected, axis=0, dtype=np.float32)
        b = np.float32(np.median(curve))
        dmin = np.float32(np.min(curve))
        d0 = np.float32(curve[ZERO_LAG_COLUMN])
    return MetricSummary(
        b_median=float(b), dmin=float(dmin), d0=float(d0),
        cbest=float(np.float32(b - dmin)), canchor=float(np.float32(b - d0)),
        valid_query_count=len(keys), support_sha256="",
    )


def _expected_samples(roster: FitRoster, sample_ids: Sequence[str] | None) -> tuple[str, ...]:
    samples = roster.sample_ids if sample_ids is None else tuple(sample_ids)
    if len(set(samples)) != len(samples) or not samples or any(s not in roster.sample_to_speaker for s in samples):
        raise FitSelectionError("requested sample subset is empty, duplicated, or outside the frozen roster")
    selected_speakers = {roster.sample_to_speaker[s] for s in samples}
    if any(sum(roster.sample_to_speaker[s] == speaker for s in samples) != 2 for speaker in selected_speakers):
        raise FitSelectionError("every selected speaker must contribute both frozen clips")
    return tuple(sorted(samples))


def _build_family(
    rows: Sequence[FitScoreRow],
    method_specs: Sequence[MethodSpec],
    roster: FitRoster,
    *,
    sample_ids: Sequence[str] | None = None,
    expected_input_pins_sha256: str | None = None,
    metric_backend: str,
) -> _FamilyData:
    if metric_backend not in METRIC_BACKENDS:
        raise FitSelectionError(f"unknown metric backend {metric_backend!r}")
    if len({m.method_id for m in method_specs}) != len(method_specs) or len(method_specs) < 3:
        raise FitSelectionError("a family requires unique methods and at least two references plus one candidate")
    samples = _expected_samples(roster, sample_ids)
    methods = tuple(m.method_id for m in method_specs)
    expected = {(sample, model) for sample in samples for model in roster.model_ids}
    specs = {m.method_id: m for m in method_specs}
    input_pins: set[str] = set()
    grouped: dict[str, dict[ContextKey, FitScoreRow]] = {m: {} for m in methods}
    errors: list[str] = []
    seen_row_ids: set[str] = set()
    for row in rows:
        if row.method_id not in specs:
            errors.append(f"unexpected method row: {row.method_id}/{row.row_id}")
            continue
        if row.sample_id not in samples or row.model_id not in roster.model_ids:
            errors.append(f"row outside expected Fit context: {row.row_id}")
            continue
        if roster.sample_to_speaker[row.sample_id] != row.speaker_id:
            errors.append(f"speaker identity differs from roster: {row.row_id}")
            continue
        if row.seed != roster.renderer_seed:
            errors.append(f"renderer seed differs: {row.row_id}")
        if row.split != "fit":
            errors.append(f"Fit-only family contains non-fit split row: {row.row_id}:{row.split}")
        spec = specs[row.method_id]
        if row.artifact_sha256 != spec.artifact_sha256 or row.binding_sha256 != spec.binding_sha256:
            errors.append(f"method artifact/binding pin differs: {row.row_id}")
        input_pins.add(row.input_pins_sha256)
        context = (row.sample_id, row.speaker_id, row.model_id)
        if context in grouped[row.method_id]:
            errors.append(f"duplicate method/context row: {row.method_id}/{context}")
            continue
        if row.row_id in seen_row_ids:
            errors.append(f"duplicate row ID: {row.row_id}")
        seen_row_ids.add(row.row_id)
        grouped[row.method_id][context] = row
    if len(input_pins) > 1:
        errors.append("method rows do not share one canonical input-pin digest")
    if expected_input_pins_sha256 is not None:
        if not _is_sha256(expected_input_pins_sha256) or input_pins != {expected_input_pins_sha256}:
            errors.append("score rows do not bind the caller's canonical input-pin digest")
    expected_contexts = {(sample, roster.sample_to_speaker[sample], model) for sample, model in expected}
    for method in methods:
        missing = expected_contexts - set(grouped[method])
        extra = set(grouped[method]) - expected_contexts
        if missing:
            errors.append(f"{method} missing {len(missing)} planned rows")
        if extra:
            errors.append(f"{method} has {len(extra)} unexpected contexts")
        if len(grouped[method]) != len(expected_contexts):
            errors.append(f"{method} denominator is {len(grouped[method])}, expected {len(expected_contexts)}")
        for context, row in grouped[method].items():
            if row.status != "PASS" or row.distance_by_query_lag is None:
                errors.append(f"{method} failed row remains in denominator: {row.row_id}:{row.status}")

    support_by_context: dict[ContextKey, tuple[QueryKey, ...]] = {}
    summaries: dict[str, dict[ContextKey, MetricSummary]] = {m: {} for m in methods}
    for context in sorted(expected_contexts):
        context_rows = [grouped[m].get(context) for m in methods]
        if any(row is None or row.status != "PASS" or row.distance_by_query_lag is None for row in context_rows):
            continue
        concrete = [row for row in context_rows if row is not None]
        for field_name in ("natural_pcm_sha256", "source_media_kind", "source_video_sha256",
                           "source_face_sha256", "split", "face_id"):
            values = {getattr(row, field_name) for row in concrete}
            if len(values) != 1:
                errors.append(f"{field_name} differs across methods for {context}")
        key_maps: list[dict[QueryKey, int]] = []
        for row in concrete:
            mapping: dict[QueryKey, int] = {}
            for index, key in enumerate(row.query_keys):
                if row.valid_query_mask[index]:
                    mapping[key] = index
            key_maps.append(mapping)
        common = set(key_maps[0])
        for mapping in key_maps[1:]:
            common.intersection_update(mapping)
        keys = tuple(sorted(common, key=_query_sort_key))
        if not keys:
            errors.append(f"empty common valid source-clock support for {context}")
            continue
        support_by_context[context] = keys
        for method, row in zip(methods, concrete):
            metric = _metric_on_keys(row, keys, metric_backend=metric_backend)
            summaries[method][context] = MetricSummary(
                metric.b_median, metric.dmin, metric.d0, metric.cbest, metric.canchor,
                metric.valid_query_count, _sha256(_canonical([list(k) for k in keys])),
            )
    if len(support_by_context) != len(expected_contexts):
        errors.append(f"common support exists for {len(support_by_context)}/{len(expected_contexts)} contexts")
    family_sha = _support_sha(methods, support_by_context) if support_by_context else _sha256(b"empty-fit-support")
    return _FamilyData(grouped, summaries, support_by_context, family_sha, tuple(dict.fromkeys(errors)))


def _speaker_equal_mean(values_by_context: Mapping[ContextKey, float], roster: Any) -> float:
    per_speaker: dict[str, list[float]] = {speaker: [] for speaker in set(roster.sample_to_speaker.values())}
    for (sample, speaker, _model), value in values_by_context.items():
        if sample in roster.sample_to_speaker and roster.sample_to_speaker[sample] == speaker:
            per_speaker[speaker].append(float(value))
    if not per_speaker or any(len(v) != 2 for v in per_speaker.values()):
        raise FitSelectionError("effect aggregation requires exactly two clips per speaker")
    return float(np.mean([np.mean(per_speaker[s], dtype=np.float64) for s in sorted(per_speaker)], dtype=np.float64))


def _effects_for_candidate(
    family: _FamilyData,
    roster: FitRoster,
    *,
    candidate_method: str,
    reference_methods: Sequence[str],
    sample_ids: Sequence[str] | None = None,
) -> MethodEffects:
    roles = _reference_role_ids(reference_methods)
    if family.errors:
        return MethodEffects({}, {}, {}, None, family.family_support_sha256,
                             {"common_contexts": len(family.support_by_context)}, False,
                             "CONTROL_FAILED_MISSING_FIT_EVIDENCE", family.errors,
                             roles["identity"], roles["rms"])
    if candidate_method not in family.summaries or any(r not in family.summaries for r in reference_methods):
        raise FitSelectionError("candidate/reference method is absent from the scored family")
    samples = set(_expected_samples(roster, sample_ids))
    selected_models = roster.model_ids
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    cbest_rms: dict[str, float] = {}
    by_model_ref_values: dict[str, dict[str, dict[str, dict[ContextKey, float]]]] = {}
    for model in selected_models:
        deltas[model] = {}
        by_model_ref_values[model] = {}
        for ref in reference_methods:
            values: dict[str, dict[ContextKey, float]] = {name: {} for name in ("D0", "Cbest", "Canchor")}
            for context in family.support_by_context:
                if context[0] not in samples or context[2] != model:
                    continue
                cand = family.summaries[candidate_method][context]
                base = family.summaries[ref][context]
                values["D0"][context] = base.d0 - cand.d0
                values["Cbest"][context] = cand.cbest - base.cbest
                values["Canchor"][context] = cand.canchor - base.canchor
            agg = {metric: _speaker_equal_mean(v, _subset_roster(roster, samples)) for metric, v in values.items()}
            deltas[model][ref] = agg
            by_model_ref_values[model][ref] = values
    rms = roles["rms"]
    for model in selected_models:
        cbest_rms[model] = deltas[model][rms]["Cbest"]
    equal: dict[str, dict[str, float]] = {}
    for ref in reference_methods:
        equal[ref] = {
            metric: float(np.mean([deltas[m][ref][metric] for m in selected_models], dtype=np.float64))
            for metric in ("D0", "Cbest", "Canchor")
        }
    worst = min(cbest_rms.values())
    return MethodEffects(
        deltas=deltas, cbest_vs_rms_by_model=cbest_rms, equal_model=equal,
        worst_model_cbest_vs_rms=worst, support_sha256=family.family_support_sha256,
        support_counts={
            "common_contexts": len(family.support_by_context),
            "common_query_keys": sum(len(x) for x in family.support_by_context.values()),
            "min_query_keys_per_context": min(map(len, family.support_by_context.values())),
            "max_query_keys_per_context": max(map(len, family.support_by_context.values())),
        },
        complete=True, status="COMPLETE",
        identity_reference_method_id=roles["identity"],
        rms_reference_method_id=roles["rms"],
    )


def _subset_roster(roster: FitRoster, samples: set[str]) -> Any:
    """Internal roster facade for equal-speaker aggregation over a whole roster/subset."""
    mapping = {sample: roster.sample_to_speaker[sample] for sample in samples}
    # Batches are four complete speakers; full comparisons are 20 complete speakers.
    return SimpleNamespace(sample_to_speaker=mapping)


@dataclass(frozen=True)
class CandidateDecision:
    method_id: str
    feasible: bool
    failed_gate_count: int
    fallback_violation: float
    worst_model_cbest_vs_rms: float
    equal_model_cbest_vs_rms: float
    source_step: int
    gates: Mapping[str, Any]


@dataclass(frozen=True)
class SnapshotSelection:
    status: str
    selected_method_id: str | None
    refinement_seed_method_id: str | None
    support_sha256: str
    accounting: Mapping[str, Any]
    candidate_decisions: tuple[CandidateDecision, ...]
    metric_backend: str
    errors: tuple[str, ...] = ()


def expected_score_accounting(candidate_count: int, *, context_count: int = 120) -> dict[str, int]:
    if type(candidate_count) is not int or candidate_count < 1 or type(context_count) is not int or context_count < 1:
        raise FitSelectionError("candidate/context counts must be positive integers")
    return {
        "fit_contexts": context_count,
        "shared_reference_methods": 2,
        "candidate_methods": candidate_count,
        "unique_official_score_rows": context_count * (candidate_count + 2),
        "shared_reference_rows": context_count * 2,
        "logical_rows_per_candidate_bundle": context_count * 3,
        "candidate_method_rows": context_count * candidate_count,
    }


def _candidate_gate(spec: SnapshotSpec, effects: MethodEffects, references: Sequence[str]) -> CandidateDecision:
    gate_items: list[tuple[str, float, str]] = []
    rms_method_id = effects.rms_reference_method_id
    gate_report: dict[str, Any] = {
        "models": {}, "equal_model_cbest_vs_rms": effects.equal_model[rms_method_id]["Cbest"],
        "reference_role_mapping": {
            "identity": effects.identity_reference_method_id,
            "rms": rms_method_id,
        },
    }
    for model, by_ref in effects.deltas.items():
        gate_report["models"][model] = {}
        for ref in references:
            m = by_ref[ref]
            gate_report["models"][model][ref] = dict(m)
            # Cbest must be strictly positive; zero is a failed gate with zero hinge cost.
            gate_items.append((f"{model}/{ref}/Cbest>0", m["Cbest"], "strict_positive"))
            gate_items.append((f"{model}/{ref}/D0>=0", m["D0"], "nonnegative"))
            gate_items.append((f"{model}/{ref}/Canchor>=0", m["Canchor"], "nonnegative"))
    equal_cbest = effects.equal_model[rms_method_id]["Cbest"]
    gate_items.append(("equal_model/Cbest_vs_RMS>=0.050", equal_cbest - EQUAL_MODEL_CBEST_RMS_MIN, "nonnegative"))
    failed = 0
    deficits: list[float] = []
    for _name, value, kind in gate_items:
        ok = value > 0 if kind == "strict_positive" else value >= 0
        if not ok:
            failed += 1
        if kind == "strict_positive":
            deficits.append(max(0.0, -value))
        else:
            deficits.append(max(0.0, -value))
    # Parent-approved fallback: mean hinge severity for the 18 model/reference
    # guards, plus the equal-model threshold shortfall.
    guard_values = [effects.deltas[m][r][metric] for m in effects.deltas for r in references
                    for metric in ("D0", "Canchor", "Cbest")]
    violation = float(np.mean([max(0.0, -x) for x in guard_values], dtype=np.float64))
    violation += max(0.0, EQUAL_MODEL_CBEST_RMS_MIN - equal_cbest)
    return CandidateDecision(
        method_id=spec.method_id,
        feasible=failed == 0,
        failed_gate_count=failed,
        fallback_violation=violation,
        worst_model_cbest_vs_rms=effects.worst_model_cbest_vs_rms,
        equal_model_cbest_vs_rms=equal_cbest,
        source_step=spec.source_step,
        gates=gate_report,
    )


def select_snapshot_family(
    rows: Sequence[FitScoreRow],
    *,
    roster: FitRoster,
    reference_specs: Sequence[MethodSpec],
    candidates: Sequence[SnapshotSpec],
    expected_input_pins_sha256: str,
    metric_backend: str,
    sample_ids: Sequence[str] | None = None,
) -> SnapshotSelection:
    """Select only a fully supported candidate; otherwise return NOT_ADMITTED.

    A complete family uses one exact common source-clock intersection across
    both references and every candidate supplied.  Therefore all four initial
    snapshots plus identity/RMS account for 720 physical rows on the full Fit
    cohort; four later centers plus the same references also account for 720.
    """
    if len(reference_specs) != 2 or len({x.method_id for x in reference_specs}) != 2:
        raise FitSelectionError("exactly two distinct identity/RMS reference methods are required")
    reference_roles = _reference_role_ids(tuple(s.method_id for s in reference_specs))
    if not candidates or len({x.method_id for x in candidates}) != len(candidates):
        raise FitSelectionError("candidate methods must be non-empty and unique")
    point_kinds = {x.point_kind for x in candidates}
    if len(point_kinds) != 1:
        raise FitSelectionError("one selection family cannot mix trained snapshots and SPSA centers")
    expected_steps = {30, 60, 90, 120} if point_kinds == {"snapshot"} else {5, 10, 15, 20}
    if len(candidates) != 4 or {x.source_step for x in candidates} != expected_steps:
        raise FitSelectionError("selection must include the full predeclared four-point family")
    if sample_ids is not None and set(sample_ids) != set(roster.sample_ids):
        raise FitSelectionError("snapshot admission is full-Fit only; mini-batches cannot be selected")
    methods: tuple[MethodSpec, ...] = (*reference_specs, *candidates)
    for spec in candidates:
        expected_binding = full_fit_candidate_binding_sha256(
            method_id=spec.method_id, artifact_sha256=spec.artifact_sha256,
            input_pins_sha256=expected_input_pins_sha256, source_step=spec.source_step,
            theta=spec.theta, point_kind=spec.point_kind,
        )
        if spec.binding_sha256 != expected_binding:
            raise FitSelectionError(f"candidate binding does not match declared theta/input pins: {spec.method_id}")
    family = _build_family(rows, methods, roster, sample_ids=sample_ids,
                           expected_input_pins_sha256=expected_input_pins_sha256,
                           metric_backend=metric_backend)
    samples = _expected_samples(roster, sample_ids)
    expected_context_count = len(samples) * len(roster.model_ids)
    accounting = expected_score_accounting(len(candidates), context_count=expected_context_count)
    expected_rows = accounting["unique_official_score_rows"]
    audit_accounting = {
        **accounting,
        "planned_methods": len(methods),
        "observed_row_records": len(rows),
        "observed_unique_rows": len({r.row_id for r in rows}),
        "failed_score_rows": sum(r.status != "PASS" for r in rows),
        "missing_score_rows": max(0, expected_rows - len(rows)),
        "unexpected_score_rows": max(0, len(rows) - expected_rows),
        "common_query_keys_total": sum(map(len, family.support_by_context.values())),
        "metric_backend": metric_backend,
        "reference_role_mapping": dict(reference_roles),
        "metric_parity_atol": METRIC_PARITY_ATOL,
        "metric_parity_rtol": METRIC_PARITY_RTOL,
    }
    if family.errors:
        return SnapshotSelection(
            status="CONTROL_FAILED_MISSING_FIT_EVIDENCE", selected_method_id=None,
            refinement_seed_method_id=None, support_sha256=family.family_support_sha256,
            accounting=audit_accounting, candidate_decisions=(), metric_backend=metric_backend,
            errors=family.errors,
        )
    if metric_backend == "numpy_diagnostic":
        return SnapshotSelection(
            status="NUMPY_DIAGNOSTIC_ONLY_NOT_ADMITTED", selected_method_id=None,
            refinement_seed_method_id=None, support_sha256=family.family_support_sha256,
            accounting={**audit_accounting, "diagnostic_only": True},
            candidate_decisions=(), metric_backend=metric_backend,
            errors=("NumPy diagnostics cannot produce gate decisions, admission, or an SPSA refinement seed; "
                    "rerun with canonical torch_cpu metrics",),
        )
    ref_ids = tuple(x.method_id for x in reference_specs)
    decisions: list[CandidateDecision] = []
    for spec in candidates:
        effects = _effects_for_candidate(family, roster, candidate_method=spec.method_id,
                                         reference_methods=ref_ids, sample_ids=samples)
        decisions.append(_candidate_gate(spec, effects, ref_ids))
    feasible = [d for d in decisions if d.feasible]
    if feasible:
        winner = sorted(feasible, key=lambda d: (-d.equal_model_cbest_vs_rms,
                                                 -d.worst_model_cbest_vs_rms, d.source_step))[0]
        return SnapshotSelection(
            "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", winner.method_id, None,
            family.family_support_sha256, audit_accounting,
            tuple(decisions), metric_backend,
        )
    seed = sorted(decisions, key=lambda d: (d.fallback_violation, d.failed_gate_count,
                                            -d.worst_model_cbest_vs_rms,
                                            -d.equal_model_cbest_vs_rms, d.source_step))[0]
    return SnapshotSelection(
        "NOT_ADMITTED", None, seed.method_id, family.family_support_sha256,
        audit_accounting,
        tuple(decisions), metric_backend,
    )


def _softplus(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _logmeanexp(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(v) for v in values):
        raise FitSelectionError("logmeanexp requires finite non-empty values")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values) / len(values))


def objective_from_effects(effects: MethodEffects, theta: Sequence[float], theta_initial: Sequence[float]) -> dict[str, float]:
    """Compute the explicitly frozen robust Fit objective from paired effects."""
    if not effects.complete or effects.status != "COMPLETE":
        raise FitSelectionError("objective cannot be computed from incomplete paired evidence")
    current = _validate_theta(theta)
    initial = _validate_theta(theta_initial)
    refs = (effects.identity_reference_method_id, effects.rms_reference_method_id)
    models = tuple(effects.deltas)
    if (set(effects.equal_model) != set(refs) or refs[0] != "identity"
            or refs[1] not in RMS_REFERENCE_METHOD_IDS or len(models) != 3):
        raise FitSelectionError("objective requires both references and exactly three models")
    lc_terms = []
    for ref in refs:
        target = EQUAL_MODEL_CBEST_RMS_MIN if ref == effects.rms_reference_method_id else 0.0
        terms = [(target - effects.deltas[model][ref]["Cbest"]) / TAU for model in models]
        lc_terms.append(TAU * _logmeanexp(terms))
    lc = float(np.mean(lc_terms, dtype=np.float64))
    guard_terms = [
        TAU * _softplus(-effects.deltas[model][ref][metric] / TAU)
        for model in models for ref in refs for metric in ("D0", "Canchor")
    ]
    lguard = 2.0 * float(np.mean(guard_terms, dtype=np.float64))
    ltheta = 0.001 * float(np.mean([(a - b) ** 2 for a, b in zip(current, initial)], dtype=np.float64))
    total = lc + lguard + ltheta
    if not all(math.isfinite(v) for v in (lc, lguard, ltheta, total)):
        raise FitSelectionError("objective produced a non-finite value")
    return {"L_c": lc, "L_guard": lguard, "L_theta": ltheta, "L_total": total}


def spsa_step_sizes(iteration: int) -> tuple[float, float]:
    if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
        raise FitSelectionError("SPSA iteration must be in 0..19")
    return 0.5 / ((iteration + 5) ** 0.602), 0.25 / ((iteration + 1) ** 0.101)


@dataclass(frozen=True)
class SPSAProposal:
    proposal_id: str
    pair_id: str
    side: str
    iteration: int
    method_id: str
    theta: tuple[float, ...]
    theta_sha256: str
    artifact_sha256: str
    binding_sha256: str
    input_pins_sha256: str
    batch_id: str
    sample_ids: tuple[str, ...]
    speakers: tuple[str, ...]
    direction: tuple[int, ...]
    c_k: float


@dataclass(frozen=True)
class SPSAProposalPair:
    pair_id: str
    iteration: int
    center_theta: tuple[float, ...]
    direction: tuple[int, ...]
    a_k: float
    c_k: float
    batch_id: str
    sample_ids: tuple[str, ...]
    speakers: tuple[str, ...]
    plus: SPSAProposal
    minus: SPSAProposal


@dataclass(frozen=True)
class ProposalEvaluation:
    proposal_id: str
    pair_id: str
    side: str
    method_id: str
    theta: tuple[float, ...]
    theta_sha256: str
    artifact_sha256: str
    binding_sha256: str
    input_pins_sha256: str
    batch_id: str
    rows: tuple[FitScoreRow, ...]


@dataclass(frozen=True)
class SPSATellResult:
    status: str
    iteration: int
    theta_before: tuple[float, ...]
    theta_after: tuple[float, ...]
    plus_objective: Mapping[str, float] | None
    minus_objective: Mapping[str, float] | None
    gradient: tuple[float, ...] | None
    clipped_update: tuple[float, ...] | None
    full_center_score_due: bool
    full_center_spec: SnapshotSpec | None
    reason: str | None
    support_sha256: str | None
    metric_backend: str


class SPSAController:
    """Deterministic CPU ask/tell controller; it never evaluates a model itself."""

    def __init__(
        self,
        *,
        initial_theta: Sequence[float],
        input_pins: Mapping[str, Mapping[str, Any]],
        candidate_artifact_sha256: str,
        roster: FitRoster,
        reference_specs: Sequence[MethodSpec],
        metric_backend: str,
        rng_seed: int = SPSA_RNG_SEED,
    ) -> None:
        if rng_seed != SPSA_RNG_SEED:
            raise FitSelectionError(f"SPSA RNG seed must remain {SPSA_RNG_SEED}")
        if not _is_sha256(candidate_artifact_sha256):
            raise FitSelectionError("SPSA requires the exact frozen candidate head/source artifact pin")
        reference_roles = _reference_role_ids(tuple(s.method_id for s in reference_specs))
        if metric_backend not in METRIC_BACKENDS:
            raise FitSelectionError(f"unknown metric backend {metric_backend!r}")
        self.theta_initial = _validate_theta(initial_theta)
        self.theta = self.theta_initial
        self.input_pins = dict(input_pins)
        self.input_pins_sha256 = canonical_input_pins_sha256(input_pins)
        self.candidate_artifact_sha256 = candidate_artifact_sha256
        self.roster = roster
        self.reference_specs = tuple(reference_specs)
        self.identity_reference_method_id = reference_roles["identity"]
        self.rms_reference_method_id = reference_roles["rms"]
        self.metric_backend = metric_backend
        self.iteration = 0
        self._rng = np.random.Generator(np.random.PCG64(rng_seed))
        self._pending: SPSAProposalPair | None = None
        self._halted_reason: str | None = None
        self.history: list[SPSATellResult] = []

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    def ask(self) -> SPSAProposalPair:
        if self.halted:
            raise FitSelectionError(f"controller halted: {self._halted_reason}")
        if self.metric_backend != "torch_cpu":
            raise FitSelectionError(
                "NUMPY_DIAGNOSTIC_ONLY: SPSA proposals require canonical torch_cpu metrics"
            )
        if self._pending is not None:
            raise FitSelectionError("tell the outstanding proposal pair before asking again")
        if self.iteration >= SPSA_ITERATIONS:
            raise FitSelectionError("all 20 predeclared SPSA iterations are complete")
        k = self.iteration
        a_k, c_k = spsa_step_sizes(k)
        delta = tuple(int(x) for x in self._rng.integers(0, 2, size=THETA_DIMENSION) * 2 - 1)
        center = self.theta
        plus_theta = tuple(center[i] + c_k * delta[i] for i in range(THETA_DIMENSION))
        minus_theta = tuple(center[i] - c_k * delta[i] for i in range(THETA_DIMENSION))
        sample_ids = self.roster.batch_sample_ids(k)
        speakers = tuple(sorted({self.roster.sample_to_speaker[s] for s in sample_ids}))
        batch_id = _sha256(_canonical({"schema": "fit40_spsa_batch_v1", "iteration": k,
                                      "sample_ids": list(sample_ids), "speakers": list(speakers),
                                      "renderer_seed": self.roster.renderer_seed}))
        pair_id = _sha256(_canonical({"schema": "fit40_spsa_pair_v1", "iteration": k,
                                     "center": list(center), "direction": list(delta),
                                     "input_pins_sha256": self.input_pins_sha256,
                                     "batch_id": batch_id}))

        def proposal(side: str, theta: tuple[float, ...]) -> SPSAProposal:
            method_id = f"fit40_d0_spsa_{side}_k{k + 1:02d}"
            theta_sha = _sha256(_canonical(list(theta)))
            proposal_id = _sha256(_canonical({"pair_id": pair_id, "side": side,
                                              "method_id": method_id, "theta_sha256": theta_sha}))
            binding_sha = _sha256(_canonical({
                "schema": "fit40_spsa_proposal_binding_v1", "proposal_id": proposal_id,
                "pair_id": pair_id, "side": side, "iteration": k, "method_id": method_id,
                "theta_sha256": theta_sha, "artifact_sha256": self.candidate_artifact_sha256,
                "input_pins_sha256": self.input_pins_sha256, "batch_id": batch_id,
                "renderer_seed": self.roster.renderer_seed, "sample_ids": list(sample_ids),
            }))
            return SPSAProposal(proposal_id, pair_id, side, k, method_id, theta, theta_sha,
                                self.candidate_artifact_sha256, binding_sha, self.input_pins_sha256,
                                batch_id, sample_ids, speakers, delta, c_k)

        pair = SPSAProposalPair(pair_id, k, center, delta, a_k, c_k, batch_id, sample_ids, speakers,
                                proposal("plus", plus_theta), proposal("minus", minus_theta))
        self._pending = pair
        return pair

    def tell(self, plus: ProposalEvaluation, minus: ProposalEvaluation) -> SPSATellResult:
        if self.halted:
            raise FitSelectionError(f"controller halted: {self._halted_reason}")
        pair = self._pending
        if pair is None:
            raise FitSelectionError("there is no outstanding SPSA pair")
        before = self.theta

        def fail(reason: str, support_sha: str | None = None) -> SPSATellResult:
            self._halted_reason = reason
            result = SPSATellResult("NO_UPDATE", self.iteration, before, before, None, None, None, None,
                                    False, None, reason, support_sha, self.metric_backend)
            self.history.append(result)
            self._pending = None
            return result

        if self.metric_backend == "numpy_diagnostic":
            return fail("NUMPY_DIAGNOSTIC_ONLY: canonical torch_cpu metrics are required for SPSA updates")

        if not _evaluation_matches(plus, pair.plus) or not _evaluation_matches(minus, pair.minus):
            return fail("proposal identity, theta, pin, side, or pair binding mismatch")
        if not plus.rows:
            return fail("plus proposal returned no rows")
        reference_ids = {self.identity_reference_method_id, self.rms_reference_method_id}
        if {row.method_id for row in plus.rows} != {*reference_ids, pair.plus.method_id}:
            return fail("plus evaluation contains an unexpected method row")
        if {row.method_id for row in minus.rows} != {*reference_ids, pair.minus.method_id}:
            return fail("minus evaluation contains an unexpected method row")
        if tuple(sorted({r.sample_id for r in plus.rows if r.method_id == pair.plus.method_id})) != tuple(sorted(pair.sample_ids)):
            return fail("plus proposal does not cover its exact eight-clip batch")
        if tuple(sorted({r.sample_id for r in minus.rows if r.method_id == pair.minus.method_id})) != tuple(sorted(pair.sample_ids)):
            return fail("minus proposal does not cover its exact eight-clip batch")
        expected_rows_per_method = len(pair.sample_ids) * len(self.roster.model_ids)
        if any(sum(r.method_id == method for r in plus.rows) != expected_rows_per_method
               for method in (*reference_ids, pair.plus.method_id)):
            return fail("plus evaluation has a missing or extra planned row")
        if any(sum(r.method_id == method for r in minus.rows) != expected_rows_per_method
               for method in (*reference_ids, pair.minus.method_id)):
            return fail("minus evaluation has a missing or extra planned row")
        if (_baseline_fingerprint(plus.rows, self.identity_reference_method_id)
                != _baseline_fingerprint(minus.rows, self.identity_reference_method_id)):
            return fail("identity reference bytes differ between paired proposals")
        if (_baseline_fingerprint(plus.rows, self.rms_reference_method_id)
                != _baseline_fingerprint(minus.rows, self.rms_reference_method_id)):
            return fail("RMS reference bytes differ between paired proposals")
        if any(row.input_pins_sha256 != self.input_pins_sha256 for row in (*plus.rows, *minus.rows)):
            return fail("evaluation input pin digest differs from the controller")
        if any(row.artifact_sha256 != pair.plus.artifact_sha256 or row.binding_sha256 != pair.plus.binding_sha256
               for row in plus.rows if row.method_id == pair.plus.method_id):
            return fail("plus rows do not bind the exact proposal artifact/theta/pair")
        if any(row.artifact_sha256 != pair.minus.artifact_sha256 or row.binding_sha256 != pair.minus.binding_sha256
               for row in minus.rows if row.method_id == pair.minus.method_id):
            return fail("minus rows do not bind the exact proposal artifact/theta/pair")
        try:
            plus_spec = MethodSpec(pair.plus.method_id, _method_artifact_for(plus.rows, pair.plus.method_id),
                                   _method_binding_for(plus.rows, pair.plus.method_id))
            minus_spec = MethodSpec(pair.minus.method_id, _method_artifact_for(minus.rows, pair.minus.method_id),
                                    _method_binding_for(minus.rows, pair.minus.method_id))
        except FitSelectionError as exc:
            return fail(f"proposal artifact/binding is inconsistent: {exc}")
        methods = (*self.reference_specs, plus_spec, minus_spec)
        shared_rows = _rows_by_method(plus.rows, (*reference_ids, pair.plus.method_id)) + _rows_by_method(minus.rows, (pair.minus.method_id,))
        # Plus and minus share one exact support intersection. Plus rows are not
        # separately scored on a larger, favorable support set.
        family = _build_family(shared_rows, methods, self.roster, sample_ids=pair.sample_ids,
                               expected_input_pins_sha256=self.input_pins_sha256,
                               metric_backend=self.metric_backend)
        if family.errors:
            return fail("paired family has incomplete rows/common support: " + "; ".join(family.errors),
                        family.family_support_sha256)
        plus_effects = _effects_for_candidate(family, self.roster, candidate_method=pair.plus.method_id,
                                              reference_methods=(self.identity_reference_method_id,
                                                                 self.rms_reference_method_id),
                                              sample_ids=pair.sample_ids)
        minus_effects = _effects_for_candidate(family, self.roster, candidate_method=pair.minus.method_id,
                                               reference_methods=(self.identity_reference_method_id,
                                                                  self.rms_reference_method_id),
                                               sample_ids=pair.sample_ids)
        try:
            plus_loss = objective_from_effects(plus_effects, pair.plus.theta, self.theta_initial)
            minus_loss = objective_from_effects(minus_effects, pair.minus.theta, self.theta_initial)
        except FitSelectionError as exc:
            return fail(f"paired objective unavailable: {exc}", family.family_support_sha256)
        gradients = tuple((plus_loss["L_total"] - minus_loss["L_total"]) / (2.0 * pair.c_k * pair.direction[i])
                          for i in range(THETA_DIMENSION))
        updates = tuple(float(np.clip(-pair.a_k * gradients[i], -0.25, 0.25)) for i in range(THETA_DIMENSION))
        updated = tuple(before[i] + updates[i] for i in range(THETA_DIMENSION))
        if not all(math.isfinite(x) for x in (*gradients, *updates, *updated)):
            return fail("paired SPSA gradient/update is non-finite", family.family_support_sha256)
        self.theta = updated
        self.iteration += 1
        self._pending = None
        center_due = self.iteration in {5, 10, 15, 20}
        center_spec = self._center_spec(self.iteration) if center_due else None
        result = SPSATellResult("UPDATED", self.iteration, before, updated, plus_loss, minus_loss,
                                gradients, updates, center_due, center_spec, None, family.family_support_sha256,
                                self.metric_backend)
        self.history.append(result)
        return result

    def _center_spec(self, iteration: int) -> SnapshotSpec:
        if iteration not in {5, 10, 15, 20}:
            raise FitSelectionError("full Fit centers are predeclared only at iterations 5, 10, 15, and 20")
        method_id = f"fit40_d0_spsa_center_k{iteration:02d}"
        binding_sha = full_fit_candidate_binding_sha256(
            method_id=method_id, artifact_sha256=self.candidate_artifact_sha256,
            input_pins_sha256=self.input_pins_sha256, source_step=iteration,
            theta=self.theta, point_kind="spsa_center",
        )
        return SnapshotSpec(method_id, self.candidate_artifact_sha256, binding_sha,
                            iteration, self.theta, "spsa_center")


def _evaluation_matches(evaluation: ProposalEvaluation, proposal: SPSAProposal) -> bool:
    return (
        evaluation.proposal_id == proposal.proposal_id
        and evaluation.pair_id == proposal.pair_id
        and evaluation.side == proposal.side
        and evaluation.method_id == proposal.method_id
        and tuple(evaluation.theta) == proposal.theta
        and evaluation.theta_sha256 == proposal.theta_sha256
        and evaluation.artifact_sha256 == proposal.artifact_sha256
        and evaluation.binding_sha256 == proposal.binding_sha256
        and evaluation.input_pins_sha256 == proposal.input_pins_sha256
        and evaluation.batch_id == proposal.batch_id
    )


def _rows_by_method(rows: Sequence[FitScoreRow], method_ids: Sequence[str]) -> list[FitScoreRow]:
    return [row for row in rows if row.method_id in method_ids]


def _baseline_fingerprint(rows: Sequence[FitScoreRow], method_id: str) -> str:
    chosen = [r for r in rows if r.method_id == method_id]
    payload = []
    for row in sorted(chosen, key=lambda x: (x.sample_id, x.model_id, x.row_id)):
        payload.append({"row_id": row.row_id, "sample_id": row.sample_id, "speaker_id": row.speaker_id,
                        "model_id": row.model_id, "seed": row.seed, "status": row.status,
                        "natural_pcm_sha256": row.natural_pcm_sha256,
                        "source_media_kind": row.source_media_kind,
                        "source_video_sha256": row.source_video_sha256,
                        "source_face_sha256": row.source_face_sha256,
                        "keys": [list(k) for k in row.query_keys], "valid": list(row.valid_query_mask),
                        "matrix_sha256": _matrix_sha(row), "artifact_sha256": row.artifact_sha256,
                        "binding_sha256": row.binding_sha256, "input_pins_sha256": row.input_pins_sha256})
    return _sha256(_canonical(payload))


def _method_artifact_for(rows: Sequence[FitScoreRow], method_id: str) -> str:
    values = {r.artifact_sha256 for r in rows if r.method_id == method_id}
    if len(values) != 1:
        raise FitSelectionError(f"method {method_id} does not have one exact artifact pin")
    return next(iter(values))


def _method_binding_for(rows: Sequence[FitScoreRow], method_id: str) -> str:
    values = {r.binding_sha256 for r in rows if r.method_id == method_id}
    if len(values) != 1:
        raise FitSelectionError(f"method {method_id} does not have one exact binding pin")
    return next(iter(values))


__all__ = [
    "CandidateDecision", "FitRoster", "FitScoreRow", "FitSelectionError", "MethodEffects",
    "MethodSpec", "METRIC_BACKENDS", "METRIC_PARITY_ATOL", "METRIC_PARITY_RTOL",
    "RMS_REFERENCE_METHOD_IDS",
    "ProposalEvaluation", "SCHEMA", "SPSAController", "SPSAProposal",
    "SPSAProposalPair", "SPSATellResult", "SnapshotSelection", "SnapshotSpec",
    "canonical_input_pins_sha256", "expected_score_accounting", "full_fit_candidate_binding_sha256",
    "objective_from_effects",
    "official_stagea_score_row", "select_snapshot_family", "spsa_step_sizes",
]
