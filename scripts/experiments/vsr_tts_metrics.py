"""Pure metrics and view construction for the frozen VSR/TTS pilot.

The runner owns file I/O and the AVSR model.  This module deliberately keeps
the experiment's mathematical pieces small enough to test with tiny tensors:
text normalisation, CTC negative log likelihood, content margins, temporal
controls, greedy CER, and paired bootstrap summaries.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


BLANK_ID = 0
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_DRAWS = 20_000


def normalize_text(text: str) -> str:
    """Apply the frozen Chinese text normalisation rule."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def token_ids(text: str, char_list: Sequence[str]) -> List[int]:
    """Map a normalised text to model character IDs without silent fallback."""

    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("empty normalised text")
    index = {str(symbol): i for i, symbol in enumerate(char_list)}
    unknown: List[str] = []
    result: List[int] = []
    for char in normalized:
        value = index.get(char)
        if value is None or value == BLANK_ID or str(char_list[value]) in {"<unk>", "<eos>"}:
            unknown.append(char)
        else:
            result.append(int(value))
    if unknown:
        raise ValueError("OOV or reserved characters: {}".format("".join(sorted(set(unknown)))))
    if not result:
        raise ValueError("text has no usable token")
    return result


def adjacent_repeated_character_count(target: Sequence[int]) -> int:
    values = [int(value) for value in target]
    return sum(left == right for left, right in zip(values, values[1:]))


def _as_logp(logp: Any) -> torch.Tensor:
    tensor = torch.as_tensor(logp, dtype=torch.float64, device="cpu")
    if tensor.ndim == 3:
        if tensor.shape[1] != 1:
            raise ValueError("logp batch dimension must have size one")
        tensor = tensor[:, 0, :]
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 1:
        raise ValueError("logp must have shape [T, C]")
    if not torch.isfinite(tensor).all():
        raise ValueError("logp contains non-finite values")
    return tensor


def ctc_nll(logp: Any, target: Sequence[int], blank: int = BLANK_ID) -> float:
    """Return a finite, unnormalised CTC NLL using the frozen loss contract.

    Length normalisation is intentionally outside this function.  The caller
    divides exactly once by the number of target characters.
    """

    values = _as_logp(logp)
    target_values = [int(value) for value in target]
    if not target_values:
        raise ValueError("empty CTC target")
    if any(value < 0 or value >= values.shape[1] for value in target_values):
        raise ValueError("CTC target ID is outside the vocabulary")
    if any(value == int(blank) for value in target_values):
        raise ValueError("CTC target contains blank")
    required = len(target_values) + adjacent_repeated_character_count(target_values)
    if values.shape[0] < required:
        raise ValueError(
            "CTC target is unreachable: T={} < required={}".format(values.shape[0], required)
        )
    target_tensor = torch.tensor(target_values, dtype=torch.long)
    input_lengths = torch.tensor([values.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([len(target_values)], dtype=torch.long)
    loss = F.ctc_loss(
        values[:, None, :],
        target_tensor,
        input_lengths,
        target_lengths,
        blank=int(blank),
        reduction="sum",
        zero_infinity=False,
    )
    result = float(loss.item())
    if not math.isfinite(result):
        raise ValueError("CTC loss is non-finite")
    return result


def length_normalized_nll(logp: Any, target: Sequence[int], blank: int = BLANK_ID) -> float:
    values = [int(value) for value in target]
    if not values:
        raise ValueError("empty target")
    return ctc_nll(logp, values, blank=blank) / float(len(values))


def content_margin(target_loss: float, decoy_losses: Iterable[float]) -> float:
    """Return M(V) = mean(decoy loss) - target loss; higher is better."""

    target_value = float(target_loss)
    decoys = [float(value) for value in decoy_losses]
    if not decoys or not math.isfinite(target_value) or not all(math.isfinite(v) for v in decoys):
        raise ValueError("content margin needs finite target and non-empty finite decoys")
    return float(np.mean(np.asarray(decoys, dtype=np.float64)) - target_value)


def greedy_decode(logp: Any, blank: int = BLANK_ID) -> List[int]:
    values = _as_logp(logp).numpy()
    path = np.argmax(values, axis=1).astype(np.int64).tolist()
    collapsed: List[int] = []
    previous: Optional[int] = None
    for value in path:
        if previous is not None and value == previous:
            continue
        previous = value
        if value != int(blank):
            collapsed.append(int(value))
    return collapsed


def levenshtein_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    ref = [int(value) for value in reference]
    hyp = [int(value) for value in hypothesis]
    previous = list(range(len(hyp) + 1))
    for i, ref_value in enumerate(ref, start=1):
        current = [i]
        for j, hyp_value in enumerate(hyp, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_value != hyp_value),
                )
            )
        previous = current
    return int(previous[-1])


def greedy_cer(logp: Any, target: Sequence[int], blank: int = BLANK_ID) -> float:
    values = [int(value) for value in target]
    if not values:
        raise ValueError("empty CER target")
    return float(levenshtein_distance(values, greedy_decode(logp, blank=blank)) / float(len(values)))


def _canonical_video_tensor(native: Any) -> np.ndarray:
    values = np.asarray(native, dtype=np.float32)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] <= 0:
        raise ValueError("video tensor must have shape [1, T, H, W] or [T, H, W]")
    if not np.isfinite(values).all():
        raise ValueError("video tensor contains non-finite values")
    return values.copy()


def _linear_time_resample(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("resampled length must be positive")
    source_length = values.shape[1]
    if source_length == length:
        return values.copy()
    if source_length == 1:
        return np.repeat(values, length, axis=1)
    source = np.linspace(0.0, 1.0, source_length, dtype=np.float64)
    target = np.linspace(0.0, 1.0, length, dtype=np.float64)
    result = np.empty((1, length, values.shape[2], values.shape[3]), dtype=np.float32)
    for row in range(values.shape[2]):
        for col in range(values.shape[3]):
            result[0, :, row, col] = np.interp(target, source, values[0, :, row, col]).astype(np.float32)
    return result


def make_views(native: Any, matched_length: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Construct the five frozen views from one already-cropped tensor."""

    values = _canonical_video_tensor(native)
    length = int(matched_length if matched_length is not None else values.shape[1])
    matched = _linear_time_resample(values, length)
    frozen_frame = values[:, values.shape[1] // 2 : values.shape[1] // 2 + 1]
    matched_frozen_frame = matched[:, matched.shape[1] // 2 : matched.shape[1] // 2 + 1]
    return {
        "native": values.copy(),
        "frozen": np.repeat(frozen_frame, values.shape[1], axis=1).astype(np.float32),
        "reversed": values[:, ::-1].copy(),
        "matched": matched.astype(np.float32),
        "matched_frozen": np.repeat(matched_frozen_frame, matched.shape[1], axis=1).astype(np.float32),
    }


def build_decoys(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build fixed length-matched decoys before any model output is inspected.

    Records must contain ``id``, ``normalized_text`` and a usable ``token_ids``
    sequence.  The function returns JSON-friendly copies keyed by string ID.
    """

    prepared = []
    for row in records:
        sample_id = int(row["id"])
        text = str(row["normalized_text"])
        ids = row.get("token_ids")
        prepared.append((sample_id, text, None if ids is None else [int(v) for v in ids]))
    output: Dict[str, List[Dict[str, Any]]] = {}
    for sample_id, text, ids in prepared:
        if ids is None:
            output[str(sample_id)] = []
            continue
        candidates = [
            (abs(len(other_text) - len(text)), other_id, other_text, other_ids)
            for other_id, other_text, other_ids in prepared
            if other_id != sample_id and other_text != text and other_ids is not None
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[: int(count)]
        output[str(sample_id)] = [
            {"id": int(other_id), "normalized_text": other_text, "token_ids": list(other_ids)}
            for _, other_id, other_text, other_ids in chosen
        ]
    return output


def _bootstrap(values: Mapping[str, float]) -> Dict[str, Any]:
    labels = sorted(str(key) for key in values)
    finite = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    if len(finite) == 0 or not np.isfinite(finite).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(finite), size=(BOOTSTRAP_DRAWS, len(finite)), dtype=np.int64)
    means = finite[indices].mean(axis=1, dtype=np.float64)
    return {
        "status": "COMPLETE",
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "positive_fraction": float(np.mean(finite > 0.0)),
        "values": {label: float(value) for label, value in zip(labels, finite)},
        "count": int(len(finite)),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def paired_summary(rows: Sequence[Mapping[str, Any]], min_pairs: int = 10) -> Dict[str, Any]:
    """Summarise paired G/B/Gmatched/R values with one pair as the unit."""

    metrics = ("g", "b", "gmatched", "r")
    ids = sorted(str(row["id"]) for row in rows)
    result: Dict[str, Any] = {
        "pair_count": len(ids),
        "required_pair_count": int(min_pairs),
        "status": "COMPLETE" if len(ids) >= int(min_pairs) else "NOT_ESTIMABLE",
        "metrics": {},
    }
    for metric in metrics:
        values: Dict[str, float] = {}
        for row in rows:
            if metric not in row or row[metric] is None or not math.isfinite(float(row[metric])):
                continue
            values[str(row["id"])] = float(row[metric])
        if len(values) != len(ids):
            result["metrics"][metric] = {"status": "INCOMPLETE", "count": len(values)}
        elif len(values) < int(min_pairs):
            result["metrics"][metric] = {"status": "NOT_ESTIMABLE", "count": len(values)}
        else:
            result["metrics"][metric] = _bootstrap(values)
    return result


def decision_status(
    *,
    engineering_ok: bool,
    calibration: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> str:
    """Apply the spec's fixed precedence to already-computed summaries."""

    if not engineering_ok or int(paired.get("pair_count", 0)) < 10:
        return "ENGINEERING_INSUFFICIENT"
    if calibration.get("status") != "CALIBRATED_ON_COHORT":
        return "INCONCLUSIVE_VSR_VALIDITY"
    metrics = paired.get("metrics", {})
    g = metrics.get("g", {})
    b = metrics.get("b", {})
    gm = metrics.get("gmatched", {})
    if g.get("status") != "COMPLETE":
        return "INCONCLUSIVE_PATTERN"
    if float(g.get("mean", 0.0)) > 0.0 and float(b.get("mean", 0.0)) <= 0.0:
        return "CONTROL_DRIVEN_DIFFERENCE"
    if float(g.get("mean", 0.0)) > 0.0 and float(gm.get("mean", 0.0)) <= 0.0:
        return "DURATION_SENSITIVE"
    low, high = [float(value) for value in g.get("ci95", [0.0, 0.0])]
    if low > 0.0 and float(b.get("mean", 0.0)) > 0.0 and float(gm.get("mean", 0.0)) > 0.0:
        return "EXPLORATORY_VISUAL_CONTENT_SUPPORT"
    if low <= 0.0 <= high:
        return "NO_CLEAR_PAIRED_GAIN"
    if high < 0.0:
        return "EXPLORATORY_REVERSE_EFFECT"
    return "INCONCLUSIVE_PATTERN"
