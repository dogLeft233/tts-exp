from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config


def selected_cores(
    sample_id: str, d_drivers: list[dict[str, Any]], masks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [row for row in d_drivers if str(row["sample_id"]) == sample_id]
    if len(rows) != 12:
        raise rt.ProtocolError(f"expected 12 D drivers: {sample_id}")
    identity_sets = []
    for row in rows:
        current = []
        for used in row.get("used_masks", []):
            item = masks.get(str(used["mask_sha256"]))
            if item is None:
                raise rt.ProtocolError(f"unknown mask: {sample_id}")
            if int(used["global_start_frame"]) != int(
                item["natural_core_start_frame"]
            ) or int(used["global_end_frame"]) != int(item["natural_core_end_frame"]):
                raise rt.ProtocolError(f"mask identity mismatch: {sample_id}")
            if (
                str(item.get("label", "")) not in config.EXCLUDED_LABELS
                and int(item["natural_core_end_frame"])
                - int(item["natural_core_start_frame"])
                >= 5
            ):
                current.append(
                    (
                        str(used["mask_sha256"]),
                        int(used["global_start_frame"]),
                        int(used["global_end_frame"]),
                        str(item.get("label", "")),
                    )
                )
        identity_sets.append(current)
    first = sorted(set(identity_sets[0]))
    if any(sorted(set(value)) != first for value in identity_sets[1:]):
        raise rt.ProtocolError(f"selected mask identities differ: {sample_id}")
    cores = [
        {"mask_sha256": key, "start": start, "end": end, "label": label}
        for key, start, end, label in first
    ]
    cores.sort(key=lambda item: (item["start"], item["end"], item["mask_sha256"]))
    if not cores:
        raise rt.ProtocolError(
            f"INPUT_DEGENERATE: empty selected core set: {sample_id}"
        )
    for left, right in pairwise(cores):
        if int(right["start"]) < int(left["end"]):
            raise rt.ProtocolError(f"selected cores overlap: {sample_id}")
    return cores


def _support_columns() -> list[np.ndarray]:
    chunks = []
    index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > 308:
            chunks.append(np.arange(292, 308, dtype=np.int64))
            break
        chunks.append(np.arange(start, start + 16, dtype=np.int64))
        index += 1
    return chunks


def exposure_diagnostics(
    natural: np.ndarray,
    candidates: dict[str, Any],
    cores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report frozen U exposure without applying a record-level gate.

    Exposure is a source-group eligibility rule.  This function deliberately
    reports a record with no exposed row instead of raising, so the caller can
    aggregate paired records before deciding whether the queue is supported.
    """
    support = _support_columns()
    positive = sorted(
        {
            int(item["start"]) + j
            for item in cores
            for j, value in enumerate(
                np.minimum(
                    1.0,
                    np.minimum(
                        np.arange(
                            int(item["end"]) - int(item["start"]), dtype=np.float64
                        )
                        / 2.0,
                        (
                            int(item["end"])
                            - int(item["start"])
                            - 1
                            - np.arange(
                                int(item["end"]) - int(item["start"]), dtype=np.float64
                            )
                        )
                        / 2.0,
                    ),
                )
            )
            if float(value) > 0
        }
    )
    positive_set = set(positive)
    exposed_rows = [
        int(row)
        for row in config.U_ROWS
        if positive_set.intersection(
            set(np.concatenate(support[row : row + 5]).tolist())
        )
    ]
    natural32 = np.asarray(natural, dtype=np.float32)
    changed_columns: dict[str, list[int]] = {}
    perturbation_norms: dict[str, dict[str, float]] = {}
    for arm in ("PHONE_CORE", "GENERIC_CORE"):
        candidate = np.asarray(candidates[arm], dtype=np.float32)
        changed = np.any(candidate != natural32, axis=0)
        changed_columns[arm] = np.flatnonzero(changed).astype(int).tolist()
        row_norms: dict[str, float] = {}
        delta = candidate.astype(np.float64) - np.asarray(natural, dtype=np.float64)
        for row in config.U_ROWS:
            columns = np.concatenate(support[row : row + 5])
            row_norms[str(row)] = float(np.linalg.norm(delta[:, columns]))
        perturbation_norms[arm] = row_norms
    return {
        "positive_columns": positive,
        "exposed_rows": exposed_rows,
        "record_exposed": bool(exposed_rows),
        "changed_columns": changed_columns,
        "u_row_perturbation_norms": perturbation_norms,
    }


def construct_candidates(
    mel: np.ndarray, cores: list[dict[str, Any]]
) -> dict[str, Any]:
    natural = np.asarray(mel, dtype=np.float64)
    if natural.shape != (80, 308) or not np.isfinite(natural).all():
        raise rt.ProtocolError("natural mel shape/finite check failed")
    if not cores:
        natural32 = natural.astype(np.float32)
        return {
            "N": natural32,
            "PHONE_CORE": natural32.copy(),
            "GENERIC_CORE": natural32.copy(),
            "metadata": {
                "cores": [],
                "norm_control_valid": False,
                "postclip_generic_phone_ratio": None,
            },
        }
    r_phone = np.zeros_like(natural)
    r_generic = np.zeros_like(natural)
    weights: list[dict[str, Any]] = []
    for core in cores:
        start, end = int(core["start"]), int(core["end"])
        length = end - start
        mu = np.mean(natural[:, start:end], axis=1, dtype=np.float64)
        w = np.minimum(
            1.0,
            np.minimum(
                np.arange(length, dtype=np.float64) / 2.0,
                (length - 1 - np.arange(length, dtype=np.float64)) / 2.0,
            ),
        )
        r_phone[:, start:end] = w[None, :] * (mu[:, None] - natural[:, start:end])
        weights.append(
            {
                "mask_sha256": core["mask_sha256"],
                "start": start,
                "end": end,
                "label": core["label"],
                "weights": w.tolist(),
                "mu": mu.tolist(),
            }
        )
    padded = np.pad(natural, ((0, 0), (2, 2)), mode="reflect")
    smooth = (
        padded[:, 0:308]
        + 4.0 * padded[:, 1:309]
        + 6.0 * padded[:, 2:310]
        + 4.0 * padded[:, 3:311]
        + padded[:, 4:312]
    ) / 16.0
    for item in weights:
        start, end = int(item["start"]), int(item["end"])
        w = np.asarray(item["weights"], dtype=np.float64)
        r_generic[:, start:end] = w[None, :] * (
            smooth[:, start:end] - natural[:, start:end]
        )
    n_phone = float(np.linalg.norm(r_phone))
    n_generic = float(np.linalg.norm(r_generic))
    if n_phone <= 1e-12 or n_generic <= 1e-12:
        raise rt.ProtocolError("INPUT_DEGENERATE: zero core residual")
    r_generic *= n_phone / n_generic
    maximum = max(float(np.max(np.abs(r_phone))), float(np.max(np.abs(r_generic))))
    a = min(0.25, 0.5 / maximum) if maximum > 0 else 0.25
    phone = np.clip(natural + a * r_phone, -4.0, 4.0).astype(np.float32)
    generic = np.clip(natural + a * r_generic, -4.0, 4.0).astype(np.float32)
    phone_delta = phone.astype(np.float64) - natural
    generic_delta = generic.astype(np.float64) - natural
    phone_norm = float(np.linalg.norm(phone_delta))
    generic_norm = float(np.linalg.norm(generic_delta))
    ratio = float(generic_norm / phone_norm) if phone_norm > 0 else 0.0
    if not (0.95 <= ratio <= 1.05) or phone_norm <= 0 or generic_norm <= 0:
        raise rt.ProtocolError("postclip norm control invalid")
    outside = np.ones(308, dtype=bool)
    for item in weights:
        outside[int(item["start"]) : int(item["end"])] = False
    if not np.array_equal(
        phone[:, outside], natural.astype(np.float32)[:, outside]
    ) or not np.array_equal(
        generic[:, outside], natural.astype(np.float32)[:, outside]
    ):
        raise rt.ProtocolError("core-external columns changed")
    candidates = {
        "N": natural.astype(np.float32),
        "PHONE_CORE": phone,
        "GENERIC_CORE": generic,
    }
    metadata = {
        "cores": weights,
        "norm_phone_preclip": n_phone,
        "norm_generic_preclip": n_generic,
        "a": float(a),
        "norm_phone_postclip": phone_norm,
        "norm_generic_postclip": generic_norm,
        "postclip_generic_phone_ratio": ratio,
        "norm_control_valid": True,
    }
    metadata.update(exposure_diagnostics(natural, candidates, cores))
    candidates["metadata"] = metadata
    return candidates
