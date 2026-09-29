"""Score-independent record lock and strict TTS-only interface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ..mfa_linear_real_video_sync.config import SEGMENT_SAMPLES
from ..mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from ..mfa_linear_real_video_sync.prepare import PreparedRecord
from ..mfa_linear_real_video_sync.protocol import (
    extract_official_bgr_frames,
    load_mfa_linear_waveform,
    sha256_file,
    write_json_once,
)
from ..mfa_linear_real_video_sync.syncnet_loss import (
    audio_embeddings,
    cached_visual_embeddings,
    official_syncnet_distance_curve,
)
from ..mfa_linear_sync_transfer.protocol import (
    calibrate_and_materialize_cohort,
    scan_structural_eligibility,
)
from .config import (
    BLOCKED_EXISTING_DATA,
    EVAL_RECORD_COUNT,
    EXPERIMENT,
    OUTCOME_SELECTION_VIOLATION,
    SELECTION_SALT,
    TRAIN_RECORD_COUNT,
)


class ExistingDataProtocolError(ValueError):
    """Raised when the existing-data experiment contract fails."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def selection_key(source_group: str, sample_id: str, *, role: str) -> str:
    if role not in {"train", "evaluation"}:
        raise ValueError("selection role must be train or evaluation")
    return sha256_bytes(f"{SELECTION_SALT}\0{role}\0{source_group}\0{sample_id}".encode("utf-8"))


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("natural_audio_values", None)
    return result


def _manifest_row(row: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    result = _public_row(row)
    result["selection_key"] = selection_key(str(row["source_group"]), str(row["sample_id"]), role=role)
    return result


def _row_key(row: Mapping[str, Any], *, role: str) -> tuple[str, bytes, bytes]:
    group = str(row["source_group"])
    sample_id = str(row["sample_id"])
    return selection_key(group, sample_id, role=role), group.encode(), sample_id.encode()


def select_cohort(
    calibrated_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    excluded_source_groups: Iterable[str],
) -> dict[str, Any]:
    excluded = {str(group) for group in excluded_source_groups}
    eligible = [
        dict(row) for row in calibrated_rows
        if str(row.get("protocol_split")) == "train"
        and str(row.get("source_group")) not in excluded
    ]
    if len(eligible) != TRAIN_RECORD_COUNT + EVAL_RECORD_COUNT:
        raise ExistingDataProtocolError(
            f"{BLOCKED_EXISTING_DATA}: expected {TRAIN_RECORD_COUNT + EVAL_RECORD_COUNT} eligible records, got {len(eligible)}"
        )
    groups = sorted({str(row["source_group"]) for row in eligible}, key=lambda group: (selection_key(group, "", role="evaluation"), group.encode()))
    if len(groups) < EVAL_RECORD_COUNT:
        raise ExistingDataProtocolError(f"{BLOCKED_EXISTING_DATA}: fewer than {EVAL_RECORD_COUNT} eligible source groups")
    evaluation_groups = groups[:EVAL_RECORD_COUNT]
    representatives: dict[str, dict[str, Any]] = {}
    for row in eligible:
        group = str(row["source_group"])
        if group in evaluation_groups:
            old = representatives.get(group)
            if old is None or _row_key(row, role="evaluation") < _row_key(old, role="evaluation"):
                representatives[group] = row
    evaluation = [representatives[group] for group in evaluation_groups]
    eval_ids = {str(row["sample_id"]) for row in evaluation}
    training = sorted(
        (row for row in eligible if str(row["sample_id"]) not in eval_ids),
        key=lambda row: _row_key(row, role="train"),
    )
    if len(training) != TRAIN_RECORD_COUNT:
        raise ExistingDataProtocolError(f"{BLOCKED_EXISTING_DATA}: training denominator changed")
    training_groups = sorted({str(row["source_group"]) for row in training})
    overlap = sorted(set(training_groups).intersection(evaluation_groups))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "manifest_type": "mfa_linear_sync_existing_data_record_heldout",
        "experiment": EXPERIMENT,
        "selection_salt": SELECTION_SALT,
        "selection_uses_outcomes": False,
        "sealed_splits_accessed": False,
        "heldout_definition": "record-heldout within fit-only LRS3; source-group overlap allowed",
        "excluded_source_groups": sorted(excluded),
        "eligible_universe": [_public_row(row) for row in eligible],
        "exclusions": [dict(item) for item in exclusions],
        "training": [_manifest_row(row, role="train") for row in training],
        "evaluation": [_manifest_row(row, role="evaluation") for row in evaluation],
        "selected_evaluation_sample_ids": [str(row["sample_id"]) for row in evaluation],
        "training_source_groups": training_groups,
        "evaluation_source_groups": evaluation_groups,
        "source_group_overlap": overlap,
        "training_record_count": TRAIN_RECORD_COUNT,
        "evaluation_record_count": EVAL_RECORD_COUNT,
    }
    write_json_once(Path(output_root) / "01_data_lock/manifest.json", manifest)
    return manifest


def prepare_training_records(
    rows: Sequence[Mapping[str, Any]],
    syncnet: torch.nn.Module,
    output_root: str | Path,
    *,
    device: torch.device,
) -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    output = Path(output_root).resolve()
    for row in rows:
        sid = str(row["sample_id"])
        scoring = extract_official_bgr_frames(
            row["visual"]["path"], output / "02_training/official_visual_frames" / sid
        )
        visual = cached_visual_embeddings(syncnet, torch.from_numpy(scoring).to(device)).detach()
        waveform, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(device)[None, None]
        with torch.no_grad():
            pristine = official_syncnet_distance_curve(audio_embeddings(syncnet, tensor), visual).detach()
        record = PreparedRecord(sid, tensor, visual, int(row["natural_target"]["target_offset"]), pristine)
        record.validate()
        records.append(record)
    return records


class FrozenTTSOnlyAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor) or tuple(waveform.shape) != (1, 1, SEGMENT_SAMPLES):
            raise ExistingDataProtocolError(
                "NATURAL_OR_SIDE_CHANNEL_LEAKAGE: adapter accepts only [1,1,61440] waveform"
            )
        if not torch.is_floating_point(waveform):
            raise ExistingDataProtocolError("adapter waveform must be floating point")
        output = self._model(waveform)
        if tuple(output.shape) != tuple(waveform.shape):
            raise ExistingDataProtocolError("candidate waveform shape does not match adapter input")
        return output

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        return self._model.state_dict(*args, **kwargs)
