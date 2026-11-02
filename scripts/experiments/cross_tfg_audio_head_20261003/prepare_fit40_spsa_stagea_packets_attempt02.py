#!/usr/bin/env python3
"""Build immutable Stage-A execution plans for one paired Fit-only SPSA step.

This CPU-side packet builder consumes a complete, byte-verified six-method
Fit40 family through ``fit_only_stagea_ingest_attempt01`` and the existing
``fit_only_spsa_selection_attempt06`` controller. It emits only the requested
plus/minus proposal rows (24 each) or one full-center row family (120 rows).
Identity and ``rms_fixed`` rows are referenced by row ID from the verified
initial 720-row family; they are never scheduled for another render here.

It does not render, score, initialize from partial evidence, change the SPSA
objective, or choose a final candidate. A separate native Stage-A execution
adapter must consume these single-method packets and bind its live runtime and
capacity receipts before any model call.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fit_only_spsa_selection_attempt06 import (
    RENDERER_SEED,
    SPSAController,
    SPSAProposalPair,
    SnapshotSpec,
    full_fit_candidate_binding_sha256,
    _canonical,
)
from fit_only_stagea_ingest_attempt01 import FamilySelection


SCHEMA = "fit40_spsa_stagea_execution_packet_v1"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
BASE_METHOD_IDS = ("identity", "rms_fixed")
PROPOSAL_METHOD_KIND = "fit40_spsa_proposal"
CENTER_METHOD_KIND = "fit40_spsa_center"
PROPOSAL_ROW_COUNT = 8 * 3
CENTER_ROW_COUNT = 40 * 3


class PacketError(ValueError):
    """A source or row packet does not match the verified Fit/SPSA contract."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))


def _read_pinned_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise PacketError(f"refusing symlinked {label}: {path}")
    raw = path.read_bytes()
    actual = _sha(raw)
    if actual != expected_sha256.lower():
        raise PacketError(f"{label} raw SHA-256 mismatch: {actual}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise PacketError(f"{label} must be a JSON object")
    return value, raw


def _file_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PacketError(f"expected a non-symlink regular file: {path}")
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": _sha(raw)}


def _finite_theta(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise PacketError(f"{label} must contain the frozen nine head parameters")
    try:
        theta = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise PacketError(f"{label} contains a nonnumeric parameter") from exc
    if not all(math.isfinite(item) for item in theta):
        raise PacketError(f"{label} contains a nonfinite parameter")
    return theta


def _validate_family_and_plan(
    base_plan: Mapping[str, Any],
    family_selection: FamilySelection,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]], str]:
    family = family_selection.family
    selection = family_selection.selection
    if (not family.complete_transport
            or family_selection.status not in {"ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", "NOT_ADMITTED"}
            or selection is None
            or selection.metric_backend != "torch_cpu"
            or family_selection.spsa_seed_method_id is None):
        raise PacketError("pair execution requires a complete verified 720-row torch_cpu Fit family")
    accounting = selection.accounting
    if (accounting.get("unique_official_score_rows") != 720
            or accounting.get("observed_unique_rows") != 720
            or accounting.get("failed_score_rows") != 0
            or accounting.get("missing_score_rows") != 0
            or family.missing_row_ids or family.producer_failure_rows
            or family.integrity_errors or family.input_artifact_failures):
        raise PacketError("initial Fit family has failed, missing, or unverifiable evidence")
    if (base_plan.get("schema_version") != PLAN_SCHEMA or base_plan.get("status") != "FROZEN"
            or base_plan.get("plan_sha256") != _self_hash(base_plan, "plan_sha256")):
        raise PacketError("base plan is not a self-sealed frozen Stage-A plan")
    if base_plan.get("scientific_input_pins_sha256") != family.input_pins_sha256:
        raise PacketError("base plan and verified selector family have different scientific input pins")
    if base_plan.get("scientific_input_pins") != family.input_pins:
        raise PacketError("base plan does not carry the exact verified selector input pins")
    if base_plan.get("execution", {}).get("primary_score") != "native_roi_original_pcm16":
        raise PacketError("base plan does not bind the official original-PCM endpoint")
    if base_plan.get("generation_repeat_qc") != []:
        raise PacketError("SPSA packet source plan must leave repeat diagnostics out of scored rows")
    if base_plan.get("supports") != [
        {"support_id": "full", "trim_each_end": 0, "minimum_queries": 1, "primary": True},
        {"support_id": "guard20", "trim_each_end": 20, "minimum_queries": 41, "primary": False},
    ]:
        raise PacketError("base plan must retain the frozen full/guard20 support definitions")
    models = base_plan.get("models")
    methods = base_plan.get("methods")
    rows = base_plan.get("rows")
    if (not isinstance(models, list) or [m.get("model_id") for m in models] != list(family.roster.model_ids)
            or not isinstance(methods, list) or len(methods) != 6
            or [m.get("method_id") for m in methods[:2]] != list(BASE_METHOD_IDS)
            or not isinstance(rows, list) or len(rows) != 720):
        raise PacketError("base plan must bind exactly 40 clips × 3 native models × six initial methods")
    if base_plan.get("protocol", {}).get("syncnet", {}).get("anchor_column") != 15:
        raise PacketError("base plan does not retain the exact zero-lag anchor column 15")
    if any(row.get("split") != "fit" or row.get("seed") != RENDERER_SEED
           or row.get("score_audio") != row.get("natural_pcm") for row in rows):
        raise PacketError("base plan contains a non-Fit row, altered seed, or non-original score audio")

    method_by_id = {method.get("method_id"): method for method in methods}
    if len(method_by_id) != 6 or None in method_by_id:
        raise PacketError("base plan has duplicate or empty method IDs")
    expected_candidate_ids = tuple(spec.method_id for spec in family.candidates)
    if tuple(method.get("method_id") for method in methods[2:]) != expected_candidate_ids:
        raise PacketError("base plan snapshot roster differs from the verified selector family")
    for spec in family.candidates:
        if method_by_id[spec.method_id].get("checkpoint_sha256") != spec.artifact_sha256:
            raise PacketError(f"snapshot source artifact differs from verified selector family: {spec.method_id}")
    syncnet_protocol = base_plan.get("protocol", {}).get("syncnet")
    if not isinstance(syncnet_protocol, Mapping) or _sha(_canonical(syncnet_protocol)) != family.score_protocol_sha256:
        raise PacketError("base plan SyncNet source-clock protocol differs from the verified selector family")
    if (method_by_id["identity"].get("config", {}).get("method") != "identity"
            or method_by_id["rms_fixed"].get("config", {}).get("method") != "rms"
            or method_by_id["rms_fixed"].get("config", {}).get("target_rms") != 0.0376838172675746):
        raise PacketError("cached baseline methods differ from exact identity/rms_fixed producer IDs")

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    sample_to_speaker: dict[str, str] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("model_id")), str(row.get("method_id")))
        if key in rows_by_key or key[2] not in method_by_id:
            raise PacketError("base plan has a duplicate row or an undeclared method")
        rows_by_key[key] = row
        prior = sample_to_speaker.setdefault(key[0], str(row.get("speaker_id")))
        if prior != row.get("speaker_id"):
            raise PacketError("base plan changes speaker identity across methods/models")
    if sample_to_speaker != dict(family.roster.sample_to_speaker):
        raise PacketError("base plan roster differs from the independently ingested 40-clip Fit roster")
    expected = {
        (sample_id, model_id, method_id)
        for sample_id in sample_to_speaker
        for model_id in family.roster.model_ids
        for method_id in method_by_id
    }
    if set(rows_by_key) != expected:
        raise PacketError("base plan does not have the exact 720 context/method rows")

    for sample_id in sample_to_speaker:
        for model_id in family.roster.model_ids:
            identity = rows_by_key[(sample_id, model_id, "identity")]
            natural = identity.get("natural_pcm")
            for method_id in method_by_id:
                candidate = rows_by_key[(sample_id, model_id, method_id)]
                if candidate.get("natural_pcm") != natural or candidate.get("score_audio") != natural:
                    raise PacketError("initial methods do not share exact natural PCM and original score audio")
                for context_key in ("speaker_id", "face_id", "source_face", "source_video", "seed"):
                    if candidate.get(context_key) != identity.get(context_key):
                        raise PacketError(f"base plan context differs from cached identity at {context_key}")

    input_sha = family.input_pins_sha256
    if len(family.reference_specs) != 2 or {x.method_id for x in family.reference_specs} != set(BASE_METHOD_IDS):
        raise PacketError("verified family does not expose the frozen identity and raw rms_fixed references")
    return dict(base_plan), rows_by_key, input_sha


def _resolve_seed_method(family_selection: FamilySelection,
                         method_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    seed_method_id = family_selection.spsa_seed_method_id
    method = method_by_id.get(seed_method_id)
    if not isinstance(method, Mapping) or method.get("kind") != "fit_parameter_snapshot":
        raise PacketError("SPSA seed must be one actual persisted Fit snapshot method")
    config = method.get("config")
    if not isinstance(config, Mapping) or config.get("method") != "fit40_parameter_snapshot":
        raise PacketError("selected seed method is not the frozen persisted-parameter transform")
    artifact = method.get("checkpoint_sha256")
    if family_selection.family.candidates:
        spec = next((entry for entry in family_selection.family.candidates
                     if entry.method_id == seed_method_id), None)
        if spec is None or spec.artifact_sha256 != artifact:
            raise PacketError("selected seed artifact differs from the verified family snapshot")
    return seed_method_id, dict(method)


def _seed_snapshot_lineage(family_selection: FamilySelection,
                           seed_method: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the selected seed to the actual materialized snapshot input pins.

    The Stage-A method's ``checkpoint_sha256`` identifies the snapshot's
    canonical artifact payload.  It is not the raw byte hash of the JSON
    record.  The raw record and snapshot-set index hashes therefore come from
    the verified plan input-pin mapping and are recorded separately.
    """
    family = family_selection.family
    seed_id = seed_method.get("method_id")
    spec = next((item for item in family.candidates if item.method_id == seed_id), None)
    if spec is None or spec.point_kind != "snapshot" or spec.source_step not in {30, 60, 90, 120}:
        raise PacketError("selected SPSA seed has no verified persisted snapshot point")
    config = seed_method.get("config")
    if (not isinstance(config, Mapping)
            or config.get("method") != "precomputed_fit_snapshot_waveform"
            or config.get("source_optimizer_step") != spec.source_step
            or config.get("snapshot_sha256") != spec.artifact_sha256
            or seed_method.get("checkpoint_sha256") != spec.artifact_sha256):
        raise PacketError("selected seed method configuration does not match its verified snapshot source")
    pins = family.input_pins
    record_name = f"snapshot_step_{spec.source_step:03d}"
    record_pin = pins.get(record_name)
    index_pin = pins.get("snapshot_index")
    for label, pin in ((record_name, record_pin), ("snapshot_index", index_pin)):
        if (not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str)
                or type(pin.get("bytes")) is not int or pin["bytes"] <= 0
                or not isinstance(pin.get("sha256"), str) or len(pin["sha256"]) != 64):
            raise PacketError(f"verified family does not carry an exact {label} raw-byte pin")
    return {
        "source_seed_method_id": seed_id,
        "source_step": spec.source_step,
        "source_theta": list(spec.theta),
        "source_theta_sha256": _sha(_canonical(list(spec.theta))),
        "source_snapshot_artifact_sha256": spec.artifact_sha256,
        "source_snapshot_record": dict(record_pin),
        "source_snapshot_index": dict(index_pin),
        "source_snapshot_record_raw_sha256": record_pin["sha256"],
        "source_snapshot_index_raw_sha256": index_pin["sha256"],
    }


def _method_for_proposal(proposal: Any, seed_method: Mapping[str, Any], input_sha: str,
                         snapshot_lineage: Mapping[str, Any]) -> dict[str, Any]:
    theta = _finite_theta(proposal.theta, f"{proposal.side} proposal theta")
    if (proposal.input_pins_sha256 != input_sha
            or proposal.artifact_sha256 != seed_method.get("checkpoint_sha256")):
        raise PacketError("proposal input pins or source checkpoint differ from the selected seed")
    config = {
        "method": "fit40_spsa_proposal",
        "schema": "fit40_spsa_proposal_transform_v1",
        "proposal_id": proposal.proposal_id,
        "pair_id": proposal.pair_id,
        "iteration": proposal.iteration,
        "side": proposal.side,
        "method_id": proposal.method_id,
        "theta": list(theta),
        "theta_sha256": proposal.theta_sha256,
        "artifact_sha256": proposal.artifact_sha256,
        "binding_sha256": proposal.binding_sha256,
        "input_pins_sha256": proposal.input_pins_sha256,
        "batch_id": proposal.batch_id,
        "sample_ids": list(proposal.sample_ids),
        "direction": list(proposal.direction),
        "c_k": proposal.c_k,
        **dict(snapshot_lineage),
        "source_training_checkpoint_sha256": seed_method["checkpoint_sha256"],
        "head_family": seed_method.get("config", {}).get("training_family"),
    }
    if _sha(_canonical(config["theta"])) != proposal.theta_sha256:
        raise PacketError("proposal theta bytes differ from the controller theta digest")
    return {
        "method_id": proposal.method_id,
        "kind": PROPOSAL_METHOD_KIND,
        "config": config,
        "config_sha256": _sha(_canonical(config)),
        "checkpoint_sha256": None,
    }


def _method_for_center(center: SnapshotSpec, seed_method: Mapping[str, Any], input_sha: str,
                       snapshot_lineage: Mapping[str, Any]) -> dict[str, Any]:
    theta = _finite_theta(center.theta, "full-center theta")
    expected_method_id = f"fit40_d0_spsa_center_k{center.source_step:02d}"
    expected_binding = full_fit_candidate_binding_sha256(
        method_id=center.method_id, artifact_sha256=center.artifact_sha256,
        input_pins_sha256=input_sha, source_step=center.source_step,
        theta=theta, point_kind="spsa_center",
    )
    if (center.point_kind != "spsa_center" or center.source_step not in {5, 10, 15, 20}
            or center.method_id != expected_method_id
            or center.artifact_sha256 != seed_method.get("checkpoint_sha256")
            or center.binding_sha256 != expected_binding):
        raise PacketError("full-center method, theta, source artifact, or input-pin binding is invalid")
    config = {
        "method": "fit40_spsa_center",
        "schema": "fit40_spsa_center_transform_v1",
        "method_id": center.method_id,
        "point_kind": "spsa_center",
        "source_step": center.source_step,
        "theta": list(theta),
        "artifact_sha256": center.artifact_sha256,
        "binding_sha256": center.binding_sha256,
        "input_pins_sha256": input_sha,
        **dict(snapshot_lineage),
        "source_training_checkpoint_sha256": seed_method["checkpoint_sha256"],
        "head_family": seed_method.get("config", {}).get("training_family"),
    }
    return {
        "method_id": center.method_id,
        "kind": CENTER_METHOD_KIND,
        "config": config,
        "config_sha256": _sha(_canonical(config)),
        "checkpoint_sha256": None,
    }


def _row_id(row: Mapping[str, Any], method_id: str) -> str:
    return "_".join(("fit", str(row["speaker_id"]), str(row["sample_id"]),
                    str(row["model_id"]), str(row["face_id"]), str(row["seed"]), method_id))


def _execution_plan(base_plan: Mapping[str, Any], base_pin: Mapping[str, Any],
                    base_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
                    method: Mapping[str, Any], sample_ids: Sequence[str],
                    scope: Mapping[str, Any]) -> dict[str, Any]:
    model_ids = [model["model_id"] for model in base_plan["models"]]
    identity_by_context = {
        (sample, model): base_rows[(sample, model, "identity")]
        for sample in sample_ids for model in model_ids
    }
    method_id = str(method["method_id"])
    planned_rows: list[dict[str, Any]] = []
    for sample in sample_ids:
        for model_id in model_ids:
            identity = identity_by_context[(sample, model_id)]
            row = copy.deepcopy(dict(identity))
            row["row_id"] = _row_id(identity, method_id)
            row["method_id"] = method_id
            row["method_config_sha256"] = method["config_sha256"]
            row["checkpoint_sha256"] = None
            row["spsa_point_binding_sha256"] = method["config"]["binding_sha256"]
            row["score_audio"] = copy.deepcopy(row["natural_pcm"])
            planned_rows.append(row)
    if len(planned_rows) != len(sample_ids) * len(model_ids):
        raise PacketError("proposal plan row materialization lost a context")
    plan = copy.deepcopy(dict(base_plan))
    plan["run_id"] = str(scope["run_id"])
    plan["methods"] = [copy.deepcopy(dict(method))]
    plan["rows"] = planned_rows
    plan["generation_repeat_qc"] = []
    plan.setdefault("execution", {})["spsa_execution_parent_plan"] = dict(base_pin)
    plan["execution"]["spsa_execution_scope"] = dict(scope)
    plan["execution"]["runtime_readiness"] = "REQUIRED_SEPARATE_SPSA_BINDING_BEFORE_EXECUTION"
    # Do not carry the original 720-row family size into these 24/120-row
    # executable derivations.  This field is descriptive plan metadata; the
    # ordered rows above remain the source of the exact denominator.
    plan["execution"]["planned_rows"] = len(planned_rows)
    plan["analysis"] = {
        "selection_status": "NOT_SELECTED",
        "scope": "fit_only_spsa_execution_packet",
        "spsa_execution_scope": dict(scope),
    }
    plan["plan_sha256"] = _self_hash(plan, "plan_sha256")
    return plan


def _write_plan_packet(output_dir: Path, base_pin: Mapping[str, Any],
                       plans: Sequence[tuple[str, dict[str, Any]]],
                       source_scope: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(os.path.abspath(output_dir))
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing existing SPSA packet directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    plan_entries = []
    for label, plan in plans:
        child = out / label
        child.mkdir()
        plan_raw = (json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n").encode("utf-8")
        rows_raw = b"".join(_canonical(row) + b"\n" for row in plan["rows"])
        plan_path = child / "plan.json"
        rows_path = child / "planned_rows.jsonl"
        with plan_path.open("xb") as stream:
            stream.write(plan_raw)
        with rows_path.open("xb") as stream:
            stream.write(rows_raw)
        plan_entries.append({
            "label": label,
            "plan": _file_pin(plan_path),
            "planned_rows": _file_pin(rows_path),
            "planned_row_count": len(plan["rows"]),
            "plan_self_sha256": plan["plan_sha256"],
            "run_id": plan["run_id"],
        })
    index: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PLANS_ONLY_REQUIRES_BOUND_RUNTIME_AND_EXECUTION",
        "parent_plan": dict(base_pin),
        "scope": dict(source_scope),
        "planned_row_total": sum(item["planned_row_count"] for item in plan_entries),
        "plans": plan_entries,
        "execution_started": False,
        "model_calls_started": False,
    }
    index["packet_sha256"] = _self_hash(index, "packet_sha256")
    index_path = out / "packet_index.json"
    with index_path.open("xb") as stream:
        stream.write((json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8"))
    return {**index, "packet_index_file": _file_pin(index_path)}


def prepare_pair_packets(
    *,
    base_plan_path: str | Path,
    expected_base_plan_sha256: str,
    family_selection: FamilySelection,
    controller: SPSAController,
    pair: SPSAProposalPair,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Emit one 8-clip plus/minus pair plan after verifying the full family.

    Each side contains exactly 8 clips × 3 native models = 24 planned rows.
    Reference row IDs are embedded for later reuse from the original verified
    720-row family, but are not scheduled for a new rendering.
    """
    base_path = Path(base_plan_path)
    base_plan, base_raw = _read_pinned_json(base_path, expected_base_plan_sha256, "base Fit plan")
    base_pin = {"path": str(base_path.resolve()), "bytes": len(base_raw), "sha256": _sha(base_raw)}
    base_plan, rows_by_key, input_sha = _validate_family_and_plan(base_plan, family_selection)
    if (controller.metric_backend != "torch_cpu"
            or controller.input_pins_sha256 != input_sha
            or controller.roster.model_ids != family_selection.family.roster.model_ids
            or pair.iteration != controller.iteration
            or controller._pending is None
            or pair != controller._pending):
        raise PacketError("controller pair is not the outstanding pair for the complete verified Fit family")
    methods = {item["method_id"]: item for item in base_plan["methods"]}
    seed_id, seed_method = _resolve_seed_method(family_selection, methods)
    snapshot_lineage = _seed_snapshot_lineage(family_selection, seed_method)
    seed_spec = next((item for item in family_selection.family.candidates if item.method_id == seed_id), None)
    if (seed_spec is None or tuple(controller.theta_initial) != tuple(seed_spec.theta)
            or controller.candidate_artifact_sha256 != seed_spec.artifact_sha256):
        raise PacketError("SPSA controller initialization does not match the selected verified Fit snapshot")
    for proposal in (pair.plus, pair.minus):
        if (proposal.pair_id != pair.pair_id or proposal.iteration != pair.iteration
                or proposal.sample_ids != pair.sample_ids or proposal.input_pins_sha256 != input_sha
                or proposal.artifact_sha256 != controller.candidate_artifact_sha256):
            raise PacketError("proposal pair members differ from the controller's frozen batch/source")
    expected_samples = tuple(pair.sample_ids)
    if (len(expected_samples) != 8
            or len({family_selection.family.roster.sample_to_speaker[s] for s in expected_samples}) != 4):
        raise PacketError("SPSA pair must contain exactly two clips from four Fit speakers")

    output_plans = []
    cached_references = []
    for side, proposal in (("plus", pair.plus), ("minus", pair.minus)):
        method = _method_for_proposal(proposal, seed_method, input_sha, snapshot_lineage)
        scope = {
            "scope_kind": "spsa_pair_proposal",
            "iteration": pair.iteration,
            "pair_id": pair.pair_id,
            "proposal_id": proposal.proposal_id,
            "side": side,
            "batch_id": pair.batch_id,
            "sample_ids": list(pair.sample_ids),
            "speaker_ids": list(pair.speakers),
            "source_seed_method_id": seed_id,
            "method_id": proposal.method_id,
            "artifact_sha256": proposal.artifact_sha256,
            "binding_sha256": proposal.binding_sha256,
            "input_pins_sha256": input_sha,
            "run_id": f"fit40_spsa_k{pair.iteration + 1:02d}_{side}_{proposal.proposal_id[:12]}",
        }
        plan = _execution_plan(base_plan, base_pin, rows_by_key, method, expected_samples, scope)
        output_plans.append((side, plan))
        for sample in expected_samples:
            for model_id in family_selection.family.roster.model_ids:
                for reference_id in BASE_METHOD_IDS:
                    reference = rows_by_key[(sample, model_id, reference_id)]
                    cached_references.append({
                        "proposal_id": proposal.proposal_id,
                        "sample_id": sample,
                        "model_id": model_id,
                        "method_id": reference_id,
                        "row_id": reference["row_id"],
                    })
    packet = _write_plan_packet(output_dir, base_pin, output_plans, {
        "scope_kind": "spsa_pair",
        "iteration": pair.iteration,
        "pair_id": pair.pair_id,
        "batch_id": pair.batch_id,
        "sample_ids": list(pair.sample_ids),
        "speaker_ids": list(pair.speakers),
        "source_seed_method_id": seed_id,
        "input_pins_sha256": input_sha,
        "cached_reference_rows": cached_references,
        "cached_reference_row_count_per_side": 8 * 3 * 2,
        "new_proposal_rows_per_side": PROPOSAL_ROW_COUNT,
        "new_proposal_rows_total": 2 * PROPOSAL_ROW_COUNT,
    })
    return packet


def prepare_center_packet(
    *,
    base_plan_path: str | Path,
    expected_base_plan_sha256: str,
    family_selection: FamilySelection,
    center: SnapshotSpec,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Emit one full 40-clip × 3-model SPSA center plan (120 planned rows)."""
    base_path = Path(base_plan_path)
    base_plan, base_raw = _read_pinned_json(base_path, expected_base_plan_sha256, "base Fit plan")
    base_pin = {"path": str(base_path.resolve()), "bytes": len(base_raw), "sha256": _sha(base_raw)}
    base_plan, rows_by_key, input_sha = _validate_family_and_plan(base_plan, family_selection)
    methods = {item["method_id"]: item for item in base_plan["methods"]}
    seed_id, seed_method = _resolve_seed_method(family_selection, methods)
    snapshot_lineage = _seed_snapshot_lineage(family_selection, seed_method)
    if center.input_pins_sha256 != input_sha:
        raise PacketError("center spec input pin identity differs from the verified initial family")
    method = _method_for_center(center, seed_method, input_sha, snapshot_lineage)
    sample_ids = family_selection.family.roster.sample_ids
    if len(sample_ids) != 40:
        raise PacketError("full SPSA center must use the frozen complete 40-clip Fit roster")
    scope = {
        "scope_kind": "spsa_full_center",
        "iteration": center.source_step,
        "method_id": center.method_id,
        "point_kind": center.point_kind,
        "source_seed_method_id": seed_id,
        "source_step": center.source_step,
        "artifact_sha256": center.artifact_sha256,
        "binding_sha256": center.binding_sha256,
        "input_pins_sha256": input_sha,
        "sample_ids": list(sample_ids),
        "speaker_ids": list(family_selection.family.roster.speaker_ids),
        "run_id": f"fit40_spsa_center_k{center.source_step:02d}_{center.binding_sha256[:12]}",
    }
    plan = _execution_plan(base_plan, base_pin, rows_by_key, method, sample_ids, scope)
    return _write_plan_packet(output_dir, base_pin, [("center", plan)], {
        **scope,
        "new_center_rows": CENTER_ROW_COUNT,
        "cached_reference_rows_for_final_selection": 40 * 3 * 2,
    })


__all__ = ["PacketError", "prepare_pair_packets", "prepare_center_packet"]
