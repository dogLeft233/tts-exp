"""Production-shaped CPU fixture checks for Muse SPSA child-plan scopes.

This test builds one proposal point and one full center from the frozen 720-row
composed family with the real packet-builder functions.  The persisted step120
theta is reused as diagnostic data; no optimizer, renderer, score, or candidate
selection is invoked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_spsa_selection_attempt06 as selection,
    musetalk_spsa_stagea_adapter_attempt07 as adapter,
    prepare_fit40_spsa_stagea_packets_attempt04 as packet_builder,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_PATH = (REPO_ROOT / ".codex/scratch/fit40_spsa_stagea_packets_attempt04"
               / "composition_attempt01/fit40_three_model_composed_plan_attempt01.json")
SNAPSHOT_PATH = (REPO_ROOT / ".codex/scratch/musetalk_point_plan_diag_attempt03"
                 / "local_project/project/evidence/snapshot_step120.json")
FAMILY_RAW_SHA256 = "b3983aeb0136f227bc3618603d1e66475ba496aac592cb381fa63838a59baa97"
FAMILY_SELF_SHA256 = "5a35c5506c2ef9411b37438af1fbc60fc3b4b72be5e5c9900083f55d52e91811"
SNAPSHOT_RAW_SHA256 = "9519c18b8c95c8291d70871a452417f4fb1cd361828284ad39feaf9cb1b178e1"
SNAPSHOT_SELF_SHA256 = "6e3e964f2eda8ce5371158e6bbce5f600409263acf0d4eb659ca2aa23c46bf82"
INPUT_PINS_SHA256 = "ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _frozen_sources():
    family_raw = FAMILY_PATH.read_bytes()
    assert _sha(family_raw) == FAMILY_RAW_SHA256
    family = json.loads(family_raw.decode("utf-8"))
    assert family["plan_sha256"] == FAMILY_SELF_SHA256
    assert family["scientific_input_pins_sha256"] == INPUT_PINS_SHA256
    assert family["status"] == "FROZEN"
    assert len(family["rows"]) == 720

    snapshot_raw = SNAPSHOT_PATH.read_bytes()
    assert len(snapshot_raw) == 9223
    assert _sha(snapshot_raw) == SNAPSHOT_RAW_SHA256
    snapshot = json.loads(snapshot_raw.decode("utf-8"))
    assert snapshot["snapshot_sha256"] == SNAPSHOT_SELF_SHA256
    assert snapshot["schema"] == "cross_tfg_fit40_parameter_snapshot_v1"
    assert snapshot["selection_status"] == "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN"
    assert snapshot["source_optimizer_step"] == 120
    theta = tuple(float(item) for row in snapshot["parameter"]["values"] for item in row)
    assert len(theta) == 9
    assert tuple(theta) == (
        -2.4561359882354736, -0.8621962070465088, -0.1879044473171234,
        -1.7196766138076782, -1.159936785697937, -0.2979220747947693,
        -2.762998580932617, -1.191292643547058, -0.1235705316066742,
    )

    seed_methods = [method for method in family["methods"]
                    if method.get("method_id") == "fit40_native_transition_head_v3_attempt11_step120"]
    assert len(seed_methods) == 1
    seed_method = seed_methods[0]
    assert seed_method["config"]["snapshot_sha256"] == SNAPSHOT_SELF_SHA256
    assert seed_method["config"]["source_optimizer_step"] == 120
    assert seed_method["config"]["training_family"] == snapshot["training_family"]

    pins = family["scientific_input_pins"]
    record_pin = pins["snapshot_step_120"]
    index_pin = pins["snapshot_index"]
    assert record_pin["bytes"] == len(snapshot_raw)
    assert record_pin["sha256"] == SNAPSHOT_RAW_SHA256
    assert record_pin["self_sha256"] == SNAPSHOT_SELF_SHA256
    lineage = {
        "source_seed_method_id": seed_method["method_id"],
        "source_snapshot_step": 120,
        "source_theta": list(theta),
        "source_theta_sha256": _sha(_canonical(list(theta))),
        "source_snapshot_artifact_sha256": SNAPSHOT_SELF_SHA256,
        "source_snapshot_record": dict(record_pin),
        "source_snapshot_index": dict(index_pin),
        "source_snapshot_record_raw_sha256": record_pin["sha256"],
        "source_snapshot_index_raw_sha256": index_pin["sha256"],
    }

    base_rows = {}
    sample_speakers = {}
    sample_order = []
    for row in family["rows"]:
        key = (row["sample_id"], row["model_id"], row["method_id"])
        base_rows[key] = row
        if row["method_id"] == "identity" and row["model_id"] == family["models"][0]["model_id"]:
            if row["sample_id"] not in sample_speakers:
                sample_speakers[row["sample_id"]] = row["speaker_id"]
                sample_order.append(row["sample_id"])
    assert len(sample_order) == 40
    assert len(sample_speakers) == 40
    return family, family_raw, base_rows, sample_order, sample_speakers, theta, seed_method, lineage


def _verify_built_child(family, family_raw, child):
    with tempfile.TemporaryDirectory() as temp:
        point_path = Path(temp) / "point_plan.json"
        point_raw = (json.dumps(child, sort_keys=True, indent=2, ensure_ascii=False,
                                allow_nan=False) + "\n").encode("utf-8")
        point_path.write_bytes(point_raw)
        family_actual_path, family_actual_raw = adapter._read_bound_plan(
            FAMILY_PATH, family, "family plan"
        )
        point_actual_path, point_actual_raw = adapter._read_bound_plan(
            point_path, child, "point plan"
        )
        assert family_actual_raw == family_raw
        adapter._verify_point_plan_parent_pin(
            family_plan=family, point_plan=child,
            family_plan_path=family_actual_path, family_plan_raw=family_actual_raw,
            point_plan_path=point_actual_path, point_plan_raw=point_actual_raw,
        )
        adapter._verify_point_plan_family_projection(family, child)
        method = child["methods"][0]
        adapter._require_exact_point_method(child, method)
        adapter._validate_point_plan_rows(
            family, child, method, child["execution"]["spsa_execution_scope"]["sample_ids"],
            method["config_sha256"],
        )


def test_real_builder_proposal_scope_projects_frozen_family_to_24_rows():
    family, family_raw, base_rows, sample_order, sample_speakers, theta, seed_method, lineage = _frozen_sources()
    speakers = sorted(set(sample_speakers.values()))[:4]
    sample_ids = [sample for sample in sample_order if sample_speakers[sample] in speakers]
    assert len(sample_ids) == 8
    assert {speaker: sum(sample_speakers[sample] == speaker for sample in sample_ids)
            for speaker in speakers} == {speaker: 2 for speaker in speakers}

    point_method_id = "fit40_spsa_proposal_diagnostic_attempt08_plus"
    pair_id = "diagnostic_pair_attempt08"
    proposal_id = "diagnostic_proposal_attempt08_plus"
    binding_sha256 = "b" * 64  # Explicit diagnostic binding; no optimizer proposal exists.
    theta_sha256 = _sha(_canonical(list(theta)))
    proposal = selection.SPSAProposal(
        proposal_id=proposal_id,
        pair_id=pair_id,
        side="plus",
        iteration=0,
        method_id=point_method_id,
        theta=theta,
        theta_sha256=theta_sha256,
        artifact_sha256=SNAPSHOT_SELF_SHA256,
        binding_sha256=binding_sha256,
        input_pins_sha256=INPUT_PINS_SHA256,
        batch_id="diagnostic_batch_attempt08",
        sample_ids=tuple(sample_ids),
        speakers=tuple(speakers),
        direction=(1,) * 9,
        c_k=0.25,
    )
    method = packet_builder._method_for_proposal(
        proposal, seed_method, INPUT_PINS_SHA256, lineage
    )
    scope = {
        "scope_kind": "spsa_pair_proposal",
        "iteration": proposal.iteration,
        "pair_id": proposal.pair_id,
        "proposal_id": proposal.proposal_id,
        "side": proposal.side,
        "batch_id": proposal.batch_id,
        "sample_ids": list(proposal.sample_ids),
        "speaker_ids": list(proposal.speakers),
        "source_seed_method_id": seed_method["method_id"],
        "method_id": proposal.method_id,
        "artifact_sha256": proposal.artifact_sha256,
        "binding_sha256": proposal.binding_sha256,
        "input_pins_sha256": INPUT_PINS_SHA256,
        "run_id": f"fit40_spsa_k01_plus_{proposal_id[:12]}",
    }
    base_pin = {"path": str(FAMILY_PATH.resolve()), "bytes": len(family_raw),
                "sha256": _sha(family_raw)}
    child = packet_builder._execution_plan(family, base_pin, base_rows, method, sample_ids, scope)
    _verify_built_child(family, family_raw, child)

    assert len(child["rows"]) == 24
    assert len({row["row_id"] for row in child["rows"]}) == 24
    assert child["execution"]["planned_rows"] == 24
    assert child["methods"][0]["config"]["theta"] == list(theta)
    assert child["execution"]["spsa_execution_scope"] == scope
    assert child["analysis"]["selection_status"] == "NOT_SELECTED"
    assert child["generation_repeat_qc"] == []


def test_real_builder_center_scope_projects_frozen_family_to_120_rows():
    family, family_raw, base_rows, sample_order, sample_speakers, theta, seed_method, lineage = _frozen_sources()
    method_id = "fit40_d0_spsa_center_k05"
    binding_sha256 = selection.full_fit_candidate_binding_sha256(
        method_id=method_id,
        artifact_sha256=SNAPSHOT_SELF_SHA256,
        input_pins_sha256=INPUT_PINS_SHA256,
        source_step=5,
        theta=theta,
        point_kind="spsa_center",
    )
    center = selection.SnapshotSpec(
        method_id=method_id,
        artifact_sha256=SNAPSHOT_SELF_SHA256,
        binding_sha256=binding_sha256,
        source_step=5,
        theta=theta,
        point_kind="spsa_center",
    )
    method = packet_builder._method_for_center(
        center, seed_method, INPUT_PINS_SHA256, lineage
    )
    speakers = sorted(set(sample_speakers.values()))
    scope = {
        "scope_kind": "spsa_full_center",
        "iteration": center.source_step,
        "method_id": center.method_id,
        "point_kind": center.point_kind,
        "source_seed_method_id": seed_method["method_id"],
        "source_step": center.source_step,
        "artifact_sha256": center.artifact_sha256,
        "binding_sha256": center.binding_sha256,
        "input_pins_sha256": INPUT_PINS_SHA256,
        "sample_ids": list(sample_order),
        "speaker_ids": speakers,
        "run_id": f"fit40_spsa_center_k{center.source_step:02d}_{center.binding_sha256[:12]}",
    }
    base_pin = {"path": str(FAMILY_PATH.resolve()), "bytes": len(family_raw),
                "sha256": _sha(family_raw)}
    child = packet_builder._execution_plan(family, base_pin, base_rows, method, sample_order, scope)
    _verify_built_child(family, family_raw, child)

    config = child["methods"][0]["config"]
    assert len(child["rows"]) == 120
    assert len({row["row_id"] for row in child["rows"]}) == 120
    assert child["execution"]["planned_rows"] == 120
    assert scope["iteration"] == config["source_step"] == 5
    assert config["theta"] == list(theta)
    assert child["execution"]["spsa_execution_scope"] == scope
    assert child["analysis"]["selection_status"] == "NOT_SELECTED"
    assert child["generation_repeat_qc"] == []
