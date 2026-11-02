from scripts.experiments.cross_tfg_audio_head_20261003 import (
    musetalk_spsa_stagea_adapter_attempt07 as adapter,
)

import json
import copy
import importlib
from pathlib import Path
import tempfile

PACKET_BUILDER = importlib.import_module(
    "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt04"
)


def _seal_plan(payload):
    payload["plan_sha256"] = adapter._sha256_bytes(adapter._canonical_bytes(payload))
    return payload


def _write_json(path: Path, payload):
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_composed_plan_keeps_model_descriptor_separate_from_source_protocol():
    descriptor_generator = {
        "renderer": "musetalk",
        "runtime_binding": "binding.json",
    }
    source_generator = {
        "audio_frontend": "frozen-wav2lip-frontend",
        "frame_count_formula": "floor(N*25/16000)",
    }
    plan = {
        "models": [{"model_id": adapter.MUSE_MODEL_ID,
                    "generator": descriptor_generator}],
        "execution": {"source_model_protocols": {
            adapter.MUSE_MODEL_ID: {"generator": source_generator},
        }},
        "protocol": {"generators_by_model": {
            adapter.MUSE_MODEL_ID: source_generator,
        }},
    }

    model, generator = adapter._resolve_muse_protocol(plan)

    assert model["generator"] == descriptor_generator
    assert generator == source_generator


def test_composed_protocol_map_must_match_exact_source_protocol():
    plan = {
        "models": [{"model_id": adapter.MUSE_MODEL_ID,
                    "generator": {"renderer": "musetalk"}}],
        "execution": {"source_model_protocols": {
            adapter.MUSE_MODEL_ID: {"generator": {"audio_frontend": "source"}},
        }},
        "protocol": {"generators_by_model": {
            adapter.MUSE_MODEL_ID: {"audio_frontend": "different"},
        }},
    }

    try:
        adapter._resolve_muse_protocol(plan)
    except adapter.MuseSPSAAdapterError as exc:
        assert "differs from source protocol" in str(exc)
    else:
        raise AssertionError("mismatched composed generator map was accepted")


def test_derived_point_plan_binds_exact_family_plan_file():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        family = _seal_plan({
            "schema_version": adapter.EXPECTED_PLAN_SCHEMA,
            "status": "FROZEN",
            "scientific_input_pins_sha256": "a" * 64,
            "models": [], "methods": [], "rows": [],
        })
        family_path = root / "family.json"
        family_raw = _write_json(family_path, family)
        point = _seal_plan({
            "schema_version": adapter.EXPECTED_PLAN_SCHEMA,
            "status": "FROZEN",
            "scientific_input_pins_sha256": family["scientific_input_pins_sha256"],
            "execution": {"spsa_execution_parent_plan": {
                "path": str(family_path.resolve()),
                "bytes": len(family_raw),
                "sha256": adapter._sha256_bytes(family_raw),
            }},
            "models": [], "methods": [], "rows": [],
        })
        point_path = root / "point.json"
        point_raw = _write_json(point_path, point)

        family_actual_path, family_actual_raw = adapter._read_bound_plan(
            family_path, family, "family plan"
        )
        point_actual_path, point_actual_raw = adapter._read_bound_plan(
            point_path, point, "point plan"
        )
        adapter._verify_point_plan_parent_pin(
            family_plan=family, point_plan=point,
            family_plan_path=family_actual_path, family_plan_raw=family_actual_raw,
            point_plan_path=point_actual_path, point_plan_raw=point_actual_raw,
        )


def test_derived_point_plan_rejects_family_roster_and_nonmatching_method():
    method = {"method_id": "point-a", "kind": "fit40_spsa_proposal"}
    adapter._require_exact_point_method({"methods": [method]}, method)

    for methods in ([method, {"method_id": "point-b"}],
                    [{"method_id": "different-point"}],
                    None):
        try:
            adapter._require_exact_point_method({"methods": methods}, method)
        except adapter.MuseSPSAAdapterError as exc:
            assert "exactly the supplied SPSA method" in str(exc)
        else:
            raise AssertionError("non-point plan method roster was accepted")


def test_derived_point_plan_rejects_wrong_family_raw_pin():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        family = _seal_plan({
            "schema_version": adapter.EXPECTED_PLAN_SCHEMA,
            "status": "FROZEN",
            "scientific_input_pins_sha256": "b" * 64,
            "models": [], "methods": [], "rows": [],
        })
        family_path = root / "family.json"
        family_raw = _write_json(family_path, family)
        point = _seal_plan({
            "schema_version": adapter.EXPECTED_PLAN_SCHEMA,
            "status": "FROZEN",
            "scientific_input_pins_sha256": family["scientific_input_pins_sha256"],
            "execution": {"spsa_execution_parent_plan": {
                "path": str(family_path.resolve()),
                "bytes": len(family_raw),
                "sha256": "0" * 64,
            }},
            "models": [], "methods": [], "rows": [],
        })
        point_path = root / "point.json"
        point_raw = _write_json(point_path, point)
        family_actual_path, family_actual_raw = adapter._read_bound_plan(
            family_path, family, "family plan"
        )
        point_actual_path, point_actual_raw = adapter._read_bound_plan(
            point_path, point, "point plan"
        )
        try:
            adapter._verify_point_plan_parent_pin(
                family_plan=family, point_plan=point,
                family_plan_path=family_actual_path, family_plan_raw=family_actual_raw,
                point_plan_path=point_actual_path, point_plan_raw=point_actual_raw,
            )
        except adapter.MuseSPSAAdapterError as exc:
            assert "exact family-plan path, bytes, and raw SHA" in str(exc)
        else:
            raise AssertionError("wrong family raw pin was accepted")


def test_actual_packet_builder_emits_8_by_3_child_rows_bound_to_family_file():
    sample_ids = [f"BAC_DIAG_{index:02d}" for index in range(8)]
    model_ids = ["wav2lip_diag", adapter.MUSE_MODEL_ID, "ditto_diag"]
    method = {"method_id": "fit40_spsa_diagnostic_only",
              "config_sha256": "c" * 64, "kind": "fit40_spsa_proposal",
              "config": {"binding_sha256": "f" * 64}}
    rows = []
    base_rows = {}
    for index, sample_id in enumerate(sample_ids):
        speaker_id = f"S{index:04d}"
        for model_id in model_ids:
            row = {
                "row_id": f"fit_{speaker_id}_{sample_id}_{model_id}_face_{20261003}_identity",
                "split": "fit", "sample_id": sample_id, "speaker_id": speaker_id,
                "model_id": model_id, "face_id": "face", "seed": 20261003,
                "method_id": "identity", "method_config_sha256": "0" * 64,
                "checkpoint_sha256": None,
                "natural_pcm": {"path": f"audio/{sample_id}.wav",
                                "file_sha256": "a" * 64, "pcm_sha256": "b" * 64,
                                "sample_count": 1000, "sample_rate_hz": 16000,
                                "channels": 1},
                "source_face": {"path": "face.png", "sha256": "d" * 64},
                "score_audio": None,
            }
            row["score_audio"] = copy.deepcopy(row["natural_pcm"])
            if model_id == adapter.MUSE_MODEL_ID:
                row["musetalk_native_frame_binding"] = {
                    "sample_count": 1000, "expected_frame_count": 1,
                }
            rows.append(row)
            base_rows[(sample_id, model_id, "identity")] = row

    family = _seal_plan({
        "schema_version": adapter.EXPECTED_PLAN_SCHEMA, "status": "FROZEN",
        "run_id": "family-diag", "scientific_input_pins_sha256": "e" * 64,
        "models": [{"model_id": model_id} for model_id in model_ids],
        "methods": [{"method_id": "identity"}], "rows": rows,
        "execution": {},
    })

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        family_path = root / "family.json"
        family_raw = _write_json(family_path, family)
        base_pin = {"path": str(family_path.resolve()), "bytes": len(family_raw),
                    "sha256": adapter._sha256_bytes(family_raw)}
        point = PACKET_BUILDER._execution_plan(
            family, base_pin, base_rows, method, sample_ids,
            {"run_id": "diagnostic-child-only"},
        )
        point_path = root / "point.json"
        point_raw = _write_json(point_path, point)
        family_actual_path, family_actual_raw = adapter._read_bound_plan(
            family_path, family, "family plan"
        )
        point_actual_path, point_actual_raw = adapter._read_bound_plan(
            point_path, point, "point plan"
        )
        adapter._verify_point_plan_parent_pin(
            family_plan=family, point_plan=point,
            family_plan_path=family_actual_path, family_plan_raw=family_actual_raw,
            point_plan_path=point_actual_path, point_plan_raw=point_actual_raw,
        )
        adapter._require_exact_point_method(point, method)
        muse_rows = adapter._validate_point_plan_rows(
            family, point, method, sample_ids, method["config_sha256"]
        )

    assert len(point["methods"]) == 1
    assert len(point["rows"]) == 24
    assert len(muse_rows) == 8
    assert [row["sample_id"] for row in muse_rows] == sample_ids


def _actual_diagnostic_plan_files():
    repo_root = Path(__file__).resolve().parents[2]
    project_root = (repo_root / ".codex/scratch/musetalk_point_plan_diag_attempt03"
                    / "local_project/project")
    return project_root, project_root / "diagnostic_family_plan.json", project_root / "diagnostic_point_plan.json"


def test_actual_diagnostic_point_preserves_family_protocol_inputs_and_supports():
    project_root, family_path, point_path = _actual_diagnostic_plan_files()
    family = json.loads(family_path.read_text(encoding="utf-8"))
    point = json.loads(point_path.read_text(encoding="utf-8"))
    family_real, family_raw = adapter._read_bound_plan(family_path, family, "family plan")
    point_real, point_raw = adapter._read_bound_plan(point_path, point, "point plan")
    adapter._verify_point_plan_parent_pin(
        family_plan=family, point_plan=point,
        family_plan_path=family_real, family_plan_raw=family_raw,
        point_plan_path=point_real, point_plan_raw=point_raw,
    )
    adapter._verify_point_plan_family_projection(family, point)

    assert point["models"] == family["models"]
    assert point["protocol"] == family["protocol"]
    assert point["scientific_input_pins"] == family["scientific_input_pins"]
    assert point["scientific_input_pins_sha256"] == family["scientific_input_pins_sha256"]
    assert point["supports"] == family["supports"]
    assert point["execution"]["planned_rows"] == len(point["rows"]) == 24
    assert point["analysis"]["selection_status"] == "NOT_SELECTED"


def test_actual_diagnostic_point_rejects_tampered_inherited_protocol_pins_or_execution():
    _, family_path, point_path = _actual_diagnostic_plan_files()
    family = json.loads(family_path.read_text(encoding="utf-8"))
    point = json.loads(point_path.read_text(encoding="utf-8"))
    mutations = []

    changed_protocol = copy.deepcopy(point)
    changed_protocol["protocol"]["unexpected_mutation"] = True
    mutations.append(changed_protocol)

    changed_pins = copy.deepcopy(point)
    changed_pins["scientific_input_pins"]["unexpected_mutation"] = {
        "path": "not-a-family-input", "bytes": 1, "sha256": "0" * 64,
    }
    mutations.append(changed_pins)

    changed_supports = copy.deepcopy(point)
    if isinstance(changed_supports["supports"], list):
        changed_supports["supports"].append({"unexpected_mutation": True})
    elif isinstance(changed_supports["supports"], dict):
        changed_supports["supports"]["unexpected_mutation"] = True
    else:
        raise AssertionError("actual family supports field has an unsupported shape")
    mutations.append(changed_supports)

    changed_execution = copy.deepcopy(point)
    changed_execution["execution"]["candidate_selection"] = "performed"
    mutations.append(changed_execution)

    for mutated in mutations:
        mutated["plan_sha256"] = adapter._sha256_bytes(
            adapter._canonical_bytes({key: value for key, value in mutated.items()
                                      if key != "plan_sha256"})
        )
        try:
            adapter._verify_point_plan_family_projection(family, mutated)
        except adapter.MuseSPSAAdapterError:
            pass
        else:
            raise AssertionError("self-hashed derived point with inherited family mutation was accepted")
