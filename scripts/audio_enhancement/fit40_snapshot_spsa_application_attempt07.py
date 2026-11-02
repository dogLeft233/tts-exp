"""Shared physical generator-drive application and per-model row projection.

This additive layer keeps the frozen attempt05 transform/receipt intact.  It
applies that transform once from one exact identity row, wraps the real
attempt05 receipt as origin provenance, and emits independent model-projection
receipts for the same physical output WAV.  It does not render, score, train,
or select a point. Attempt07 also exports the attempt05 snapshot-point
validator for model adapters that need the same pinned validation entry point.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping


SCHEMA = "cross_tfg_fit_snapshot_shared_application_v1"
STATUS = "PASS_FIT_SNAPSHOT_SHARED_APPLICATION_ONLY"
PROJECTION_SCHEMA = "cross_tfg_fit_snapshot_model_projection_v1"
PROJECTION_STATUS = "PASS_FIT_SNAPSHOT_MODEL_PROJECTION_ONLY"
HELPER05_REL = PurePosixPath("scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py")
HELPER05_BYTES = 56_593
HELPER05_SHA256 = "3a33cf6f148de36d0433074c2efadb98167880ed5b7f1f66138a797a047afc5b"
EXPECTED_MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
POINT_FIELDS = (
    "proposal_id", "pair_id", "iteration", "side", "batch_id", "c_k",
    "point_kind", "source_step",
)


class SharedApplicationError(RuntimeError):
    """Shared waveform or model-specific projection lineage failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_sha(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    return sha256_bytes(canonical_bytes({key: item for key, item in value.items() if key != field}))


def _read_regular(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SharedApplicationError(f"expected a non-symlink regular file: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > max_bytes:
        raise SharedApplicationError(f"file is empty or exceeds the byte limit: {path}")
    return raw, {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _read_json_pin(pin: Mapping[str, Any], label: str, *, self_field: str | None = None,
                   max_bytes: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes, Path, dict[str, Any]]:
    if not isinstance(pin, Mapping):
        raise SharedApplicationError(f"{label} pin is missing")
    required_pin_keys = {"path", "bytes", "sha256"}
    if self_field is not None:
        required_pin_keys.add(self_field)
    if set(pin) != required_pin_keys:
        raise SharedApplicationError(f"{label} pin has missing or unexpected fields")
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise SharedApplicationError(f"{label} path must be absolute")
    path = Path(path_text)
    raw, actual = _read_regular(path, max_bytes=max_bytes)
    if (type(pin.get("bytes")) is not int or actual["bytes"] != pin.get("bytes")
            or actual["sha256"] != pin.get("sha256")):
        raise SharedApplicationError(f"{label} bytes/raw SHA differ from the caller pin")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SharedApplicationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SharedApplicationError(f"{label} root must be a JSON object")
    if self_field is not None:
        expected = value.get(self_field)
        if (not isinstance(expected, str) or _self_sha(value, self_field) != expected
                or pin.get("self_sha256") != expected):
            raise SharedApplicationError(f"{label} canonical self hash differs")
    return value, raw, path.resolve(strict=True), actual


def _write_exclusive_json(path: str | Path, packet: Mapping[str, Any], *, self_field: str = "self_sha256") -> dict[str, Any]:
    out = Path(path)
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise SharedApplicationError(f"receipt destination must be fresh and absolute: {out}")
    if out.parent.is_symlink() or not out.parent.is_dir():
        raise SharedApplicationError(f"receipt parent must be an existing real directory: {out.parent}")
    value = dict(packet)
    value[self_field] = _self_sha(value, self_field)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(out, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return {"path": str(out.resolve(strict=True)), "bytes": len(raw),
            "sha256": sha256_bytes(raw), "self_sha256": value[self_field]}


def _load_helper05(project_root: Path):
    path = project_root.joinpath(*HELPER05_REL.parts)
    raw, pin = _read_regular(path, max_bytes=HELPER05_BYTES)
    if (pin["bytes"], pin["sha256"]) != (HELPER05_BYTES, HELPER05_SHA256):
        raise SharedApplicationError("frozen helper05 source differs from its exact raw pin")
    name = "fit40_snapshot_spsa_application_attempt05_for_shared07"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve(strict=True) != path.resolve(strict=True):
            raise SharedApplicationError("helper05 module name is already bound to another source path")
        return existing, pin
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SharedApplicationError("could not load exact helper05 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, pin


def _plan_pin(pin: Mapping[str, Any], plan: Mapping[str, Any], label: str):
    raw, actual = _read_regular(Path(str(pin.get("path", ""))))
    if actual != {"path": str(Path(str(pin["path"])).resolve(strict=True)),
                  "bytes": pin.get("bytes"), "sha256": pin.get("sha256")}:
        raise SharedApplicationError(f"{label} raw pin differs from its file")
    if plan.get("plan_sha256") != sha256_bytes(canonical_bytes(
            {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise SharedApplicationError(f"{label} canonical self hash differs")
    try:
        disk = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SharedApplicationError(f"{label} file is not JSON") from exc
    if canonical_bytes(disk) != canonical_bytes(plan):
        raise SharedApplicationError(f"{label} in-memory plan differs from its pinned bytes")
    return {"path": actual["path"], "bytes": actual["bytes"],
            "sha256": actual["sha256"], "self_sha256": plan["plan_sha256"]}


def _validate_plan_pair(*, project_root: str | Path, family_plan: Mapping[str, Any],
                        family_plan_pin: Mapping[str, Any], point_plan: Mapping[str, Any],
                        point_plan_pin: Mapping[str, Any], method: Mapping[str, Any],
                        helper05: Any) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(project_root)
    if root.is_symlink() or not root.is_dir():
        raise SharedApplicationError("snapshot project root must be an existing real directory")
    root = root.resolve(strict=True)
    family_identity = _plan_pin(family_plan_pin, family_plan, "family plan")
    point_identity = _plan_pin(point_plan_pin, point_plan, "point plan")
    if (family_plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or family_plan.get("status") != "FROZEN"
            or len(family_plan.get("rows", [])) != 720
            or family_plan.get("execution", {}).get("planned_rows") != 720
            or family_plan.get("execution", {}).get("candidate_selection") != "not performed"
            or family_plan.get("execution", {}).get("dev_or_confirm_access") is not False):
        raise SharedApplicationError("family plan is not the actual frozen 720-row Fit-only parent")
    models = family_plan.get("models")
    if (not isinstance(models, list)
            or tuple(item.get("model_id") for item in models if isinstance(item, Mapping)) != EXPECTED_MODEL_ORDER):
        raise SharedApplicationError("family plan model order differs from its frozen three-model contract")
    if (point_plan.get("schema_version") != family_plan.get("schema_version")
            or point_plan.get("status") != "FROZEN"
            or point_plan.get("scientific_input_pins_sha256") != family_plan.get("scientific_input_pins_sha256")
            or point_plan.get("generation_repeat_qc") != []):
        raise SharedApplicationError("point plan changed the frozen schema/input identity or repeat ledger")
    parent_ref = point_plan.get("execution", {}).get("spsa_execution_parent_plan")
    expected_parent_ref = {key: family_identity[key] for key in ("path", "bytes", "sha256")}
    if parent_ref != expected_parent_ref:
        raise SharedApplicationError("point plan does not bind exact family parent raw bytes/path")
    if len(point_plan.get("methods", [])) != 1 or canonical_bytes(point_plan["methods"][0]) != canonical_bytes(method):
        raise SharedApplicationError("point plan does not contain the exact single method")
    _verify_point_plan_family_projection(family_plan, point_plan, method)
    helper05.validate_snapshot_point(project_root=root, plan=point_plan, method=method)
    return root, family_identity, point_identity, dict(method["config"])


def _verify_point_plan_family_projection(family_plan: Mapping[str, Any],
                                         point_plan: Mapping[str, Any],
                                         method: Mapping[str, Any]) -> None:
    """Recompute the composer04 child plan projection from the frozen family.

    The child plan is allowed to change only the single method, its exact
    ordered rows, its execution scope/readiness metadata, the run id, the
    empty repeat ledger, and the self hash.  Every other inherited family
    field is compared exactly, including models, protocols, supports,
    renderer/scorer bindings, and scientific input pins.
    """
    mutable_top = {"run_id", "methods", "rows", "generation_repeat_qc",
                   "plan_sha256", "execution", "analysis"}
    if set(point_plan) != set(family_plan) | {"analysis"}:
        raise SharedApplicationError("point plan top-level keys differ from composer04 projection")
    for key in set(family_plan) | set(point_plan):
        if key not in mutable_top and family_plan.get(key) != point_plan.get(key):
            raise SharedApplicationError(f"point plan altered frozen family field {key}")
    if point_plan.get("generation_repeat_qc") != []:
        raise SharedApplicationError("point plan repeat ledger must begin empty")

    family_execution = family_plan.get("execution")
    point_execution = point_plan.get("execution")
    scope = point_execution.get("spsa_execution_scope") if isinstance(point_execution, Mapping) else None
    if not isinstance(family_execution, Mapping) or not isinstance(point_execution, Mapping):
        raise SharedApplicationError("family or point execution metadata is malformed")
    if not isinstance(scope, Mapping):
        raise SharedApplicationError("point plan has no immutable SPSA execution scope")
    point_only = {"spsa_execution_parent_plan", "spsa_execution_scope", "runtime_readiness"}
    if set(point_execution) != set(family_execution) | point_only:
        raise SharedApplicationError("point execution keys differ from composer04 projection")
    allowed_execution_changes = point_only | {"planned_rows"}
    for key, value in family_execution.items():
        if key not in allowed_execution_changes and point_execution.get(key) != value:
            raise SharedApplicationError(f"point plan altered frozen execution field {key}")
    rows = point_plan.get("rows")
    methods = point_plan.get("methods")
    config = method.get("config")
    analysis = point_plan.get("analysis")
    if (not isinstance(rows, list) or not isinstance(methods, list) or len(methods) != 1
            or not isinstance(config, Mapping) or not isinstance(analysis, Mapping)):
        raise SharedApplicationError("point method, ordered rows, or analysis is malformed")
    if (type(point_execution.get("planned_rows")) is not int
            or point_execution.get("planned_rows") != len(rows)
            or point_execution.get("runtime_readiness")
            != "REQUIRED_SEPARATE_SPSA_BINDING_BEFORE_EXECUTION"):
        raise SharedApplicationError("point execution row count/readiness differs from its rows")
    if (dict(analysis) != {"selection_status": "NOT_SELECTED",
                          "scope": "fit_only_spsa_execution_packet",
                          "spsa_execution_scope": dict(scope)}
            or point_plan.get("run_id") != scope.get("run_id")):
        raise SharedApplicationError("point plan analysis/run identity differs from its scope")

    method_kind = config.get("method")
    common_scope = {
        "method_id": method.get("method_id"),
        "input_pins_sha256": family_plan.get("scientific_input_pins_sha256"),
        "artifact_sha256": config.get("artifact_sha256"),
        "binding_sha256": config.get("binding_sha256"),
        "source_seed_method_id": config.get("source_seed_method_id"),
    }
    if any(scope.get(key) != value for key, value in common_scope.items()):
        raise SharedApplicationError("point scope differs from method/family identity")
    if method_kind == "fit40_spsa_proposal":
        scope_expected = {
            "scope_kind": "spsa_pair_proposal",
            "iteration": config.get("iteration"),
            "pair_id": config.get("pair_id"),
            "proposal_id": config.get("proposal_id"),
            "side": config.get("side"),
            "batch_id": config.get("batch_id"),
            "sample_ids": config.get("sample_ids"),
        }
        sample_ids = config.get("sample_ids")
        if (not isinstance(sample_ids, list) or not sample_ids
                or len(sample_ids) != len(set(sample_ids))
                or any(not isinstance(item, str) for item in sample_ids)):
            raise SharedApplicationError("proposal has no unique ordered sample roster")
    elif method_kind == "fit40_spsa_center":
        scope_expected = {
            "scope_kind": "spsa_full_center",
            "iteration": config.get("source_step"),
            "point_kind": "spsa_center",
            "source_step": config.get("source_step"),
        }
        # The full center's roster is inherited from the family identity rows;
        # its method config intentionally has no sample_ids field.
        sample_ids = sorted({row.get("sample_id") for row in family_plan.get("rows", [])
                             if isinstance(row, Mapping) and row.get("method_id") == "identity"
                             and row.get("split") == "fit"})
    else:
        raise SharedApplicationError("point method is not a frozen SPSA proposal or center")
    if any(scope.get(key) != value for key, value in scope_expected.items()):
        raise SharedApplicationError("point scope does not match its proposal/center method fields")

    family_methods = family_plan.get("methods")
    seed_id = config.get("source_seed_method_id")
    seed_methods = ([item for item in family_methods if isinstance(item, Mapping)
                     and item.get("method_id") == seed_id]
                    if isinstance(family_methods, list) else [])
    if len(seed_methods) != 1:
        raise SharedApplicationError("point does not bind one exact legacy snapshot seed method")
    seed_config = seed_methods[0].get("config")
    if (not isinstance(seed_config, Mapping)
            or seed_config.get("method") != "precomputed_fit_snapshot_waveform"
            or seed_config.get("source_optimizer_step") != config.get("source_snapshot_step")
            or seed_config.get("snapshot_sha256") != config.get("source_snapshot_artifact_sha256")
            or seed_config.get("training_family") != config.get("head_family")):
        raise SharedApplicationError("point snapshot lineage differs from its exact family seed method")

    family_models = [item.get("model_id") for item in family_plan.get("models", [])
                     if isinstance(item, Mapping)]
    if tuple(family_models) != EXPECTED_MODEL_ORDER:
        raise SharedApplicationError("family model order differs from the shared projection contract")
    all_family_speakers: dict[str, str] = {}
    for row in family_plan.get("rows", []):
        if (isinstance(row, Mapping) and row.get("method_id") == "identity"
                and row.get("split") == "fit" and row.get("model_id") == EXPECTED_MODEL_ORDER[0]):
            sample, speaker = row.get("sample_id"), row.get("speaker_id")
            if not isinstance(sample, str) or not isinstance(speaker, str) or sample in all_family_speakers:
                raise SharedApplicationError("family identity roster is malformed or duplicated")
            all_family_speakers[sample] = speaker
    if len(all_family_speakers) != 40 or len(set(all_family_speakers.values())) != 20:
        raise SharedApplicationError("family identity roster is not the frozen 40-clip/20-speaker Fit roster")
    all_sample_ids = sorted(all_family_speakers)
    all_speaker_ids = sorted(set(all_family_speakers.values()))
    if method_kind == "fit40_spsa_proposal":
        iteration = config.get("iteration")
        if type(iteration) is not int or not 0 <= iteration < 20:
            raise SharedApplicationError("proposal iteration is outside the frozen 0..19 Fit batch cycle")
        start = (iteration * 4) % len(all_speaker_ids)
        expected_batch_speakers = [all_speaker_ids[(start + offset) % len(all_speaker_ids)]
                                   for offset in range(4)]
        selected_speakers = set(expected_batch_speakers)
        expected_batch_samples = [sample for sample in all_sample_ids
                                  if all_family_speakers[sample] in selected_speakers]
        if sample_ids != expected_batch_samples:
            raise SharedApplicationError("proposal sample batch differs from frozen cyclic 4-speaker roster")
        speaker_ids_for_scope = expected_batch_speakers
    else:
        if sample_ids != all_sample_ids:
            raise SharedApplicationError("full-center sample roster differs from the complete sorted Fit roster")
        speaker_ids_for_scope = all_speaker_ids

    identity_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in family_plan.get("rows", []):
        if (isinstance(row, Mapping) and row.get("method_id") == "identity"
                and row.get("split") == "fit" and row.get("sample_id") in sample_ids):
            key = (str(row.get("sample_id")), str(row.get("model_id")))
            if key in identity_rows:
                raise SharedApplicationError("family plan has duplicate identity row for a point context")
            identity_rows[key] = row
    expected_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        for model_id in EXPECTED_MODEL_ORDER:
            identity = identity_rows.get((sample_id, model_id))
            if identity is None:
                raise SharedApplicationError("point roster lacks one model's exact Fit identity row")
            expected = copy.deepcopy(dict(identity))
            expected.update({
                "row_id": "_".join(("fit", str(identity["speaker_id"]), str(identity["sample_id"]),
                                    str(model_id), str(identity["face_id"]), str(identity["seed"]),
                                    str(method["method_id"]))),
                "method_id": method["method_id"],
                "method_config_sha256": method["config_sha256"],
                "checkpoint_sha256": None,
                "spsa_point_binding_sha256": config["binding_sha256"],
                "score_audio": copy.deepcopy(identity["natural_pcm"]),
            })
            expected_rows.append(expected)
    scope_expected.update({"sample_ids": list(sample_ids), "speaker_ids": speaker_ids_for_scope,
                           "source_seed_method_id": seed_id,
                           "method_id": method.get("method_id"),
                           "artifact_sha256": config.get("artifact_sha256"),
                           "binding_sha256": config.get("binding_sha256"),
                           "input_pins_sha256": family_plan.get("scientific_input_pins_sha256")})
    if method_kind == "fit40_spsa_proposal":
        proposal_id = config.get("proposal_id")
        side = config.get("side")
        iteration = config.get("iteration")
        if (not isinstance(proposal_id, str) or side not in {"plus", "minus"}
                or type(iteration) is not int or iteration < 0):
            raise SharedApplicationError("proposal identity fields are malformed")
        scope_expected["run_id"] = f"fit40_spsa_k{iteration + 1:02d}_{side}_{proposal_id[:12]}"
    else:
        source_step = config.get("source_step")
        binding = config.get("binding_sha256")
        if type(source_step) is not int or not isinstance(binding, str):
            raise SharedApplicationError("center source step or binding is malformed")
        scope_expected.update({
            "run_id": f"fit40_spsa_center_k{source_step:02d}_{binding[:12]}",
            "sample_ids": list(sample_ids),
            "speaker_ids": speaker_ids_for_scope,
        })
    if canonical_bytes(dict(scope)) != canonical_bytes(scope_expected):
        actual_scope = dict(scope)
        changed = {key: {"actual": actual_scope.get(key), "expected": scope_expected.get(key)}
                   for key in set(actual_scope) | set(scope_expected)
                   if actual_scope.get(key) != scope_expected.get(key)}
        raise SharedApplicationError(
            "execution scope has missing, extra, or altered composer04 fields: "
            + json.dumps(changed, sort_keys=True, ensure_ascii=False, default=str)
        )
    if (len(rows) != len(expected_rows)
            or canonical_bytes(rows) != canonical_bytes(expected_rows)):
        raise SharedApplicationError("ordered point rows are not the exact composer04 identity-row projection")


def _source_context(*, project_root: Path, family_plan: Mapping[str, Any], sample_id: str,
                    parent_rows_by_model: Mapping[str, Mapping[str, Any]], point_plan: Mapping[str, Any],
                    method: Mapping[str, Any], helper05: Any
                    ) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if tuple(parent_rows_by_model.keys()) != EXPECTED_MODEL_ORDER:
        raise SharedApplicationError("source context must preserve the frozen ordered three-model rows")
    origin_point = helper05.validate_snapshot_point(
        project_root=project_root, plan=point_plan, method=method,
        parent_row=parent_rows_by_model[EXPECTED_MODEL_ORDER[0]])
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    audio_io = helper05._verify_loaded_module(
        "scripts.audio_enhancement.audio_io", origin_point["audio_io_path"],
        origin_point["audio_io_pin"], "audio_io")
    source_identity = None
    source_path = None
    parent_ids = {}
    speakers = set()
    seeds = set()
    for model_id in EXPECTED_MODEL_ORDER:
        row = parent_rows_by_model[model_id]
        if (row.get("sample_id") != sample_id or row.get("model_id") != model_id
                or row.get("method_id") != "identity" or row.get("split") != "fit"
                or row.get("score_audio") != row.get("natural_pcm")):
            raise SharedApplicationError(f"{model_id} is not the exact Fit identity row for {sample_id}")
        matches = [item for item in family_plan.get("rows", [])
                   if item.get("sample_id") == sample_id and item.get("model_id") == model_id
                   and item.get("method_id") == "identity"]
        if len(matches) != 1 or canonical_bytes(matches[0]) != canonical_bytes(row):
            raise SharedApplicationError(f"{model_id} parent row is not the exact family identity row")
        path, plan_input_pin = helper05._resolve_plan_pinned_natural_pcm(
            project_root, family_plan, row)
        raw, file_pin = _read_regular(path, max_bytes=32 * 1024 * 1024)
        natural = row["natural_pcm"]
        samples, audio_info = audio_io.read_canonical_wav(path, target_rate=16000)
        if (file_pin["sha256"] != natural.get("file_sha256")
                or file_pin["sha256"] != plan_input_pin.get("sha256")
                or file_pin["bytes"] != plan_input_pin.get("bytes")
                or audio_info.pcm_float32_sha256 != natural.get("pcm_sha256")
                or audio_info.frames != natural.get("sample_count")
                or samples.size != natural.get("sample_count")
                or audio_info.sample_rate != 16000 or audio_info.source_sample_rate != 16000
                or audio_info.source_channels != 1
                or natural.get("sample_rate_hz") != 16000 or natural.get("channels") != 1
                or not isinstance(natural.get("sample_count"), int)
                or natural.get("sample_count") <= 0):
            raise SharedApplicationError(f"{model_id} source file/clock metadata differs from its plan pin")
        identity = {
            "path": str(path), "bytes": file_pin["bytes"], "file_sha256": file_pin["sha256"],
            "pcm_sha256": natural.get("pcm_sha256"), "sample_count": natural.get("sample_count"),
            "sample_rate_hz": natural.get("sample_rate_hz"), "channels": natural.get("channels"),
            "subtype": natural.get("subtype"),
        }
        if source_identity is None:
            source_identity, source_path = identity, path
        elif identity != source_identity or path != source_path:
            raise SharedApplicationError("three model identity rows do not bind the same physical natural input")
        parent_ids[model_id] = row.get("row_id")
        speakers.add(row.get("speaker_id"))
        seeds.add(row.get("seed"))
    if len(speakers) != 1 or len(seeds) != 1 or source_identity is None or source_path is None:
        raise SharedApplicationError("source sample speaker/seed context is not common across models")
    return ({"sample_id": sample_id, "speaker_id": next(iter(speakers)),
             "renderer_seed": next(iter(seeds)), "identity_parent_row_ids_by_model": parent_ids,
             "natural_pcm": source_identity}, dict(source_identity), source_path)


def _point_rows_by_model(point_plan: Mapping[str, Any], method: Mapping[str, Any],
                         sample_id: str, parent_rows_by_model: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    rows = [row for row in point_plan.get("rows", []) if row.get("sample_id") == sample_id]
    if len(rows) != len(EXPECTED_MODEL_ORDER):
        raise SharedApplicationError("point plan does not contain exactly three model rows for the sample")
    by_model = {row.get("model_id"): row for row in rows}
    if tuple(by_model.keys()) != EXPECTED_MODEL_ORDER:
        raise SharedApplicationError("point rows do not preserve frozen model order")
    for model_id in EXPECTED_MODEL_ORDER:
        parent = parent_rows_by_model[model_id]
        row = by_model[model_id]
        expected = copy.deepcopy(dict(parent))
        expected.update({
            "row_id": f"{parent['row_id'].rsplit('_', 1)[0]}_{method['method_id']}",
            "method_id": method["method_id"],
            "method_config_sha256": method["config_sha256"],
            "checkpoint_sha256": None,
            "spsa_point_binding_sha256": method["config"]["binding_sha256"],
            "score_audio": copy.deepcopy(parent["natural_pcm"]),
        })
        # Composer04 computes the point ID from its exact source-row fields.
        # Validate all fields except row_id here; the actual point row ID is
        # itself bound by the point-plan self hash and projection receipt.
        expected_id = "_".join(("fit", str(parent["speaker_id"]), str(parent["sample_id"]),
                                str(model_id), str(parent["face_id"]), str(parent["seed"]),
                                str(method["method_id"])))
        expected["row_id"] = expected_id
        if canonical_bytes(row) != canonical_bytes(expected):
            raise SharedApplicationError(f"{model_id} point row differs from exact composer projection")
    return by_model


def _method_binding(config: Mapping[str, Any], config_sha256: str | None = None) -> dict[str, Any]:
    result = {key: config[key] for key in POINT_FIELDS if key in config}
    for key in ("method_id", "method", "theta", "theta_sha256", "method_config_sha256",
                "artifact_sha256", "source_snapshot_artifact_sha256", "source_snapshot_step",
                "source_seed_method_id", "binding_sha256", "input_pins_sha256", "head_family",
                "source_theta", "source_theta_sha256"):
        if key in config:
            result[key] = config[key]
    if config_sha256 is not None:
        result["method_config_sha256"] = config_sha256
    return result


def _model_context(family_plan: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    execution = family_plan.get("execution", {})
    models = [item for item in family_plan.get("models", []) if item.get("model_id") == model_id]
    protocols = execution.get("source_model_protocols", {})
    model_execution = execution.get("source_model_execution", {})
    runtimes = execution.get("runtime_bindings_by_model", {})
    if (len(models) != 1 or model_id not in protocols or model_id not in model_execution
            or model_id not in runtimes):
        raise SharedApplicationError(f"family plan lacks full model descriptor/protocol/execution/runtime for {model_id}")
    values = {
        "model_descriptor": models[0],
        "source_model_protocol": protocols[model_id],
        "source_model_execution": model_execution[model_id],
        "runtime_binding_reference": runtimes[model_id],
    }
    return {key + "_sha256": sha256_bytes(canonical_bytes(value)) for key, value in values.items()}


def _candidate_bindings(*, model_id: str, parent_row: Mapping[str, Any],
                        method: Mapping[str, Any], shared_receipt_pin: Mapping[str, Any],
                        projection_pin: Mapping[str, Any],
                        shared_output: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = method["config"]
    base = _method_binding(config, method["config_sha256"])
    base.update({
        "parent_row_id": parent_row["row_id"],
        "sample_id": parent_row["sample_id"],
        "speaker_id": parent_row["speaker_id"],
        "renderer_seed": parent_row.get("seed"),
        "model_id": model_id,
        "shared_application_receipt_raw_sha256": shared_receipt_pin["sha256"],
        "shared_application_receipt_self_sha256": shared_receipt_pin["self_sha256"],
    })
    generator_condition_input = {
        **dict(shared_output),
        "pcm_sha256": shared_output["pcm_float32_sha256"],
        **{key: value for key, value in base.items() if key not in {
            "parent_row_id", "sample_id", "speaker_id", "renderer_seed", "model_id",
            "shared_application_receipt_raw_sha256", "shared_application_receipt_self_sha256"}},
        "model_id": model_id,
        "parent_row_id": parent_row["row_id"],
        "sample_id": parent_row["sample_id"],
        "speaker_id": parent_row["speaker_id"],
        "renderer_seed": parent_row.get("seed"),
        "shared_application_receipt": dict(shared_receipt_pin),
        "shared_application_receipt_raw_sha256": shared_receipt_pin["sha256"],
        "shared_application_receipt_self_sha256": shared_receipt_pin["self_sha256"],
        "model_projection_receipt": dict(projection_pin),
        "model_projection_receipt_raw_sha256": projection_pin["sha256"],
        "model_projection_receipt_self_sha256": projection_pin["self_sha256"],
    }
    spsa_binding = {
        **base,
        "shared_application_receipt": dict(shared_receipt_pin),
        "model_projection_receipt": dict(projection_pin),
        "model_projection_receipt_raw_sha256": projection_pin["sha256"],
        "model_projection_receipt_self_sha256": projection_pin["self_sha256"],
    }
    return spsa_binding, generator_condition_input


def _reconstruct_v05_origin_candidate(parent_row: Mapping[str, Any], method: Mapping[str, Any],
                                      origin_packet: Mapping[str, Any],
                                      origin_receipt_pin: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild v05's deterministic candidate projection from its actual receipt.

    The origin receipt intentionally does not embed the whole Stage-A row.  Its
    source, theta, output, and receipt pins are sufficient to reconstruct the
    exact candidate shape expected by helper05's no-apply lineage validator.
    """
    config = method["config"]
    output = origin_packet.get("output_generator_drive_wav")
    if not isinstance(output, Mapping):
        raise SharedApplicationError("v05 origin receipt omits its output waveform pin")
    point_fields = {key: config[key] for key in POINT_FIELDS if key in config}
    common = {
        **point_fields,
        "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"],
        "artifact_sha256": config["artifact_sha256"],
        "binding_sha256": config["binding_sha256"],
        "input_pins_sha256": config["input_pins_sha256"],
        "theta_sha256": config["theta_sha256"],
        "parent_row_id": parent_row["row_id"],
        "source_snapshot_artifact_sha256": config["source_snapshot_artifact_sha256"],
        "source_snapshot_step": config["source_snapshot_step"],
        "source_seed_method_id": config["source_seed_method_id"],
        "seed": parent_row.get("seed"),
        "application_receipt_raw_sha256": origin_receipt_pin["sha256"],
        "application_receipt_self_sha256": origin_receipt_pin["self_sha256"],
    }
    input_ref = {
        **dict(output),
        "pcm_sha256": output["pcm_float32_sha256"],
        **point_fields,
        **{key: common[key] for key in (
            "method_id", "method_config_sha256", "artifact_sha256", "binding_sha256",
            "input_pins_sha256", "theta_sha256", "parent_row_id",
            "source_snapshot_artifact_sha256", "source_snapshot_step",
            "source_seed_method_id", "seed", "application_receipt_raw_sha256",
            "application_receipt_self_sha256")},
    }
    binding = {
        **point_fields,
        "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"],
        "parent_row_id": parent_row["row_id"],
        "seed": parent_row.get("seed"),
        "theta": list(config["theta"]),
        "theta_sha256": config["theta_sha256"],
        "artifact_sha256": config["artifact_sha256"],
        "binding_sha256": config["binding_sha256"],
        "input_pins_sha256": config["input_pins_sha256"],
        "source_snapshot_artifact_sha256": config["source_snapshot_artifact_sha256"],
        "source_seed_method_id": config["source_seed_method_id"],
        "source_snapshot_step": config["source_snapshot_step"],
        "application_receipt": dict(origin_receipt_pin),
        "application_receipt_raw_sha256": origin_receipt_pin["sha256"],
        "application_receipt_self_sha256": origin_receipt_pin["self_sha256"],
    }
    candidate = copy.deepcopy(dict(parent_row))
    candidate.update({
        "row_id": f"{parent_row['row_id']}__{method['method_id']}",
        "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"],
        "score_audio": copy.deepcopy(parent_row["natural_pcm"]),
        "spsa_point_binding_sha256": config["binding_sha256"],
        "spsa_binding": binding,
        "generator_condition_input": input_ref,
    })
    return candidate


def apply_snapshot_theta_shared(*, project_root: str | Path,
                                family_plan: Mapping[str, Any], family_plan_pin: Mapping[str, Any],
                                point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
                                method: Mapping[str, Any],
                                parent_rows_by_model: Mapping[str, Mapping[str, Any]],
                                output_wav_path: str | Path, origin_receipt_path: str | Path,
                                shared_receipt_path: str | Path,
                                origin_model_id: str = EXPECTED_MODEL_ORDER[0],
                                device: str = "cpu") -> dict[str, Any]:
    """Apply one point to one natural input once, retaining the real v05 origin.

    The origin receipt remains exactly the per-row v05 artifact it was.  The
    returned shared receipt records that provenance and binds the same single
    output path to every model identity row for this sample.
    """
    if origin_model_id not in EXPECTED_MODEL_ORDER:
        raise SharedApplicationError("origin model must be one of the frozen family models")
    root, family_identity, point_identity, config = _validate_plan_pair(
        project_root=project_root, family_plan=family_plan, family_plan_pin=family_plan_pin,
        point_plan=point_plan, point_plan_pin=point_plan_pin, method=method,
        helper05=_load_helper05(Path(project_root).resolve(strict=True))[0])
    helper05, helper05_pin = _load_helper05(root)
    sample_ids = {row.get("sample_id") for row in parent_rows_by_model.values()}
    if len(sample_ids) != 1 or None in sample_ids:
        raise SharedApplicationError("shared application requires exactly one source sample across model parents")
    sample_id = str(next(iter(sample_ids)))
    context, source_pcm, source_path = _source_context(
        project_root=root, family_plan=family_plan, sample_id=sample_id,
        parent_rows_by_model=parent_rows_by_model, point_plan=point_plan, method=method,
        helper05=helper05)
    point_rows = _point_rows_by_model(point_plan, method, sample_id, parent_rows_by_model)
    model_contexts = {model_id: _model_context(family_plan, model_id)
                      for model_id in EXPECTED_MODEL_ORDER}
    point = helper05.validate_snapshot_point(project_root=root, plan=point_plan,
                                             method=method,
                                             parent_row=parent_rows_by_model[origin_model_id])
    parent = parent_rows_by_model[origin_model_id]
    output = helper05.apply_snapshot_theta(
        project_root=root, plan=point_plan, method=method, parent_row=parent,
        output_wav_path=output_wav_path, receipt_path=origin_receipt_path, device=device)
    origin_pin = dict(output["application_receipt"])
    origin_packet = dict(output["application_packet"])
    # Re-run v05 lineage on the actual producer row to ensure the origin
    # artifact itself is authentic before wrapping it as a shared output.
    origin_candidate = copy.deepcopy(dict(parent))
    origin_candidate.update({
        "row_id": f"{parent['row_id']}__{method['method_id']}",
        "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"],
        "checkpoint_sha256": None,
        "spsa_point_binding_sha256": config["binding_sha256"],
        "score_audio": copy.deepcopy(parent["natural_pcm"]),
        "spsa_binding": output["spsa_binding"],
        "generator_condition_input": output["generator_condition_input"],
    })
    lineage = helper05.validate_application_lineage(
        project_root=root, plan=point_plan, method=method, parent_row=parent,
        candidate_row=origin_candidate, snapshot_point=point)
    if (lineage.get("application_receipt") != origin_pin
            or lineage.get("generator_pcm_sha256") != origin_packet.get("output_pcm_float32_sha256")
            or lineage.get("sample_count") != source_pcm.get("sample_count")):
        raise SharedApplicationError("attempt05 origin receipt did not replay its exact source/output lineage")
    if (origin_packet.get("schema") != helper05.SCHEMA
            or origin_packet.get("status") != helper05.STATUS
            or origin_packet.get("parent_row_id") != parent.get("row_id")
            or origin_packet.get("sample_id") != sample_id):
        raise SharedApplicationError("attempt05 origin receipt is not bound to its actual source identity row")
    output_pin = dict(origin_packet["output_generator_drive_wav"])
    for model_id in EXPECTED_MODEL_ORDER:
        if point_rows[model_id].get("sample_id") != sample_id:
            raise SharedApplicationError("point row sample changed across models")
    packet = {
        "schema": SCHEMA,
        "status": STATUS,
        "method_binding": _method_binding(config, method["config_sha256"]),
        "family_plan": family_identity,
        "point_plan": point_identity,
        "family_plan_self_sha256": family_plan["plan_sha256"],
        "point_plan_self_sha256": point_plan["plan_sha256"],
        "sample_id": sample_id,
        "speaker_id": context["speaker_id"],
        "renderer_seed": context["renderer_seed"],
        "source_natural_pcm": source_pcm,
        "source_natural_pcm_path": str(source_path),
        "identity_parent_row_ids_by_model": context["identity_parent_row_ids_by_model"],
        "model_order": list(EXPECTED_MODEL_ORDER),
        "model_context_hashes_by_model": model_contexts,
        "origin_v05_application_receipt": origin_pin,
        "origin_v05_model_id": origin_model_id,
        "origin_v05_parent_row_id": parent["row_id"],
        "origin_v05_schema": origin_packet["schema"],
        "origin_v05_self_sha256": origin_packet["self_sha256"],
        "helper05_source": helper05_pin,
        "output_generator_drive_wav": output_pin,
        "output_pcm_float32_sha256": origin_packet["output_pcm_float32_sha256"],
        "output_sample_count": origin_packet["output_sample_count"],
        "source_clock": dict(origin_packet["source_clock"]),
        "source_clock_preserved_after_canonicalization": origin_packet["source_clock_preserved_after_canonicalization"],
        "rms_evidence": dict(origin_packet["rms_evidence"]),
        "projection_evidence": dict(origin_packet["projection_evidence"]),
        "application_device": origin_packet.get("runtime", {}).get("device"),
        "render_or_score_performed": False,
        "optimizer_created_or_stepped": False,
        "candidate_selection_performed": False,
        "shared_output_created_once": True,
        "shared_across_model_rows_as_physical_path": True,
    }
    shared_pin = _write_exclusive_json(shared_receipt_path, packet)
    return {
        "shared_application_receipt": shared_pin,
        "shared_application_packet": packet | {"self_sha256": shared_pin["self_sha256"]},
        "origin_application_receipt": origin_pin,
        "origin_application_packet": origin_packet,
        "output_generator_drive_wav": output_pin,
        "output_samples": output["output_samples"],
        "snapshot_point": point,
        "config": config,
        "sample_count": lineage["sample_count"],
    }


def validate_shared_application_lineage(*, project_root: str | Path,
                                        family_plan: Mapping[str, Any], family_plan_pin: Mapping[str, Any],
                                        point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
                                        method: Mapping[str, Any],
                                        parent_rows_by_model: Mapping[str, Mapping[str, Any]],
                                        shared_application_receipt_pin: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a model-neutral shared receipt and its actual v05 origin."""
    root, family_identity, point_identity, config = _validate_plan_pair(
        project_root=project_root, family_plan=family_plan, family_plan_pin=family_plan_pin,
        point_plan=point_plan, point_plan_pin=point_plan_pin, method=method,
        helper05=_load_helper05(Path(project_root).resolve(strict=True))[0])
    helper05, helper05_pin = _load_helper05(root)
    sample_ids = {row.get("sample_id") for row in parent_rows_by_model.values()}
    if len(sample_ids) != 1 or None in sample_ids:
        raise SharedApplicationError("shared lineage validation requires one source sample across three parents")
    sample_id = str(next(iter(sample_ids)))
    context, source_pcm, source_path = _source_context(
        project_root=root, family_plan=family_plan, sample_id=sample_id,
        parent_rows_by_model=parent_rows_by_model, point_plan=point_plan, method=method,
        helper05=helper05)
    point_rows = _point_rows_by_model(point_plan, method, sample_id, parent_rows_by_model)
    model_contexts = {model_id: _model_context(family_plan, model_id)
                      for model_id in EXPECTED_MODEL_ORDER}
    packet, shared_raw, shared_path, shared_actual = _read_json_pin(
        shared_application_receipt_pin, "shared application receipt", self_field="self_sha256")
    if (packet.get("schema") != SCHEMA or packet.get("status") != STATUS
            or packet.get("family_plan") != family_identity
            or packet.get("point_plan") != point_identity
            or packet.get("family_plan_self_sha256") != family_plan.get("plan_sha256")
            or packet.get("point_plan_self_sha256") != point_plan.get("plan_sha256")
            or packet.get("method_binding") != _method_binding(config, method["config_sha256"])
            or packet.get("sample_id") != sample_id
            or packet.get("speaker_id") != context["speaker_id"]
            or packet.get("renderer_seed") != context["renderer_seed"]
            or packet.get("source_natural_pcm") != source_pcm
            or packet.get("source_natural_pcm_path") != str(source_path)
            or packet.get("identity_parent_row_ids_by_model") != context["identity_parent_row_ids_by_model"]
            or packet.get("model_order") != list(EXPECTED_MODEL_ORDER)
            or packet.get("model_context_hashes_by_model") != model_contexts
            or packet.get("helper05_source") != helper05_pin
            or packet.get("render_or_score_performed") is not False
            or packet.get("optimizer_created_or_stepped") is not False
            or packet.get("candidate_selection_performed") is not False
            or packet.get("shared_output_created_once") is not True
            or packet.get("shared_across_model_rows_as_physical_path") is not True):
        raise SharedApplicationError("shared receipt differs from actual family/point/source/model context")
    origin_pin = packet.get("origin_v05_application_receipt")
    origin, origin_raw, origin_path, origin_actual = _read_json_pin(
        origin_pin, "v05 origin application receipt", self_field="self_sha256")
    origin_model_id = packet.get("origin_v05_model_id")
    if origin_model_id not in EXPECTED_MODEL_ORDER:
        raise SharedApplicationError("shared receipt origin model is outside the frozen family")
    origin_parent = parent_rows_by_model[origin_model_id]
    if (origin.get("schema") != helper05.SCHEMA or origin.get("status") != helper05.STATUS
            or origin.get("parent_row_id") != origin_parent.get("row_id")
            or origin.get("sample_id") != sample_id
            or origin.get("self_sha256") != packet.get("origin_v05_self_sha256")
            or packet.get("origin_v05_parent_row_id") != origin_parent.get("row_id")
            or packet.get("origin_v05_schema") != helper05.SCHEMA):
        raise SharedApplicationError("shared receipt does not preserve the exact v05 origin provenance")
    output_pin = packet.get("output_generator_drive_wav")
    origin_output = origin.get("output_generator_drive_wav")
    if not isinstance(output_pin, Mapping):
        raise SharedApplicationError("shared receipt output waveform pin is not an object")
    output_raw, output_actual = _read_regular(Path(str(output_pin.get("path", ""))), max_bytes=8 * 1024 * 1024)
    if (not isinstance(output_pin, Mapping)
            or output_actual["bytes"] != output_pin.get("bytes")
            or output_actual["sha256"] != output_pin.get("sha256")
            or origin_output != output_pin
            or packet.get("output_pcm_float32_sha256") != origin.get("output_pcm_float32_sha256")
            or packet.get("output_sample_count") != source_pcm.get("sample_count")
            or packet.get("source_clock") != origin.get("source_clock")
            or packet.get("source_clock_preserved_after_canonicalization") is not True
            or packet.get("rms_evidence") != origin.get("rms_evidence")
            or packet.get("projection_evidence") != origin.get("projection_evidence")):
        raise SharedApplicationError("shared output file/PCM/N/clock/RMS/projection differs from v05 origin")
    if (origin.get("method_id") != method.get("method_id")
            or origin.get("method_config_sha256") != method.get("config_sha256")
            or origin.get("theta_sha256") != config.get("theta_sha256")
            or origin.get("artifact_sha256") != config.get("artifact_sha256")
            or origin.get("binding_sha256") != config.get("binding_sha256")
            or origin.get("source_snapshot_step") != config.get("source_snapshot_step")):
        raise SharedApplicationError("v05 origin does not use the exact point method/theta/snapshot")
    normalized_origin_pin = {**origin_actual, "self_sha256": origin["self_sha256"]}
    origin_candidate = _reconstruct_v05_origin_candidate(
        origin_parent, method, origin, normalized_origin_pin)
    try:
        origin_point = helper05.validate_snapshot_point(
            project_root=root, plan=point_plan, method=method, parent_row=origin_parent)
        origin_lineage = helper05.validate_application_lineage(
            project_root=root, plan=point_plan, method=method, parent_row=origin_parent,
            candidate_row=origin_candidate, snapshot_point=origin_point)
    except Exception as exc:
        raise SharedApplicationError(f"actual v05 origin application lineage failed replay: {exc}") from exc
    if (origin_lineage.get("application_receipt") != normalized_origin_pin
            or origin_lineage.get("generator_pcm_sha256") != origin.get("output_pcm_float32_sha256")
            or origin_lineage.get("sample_count") != source_pcm.get("sample_count")):
        raise SharedApplicationError("replayed v05 origin lineage differs from shared receipt pins")
    for model_id, row in parent_rows_by_model.items():
        helper05.validate_snapshot_point(project_root=root, plan=point_plan,
                                         method=method, parent_row=row)
        _ = point_rows[model_id]
    return {
        "shared_application_receipt": {**shared_actual, "self_sha256": packet["self_sha256"]},
        "shared_application_packet": packet,
        "origin_application_receipt": normalized_origin_pin,
        "origin_application_packet": origin,
        "output_generator_drive_wav": dict(output_pin),
        "snapshot_point": helper05.validate_snapshot_point(project_root=root,
            plan=point_plan, method=method, parent_row=origin_parent),
        "config": config,
        "config_sha256": method["config_sha256"],
        "source_natural_pcm": source_pcm,
        "sample_count": source_pcm["sample_count"],
        "model_point_rows": point_rows,
    }


def project_shared_application_to_model_row(*, project_root: str | Path,
                                            family_plan: Mapping[str, Any], family_plan_pin: Mapping[str, Any],
                                            point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
                                            method: Mapping[str, Any], model_id: str,
                                            parent_rows_by_model: Mapping[str, Mapping[str, Any]],
                                            parent_row: Mapping[str, Any], point_row: Mapping[str, Any],
                                            shared_application_receipt_pin: Mapping[str, Any],
                                            projection_receipt_path: str | Path) -> dict[str, Any]:
    """Create one separate model-specific projection over a shared output."""
    if model_id not in EXPECTED_MODEL_ORDER or parent_rows_by_model.get(model_id) != parent_row:
        raise SharedApplicationError("projection parent does not match the requested frozen model row")
    shared = validate_shared_application_lineage(
        project_root=project_root, family_plan=family_plan, family_plan_pin=family_plan_pin,
        point_plan=point_plan, point_plan_pin=point_plan_pin, method=method,
        parent_rows_by_model=parent_rows_by_model,
        shared_application_receipt_pin=shared_application_receipt_pin)
    expected_point = shared["model_point_rows"].get(model_id)
    if not isinstance(expected_point, Mapping) or canonical_bytes(point_row) != canonical_bytes(expected_point):
        raise SharedApplicationError("projection point row is not the exact model row in the bound point plan")
    model_context = _model_context(family_plan, model_id)
    projection_path = Path(projection_receipt_path)
    shared_output = shared["output_generator_drive_wav"]
    model_binding = _model_binding_packet_method(
        method, parent_row, model_id, shared["shared_application_receipt"])
    projection_packet = {
        "schema": PROJECTION_SCHEMA,
        "status": PROJECTION_STATUS,
        "shared_application_receipt": dict(shared["shared_application_receipt"]),
        "family_plan": _plan_pin(family_plan_pin, family_plan, "family plan"),
        "point_plan": _plan_pin(point_plan_pin, point_plan, "point plan"),
        "model_id": model_id,
        "parent_row_id": parent_row["row_id"],
        "point_row_id": point_row["row_id"],
        "sample_id": parent_row["sample_id"],
        "speaker_id": parent_row["speaker_id"],
        "renderer_seed": parent_row.get("seed"),
        "method_binding": model_binding,
        "model_context": dict(model_context),
        "source_natural_pcm": dict(shared["source_natural_pcm"]),
        "shared_output_generator_drive_wav": dict(shared_output),
        "score_audio": copy.deepcopy(parent_row["natural_pcm"]),
        "render_or_score_performed": False,
        "optimizer_created_or_stepped": False,
    }
    projection_pin = _write_exclusive_json(projection_path, projection_packet)
    spsa_binding, generator_condition_input = _candidate_bindings(
        model_id=model_id, parent_row=parent_row, method=method,
        shared_receipt_pin=shared["shared_application_receipt"],
        projection_pin=projection_pin, shared_output=shared_output)
    build = {"spsa_binding": spsa_binding,
             "generator_condition_input": generator_condition_input,
             "model_projection_receipt": projection_packet,
             "model_projection_receipt_pin": projection_pin}
    output = shared["output_generator_drive_wav"]
    candidate = copy.deepcopy(dict(parent_row))
    candidate.update({
        "row_id": point_row["row_id"],
        "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"],
        "checkpoint_sha256": None,
        "spsa_point_binding_sha256": method["config"]["binding_sha256"],
        "score_audio": copy.deepcopy(parent_row["natural_pcm"]),
        "spsa_binding": build["spsa_binding"],
        "generator_condition_input": build["generator_condition_input"],
    })
    candidate_input = candidate.get("generator_condition_input")
    if not isinstance(output, Mapping) or not isinstance(candidate_input, Mapping):
        raise SharedApplicationError(
            "shared projection requires object-valued output and candidate waveform pins; "
            f"got output={type(output).__name__}, candidate_input={type(candidate_input).__name__}"
        )
    # This is a no-apply structural check; the physical output is pinned by the
    # shared receipt and verified again by validate_model_projection_lineage.
    if (candidate["score_audio"] != parent_row["natural_pcm"]
            or candidate["generator_condition_input"].get("path") != output.get("path")
            or candidate["generator_condition_input"].get("sha256") != output.get("sha256")
            or candidate["generator_condition_input"].get("pcm_sha256") != output.get("pcm_float32_sha256")
            or candidate["generator_condition_input"].get("sample_count") != output.get("sample_count")):
        raise SharedApplicationError("model candidate does not point to the shared physical output and natural score audio")
    validate_model_projection_lineage(
        project_root=project_root, family_plan=family_plan, family_plan_pin=family_plan_pin,
        point_plan=point_plan, point_plan_pin=point_plan_pin, method=method,
        model_id=model_id, parent_rows_by_model=parent_rows_by_model,
        parent_row=parent_row, point_row=point_row, candidate_row=candidate,
        shared_application_receipt_pin=shared["shared_application_receipt"],
        model_projection_receipt_pin=build["model_projection_receipt_pin"])
    return {"candidate_row": candidate,
            "model_projection_receipt": build["model_projection_receipt"],
            "model_projection_receipt_pin": build["model_projection_receipt_pin"],
            "generator_condition_input": build["generator_condition_input"],
            "spsa_binding": build["spsa_binding"],
            "shared_application_receipt": shared["shared_application_receipt"],
            "output_generator_drive_wav": dict(output)}


def validate_model_projection_lineage(*, project_root: str | Path,
                                      family_plan: Mapping[str, Any], family_plan_pin: Mapping[str, Any],
                                      point_plan: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
                                      method: Mapping[str, Any], model_id: str,
                                      parent_rows_by_model: Mapping[str, Mapping[str, Any]],
                                      parent_row: Mapping[str, Any], point_row: Mapping[str, Any],
                                      candidate_row: Mapping[str, Any],
                                      shared_application_receipt_pin: Mapping[str, Any],
                                      model_projection_receipt_pin: Mapping[str, Any]) -> dict[str, Any]:
    """Validate projection receipts without applying the transform or rendering."""
    if model_id not in EXPECTED_MODEL_ORDER or parent_rows_by_model.get(model_id) != parent_row:
        raise SharedApplicationError("projection validator parent row is not the exact model parent")
    shared = validate_shared_application_lineage(
        project_root=project_root, family_plan=family_plan, family_plan_pin=family_plan_pin,
        point_plan=point_plan, point_plan_pin=point_plan_pin, method=method,
        parent_rows_by_model=parent_rows_by_model,
        shared_application_receipt_pin=shared_application_receipt_pin)
    expected_point = shared["model_point_rows"].get(model_id)
    if not isinstance(expected_point, Mapping) or canonical_bytes(expected_point) != canonical_bytes(point_row):
        raise SharedApplicationError("projection validator received a changed point row")
    projection, projection_raw, _projection_path, projection_actual = _read_json_pin(
        model_projection_receipt_pin, "model projection receipt", self_field="self_sha256")
    model_context = _model_context(family_plan, model_id)
    expected_parent_pin = _plan_pin(family_plan_pin, family_plan, "family plan")
    expected_point_pin = _plan_pin(point_plan_pin, point_plan, "point plan")
    expected_config = method["config"]
    if (projection.get("schema") != PROJECTION_SCHEMA
            or projection.get("status") != PROJECTION_STATUS
            or projection.get("shared_application_receipt") != shared["shared_application_receipt"]
            or projection.get("family_plan") != expected_parent_pin
            or projection.get("point_plan") != expected_point_pin
            or projection.get("model_id") != model_id
            or projection.get("parent_row_id") != parent_row.get("row_id")
            or projection.get("point_row_id") != point_row.get("row_id")
            or projection.get("sample_id") != parent_row.get("sample_id")
            or projection.get("speaker_id") != parent_row.get("speaker_id")
            or projection.get("renderer_seed") != parent_row.get("seed")
            or projection.get("method_binding") != _model_binding_packet_method(method, parent_row, model_id, shared["shared_application_receipt"])
            or projection.get("model_context") != model_context
            or projection.get("source_natural_pcm") != shared["source_natural_pcm"]
            or projection.get("shared_output_generator_drive_wav") != shared["output_generator_drive_wav"]
            or projection.get("score_audio") != parent_row.get("natural_pcm")
            or projection.get("render_or_score_performed") is not False
            or projection.get("optimizer_created_or_stepped") is not False):
        raise SharedApplicationError("model projection receipt differs from exact family/point/output lineage")
    expected_candidate = copy.deepcopy(dict(parent_row))
    expected_candidate.update({
        "row_id": point_row["row_id"], "method_id": method["method_id"],
        "method_config_sha256": method["config_sha256"], "checkpoint_sha256": None,
        "spsa_point_binding_sha256": expected_config["binding_sha256"],
        "score_audio": copy.deepcopy(parent_row["natural_pcm"]),
    })
    spsa_binding, generator_condition_input = _candidate_bindings(
        model_id=model_id, parent_row=parent_row, method=method,
        shared_receipt_pin=shared["shared_application_receipt"],
        projection_pin=dict(model_projection_receipt_pin),
        shared_output=shared["output_generator_drive_wav"])
    built = {"spsa_binding": spsa_binding,
             "generator_condition_input": generator_condition_input}
    expected_candidate.update({"spsa_binding": built["spsa_binding"],
                               "generator_condition_input": built["generator_condition_input"]})
    if canonical_bytes(candidate_row) != canonical_bytes(expected_candidate):
        raise SharedApplicationError("candidate row does not exactly project the validated shared output")
    return {
        "shared_application_receipt": shared["shared_application_receipt"],
        "model_projection_receipt": {**projection_actual, "self_sha256": projection["self_sha256"]},
        "generator_condition_input": dict(candidate_row["generator_condition_input"]),
        "spsa_binding": dict(candidate_row["spsa_binding"]),
        "output_generator_drive_wav": shared["output_generator_drive_wav"],
        "config": shared["config"],
        "config_sha256": method["config_sha256"],
        "sample_count": shared["sample_count"],
    }


def _model_binding_packet_method(method: Mapping[str, Any], parent_row: Mapping[str, Any],
                                 model_id: str, shared_receipt_pin: Mapping[str, Any]) -> dict[str, Any]:
    base = _method_binding(method["config"], method["config_sha256"])
    base.update({
        "parent_row_id": parent_row["row_id"], "sample_id": parent_row["sample_id"],
        "speaker_id": parent_row["speaker_id"], "renderer_seed": parent_row.get("seed"),
        "model_id": model_id,
        "shared_application_receipt_raw_sha256": shared_receipt_pin["sha256"],
        "shared_application_receipt_self_sha256": shared_receipt_pin["self_sha256"],
    })
    return base


def validate_snapshot_point(*, project_root: str | Path, plan: Mapping[str, Any],
                            method: Mapping[str, Any],
                            parent_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the exact frozen attempt05 point/snapshot validation.

    This is a narrow compatibility entry point for model adapters. The full
    family-to-point projection and shared application lineage remain enforced
    by ``validate_shared_application_lineage`` and
    ``validate_model_projection_lineage``; this function does not replace
    either of those gates.
    """
    root = Path(project_root)
    if root.is_symlink() or not root.is_dir():
        raise SharedApplicationError("snapshot project root must be a real directory")
    root = root.resolve(strict=True)
    helper05, helper05_pin = _load_helper05(root)
    context = helper05.validate_snapshot_point(
        project_root=root, plan=plan, method=method, parent_row=parent_row)
    if not isinstance(context, dict):
        raise SharedApplicationError("frozen attempt05 snapshot validator returned a malformed context")
    context = dict(context)
    context["frozen_helper05_source_pin"] = dict(helper05_pin)
    return context


__all__ = [
    "SharedApplicationError", "SCHEMA", "STATUS", "PROJECTION_SCHEMA", "PROJECTION_STATUS",
    "validate_snapshot_point",
    "apply_snapshot_theta_shared", "validate_shared_application_lineage",
    "project_shared_application_to_model_row", "validate_model_projection_lineage",
    "canonical_bytes", "sha256_bytes",
]
