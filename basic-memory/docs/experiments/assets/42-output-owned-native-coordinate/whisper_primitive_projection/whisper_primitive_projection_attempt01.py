"""One-shot CPU diagnostic for MuseTalk Whisper-window alias consistency."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = Path(__file__).resolve().parent
OUT = ATTEMPT / "whisper_primitive_projection_attempt01.json"
CANDIDATE_ROOT = ATTEMPT / "output_owned_candidates_attempt01"
CANDIDATE_RECEIPT = CANDIDATE_ROOT / "receipt.json"
CPU6_RECEIPT = ATTEMPT / "readback_attempt04" / "receipt.json"
CPU6_QA = ATTEMPT / "root_actual_cpu6_qa.json"
CPU6_READBACK = ATTEMPT / "readback_attempt04"
HELPER_PATH = ATTEMPT / "rotation_candidate_source_attempt02.py"
HELPER_SHA = "6f782f2f89467914f3e9c70e483c2efd28b160789b3e64f357e0256902acdcd7"
ART_PATH = ROOT / ".codex/scratch/native_articulatory_coordinate_cpu_20261015_attempt02/source.py"
ART_SHA = "38fe8b6a7f5d1ccb926b3250feadb2a583448e17057ee8ec41fbbee0c8cb6c3c"
PARSER_PATH = ROOT / ".codex/scratch/phone_conditioned_visual_response_20261015_attempt01/support_source.py"
PARSER_SHA = "f138a2088cc8d7ac07c2886ab50569362d39599e9a3e3c9b256251d071d538eb"
FRONTEND_PATH = ROOT / ".codex/scratch/root_frontend_actual_source_review_attempt01/08_audio_processor.py"
FRONTEND_SHA = "b7b62f3daa99678576138bf581a6394a10d346571a2530a760d1299ffd622b81"
WHISPER_IMPL_PATH = ROOT / ".codex/scratch/root_frontend_actual_source_review_attempt01/10_modeling_whisper.py"
WHISPER_IMPL_SHA = "9f8a41810e9508933cc07b96618718701c16ba4e17154bc81db03230284013a6"
WHISPER_CONFIG_PATH = ROOT / ".codex/scratch/root_frontend_actual_source_review_attempt01/12_config.json"
WHISPER_CONFIG_SHA = "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98"
SOURCE_QC_PATH = ROOT / ".codex/scratch/root_frontend_actual_source_review_attempt01/root_actual_qc.json"
SOURCE_QC_SHA = "f009da64e826f7ae0a1e460c6e9a6f04b4b4f82b856a89c8eaca93b3e152db97"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def import_pinned(name: str, path: Path, expected_sha: str) -> Any:
    actual = sha_file(path)
    if actual != expected_sha:
        raise ValueError(f"source pin mismatch: {path} {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def primitive_groups(frame_count: int, actual_length: int, hidden_state_count: int) -> dict[tuple[Any, ...], list[tuple[int, int]]]:
    fps = 25
    audio_fps = 50
    multiplier = audio_fps / fps
    left_padding = math.ceil(multiplier) * 2
    right_padding = math.ceil(multiplier) * 3 * 2
    context_length = 2 * (2 + 2 + 1)
    native_width = context_length * hidden_state_count
    padded_length = left_padding + actual_length + right_padding
    groups: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    for frame in range(frame_count):
        audio_index = math.floor(frame * multiplier)
        for feature_index in range(native_width):
            context_index, layer_index = divmod(feature_index, hidden_state_count)
            padded_index = audio_index + context_index
            if padded_index < 0 or padded_index >= padded_length:
                raise ValueError(f"source-derived Whisper slice is out of padded bounds: {frame}/{feature_index}")
            if padded_index < left_padding:
                key = ("left_zero_pad", padded_index, layer_index)
            elif padded_index < left_padding + actual_length:
                key = ("whisper_hidden", 0, padded_index - left_padding, layer_index)
            else:
                key = ("right_zero_pad", padded_index - left_padding - actual_length, layer_index)
            groups.setdefault(key, []).append((frame, feature_index))
    if sum(len(v) for v in groups.values()) != frame_count * native_width:
        raise AssertionError("primitive gather map failed to cover every prompt slot exactly once")
    return groups


def gather_reconstruct(base: np.ndarray, groups: dict[tuple[Any, ...], list[tuple[int, int]]]) -> np.ndarray:
    rebuilt = np.empty_like(base)
    for locations in groups.values():
        source_frame, source_feature = locations[0]
        vector = base[source_frame, source_feature, :]
        for frame, feature in locations:
            rebuilt[frame, feature, :] = vector
    return rebuilt


def project_delta(delta: np.ndarray, groups: dict[tuple[Any, ...], list[tuple[int, int]]]) -> np.ndarray:
    projected = np.zeros(delta.shape, dtype=np.float64)
    for locations in groups.values():
        coords = np.asarray(locations, dtype=np.int64)
        mean = np.mean(delta[coords[:, 0], coords[:, 1], :], axis=0, dtype=np.float64)
        projected[coords[:, 0], coords[:, 1], :] = mean
    return projected


def energy(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.sum(x * x, dtype=np.float64))


def margin(flat: np.ndarray, indices: np.ndarray, axis_signed: np.ndarray, helper: Any, art: Any) -> float | None:
    direction = helper.token_direction(flat, indices, art)
    return None if direction is None else float(direction @ axis_signed)


def arm_projection_summary(
    base: np.ndarray,
    candidate: np.ndarray,
    projected_delta: np.ndarray,
    owned_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    delta = candidate.astype(np.float64) - base.astype(np.float64)
    residual = delta - projected_delta
    total_e = energy(delta)
    projected_e = energy(projected_delta)
    residual_e = energy(residual)
    owned_e = energy(delta[owned_mask])
    projected_owned_e = energy(projected_delta[owned_mask])
    nonowned = ~owned_mask
    projected_nonowned = projected_delta[nonowned]
    nonowned_e = energy(projected_nonowned)
    nonowned_frames = np.flatnonzero(np.any(projected_nonowned != 0.0, axis=(1, 2)))
    nonowned_indices = np.flatnonzero(nonowned)
    affected_nonowned = nonowned_indices[nonowned_frames]
    n_nonowned_values = int(projected_nonowned.size)
    summary = {
        "candidate_delta_energy": total_e,
        "projected_delta_energy": projected_e,
        "residual_energy": residual_e,
        "residual_energy_ratio": None if total_e == 0.0 else residual_e / total_e,
        "projected_energy_ratio": None if total_e == 0.0 else projected_e / total_e,
        "owned_candidate_energy": owned_e,
        "owned_projected_energy": projected_owned_e,
        "owned_target_retained_energy_ratio": None if owned_e == 0.0 else projected_owned_e / owned_e,
        "owned_target_retained_norm_ratio": None if owned_e == 0.0 else math.sqrt(projected_owned_e / owned_e),
        "projected_nonowned_energy": nonowned_e,
        "projected_nonowned_energy_fraction_of_projected": None if projected_e == 0.0 else nonowned_e / projected_e,
        "projected_nonowned_energy_over_original_owned_energy": None if owned_e == 0.0 else nonowned_e / owned_e,
        "projected_nonowned_affected_frame_indices": affected_nonowned.tolist(),
        "projected_nonowned_affected_frame_count": int(affected_nonowned.size),
        "projected_nonowned_max_abs": float(np.max(np.abs(projected_nonowned))) if projected_nonowned.size else 0.0,
        "projected_nonowned_rms": math.sqrt(nonowned_e / n_nonowned_values) if n_nonowned_values else 0.0,
    }
    stored_candidate = np.ascontiguousarray((base.astype(np.float64) + projected_delta).astype(base.dtype))
    return summary, stored_candidate


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"create-only output already exists: {OUT}")
    pinned = {
        "candidate_receipt": (CANDIDATE_RECEIPT, "37ff2a054bb1e4f1dcca1d37de4fe6f292bee433832371886109b879841be705"),
        "cpu6_receipt": (CPU6_RECEIPT, "892720030a6e78b109607bda0f447af7faeb9080283f8258a6a1bd0cd2dcf12b"),
        "cpu6_qa": (CPU6_QA, "e9735787aa9d6c33ce591feec104864319e4c51a425ff3beec6ff668b47e5b72"),
        "audio_processor": (FRONTEND_PATH, FRONTEND_SHA),
        "transformers_whisper_impl": (WHISPER_IMPL_PATH, WHISPER_IMPL_SHA),
        "whisper_config": (WHISPER_CONFIG_PATH, WHISPER_CONFIG_SHA),
        "source_qc": (SOURCE_QC_PATH, SOURCE_QC_SHA),
    }
    pins = {name: {"path": str(path), "sha256": sha_file(path)} for name, (path, expected) in pinned.items()}
    for name, (_, expected) in pinned.items():
        if pins[name]["sha256"] != expected:
            raise ValueError(f"pinned input changed: {name} {pins[name]['sha256']}")

    candidate_receipt = read_json(CANDIDATE_RECEIPT)
    cpu_receipt = read_json(CPU6_RECEIPT)
    source_qc = read_json(SOURCE_QC_PATH)
    if candidate_receipt.get("status") != "OUTPUT_OWNED_CPU_CANDIDATES_ONLY_NOT_RENDERED_OR_SCORED":
        raise ValueError("unexpected output-owned candidate receipt status")
    if cpu_receipt.get("status") != "MATERIALIZED_UNMODIFIED_NATIVE_FEATURES_ONLY":
        raise ValueError("unexpected CPU6 receipt status")
    config = read_json(WHISPER_CONFIG_PATH)
    if int(config["d_model"]) != 384 or int(config["encoder_layers"]) != 4:
        raise ValueError(f"unexpected Whisper tiny dimensions: d_model={config.get('d_model')}, layers={config.get('encoder_layers')}")
    if int(config["max_source_positions"]) != 1500:
        raise ValueError("unexpected Whisper encoder position count")
    source_records = {item["path"]: item for item in source_qc["files"]}
    hf_remote = next((p for p in source_records if p.endswith("/transformers/models/whisper/modeling_whisper.py")), None)
    if hf_remote is None or source_records[hf_remote]["sha256"] != WHISPER_IMPL_SHA:
        raise ValueError("root frontend QC does not bind the expected Transformers Whisper source")
    audio_remote = "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/model_sources/musetalk/musetalk/utils/audio_processor.py"
    if audio_remote not in source_records or source_records[audio_remote]["sha256"] != FRONTEND_SHA:
        raise ValueError("root frontend QC does not bind the expected MuseTalk AudioProcessor source")

    helper = import_pinned("accepted_rotation_candidate_helper", HELPER_PATH, HELPER_SHA)
    art = helper.import_pinned("native_articulatory_attempt02", helper.ART_SOURCE, helper.ART_SOURCE_SHA256)
    parser = helper.import_pinned("phone_support_time_parser", helper.TIME_PARSER, helper.TIME_PARSER_SHA256)
    fit_tokens = helper.fit_model_tokens("musetalk", art, parser)
    cpu_rows = {row["sample_id"]: row for row in cpu_receipt["rows"] if row["model_key"] == "musetalk"}

    rows_out: list[dict[str, Any]] = []
    for candidate_row in candidate_receipt["rows"]:
        sample = candidate_row["sample_id"]
        cpu_row = cpu_rows[sample]
        frontend_detail = cpu_row["frontend_detail"]["detail"]
        if frontend_detail["source_files"].get("musetalk/utils/audio_processor.py") != FRONTEND_SHA:
            raise ValueError(f"CPU6 row does not bind the pinned AudioProcessor: {sample}")
        if frontend_detail["runtime"].get("transformers") != "4.39.2":
            raise ValueError(f"CPU6 row has unexpected Whisper runtime: {sample}")
        npz_path = CPU6_READBACK / Path(cpu_row["output_path"]).name
        if sha_file(npz_path) != cpu_row["output_file_sha256"]:
            raise ValueError(f"CPU6 NPZ file hash mismatch: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as archive:
            base = np.array(archive["features"], copy=True)
            clock = np.array(archive["native_time_s"], copy=True)
        if list(base.shape) != candidate_row["source_shape"] or base.dtype != np.float32:
            raise ValueError(f"unexpected CPU6 array layout for {sample}: {base.shape}/{base.dtype}")
        if raw_array_sha(base) != cpu_row["feature_array_bytes_sha256"]:
            raise ValueError(f"CPU6 feature body mismatch: {sample}")

        arms: dict[str, np.ndarray] = {}
        for arm in ("zero", "plus", "minus"):
            desc = candidate_row["arrays"][f"output_owned_height_{arm}"]
            path = Path(desc["path"])
            if sha_file(path) != desc["file_sha256"]:
                raise ValueError(f"candidate file SHA mismatch: {path}")
            value = np.load(path, allow_pickle=False)
            if value.shape != base.shape or value.dtype != base.dtype or raw_array_sha(value) != desc["array_sha256"]:
                raise ValueError(f"candidate array body/shape/dtype mismatch: {sample}/{arm}")
            arms[arm] = value
        if not np.array_equal(arms["zero"].view(np.uint8), base.view(np.uint8)):
            raise ValueError(f"candidate zero is not the exact CPU6 baseline: {sample}")
        clock_desc = candidate_row["native_time_s"]
        clock_path = Path(clock_desc["path"])
        if sha_file(clock_path) != clock_desc["file_sha256"]:
            raise ValueError(f"candidate clock file SHA mismatch: {sample}")
        candidate_clock = np.load(clock_path, allow_pickle=False)

        sample_count = int(cpu_row["input_detail"]["sample_count"])
        frame_count = int(base.shape[0])
        actual_length = sample_count * 50 // 16000
        expected_frames = sample_count * 25 // 16000
        if frame_count != expected_frames:
            raise ValueError(f"source clock/frame formula mismatch: {sample} {frame_count} vs {expected_frames}")
        if sample_count >= 30 * 16000 or len(cpu_row["frontend_detail"]["detail"]["whisper_input_shapes"]) != 1:
            raise ValueError(f"expected a single Whisper 30s segment for {sample}")
        if clock.shape != (frame_count,) or not np.array_equal(clock, np.arange(frame_count, dtype=np.float64) / 25.0):
            raise ValueError(f"CPU6 Whisper prompt clock is not unchanged j/25: {sample}")
        if candidate_clock.shape != clock.shape or not np.array_equal(candidate_clock, clock):
            raise ValueError(f"output-owned candidate changed the CPU6 native clock: {sample}")
        hidden_state_count = int(config["encoder_layers"]) + 1
        if base.shape[1] != 10 * hidden_state_count or base.shape[2] != int(config["d_model"]):
            raise ValueError(f"native prompt layout does not match source-derived 10 x 5 x 384: {sample}")

        groups = primitive_groups(frame_count, actual_length, hidden_state_count)
        reconstructed = gather_reconstruct(base, groups)
        exact_reconstruction = bool(np.array_equal(reconstructed.view(np.uint8), base.view(np.uint8)))
        owned = sorted({
            int(frame)
            for token in candidate_row["tokens"]
            if token["status"] == "ROTATED_OUTPUT_OWNED"
            for frame in token["output_owned_prompt_rows_I_plus_p"]
        })
        owned_mask = np.zeros(frame_count, dtype=bool)
        owned_mask[owned] = True
        if not exact_reconstruction:
            rows_out.append({
                "sample_id": sample,
                "status": "ABORTED_ZERO_GATHER_RECONSTRUCTION_NOT_EXACT",
                "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed.astype(np.float64) - base.astype(np.float64)))),
            })
            continue

        multiplicity = Counter(len(locations) for locations in groups.values())
        base_flat = helper.flatten_native(base, "musetalk")
        speaker = candidate_row["speaker_id"]
        anchors, supporters = art.class_anchor_for_fold(fit_tokens, helper.HEIGHT_MAP, speaker, int(np.prod(base.shape[1:])))
        if anchors.get("high") is None or anchors.get("low") is None:
            raise ValueError(f"natural FIT anchors missing for {sample}: {supporters}")
        token_receipts = [t for t in candidate_row["tokens"] if t["status"] == "ROTATED_OUTPUT_OWNED"]
        arm_data = {}
        projected_native_by_arm = {}
        for arm in ("plus", "minus"):
            delta = arms[arm].astype(np.float64) - base.astype(np.float64)
            if np.any(delta[~owned_mask] != 0.0):
                raise ValueError(f"candidate delta is nonzero outside effective rotated-token rows: {sample}/{arm}")
            projected_delta = project_delta(delta, groups)
            summary, projected_native = arm_projection_summary(base, arms[arm], projected_delta, owned_mask)
            projected_native_by_arm[arm] = projected_native
            summary["candidate_delta_nonzero_frame_indices"] = np.flatnonzero(np.any(delta != 0.0, axis=(1, 2))).tolist()
            arm_data[arm] = summary

        token_margins = []
        for token in token_receipts:
            indices = np.asarray(token["output_owned_prompt_rows_I_plus_p"], dtype=np.int64)
            label = token["height_class"]
            other = "low" if label == "high" else "high"
            signed_class = anchors[label] - anchors[other]
            base_margin = margin(base_flat, indices, signed_class, helper, art)
            entry: dict[str, Any] = {
                "phone_index": token["phone_index"],
                "base_phone": token["base_phone_after_tone_only_strip"],
                "class": label,
                "owned_prompt_rows": indices.tolist(),
                "source_candidate_baseline_signed_margin": token["baseline_prompt_direction_signed_class_margin"],
                "recomputed_zero_signed_margin": base_margin,
                "arms": {},
            }
            for arm in ("plus", "minus"):
                raw_flat = helper.flatten_native(arms[arm], "musetalk")
                proj_flat = helper.flatten_native(projected_native_by_arm[arm], "musetalk")
                raw_margin = margin(raw_flat, indices, signed_class, helper, art)
                projected_margin = margin(proj_flat, indices, signed_class, helper, art)
                entry["arms"][arm] = {
                    "candidate_signed_margin": raw_margin,
                    "candidate_delta_vs_zero": None if raw_margin is None or base_margin is None else raw_margin - base_margin,
                    "projected_signed_margin": projected_margin,
                    "projected_delta_vs_zero": None if projected_margin is None or base_margin is None else projected_margin - base_margin,
                }
            token_margins.append(entry)

        rows_out.append({
            "sample_id": sample,
            "status": "PASS_SOURCE_DERIVED_GATHER_RECONSTRUCTS_CPU6_ZERO_EXACTLY",
            "cpu6_native_npz": {"path": str(npz_path), "file_sha256": cpu_row["output_file_sha256"], "feature_array_sha256": cpu_row["feature_array_bytes_sha256"]},
            "source_pcm": {"path": cpu_row["input_detail"]["path"], "file_sha256": cpu_row["input_detail"]["file_sha256"], "pcm_float32_sha256": cpu_row["input_detail"]["pcm_float32_sha256"], "sample_count": sample_count},
            "candidate_arrays": {arm: {"path": candidate_row["arrays"][f"output_owned_height_{arm}"]["path"], "file_sha256": candidate_row["arrays"][f"output_owned_height_{arm}"]["file_sha256"], "array_sha256": candidate_row["arrays"][f"output_owned_height_{arm}"]["array_sha256"]} for arm in ("zero", "plus", "minus")},
            "native_clock": {"path": clock_desc["path"], "file_sha256": clock_desc["file_sha256"], "exact_unchanged_j_over_25": True},
            "native_shape": list(base.shape),
            "native_dtype": str(base.dtype),
            "sample_count_16k": sample_count,
            "frame_count_25fps": frame_count,
            "actual_whisper_length_50hz": actual_length,
            "left_pad_tokens": 4,
            "right_pad_tokens": 12,
            "hidden_state_count": hidden_state_count,
            "flatten_rule": "feature_index = context_index * 5 + hidden_state_index",
            "source_hidden_key_rule": "(segment=0, hidden_time=q-4, hidden_state_index) for real tokens; side-specific padded-sequence slot keys for zero padding",
            "primitive_count": len(groups),
            "prompt_slot_count": int(frame_count * base.shape[1]),
            "primitive_copy_multiplicity_histogram": {str(k): int(v) for k, v in sorted(multiplicity.items())},
            "zero_gather_reconstruction": {
                "exact_bitwise": exact_reconstruction,
                "max_abs_error": float(np.max(np.abs(reconstructed.astype(np.float64) - base.astype(np.float64)))),
                "mismatch_scalar_count": int(np.count_nonzero(reconstructed != base)),
            },
            "effective_owned_output_rows_from_rotated_status_only": owned,
            "arms": arm_data,
            "projected_token_signed_margins": token_margins,
        })

    result = {
        "schema": "musetalk_whisper_primitive_alias_projection_v1",
        "status": "COMPLETED_CPU_FEASIBILITY_ONLY" if all(r.get("status", "").startswith("PASS") for r in rows_out) else "ZERO_RECONSTRUCTION_FAILED_NO_PROJECTION_RESULT",
        "source_derived_operator": "A has one row per (video frame, flattened prompt feature) and one column per shared (encoder segment, hidden timestep, hidden-state tuple index); AA+ averages each delta over source-identified copies of the same primitive.",
        "scope_limit": "necessary slice-consistency condition only; not evidence that any projected native tensor is reachable from a waveform or from the frozen Whisper encoder; no waveform, encoder, renderer, score, or GPU executed",
        "source_mapping": {
            "audio_processor_path_remote": "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/model_sources/musetalk/musetalk/utils/audio_processor.py",
            "audio_processor_sha256": FRONTEND_SHA,
            "transformers_version": "4.39.2",
            "transformers_whisper_modeling_sha256": WHISPER_IMPL_SHA,
            "whisper_config_sha256": WHISPER_CONFIG_SHA,
            "config_d_model": int(config["d_model"]),
            "config_encoder_layers": int(config["encoder_layers"]),
            "hidden_states_tuple_count": int(config["encoder_layers"]) + 1,
            "fps_ratio": "50/25=2; frame j slices padded indices [2j,2j+10)",
            "time_mapping": "context c maps q=2j+c; q in [4,4+actual_length) maps hidden timestep q-4; native clock remains j/25",
            "layers": "hidden-state tuple entries: initial conv+position state; states entering encoder layers 1-3; final post-layernorm state",
        },
        "pins": pins,
        "candidate_receipt_sha256": sha_file(CANDIDATE_RECEIPT),
        "cpu6_receipt_sha256": sha_file(CPU6_RECEIPT),
        "cpu6_qa_sha256": sha_file(CPU6_QA),
        "rotation_helper_sha256": HELPER_SHA,
        "articulatory_anchor_source_sha256": ART_SHA,
        "phone_time_parser_sha256": PARSER_SHA,
        "rows": rows_out,
    }
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "rows": len(rows_out), "output": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
