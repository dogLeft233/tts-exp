"""Build a pinned, score-blind inventory for the recovered 74-pair Wav2Lip cohort.

This module only joins frozen manifests and existing score receipts. It does not
decode audio, run SyncNet, or select rows by outcome.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO / "runs/processing_chain_predictive_audit_20261015_attempt01"
INV_ROOT = TASK_ROOT / "inventory_A"
MIRROR = INV_ROOT / "remote_mirror"
RESIDUAL = MIRROR / "tts_pcm_residual_20260926"
OUTPUTS = {
    "assets": INV_ROOT / "asset_inventory_attempt02.json",
    "scores": INV_ROOT / "paired_scores_attempt02.csv",
    "coverage": INV_ROOT / "coverage_attempt02.json",
    "manifest": INV_ROOT / "inventory_output_manifest_attempt02.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    protocol_path = RESIDUAL / "protocol.json"
    cohort_path = RESIDUAL / "aishell1_cohort.json"
    tts_path = RESIDUAL / "aishell1_tts_meta.json"
    render_path = RESIDUAL / "wav2lip_manifest.json"
    body_readback_path = INV_ROOT / "source_body_readback_attempt01.json"
    score_dir = RESIDUAL / "scores"

    protocol = read_json(protocol_path)
    cohort = read_json(cohort_path)
    tts_meta = read_json(tts_path)
    render = read_json(render_path)
    body_readback = read_json(body_readback_path)
    schema_path = TASK_ROOT / "schema_design_v1.json"

    protocol_sha = sha256_file(protocol_path)
    cohort_sha = sha256_file(cohort_path)
    tts_sha = sha256_file(tts_path)
    render_sha = sha256_file(render_path)
    schema_sha = sha256_file(schema_path)

    if protocol_sha != body_readback["source_protocol_sha256"]:
        raise ValueError("Local protocol mirror does not match the body-readback pin")
    if tts_sha != body_readback["tts_meta_sha256"]:
        raise ValueError("Local TTS metadata mirror does not match the body-readback pin")
    if cohort_sha != tts_meta["cohort_manifest_sha256"]:
        raise ValueError("Cohort manifest hash disagrees with frozen TTS metadata")
    if render["cohort_manifest_sha256"] != cohort_sha or render["tts_meta_sha256"] != tts_sha:
        raise ValueError("Wav2Lip manifest source pins disagree with the selected manifests")

    protocol_rows = protocol["rows"]
    body_rows = {row["sample_id"]: row for row in body_readback["source_rows"]}
    score_readbacks = {row["sample_id"]: row for row in body_readback["score_receipts"]}
    cohort_rows = {row["sample_id"]: row for row in cohort["records"]}
    tts_rows = tts_meta["results"]
    render_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in render["records"]:
        render_rows[(row["sample_id"], row["source_condition"])] = row

    speaker_members: dict[str, list[str]] = defaultdict(list)
    assets: list[dict[str, Any]] = []
    upstream_artifacts: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    score_contract = {
        "native_or_unified_pcm_or_replacement": "native experimental source PCM WAVs used as N_source and T_source; legacy mux-derived cells are a separate contract",
        "scorer_source_version": "scripts/experiments/tts_pcm_residual.py@bab399d8a521fbfac2a64367f32b55bed37f76e9b720679ec441b3b438c9c573",
        "scorer_weight_identity": render["syncnet_model_sha256"],
        "roi_policy": (
            "dynamic face-reference MP4 used by the historical run; "
            f"face_sha256={render['face_sha256']}; archival manifest label="
            f"{render['face_protocol']!r} is metadata and does not mean a static portrait"
        ),
        "window_support": "guard20; official per-arm window support; N/T primary arms are not forced to equal absolute-window counts",
        "min_track": render["min_track"],
        "fps": None,
        "fps_missing_reason": "not stated in the selected scalar receipts or bound manifests",
        "audio_chain": "N_source/T_source score cells consume recovered canonical source WAV bodies; compressed video-audio legacy cells and MFA-linear replacement cells remain separate",
    }
    contract_fingerprint = sha256_bytes(canonical_json(score_contract).encode("utf-8"))

    for protocol_row in protocol_rows:
        sample_id = protocol_row["id"]
        speaker = protocol_row["speaker"]
        pair_uid = f"w2l_tts_pcm74:{sample_id}"
        speaker_group = f"AISHELL-1:{speaker}"
        speaker_members[speaker_group].append(sample_id)
        cohort_row = cohort_rows[sample_id]
        tts_row = tts_rows[sample_id]
        body_row = body_rows[sample_id]
        score_readback = score_readbacks[sample_id]
        score_path = score_dir / f"{sample_id}.json"
        score_raw = score_path.read_bytes()
        score_sha = sha256_bytes(score_raw)
        score_obj = json.loads(score_raw)

        if score_sha != score_readback["sha256"]:
            raise ValueError(f"Local scalar receipt body mismatch: {sample_id}")
        if score_obj["id"] != sample_id or score_obj["speaker"] != speaker:
            raise ValueError(f"Scalar receipt identity mismatch: {sample_id}")

        row_audio_uids: dict[str, str] = {}
        for arm, condition in (("N", "natural"), ("T", "tts")):
            # Both recovered WAVs are canonical generator/run-input bodies.
            # Their natural/original and provider-native assets are separate
            # upstream lineage records and are not substituted for these.
            source_layer = "generator"
            protocol_cell = protocol_row["cells"][arm]
            body_cell = body_row["arms"][arm]
            audio_uid = f"w2l_tts_pcm74:{sample_id}:{arm}"
            row_audio_uids[arm] = audio_uid
            if not body_cell["exists"] or not body_cell["matches_expected"]:
                raise ValueError(f"Source audio body did not pass frozen hash binding: {sample_id}/{arm}")
            if body_cell["sha256"] != protocol_cell["audio_sha256"]:
                raise ValueError(f"Protocol/source body SHA mismatch: {sample_id}/{arm}")

            if arm == "N":
                canonical = cohort_row["canonical"]
                if canonical["canonical_audio_sha256"] != body_cell["sha256"]:
                    raise ValueError(f"Natural canonical source binding mismatch: {sample_id}")
                source_metadata = {
                    "sample_rate_hz": canonical["sample_rate_hz"],
                    "channels": canonical["channels"],
                    "sample_count": canonical["samples"],
                    "duration_s": canonical["duration_s"],
                }
                lineage = {
                    "role": "canonical 16 kHz natural-content generator/run-input supplied to historical Wav2Lip",
                    "run_input_path": canonical["canonical_16k_audio"],
                    "run_input_sha256": canonical["canonical_audio_sha256"],
                    "archive_copy_path": body_cell["path"],
                    "archive_copy_body_sha256_verified": True,
                    "upstream_original_source_ref": {
                        "path": cohort_row.get("source_audio_path"),
                        "expected_sha256": cohort_row.get("source_audio_sha256"),
                        "source_layer": "original",
                        "body_status": "provenance only; distinct from the recovered canonical run-input and not body-verified here",
                    },
                }
                upstream_artifacts.append(
                    {
                        "audio_uid": f"w2l_tts_pcm74:{sample_id}:upstream_original",
                        "pair_uid": pair_uid,
                        "condition": "natural",
                        "source_layer": "original",
                        "source_path": cohort_row.get("source_audio_path"),
                        "expected_sha256": cohort_row.get("source_audio_sha256"),
                        "file_sha256": None,
                        "status": "PROVENANCE_ONLY_NOT_BODY_VERIFIED",
                        "missing_reason": "The selected model input is the separately recovered canonical 16 kHz run-input; upstream dataset source body was not recovered in this inventory.",
                    }
                )
            else:
                if tts_row["canonical_audio_sha256"] != body_cell["sha256"]:
                    raise ValueError(f"TTS canonical source binding mismatch: {sample_id}")
                source_metadata = {
                    "sample_rate_hz": tts_row["sample_rate_hz"],
                    "channels": 1,
                    "sample_count": tts_row["samples"],
                    "duration_s": tts_row["duration_s"],
                }
                lineage = {
                    "role": "canonical 16 kHz generator audio supplied to the historical Wav2Lip run",
                    "canonical_run_input_path": tts_row["canonical_16k_audio"],
                    "canonical_run_input_sha256": tts_row["canonical_audio_sha256"],
                    "archive_copy_path": body_cell["path"],
                    "archive_copy_body_sha256_verified": True,
                    "provider": tts_row["provider"],
                    "model": tts_row["model"],
                    "reference_audio_path": tts_row["reference_audio"],
                    "reference_audio_sha256": tts_row["reference_audio_sha256"],
                    "provider_native_output": {
                        "path": tts_row["provider_audio"],
                        "expected_sha256": tts_row["provider_audio_sha256"],
                        "sample_rate_hz": tts_row["provider_sample_rate_hz"],
                        "available_at_referenced_path": False,
                        "absence_check": "one manifest-led check of the configured tts output directory found zero *.provider.wav files",
                        "role_note": "upstream 24 kHz provider output is not the 16 kHz canonical scoring/generator input",
                    },
                    "canonicalization_method": "not specified in the selected run metadata; do not infer a resampling algorithm",
                }
                upstream_artifacts.append(
                    {
                        "audio_uid": f"w2l_tts_pcm74:{sample_id}:provider_native",
                        "pair_uid": pair_uid,
                        "condition": "tts",
                        "source_layer": "generator",
                        "source_path": tts_row["provider_audio"],
                        "expected_sha256": tts_row["provider_audio_sha256"],
                        "sample_rate_hz": tts_row["provider_sample_rate_hz"],
                        "file_sha256": None,
                        "status": "UNAVAILABLE_AT_REFERENCED_PATH",
                        "missing_reason": "Targeted configured TTS output-directory check found zero *.provider.wav files; the expected hash is metadata only and is not body-verified.",
                    }
                )

            face = render["face"]
            assets.append(
                {
                    "audio_uid": audio_uid,
                    "pair_uid": pair_uid,
                    "source_group": speaker_group,
                    "source_group_basis": "speaker_id from the selected AISHELL-1 cohort record; N and its per-sample self-clone T remain connected",
                    "dataset": "AISHELL-1",
                    "language": "zh",
                    "condition": condition,
                    "source_layer": source_layer,
                    "source_path": body_cell["path"],
                    "location": "LAN remote source archive at wjj@192.168.50.2:22; body remains remote for later CPU decode",
                    "file_sha256": body_cell["sha256"],
                    "pcm_sha256": None,
                    "lineage": lineage,
                    "missing_reason": "pending_frozen_decode: SHA256 of unified contiguous little-endian float32 mono16k decoder output has not been materialized",
                    "source_file_size_bytes": body_cell["bytes"],
                    "source_media_metadata": source_metadata,
                    "paired_key": cohort_row["paired_key"],
                    "context": {
                        "tts_provider": tts_meta["provider"],
                        "tts_version": tts_meta["model"],
                        "tfg": "Wav2Lip",
                        "tfg_version": render["wav2lip_checkpoint_sha256"],
                        "run": protocol_row["cells"][arm]["base"],
                        "portrait_reference": {
                            "kind": "dynamic_video_reference",
                            "path": face,
                            "sha256": render["face_sha256"],
                            "is_static_portrait": False,
                            "archival_manifest_face_protocol_label": render["face_protocol"],
                        },
                        "score_contract": "native source PCM guard20; source N/T only",
                    },
                    "body_verification_ref": "source_body_readback_attempt01.json",
                }
            )

        n = score_obj["cells"]["N_source"]["20"]
        t = score_obj["cells"]["T_source"]["20"]
        paired_rows.append(
            {
                "score_uid": f"{pair_uid}:native_source_pcm_guard20",
                "pair_uid": pair_uid,
                "source_group": speaker_group,
                "dataset": "AISHELL-1",
                "language": "zh",
                "tts_provider": tts_meta["provider"],
                "tts_version": tts_meta["model"],
                "tfg": "Wav2Lip",
                "tfg_version": render["wav2lip_checkpoint_sha256"],
                "run": protocol_row["cells"]["N"]["base"],
                "score_contract": "native_source_pcm_guard20_T_minus_N",
                "contract_fingerprint": contract_fingerprint,
                "feature_layer": "generator",
                "natural_audio_uid": row_audio_uids["N"],
                "tts_audio_uid": row_audio_uids["T"],
                "C_n": n["C"],
                "C_t": t["C"],
                "delta_c": t["C"] - n["C"],
                "score_source": score_readback["path"],
                "score_source_sha256": score_readback["sha256"],
                "missing_reason": "",
                "B_n": n.get("B"),
                "B_t": t.get("B"),
                "Dmin_n": None,
                "Dmin_t": None,
                "D0_n": None,
                "D0_t": None,
                "best_offset_n": n.get("offset"),
                "best_offset_t": t.get("offset"),
                "query_count_n": n.get("rows"),
                "query_count_t": t.get("rows"),
                "score_D_n_raw": n.get("D"),
                "score_D_t_raw": t.get("D"),
                "auxiliary_missing_reason": "historical scalar receipt stores C, B, D, offset, rows; Dmin and D0 are not present and were not inferred",
                "score_source_local_mirror": str(score_path.relative_to(REPO)),
                "score_features_sha256": score_obj["features_sha256"],
                "score_matrices_sha256": score_obj["matrices_sha256"],
                "native_or_unified_pcm_or_replacement": score_contract["native_or_unified_pcm_or_replacement"],
                "scorer_source_version": score_contract["scorer_source_version"],
                "scorer_weight_identity": score_contract["scorer_weight_identity"],
                "roi_policy": score_contract["roi_policy"],
                "window_support": score_contract["window_support"],
                "min_track": score_contract["min_track"],
                "fps": "",
                "audio_chain": score_contract["audio_chain"],
                "score_contract_fields_json": canonical_json(score_contract),
            }
        )

    if len(protocol_rows) != 74 or len(assets) != 148 or len(paired_rows) != 74:
        raise ValueError("Primary cohort row counts differ from the frozen 74-pair contract")

    raw_provider = body_readback["provider_native_outputs"]
    if len(raw_provider) != 74 or any(row["exists"] for row in raw_provider):
        raise ValueError("Provider-native output availability changed; update the explicit upstream-layer status")

    score_contracts = {
        "primary_native_source_pcm_guard20": {
            "contract_fingerprint": contract_fingerprint,
            "fields": score_contract,
            "included_in_paired_scores_csv": True,
        },
        "legacy_mux_audio_guard20": {
            "available_in_same_74_scalar_receipts": 74,
            "included_in_paired_scores_csv": False,
            "semantics": "compressed video-audio legacy comparator, never an original or generator WAV",
        },
        "mfa_linear_replacement": {
            "available_in_same_74_scalar_receipts": 74,
            "included_in_paired_scores_csv": False,
            "condition_semantics": "MFA-linear TTS-derived natural-clock replacement; not raw/native TTS",
        },
    }

    assets_doc = {
        "schema_version": "processing_chain_audit_v1",
        "inventory_version": "inventory_A_main74_attempt02",
        "status": "READY_FOR_ROOT_COHORT_BINDING_REVIEW",
        "scope": "Score-blind inventory of the selected Wav2Lip natural/TTS canonical source-Pcm pairs; no extraction or new scoring",
        "cohort_selection": {
            "source": "runs/tts_pcm_residual_20260926/protocol.json",
            "selection_rule": "historical eligibility and asset completeness; protocol states no score selection",
            "eligible_pair_count": 74,
            "universe_count": protocol["universe"],
            "excluded_or_missing_count": len(protocol["missing"]),
            "score_values_used_for_selection": False,
        },
        "counts": {
            "assets": len(assets),
            "natural_assets": 74,
            "tts_assets": 74,
            "paired_source_groups": len(speaker_members),
            "source_body_sha256_verified": body_readback["counts"]["source_body_sha256_matches"],
            "source_body_readback_receipt": "source_body_readback_attempt01.json",
            "provider_native_24k_assets_available": body_readback["counts"]["provider_native_present"],
            "canonical_tts_16k_generator_assets_available": 74,
            "unified_pcm_sha256_materialized": 0,
        },
        "source_input_pins": {
            "schema_design_v1_sha256": schema_sha,
            "residual_protocol_sha256": protocol_sha,
            "cohort_manifest_sha256": cohort_sha,
            "tts_meta_sha256": tts_sha,
            "wav2lip_manifest_sha256": render_sha,
            "source_body_readback_sha256": sha256_file(body_readback_path),
            "v1_asset_inventory_sha256": sha256_file(INV_ROOT / "asset_inventory.json"),
            "v1_paired_scores_sha256": sha256_file(INV_ROOT / "paired_scores.csv"),
            "v1_builder_preserved_sha256": sha256_file(Path(__file__).with_name("inventory_attempt01.py")),
        },
        "score_contracts": score_contracts,
        "upstream_layer_status": {
            "natural_original": "The upstream AISHELL original path/hash is retained separately as provenance only; it is not the recovered run-input body and was not body-verified here.",
            "canonical_generator_inputs": "Both conditions have 74/74 canonical 16 kHz generator/run-input bodies recovered and SHA-verified; condition remains natural or TTS in the paired design.",
            "tts_provider_native": "UNAVAILABLE: TTS metadata declares a 24 kHz provider output and hashes, but the targeted configured TTS directory contained zero referenced *.provider.wav outputs; do not treat canonical 16 kHz bodies as 24 kHz provider outputs",
            "pcm_sha256": "pending frozen unified decode: contiguous little-endian float32 mono16k bytes without a header; file_sha256 remains the WAV container body hash",
        },
        "upstream_artifacts": upstream_artifacts,
        "assets": assets,
    }

    # Main related score-only coverage is pinned separately. It is not pooled
    # into the 74-pair generator corpus or selected by its score values.
    multiset_root = MIRROR / "multiset_pipeline"
    multiset_manifest = read_json(multiset_root / "video_manifest_250.json")
    multiset_scores = read_json(multiset_root / "scores_250.json")
    multiset_meta = read_json(multiset_root / "eval_meta.json")
    leaptalk_scores = read_json(MIRROR / "leaptalk_eval/scores_leaptalk.json")
    english_n = list((REPO / "data/data/audio_en").glob("*.wav"))
    english_t = list((REPO / "data/data/audio_en_qwen3_tts").glob("*.wav"))

    manifest_datasets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in multiset_manifest["records"]:
        manifest_datasets[row["dataset"]].append(row)
    multiset_score_rows = multiset_scores if isinstance(multiset_scores, list) else multiset_scores.get("scores", [])
    multiset_score_counts: dict[str, int] = defaultdict(int)
    for row in multiset_score_rows:
        multiset_score_counts[str(row.get("condition", row.get("cond", "unknown")))] += 1

    coverage = {
        "schema_version": "processing_chain_audit_coverage_v1",
        "inventory_version": "inventory_A_main74_attempt02",
        "status": "PRIMARY_74_READY; EXTENSION_COHORTS_RECORDED_WITH_EXPLICIT_GAPS",
        "scope_policy": "No outcome-based selection; no L1/L2 model has run; separate source layers and contracts remain separate.",
        "primary_cohort": {
            "dataset": "AISHELL-1",
            "language": "zh",
            "tfg": "Wav2Lip",
            "tts_provider": tts_meta["provider"],
            "tts_version": tts_meta["model"],
            "pair_count": 74,
            "asset_count": 148,
            "source_group_count": len(speaker_members),
            "source_group_basis": "speaker_id in AISHELL-1 cohort; linked self-clone natural/TTS pairs are kept with that speaker",
            "source_group_membership": {k: sorted(v) for k, v in sorted(speaker_members.items())},
            "source_audio_status": {
                "natural_condition_canonical_16k_generator_input": {"available": 74, "sha256_verified": 74, "source_layer": "generator"},
                "tts_condition_canonical_16k_generator_input": {"available": 74, "sha256_verified": 74, "source_layer": "generator"},
                "upstream_aishell_original_body": {"available": 0, "expected_provenance_rows": 74, "body_verified": 0},
                "tts_provider_native_24k": {"available": 0, "expected": 74, "missing_reason": "provider outputs named by tts_meta are absent from the targeted configured tts directory"},
                "unified_pcm_sha256": {"available": 0, "reason": "frozen decode not yet run"},
            },
            "score_status": {
                "native_source_pcm_guard20_pairs": 74,
                "per_pair_scalar_receipts_verified": 74,
                "primary_contract": "native source PCM N_source/T_source, guard20, T minus N; feature_layer=generator for both recovered canonical run-inputs",
                "legacy_compressed_contract_pairs": 74,
                "mfa_linear_replacement_contract_pairs": 74,
                "Dmin_D0_fields": "not present in these per-pair scalar receipts; left null, not inferred from raw D",
            },
            "protocol_missing_or_ineligible": len(protocol["missing"]),
            "historical_m_arm": {
                "available_in_score_receipts": 74,
                "condition_semantics": "MFA-linear TTS-derived natural-clock replacement",
                "native_tts_l1_or_primary_l2": False,
                "audio_body_in_main_148": False,
            },
        },
        "extension_cohorts_not_in_main74": [
            {
                "cohort": "multiset_pipeline",
                "status": "L2_SCORE_ONLY; PAIRED_TTS_SOURCE_AUDIO_UNAVAILABLE",
                "source_manifest": "inventory_A/remote_mirror/multiset_pipeline/video_manifest_250.json",
                "source_manifest_sha256": sha256_file(multiset_root / "video_manifest_250.json"),
                "sample_count": len(multiset_manifest["records"]),
                "dataset_counts": {k: len(v) for k, v in sorted(manifest_datasets.items())},
                "audio_source_counts": {
                    "embedded_video_audio": sum(1 for r in multiset_manifest["records"] if r.get("audio_source") == "embedded"),
                    "external_wav_reference": sum(1 for r in multiset_manifest["records"] if r.get("audio_source") != "embedded"),
                },
                "lrs3_scope": "50 samples within the 250, 45 parent video/source groups; not 250 LRS3 clips",
                "tts_provider": "faster_qwen3 is configured for this run; run-level TTS metadata and paired TTS WAV outputs were not found at the targeted configured run locations",
                "score_rows": len(multiset_score_rows),
                "score_condition_counts": dict(sorted(multiset_score_counts.items())),
                "eval_meta_complete": multiset_meta.get("complete"),
                "eval_meta_total": multiset_meta.get("total"),
                "score_file_sha256": sha256_file(multiset_root / "scores_250.json"),
                "limitations": "Original embedded-audio extraction is mux-derived; full paired native/generated WAV sources are not available for the current predictor audit.",
            },
            {
                "cohort": "leaptalk_eval",
                "status": "L2_SCORE_ONLY; PAIRED_SOURCE_AUDIO_UNVERIFIED",
                "score_file": "inventory_A/remote_mirror/leaptalk_eval/scores_leaptalk.json",
                "score_file_sha256": sha256_file(MIRROR / "leaptalk_eval/scores_leaptalk.json"),
                "score_rows": len(leaptalk_scores),
                "pair_count": len(leaptalk_scores) // 2,
                "limitation": "Historical score table only in selected mirror; no source/generator audio pair bodies were verified for this inventory.",
            },
            {
                "cohort": "local_english_qwen3",
                "status": "L1_AUDIO_PAIRS_AVAILABLE; L2_SCORE_LINK_UNAVAILABLE",
                "natural_wav_count": len(english_n),
                "tts_wav_count": len(english_t),
                "pair_count": min(len(english_n), len(english_t)),
                "tfg_score_link": "no exact matched TFG/SyncNet row contract found in this inventory pass",
            },
            {
                "cohort": "local_aishell1_phoneme_tfg_link",
                "status": "SCORE_ONLY; AUDIO_SOURCE_BODIES_NOT_VERIFIED_HERE",
                "score_remote_path": "/home/wjj/tts-audio/tts-exp/results/aishell100_phoneme/tfg_link/syncnet_100.json",
                "speaker_count_in_local_aishell_manifest": 16,
                "warning": "Separate manifest scope from the 15-speaker Wav2Lip 100-run manifest and the 13-speaker 74-pair residual subset; do not merge counts.",
            },
            {
                "cohort": "historical_codec_47",
                "status": "HISTORICAL_SCORE_LAYER; NOT_47_INDEPENDENT_GROUPS",
                "source_group_count": 1,
                "source_group": "AISHELL-1:S0765",
                "warning": "The 47 label denotes items/rows in that mechanism cohort, not 47 source groups.",
            },
        ],
        "input_hashes": {
            "protocol_sha256": protocol_sha,
            "cohort_sha256": cohort_sha,
            "tts_meta_sha256": tts_sha,
            "wav2lip_manifest_sha256": render_sha,
            "source_body_readback_sha256": sha256_file(body_readback_path),
            "schema_sha256": schema_sha,
            "v1_asset_inventory_sha256": sha256_file(INV_ROOT / "asset_inventory.json"),
            "v1_paired_scores_sha256": sha256_file(INV_ROOT / "paired_scores.csv"),
        },
    }

    write_json(OUTPUTS["assets"], assets_doc)
    score_columns = list(paired_rows[0])
    with OUTPUTS["scores"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_rows)
    write_json(OUTPUTS["coverage"], coverage)

    output_manifest = {
        "schema_version": "processing_chain_inventory_output_manifest_v2",
        "status": "COMPLETE",
        "source_code": {"path": str(Path(__file__).relative_to(REPO)), "sha256": sha256_file(Path(__file__))},
        "inputs": assets_doc["source_input_pins"],
        "outputs": {
            key: {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in OUTPUTS.items()
            if key != "manifest"
        },
        "row_counts": {"assets": len(assets), "paired_scores": len(paired_rows), "source_groups": len(speaker_members)},
    }
    write_json(OUTPUTS["manifest"], output_manifest)


if __name__ == "__main__":
    main()
