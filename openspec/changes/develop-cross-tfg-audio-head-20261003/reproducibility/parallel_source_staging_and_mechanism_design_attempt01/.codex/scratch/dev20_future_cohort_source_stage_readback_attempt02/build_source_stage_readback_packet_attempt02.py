from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

WORKSPACE = Path(r"E:\Documents\tts-exp")
PACKET = Path(__file__).resolve().parent
UPLOAD_PACKET = WORKSPACE / ".codex/scratch/dev20_future_cohort_upload_packet_attempt04"
DATA = WORKSPACE / ".codex/scratch/dev20_future_cohort_metadata_attempt06"
QC_LOCAL = WORKSPACE / ".codex/scratch/dev20_future_cohort_independent_qc_attempt01/dev20_full_independent_pcm_f32_qc_attempt01.json"
QC_ROOT = WORKSPACE / ".codex/scratch/root_future_dev20_full_pcm_qc_attempt01.json"
SOURCE_PLAN = DATA / "aishell_future_dev20_exact_source_download_plan_attempt01.json"
DOWNLOAD_SOURCE_RECEIPT = DATA / "aishell_future_dev20_download_readback_attempt01.json"
HEADER_RECEIPT = DATA / "aishell_future_dev20_header_eligibility_attempt01.json"
FIT_RECEIPT = DATA / "fit40_existing_source_hash_reconciliation_attempt01.json"
COHORT_PLAN = WORKSPACE / ".codex/scratch/dev20_future_cohort_metadata_attempt02/dev20_future_cohort_metadata_plan_attempt02.json"
README = DATA / "pinned_README_attempt01.md"
AUDIO_IO = WORKSPACE / "scripts/audio_enhancement/audio_io.py"
PORTABLE_BUILDER = DATA / "build_dev20_portable_source_bundle_attempt02.py"
ROOT_QC_REVIEW = WORKSPACE / ".codex/scratch/dev20_future_cohort_independent_qc_attempt01/reconstruct_verify_dev20_pcm_attempt01.py"
PACKET04_REVIEW = WORKSPACE / ".codex/scratch/dev20_future_cohort_upload_packet_review_attempt02/packet04_source_review_attempt01.json"
PACKET03_FAILED_Q00 = WORKSPACE / ".codex/scratch/dev20_future_cohort_upload_packet_attempt03/00_create_remote_source_directory.json.response.json"
PACKET04_INDEX = UPLOAD_PACKET / "request_index_attempt04.json"
BUNDLE = DATA / "future_dev20_source_bundle_attempt02.zip"
MANIFEST = DATA / "future_dev20_portable_source_manifest_attempt02.json"
BUILD_RECEIPT = DATA / "future_dev20_source_bundle_build_receipt_attempt02.json"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compact_sha(obj: dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw)


def raw_pin(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw)}


def write_create_only(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)


def response_ok(response: dict[str, Any]) -> bool:
    trunc = response.get("truncated") or {}
    return (response.get("ok") is True and response.get("code") == 0 and not response.get("timed_out")
            and not response.get("incomplete_stream") and response.get("stream_error") is None
            and not response.get("stderr") and trunc.get("stdout") is False and trunc.get("stderr") is False)


def main() -> None:
    req_dir = PACKET / "requests_attempt02"
    download_dir = PACKET / "downloads_attempt02"
    index_path = PACKET / "request_index_attempt02.json"
    if req_dir.exists() or download_dir.exists() or index_path.exists():
        raise FileExistsError("readback packet outputs already exist; no overwrite/retry")
    req_dir.mkdir()
    download_dir.mkdir()
    local_target = download_dir / "source_stage_qc_attempt01.json"
    if local_target.exists() or local_target.is_symlink():
        raise FileExistsError("download target must be absent")

    upload_index = json.loads(PACKET04_INDEX.read_bytes())
    upload_index_self = upload_index.pop("self_sha256", None)
    if compact_sha(upload_index) != upload_index_self:
        raise RuntimeError("packet04 request-index self-hash mismatch")
    upload_index["self_sha256"] = upload_index_self
    if upload_index["status"] != "PREPARED_ONLY_NOT_DISPATCHED":
        raise RuntimeError("packet04 index status unexpected")
    if len(upload_index["requests"]) != 5 or len(upload_index["source_evidence"]) != 38:
        raise RuntimeError("packet04 request/evidence count mismatch")
    for pin in upload_index["source_evidence"].values():
        observed = raw_pin(Path(pin["path"]))
        if observed["bytes"] != pin["bytes"] or observed["sha256"] != pin["sha256"]:
            raise RuntimeError(f"packet04 source-evidence pin mismatch: {pin['path']}")
    for pin in upload_index["requests"]:
        if (pin["path"] + ".response.json") == str(local_target):
            raise RuntimeError("unexpected response path collision")

    request_responses: list[tuple[dict[str, Any], dict[str, Any], Path, Path]] = []
    for pin in upload_index["requests"]:
        request_path = Path(pin["path"])
        request_raw = request_path.read_bytes()
        if len(request_raw) != pin["bytes"] or sha256(request_raw) != pin["sha256"]:
            raise RuntimeError(f"packet04 request pin mismatch: {request_path.name}")
        req = json.loads(request_raw)
        response_path = Path(str(request_path) + ".response.json")
        response_raw = response_path.read_bytes()
        response = json.loads(response_raw)
        if req.get("op") == "exec":
            if not response_ok(response):
                raise RuntimeError(f"packet04 exec not successful: {request_path.name}")
        elif req.get("op") == "upload":
            if (response.get("ok") is not True or response.get("local") != req.get("local")
                    or response.get("remote") != req.get("remote")
                    or response.get("size") != req.get("expected_bytes")
                    or response.get("sha256") != req.get("expected_sha256")):
                raise RuntimeError(f"packet04 upload response mismatch: {request_path.name}")
        else:
            raise RuntimeError(f"unexpected packet04 operation: {request_path.name}")
        request_responses.append((req, response, request_path, response_path))
    if len(request_responses) != 5:
        raise RuntimeError("expected five actual packet04 responses")

    # Validate q00's actual diagnostic and create-only directory result.
    q00_req, q00_response, _, _ = request_responses[0]
    q00_summary = json.loads(q00_response["stdout"].strip())
    expected_root = upload_index["remote_root"]
    if (q00_req["op"] != "exec" or q00_summary.get("status") != "REMOTE_SOURCE_DIRECTORY_CREATED_CREATE_ONLY"
            or q00_summary.get("path") != expected_root):
        raise RuntimeError("actual q00 response is not the expected create-only directory result")
    checks = q00_summary.get("checks", {})
    if (checks.get("old_ditto_interpreter", {}).get("exists") is not False
            or checks.get("selected_interpreter", {}).get("executable") is not True
            or checks.get("research_namespace_parent", {}).get("is_dir") is not True
            or checks.get("remote_root", {}).get("exists") is not False):
        raise RuntimeError("q00 diagnostic checks differ from expected interpreter/namespace outcome")

    # Verify q01-q03 transfers match each request and source file bytes.
    for index in (1, 2, 3):
        req, response, _, _ = request_responses[index]
        if req["op"] != "upload" or response.get("local") != req["local"] or response.get("remote") != req["remote"]:
            raise RuntimeError(f"upload response/request mismatch at q{index:02d}")
        if response.get("size") != req["expected_bytes"] or response.get("sha256") != req["expected_sha256"]:
            raise RuntimeError(f"upload response pin mismatch at q{index:02d}")
        source_pin = raw_pin(Path(req["local"]))
        if source_pin["bytes"] != response["size"] or source_pin["sha256"] != response["sha256"]:
            raise RuntimeError(f"local upload source differs from actual response at q{index:02d}")

    q04_req, q04_response, q04_path, q04_response_path = request_responses[4]
    if q04_req["op"] != "exec":
        raise RuntimeError("q04 is not an exec request")
    out = q04_response.get("stdout", "").strip()
    summary = json.loads(out)
    if (summary.get("status") != "ALL_20_SOURCE_AND_RAW_F32_VERIFIED_CREATE_ONLY"
            or summary.get("source_file_count") != 20 or summary.get("canonical_f32_file_count") != 20
            or summary.get("sample_clock_preserved") is not True or summary.get("model_or_gpu_run") is not False):
        raise RuntimeError("q04 stdout did not advertise full 20-file source/F32 receipt")
    if not summary.get("receipt_path", "").startswith(expected_root + "/source_stage_attempt01/"):
        raise RuntimeError("q04 advertised receipt outside the create-only source-stage root")
    advertised_bytes = summary.get("receipt_bytes")
    advertised_sha = summary.get("receipt_sha256")
    if not isinstance(advertised_bytes, int) or advertised_bytes <= 0 or not isinstance(advertised_sha, str) or len(advertised_sha) != 64:
        raise RuntimeError("q04 advertised receipt size/hash malformed")
    if q04_response.get("stderr") or q04_response.get("code") != 0:
        raise RuntimeError("q04 did not complete cleanly")

    source_manifest = json.loads(MANIFEST.read_bytes())
    manifest_raw = MANIFEST.read_bytes()
    bundle_raw = BUNDLE.read_bytes()
    if len(manifest_raw) != 24072 or sha256(manifest_raw) != "fff18619b7be8f992002f99c33121ef3eb25acce6aa3caa3da55f398eb3faffa":
        raise RuntimeError("portable manifest changed from sealed pin")
    if len(bundle_raw) != 3548546 or sha256(bundle_raw) != "6f9de4953da6fd52a4e916604c21b0fd187ab145fefc92ea47cd8e274a460d9a":
        raise RuntimeError("portable source bundle changed from sealed pin")
    q01 = request_responses[1][1]
    q02 = request_responses[2][1]
    if q01.get("sha256") != sha256(bundle_raw) or q02.get("sha256") != sha256(manifest_raw):
        raise RuntimeError("actual uploads do not match local source/manifest bytes")

    receipt_expectation = {
        "remote_path": summary["receipt_path"],
        "bytes": advertised_bytes,
        "sha256": advertised_sha,
        "advertised_by_request": {"path": str(q04_path.resolve()), "bytes": len(q04_path.read_bytes()), "sha256": sha256(q04_path.read_bytes())},
        "advertised_by_response": {"path": str(q04_response_path.resolve()), "bytes": len(q04_response_path.read_bytes()), "sha256": sha256(q04_response_path.read_bytes())},
        "summary_status": summary["status"],
    }
    request = {
        "op": "download",
        "remote": receipt_expectation["remote_path"],
        "local": str(local_target.resolve()),
        "timeout": 120,
    }
    request_path = req_dir / "00_download_source_stage_qc_receipt.json"
    request_raw = json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    write_create_only(request_path, request_raw)
    if local_target.exists() or local_target.is_symlink():
        raise RuntimeError("download target must remain absent until ROOT dispatch")

    evidence_paths: dict[str, Path] = {
        "portable_source_bundle": BUNDLE,
        "portable_source_manifest": MANIFEST,
        "source_bundle_build_receipt": BUILD_RECEIPT,
        "source_download_readback_receipt": DOWNLOAD_SOURCE_RECEIPT,
        "source_download_plan": SOURCE_PLAN,
        "source_header_selection_receipt": HEADER_RECEIPT,
        "fit40_lfs_reconciliation_receipt": FIT_RECEIPT,
        "frozen_cohort_metadata_plan": COHORT_PLAN,
        "pinned_source_readme": README,
        "audio_io_reader_source": AUDIO_IO,
        "portable_bundle_builder_source": PORTABLE_BUILDER,
        "independent_local_pcm_f32_qc_source": ROOT_QC_REVIEW,
        "independent_local_pcm_f32_qc_receipt": QC_LOCAL,
        "root_local_pcm_f32_qc_receipt": QC_ROOT,
        "upload_packet04_builder": UPLOAD_PACKET / "build_dev20_upload_packet_attempt04.py",
        "upload_packet04_worker02": UPLOAD_PACKET / "extract_and_qc_dev20_source_stage_attempt02.py",
        "upload_packet04_index": PACKET04_INDEX,
        "packet03_q00_failed_response": PACKET03_FAILED_Q00,
        "packet04_independent_review_receipt": PACKET04_REVIEW,
        "previous_readback_builder_attempt01": WORKSPACE / ".codex/scratch/dev20_future_cohort_source_stage_readback_attempt01/build_source_stage_readback_packet_attempt01.py",
        "current_readback_packet_builder": Path(__file__).resolve(),
    }
    for pin in upload_index["requests"]:
        p = Path(pin["path"])
        evidence_paths[f"packet04_request_{pin['name']}"] = p
        evidence_paths[f"packet04_response_{pin['name']}"] = Path(str(p) + ".response.json")
    download_receipt = json.loads(DOWNLOAD_SOURCE_RECEIPT.read_bytes())
    if download_receipt["status"] != "ALL_20_DOWNLOADED_HASH_VERIFIED_AND_CANONICALIZED" or len(download_receipt["files"]) != 20:
        raise RuntimeError("local source download receipt is incomplete")
    for row in download_receipt["files"]:
        evidence_paths[f"source_wav_{row['sample_id']}"] = Path(row["target_path"])

    source_evidence: dict[str, dict[str, Any]] = {}
    for key, path in evidence_paths.items():
        source_evidence[key] = raw_pin(path)
    request_pin = {"name": "00_download_source_stage_qc_receipt", "path": str(request_path.resolve()), "bytes": len(request_raw), "sha256": sha256(request_raw)}
    index: dict[str, Any] = {
        "schema": "aishell_future_dev20_source_stage_receipt_readback_attempt02_v1",
        "status": "READY_FOR_ROOT_DISPATCH_NOT_DISPATCHED",
        "remote_stage_root": upload_index["remote_stage_root"],
        "receipt_expectation": receipt_expectation,
        "download_target": {"path": str(local_target.resolve()), "must_be_absent_before_dispatch": True, "currently_absent": not local_target.exists()},
        "source_evidence": source_evidence,
        "requests": [request_pin],
        "independent_qc_contract": {
            "verify_download_bytes_against_advertised_q04_receipt_bytes_sha256": True,
            "verify_receipt_json_self_hash": True,
            "verify_all_20_ordered_manifest_rows_and_sample_clocks": True,
            "verify_exactly_48_written_output_pins": {"source_manifest": 1, "source_wav": 20, "canonical_float32": 20, "evidence": 7},
            "verify_each_wav_and_f32_pin_against_local_manifest_bundle_and_root_local_raw_f32_qc": True,
            "verify_all_seven_evidence_output_hashes_against_zip_manifest_and_local_evidence": True,
            "asr_scores_model_gpu_or_remote_operation_by_this_builder": False,
        },
        "remote_download_dispatched": False,
    }
    index["self_sha256"] = compact_sha(index)
    index_raw = json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    write_create_only(index_path, index_raw)
    print(json.dumps({"status": index["status"], "index_path": str(index_path.resolve()), "index_bytes": len(index_raw), "index_sha256": sha256(index_raw), "self_sha256": index["self_sha256"], "request_path": str(request_path.resolve()), "request_bytes": len(request_raw), "request_sha256": sha256(request_raw), "expected_receipt_bytes": advertised_bytes, "expected_receipt_sha256": advertised_sha, "source_evidence_count": len(source_evidence), "remote_dispatch": False}, ensure_ascii=False, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
