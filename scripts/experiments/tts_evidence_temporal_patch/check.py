from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    MAX_GPU_MINUTES,
    MAX_NEW_RENDER_CELLS,
    PROTOCOL_ID,
    ProtocolError,
    RunPaths,
    canonical_hash,
    decoded_pcm_sha256,
    file_sha256,
    read_json,
    verify_artifact_hash,
    write_json,
)


def _check_file_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.is_file():
        raise ProtocolError(f"missing {label}: {path}")
    if expected and file_sha256(path) != expected:
        raise ProtocolError(f"{label} hash changed: {path}")


def validate_run(run_root: Path, *, require_human: bool = False) -> dict[str, Any]:
    """Validate immutable inputs, patch receipts, budgets, and evidence status."""

    paths = RunPaths(run_root.resolve())
    protocol_path = paths.protocol / "protocol.json"
    if not protocol_path.is_file():
        raise ProtocolError(f"protocol is missing: {protocol_path}")
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("protocol_id mismatch")
    verify_artifact_hash(protocol)
    code_bindings = protocol.get("code", {})
    if isinstance(code_bindings, Mapping):
        for name, expected in code_bindings.items():
            current_path = protocol_path.parents[3] / "scripts" / "experiments" / "tts_evidence_temporal_patch" / str(name)
            if expected and (not current_path.is_file() or file_sha256(current_path) != expected):
                raise ProtocolError(f"implementation code changed after protocol freeze: {name}")
    parent_inventory_path = paths.protocol / "parent_inventory.json"
    if not parent_inventory_path.is_file():
        raise ProtocolError("parent inventory is missing")
    inventory = read_json(parent_inventory_path)
    verify_artifact_hash(inventory)
    if int(inventory.get("video_cells", -1)) != 108 or int(inventory.get("source_group_count", -1)) != 36:
        raise ProtocolError("parent inventory does not contain 108 cells/36 source groups")
    selection_path = paths.protocol / "selection.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        verify_artifact_hash(selection)
        rows = selection.get("rows", [])
        groups = [str(row["source_group"]) for row in rows]
        if len(groups) != len(set(groups)):
            raise ProtocolError("selection reuses a source group")
        if len(rows) != 16:
            raise ProtocolError(f"selection has {len(rows)} groups, expected 16")
    render_count = 0
    elapsed_gpu_minutes = 0.0
    receipt_rows: list[dict[str, Any]] = []
    inventory_cells = {
        (str(row.get("source_group")), str(row.get("sample_id")), str(row.get("arm"))): row
        for row in inventory.get("cells", [])
    }
    for path in sorted(paths.patch_video.rglob("receipt.json")) if paths.patch_video.is_dir() else []:
        receipt = read_json(path)
        verify_artifact_hash(receipt)
        if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("status") != "complete":
            raise ProtocolError(f"invalid patch receipt: {path}")
        if receipt.get("latent_layer") != "audio_encoder.output":
            raise ProtocolError(f"patch receipt uses an unregistered layer: {path}")
        if float(receipt.get("lambda", -1)) != 0.5:
            raise ProtocolError(f"patch lambda changed: {path}")
        mapping = receipt.get("mapping")
        if not isinstance(mapping, Mapping) or canonical_hash(mapping) != receipt.get("mapping_hash"):
            raise ProtocolError(f"patch mapping/support hash changed: {path}")
        expected_support_hash = canonical_hash(mapping.get("support_blocks", []))
        if receipt.get("support_hash") != expected_support_hash:
            raise ProtocolError(f"patch support hash changed: {path}")
        condition_names = tuple(str(item) for item in receipt.get("condition_names", []))
        if receipt.get("mode") == "technical_controls" and condition_names != ("BASE", "IDENTITY", "SHIFT_PLUS3", "SHIFT_MINUS3"):
            raise ProtocolError(f"technical patch conditions changed: {path}")
        if receipt.get("mode") == "science_conditions" and condition_names != ("BASE", "COHERENT", "SCRAMBLED", "ERASE"):
            raise ProtocolError(f"science patch conditions changed: {path}")
        recipient_key = (str(receipt.get("source_group")), str(receipt.get("sample_id")), str(receipt.get("recipient_arm")))
        recipient_binding = inventory_cells.get(recipient_key)
        if recipient_binding is None:
            raise ProtocolError(f"patch receipt recipient is not in frozen inventory: {path}")
        recipient_audio = Path(str(receipt.get("recipient_audio", "")))
        if not receipt.get("recipient_audio_sha256") or not receipt.get("recipient_audio_pcm_sha256"):
            raise ProtocolError(f"patch receipt lacks complete recipient audio binding: {path}")
        if not recipient_audio.is_file() or file_sha256(recipient_audio) != receipt.get("recipient_audio_sha256"):
            raise ProtocolError(f"recipient audio binding changed: {path}")
        if str(recipient_binding.get("audio")) != str(recipient_audio) or str(recipient_binding.get("audio_sha256")) != str(receipt.get("recipient_audio_sha256")):
            raise ProtocolError(f"recipient audio is not the frozen inventory cell: {path}")
        donor_key = (str(receipt.get("source_group")), str(receipt.get("sample_id")), str(receipt.get("donor_arm")))
        donor_binding = inventory_cells.get(donor_key)
        if donor_binding is None or str(donor_binding.get("audio")) != str(receipt.get("donor_audio")) or str(donor_binding.get("audio_sha256")) != str(receipt.get("donor_audio_sha256")):
            raise ProtocolError(f"donor audio is not the frozen inventory cell: {path}")
        if receipt.get("recipient_audio_pcm_sha256"):
            ffmpeg = Path("/home/wjj/miniconda3/bin/ffmpeg")
            if ffmpeg.is_file() and decoded_pcm_sha256(recipient_audio, ffmpeg) != receipt["recipient_audio_pcm_sha256"]:
                raise ProtocolError(f"recipient PCM binding changed: {path}")
        for condition, output in receipt.get("outputs", {}).items():
            media = Path(str(output.get("video", "")))
            if not media.is_file() or file_sha256(media) != output.get("video_sha256"):
                raise ProtocolError(f"patch video binding changed: {path}/{condition}")
            if receipt.get("recipient_audio_pcm_sha256") and output.get("audio_pcm_sha256") != receipt["recipient_audio_pcm_sha256"]:
                raise ProtocolError(f"patch output PCM differs from recipient: {path}/{condition}")
            if receipt.get("recipient_audio_pcm_sha256"):
                ffmpeg = Path("/home/wjj/miniconda3/bin/ffmpeg")
                if ffmpeg.is_file() and decoded_pcm_sha256(media, ffmpeg) != receipt["recipient_audio_pcm_sha256"]:
                    raise ProtocolError(f"decoded patch output PCM differs from recipient: {path}/{condition}")
        if set(receipt.get("outputs", {})) != set(condition_names):
            raise ProtocolError(f"patch output conditions do not match receipt declaration: {path}")
        render_count += len(receipt.get("outputs", {}))
        elapsed_gpu_minutes += float(receipt.get("elapsed_seconds", 0.0)) / 60.0
        if not receipt.get("hook_removed", False):
            raise ProtocolError(f"patch hook cleanup is not proven: {path}")
        receipt_rows.append({"path": str(path), "source_group": receipt.get("source_group"), "direction": receipt.get("direction"), "outputs": sorted(receipt.get("outputs", {}))})
    if render_count > MAX_NEW_RENDER_CELLS:
        raise ProtocolError(f"new render count exceeds budget: {render_count}")
    if elapsed_gpu_minutes > MAX_GPU_MINUTES + 1e-9:
        raise ProtocolError(f"GPU budget exceeded: {elapsed_gpu_minutes:.3f} minutes")
    ratings_path = paths.blind / "ratings.json"
    human_status = "MEASURED" if ratings_path.is_file() else "PENDING_HUMAN"
    if require_human and human_status != "MEASURED":
        raise ProtocolError("human ratings are required but have not been imported")
    analysis_path = paths.analysis / "analysis.json"
    analysis_status = "MISSING"
    if analysis_path.is_file():
        analysis = read_json(analysis_path)
        verify_artifact_hash(analysis)
        analysis_status = str(analysis.get("status", "UNKNOWN"))
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE_ENGINEERING" if analysis_status != "MISSING" else "INCOMPLETE",
        "scientific_status": "PENDING_HUMAN" if human_status != "MEASURED" else "HUMAN_EVIDENCE_AVAILABLE",
        "human_status": human_status,
        "parent_cells": int(inventory["video_cells"]),
        "parent_groups": int(inventory["source_group_count"]),
        "new_render_cells": render_count,
        "gpu_minutes": elapsed_gpu_minutes,
        "patch_receipts": receipt_rows,
        "analysis_status": analysis_status,
        "checked_at_epoch": time.time(),
    }
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    write_json(paths.validation, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a TTS evidence/temporal patch run")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--require-human", action="store_true")
    args = parser.parse_args(argv)
    validate_run(args.run_root, require_human=args.require_human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
