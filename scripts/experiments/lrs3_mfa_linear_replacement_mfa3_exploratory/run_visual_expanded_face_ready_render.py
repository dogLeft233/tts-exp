from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    PROTOCOL_ID,
    file_sha256,
    load_json,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_strict_replacement import (
    _assert_gpu_ready,
    _render,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_visual_expanded_render import (
    ALIGNMENT_PATH,
    CANDIDATE_PATH,
    STAGE00_PATH,
    _build_render_manifest,
    _validate_inputs,
)

REPO = Path(__file__).resolve().parents[3]
STAGE_ID = "03_visual_expanded_wav2lip_render_retry2"
FACE_PREFLIGHT_ID = "03_visual_expanded_face_preflight_retry1"
FACE_PREFLIGHT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / FACE_PREFLIGHT_ID
    / "face_preflight_manifest.json"
)
OUTPUT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / STAGE_ID
)


def _validate_face_preflight(
    stage00_path: Path,
    candidate_path: Path,
    alignment_path: Path,
    face_preflight_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    if face_preflight_path != FACE_PREFLIGHT_PATH.resolve():
        raise ValueError("render retry2 requires the frozen face preflight path")
    preflight = load_json(face_preflight_path)
    expected_parents = {
        "stage00": stage00_path,
        "candidate": candidate_path,
        "alignment": alignment_path,
    }
    parents = preflight.get("parents")
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("stage_id") != FACE_PREFLIGHT_ID
        or preflight.get("status") != "complete"
        or preflight.get("engineering_decision") != "GO"
        or preflight.get("input_record_count") != 62
        or preflight.get("face_ready_record_count") != 59
        or preflight.get("face_ready_group_count") != 23
        or preflight.get("selection", {}).get("score_based_selection") is not False
        or preflight.get("selection", {}).get("visual_metric_used_for_selection") is not False
        or preflight.get("selection", {}).get("syncnet_used_for_selection") is not False
    ):
        raise ValueError("face preflight is not the complete 59-record engineering GO")
    for key, path in expected_parents.items():
        binding = parents.get(key) if isinstance(parents, dict) else None
        if (
            not isinstance(binding, dict)
            or Path(str(binding.get("path"))).resolve() != path.resolve()
            or str(binding.get("sha256")) != file_sha256(path)
        ):
            raise ValueError(f"face preflight {key} parent mismatch")
    results = list(preflight.get("results", []))
    ready_ids = [str(row["sample_id"]) for row in results if row.get("face_ready") is True]
    manifest_ids = [str(row["sample_id"]) for row in results]
    if len(manifest_ids) != 62 or len(ready_ids) != 59 or len(set(ready_ids)) != 59:
        raise ValueError("face preflight IDs/counts are invalid")
    if preflight.get("face_ready_sample_ids") != ready_ids:
        raise ValueError("face preflight ready ID order mismatch")
    return preflight, ready_ids


def run(
    stage00_path: Path = STAGE00_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    alignment_path: Path = ALIGNMENT_PATH,
    face_preflight_path: Path = FACE_PREFLIGHT_PATH,
    output_dir: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty visual render output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage00_path = stage00_path.resolve()
        candidate_path = candidate_path.resolve()
        alignment_path = alignment_path.resolve()
        face_preflight_path = face_preflight_path.resolve()
        stage00, candidate, _ = _validate_inputs(stage00_path, candidate_path, alignment_path)
        preflight, ready_ids = _validate_face_preflight(
            stage00_path,
            candidate_path,
            alignment_path,
            face_preflight_path,
        )
        manifest = _build_render_manifest(
            stage00_path,
            candidate_path,
            alignment_path,
            stage00,
            candidate,
        )
        ready_set = set(ready_ids)
        manifest["cohort"]["records"] = [
            row for row in manifest["cohort"]["records"] if str(row["sample_id"]) in ready_set
        ]
        if len(manifest["cohort"]["records"]) != 59:
            raise ValueError("face-ready render cohort count mismatch")
        if len({str(row["source_group"]) for row in manifest["cohort"]["records"]}) != 23:
            raise ValueError("face-ready render cohort lost group coverage")
        manifest["cohort"]["record_count"] = len(manifest["cohort"]["records"])
        manifest["cohort"]["source_group_count"] = len({
            str(row["source_group"]) for row in manifest["cohort"]["records"]
        })
        manifest["stage_id"] = STAGE_ID
        manifest["parents"]["face_preflight"] = {
            "path": str(face_preflight_path),
            "sha256": file_sha256(face_preflight_path),
        }
        manifest["selection"] = {
            **manifest["selection"],
            "face_preflight_used": True,
            "face_ready_record_count": 59,
            "face_failed_record_count": 3,
        }
        gpu_gate = _assert_gpu_ready()
        renders = _render(manifest, output_dir)
        if len(renders) != 118:
            raise ValueError(f"expected 118 rendered videos, found {len(renders)}")
        write_json(output_dir / "protocol_manifest.json", manifest)
        (output_dir / "protocol_manifest.sha256").write_text(
            file_sha256(output_dir / "protocol_manifest.json") + "\n",
            encoding="utf-8",
        )
        write_json(
            output_dir / "render_manifest.json",
            {
                "schema_version": 1,
                "manifest_type": "lrs3_mfa3_visual_expanded_face_ready_wav2lip_render_results",
                "stage_id": STAGE_ID,
                "protocol_id": PROTOCOL_ID,
                "record_count": 59,
                "render_count": len(renders),
                "renders": renders,
                "gpu_gate": gpu_gate,
                "parents": manifest["parents"],
                "face_ready_sample_ids": ready_ids,
            },
        )
        summary = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "input_record_count": 62,
            "record_count": 59,
            "excluded_face_failed_record_count": 3,
            "source_group_count": 23,
            "render_count": len(renders),
            "expected_render_count": 118,
            "arms": ["natural", "candidate"],
            "gpu_used": True,
            "syncnet_used": False,
            "media_access": manifest["media_access"],
            "face_preflight": {
                "stage_id": preflight["stage_id"],
                "sha256": file_sha256(face_preflight_path),
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "decision.json",
            {
                **summary,
                "next_allowed_stage": "01_visual_teacher_audit",
                "reason": "59 face-ready records rendered with frozen Wav2Lip after score-free preflight",
            },
        )
        return summary
    except Exception as exc:
        partial = len(list((output_dir / "renders").glob("*/*.mp4")))
        payload = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "error": str(exc),
            "partial_render_count": partial,
            "gpu_used": partial > 0,
            "syncnet_used": False,
        }
        write_json(output_dir / "failure.json", payload)
        write_json(output_dir / "summary.json", payload)
        write_json(output_dir / "decision.json", {**payload, "next_allowed_stage": None})
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=STAGE00_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument("--face-preflight", type=Path, default=FACE_PREFLIGHT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run(args.stage00, args.candidate, args.alignment, args.face_preflight, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
