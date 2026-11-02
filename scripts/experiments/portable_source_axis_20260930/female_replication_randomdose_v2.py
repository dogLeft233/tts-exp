"""Female cohort runner adapter using the frozen actual-float32 dose op."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import evaluate, female_replication as female, ops, randomdose_f32_v2

EXPECTED_RANDOM_SPEC_SHA256 = "679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79"
_ORIGINAL_MAKE_CONTEXT = evaluate._make_context
_ORIGINAL_CODE_HASHES = female._code_hashes
_ORIGINAL_WRITE_SEAL = female._write_seal


def _sha(path: str | Path) -> str:
    return female._sha(path)


def _make_context_with_corrected_random_dose(*args: Any, **kwargs: Any):
    """Route only the three random interventions through the v2 float32 op."""
    original_apply = ops.apply_random_direction

    def apply_v2(z, speech_mask, direction, u, orth_plus, **operation_kwargs):
        return randomdose_f32_v2.apply_random_direction_f32_dose_v2(
            z, speech_mask, direction, u, orth_plus, **operation_kwargs,
        )

    ops.apply_random_direction = apply_v2
    try:
        return _ORIGINAL_MAKE_CONTEXT(*args, **kwargs)
    finally:
        ops.apply_random_direction = original_apply


def _code_hashes_v2() -> dict[str, str]:
    hashes = _ORIGINAL_CODE_HASHES()
    package = Path(__file__).resolve().parent
    hashes["female_replication_randomdose_v2.py"] = _sha(__file__)
    hashes["randomdose_f32_v2.py"] = _sha(package / "randomdose_f32_v2.py")
    return hashes


def _write_seal_v2(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    adapter_map: list[dict[str, Any]],
    adapter_manifest_sha: str,
    paired_hash: str,
    shard_shas: tuple[str, str],
    calibration: dict[str, Any],
    d,
    u,
) -> dict[str, Any]:
    protocol = _ORIGINAL_WRITE_SEAL(
        args, rows, manifest, adapter_map, adapter_manifest_sha,
        paired_hash, shard_shas, calibration, d, u,
    )
    spec_path = Path(args.random_specificity_spec).resolve(strict=True)
    protocol["random_specificity"] = {
        "spec_path": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "spec_expected_sha256": EXPECTED_RANDOM_SPEC_SHA256,
        "operation_version": randomdose_f32_v2.OPERATION_VERSION,
        "operation_module_path": str(Path(randomdose_f32_v2.__file__).resolve()),
        "operation_module_sha256": _sha(randomdose_f32_v2.__file__),
        "female_adapter_path": str(Path(__file__).resolve()),
        "female_adapter_sha256": _sha(__file__),
        "target": "saved float32 orth_plus minus exact saved float32 baseline displacement per active frame",
        "candidate_measurement": "float64 norm of cast float32 candidate minus exact saved float32 baseline",
        "acceptance_tolerance": "max(1e-6, 1e-6*target_norm)",
        "multiplier_bounds": [0.0, 64.0],
        "doubling_bracket_start": 1.0,
        "maximum_bisections": 64,
        "failure_policy": "retain each planned row and mark frame-level misses or capped infeasibility",
    }
    protocol["method"]["random_dose_operation_version"] = randomdose_f32_v2.OPERATION_VERSION
    protocol["method"]["random_dose_spec_sha256"] = EXPECTED_RANDOM_SPEC_SHA256
    run_dir = Path(args.run_dir).resolve()
    female._atomic_json(run_dir / "evaluation_protocol.json", protocol)
    female._atomic_json(run_dir / "code_seal.json", {
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": protocol["row_manifest_sha256"],
        "production_code_sha256": protocol["production_code_sha256"],
        "random_specificity_spec_sha256": EXPECTED_RANDOM_SPEC_SHA256,
    })
    return protocol


def build_parser() -> argparse.ArgumentParser:
    parser = female.build_parser()
    parser.add_argument("--random-specificity-spec", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.random_specificity_spec).resolve(strict=True)
    if _sha(spec_path) != EXPECTED_RANDOM_SPEC_SHA256:
        raise ValueError(f"frozen random-specificity spec SHA mismatch: {_sha(spec_path)}")
    if _sha(Path(args.spec).resolve(strict=True)) != female.EXPECTED_SPEC_SHA256:
        raise ValueError("female replication spec SHA differs from its frozen seal")
    summary_path = Path(args.calibration_summary).resolve(strict=True)
    axes_path = Path(args.calibration_axes).resolve(strict=True)
    if _sha(summary_path) != female.EXPECTED_CALIBRATION_SUMMARY_SHA256:
        raise ValueError("male calibration summary SHA differs from its frozen seal")
    if _sha(axes_path) != female.EXPECTED_CALIBRATION_AXES_SHA256:
        raise ValueError("male calibration axes SHA differs from their frozen seal")
    calibration = json.loads(summary_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "calibration_complete":
        raise ValueError("male external CAL summary is not complete")
    anchor = calibration.get("global_anchor", {})
    observed_anchor = {
        "k": int(anchor.get("selected_k", 999)),
        "column": int(anchor.get("anchor_column", -1)),
        "official_offset": int(anchor.get("official_syncnet_offset", 999)),
    }
    if observed_anchor != {"k": 3, "column": 18, "official_offset": -3}:
        raise ValueError(f"female preflight requires frozen male CAL anchor k=3/column18/offset=-3; got {observed_anchor}")
    with np.load(axes_path, allow_pickle=False) as archive:
        d = np.asarray(archive["d_source"], dtype=np.float64)
        u = np.asarray(archive["u_level"], dtype=np.float64)
    if ops.array_sha256(d) != calibration["axes"]["d_sha256"] or ops.array_sha256(u) != calibration["axes"]["u_sha256"]:
        raise ValueError("external male CAL axis arrays differ from their summary bindings")
    if not np.isclose(ops.TARGET_RMS, 0.0376838172675746, rtol=0.0, atol=0.0):
        raise ValueError("female preflight observed a different frozen RMS target")

    input_root = Path(args.input_root).resolve(strict=True)
    manifest_path = Path(args.input_manifest).resolve(strict=True)
    shard_paths = (Path(args.shard_one).resolve(strict=True), Path(args.shard_two).resolve(strict=True))
    input_adapter_root = Path(args.input_adapter_root).resolve()
    manifest, records, _, _, rows, _, paired_hash, shard_shas = female._prepare_rows(
        manifest_path, input_root, input_adapter_root, shard_paths,
    )
    if len(records) != 40 or len({row["speaker"] for row in records}) != 20 or len(rows) != 640:
        raise ValueError("female preflight did not produce the frozen 40-clip/20-speaker/640-row grid")
    for path in (args.wav2lip_checkpoint, args.syncnet_checkpoint, args.ffmpeg, args.primary_face):
        if not Path(path).resolve().is_file():
            raise FileNotFoundError(f"required female run asset missing: {Path(path).resolve()}")
    for path in (args.wav2lip_root, args.syncnet_root):
        if not Path(path).resolve().is_dir():
            raise FileNotFoundError(f"required model checkout missing: {Path(path).resolve()}")
    face = cv2.imread(str(Path(args.primary_face).resolve()))
    if face is None:
        raise ValueError("female primary face image cannot be decoded")
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"female preflight output must be empty: {run_dir}")
    return {
        "status": "preflight_pass",
        "classification": "non-effect female cohort/input/CAL/asset preflight",
        "planned_samples": len(records), "planned_speakers": 20, "planned_rows": len(rows),
        "cohort_pair_identity_sha256": female.EXPECTED_NATURAL_PAIR_IDENTITY_SHA256,
        "paired_audio_identity_sha256": paired_hash,
        "input_manifest_sha256": _sha(manifest_path),
        "embedded_manifest_fingerprint": manifest["manifest_sha256"],
        "transport_shard_sha256": list(shard_shas),
        "calibration_summary_sha256": _sha(summary_path), "calibration_axes_sha256": _sha(axes_path),
        "anchor": observed_anchor,
        "randomdose_operation_version": randomdose_f32_v2.OPERATION_VERSION,
        "randomdose_module_sha256": _sha(randomdose_f32_v2.__file__),
        "female_adapter_sha256": _sha(__file__),
        "production_code_sha256": _code_hashes_v2(),
        "face_sha256": _sha(args.primary_face),
        "wav2lip_checkpoint_sha256": _sha(args.wav2lip_checkpoint),
        "syncnet_checkpoint_sha256": _sha(args.syncnet_checkpoint),
        "python": sys.version.split()[0],
        "output_run_dir": str(run_dir),
    }


def main() -> int:
    args = build_parser().parse_args()
    spec_path = Path(args.random_specificity_spec).resolve(strict=True)
    if _sha(spec_path) != EXPECTED_RANDOM_SPEC_SHA256:
        raise ValueError(f"frozen random-specificity spec SHA mismatch: {_sha(spec_path)}")
    if args.preflight_only:
        print(json.dumps(_preflight(args), ensure_ascii=False, separators=(",", ":")))
        return 0
    evaluate._make_context = _make_context_with_corrected_random_dose
    female._code_hashes = _code_hashes_v2
    female._write_seal = _write_seal_v2
    result = female._run(args)
    print(json.dumps({
        "status": result["status"],
        "completed_rows": result["completed_rows"],
        "failed_rows": result["failed_rows"],
        "run_dir": result["run_dir"],
        "production_code_sha256": result["production_code_sha256"],
    }, separators=(",", ":")))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
