#!/usr/bin/env python3
"""22_dose_response_syncnet.py — Downstream causal evaluation of audio interventions.

For each of the 8 audio interventions defined in script 20, this script:
  1. Applies the transformation to the source audio (TTS or natural).
  2. Writes the transformed audio to disk.
  3. Feeds it through Ditto TFG inference with the sample's reference image.
  4. Runs SyncNet on the resulting video.
  5. Computes ΔSync-C / ΔSync-D versus the original diagonal baseline
     (G_tts_E_tts for TTS-source interventions, G_nat_E_nat for natural-source).

The Ditto pipeline used here mirrors the strict run config (``ditto.loudnorm:
false``): no loudness normalisation is applied to the audio before Ditto, so
LUFS interventions remain causally meaningful.

Output: data/wav2sem_analysis/metrics/dose_response.json

Usage:
    python scripts/22_dose_response_syncnet.py --run-id aishell1_strict_20260707T081223Z --dry-run
    python scripts/22_dose_response_syncnet.py --run-id aishell1_strict_20260707T081223Z --samples 3
    python scripts/22_dose_response_syncnet.py --run-id aishell1_strict_20260707T081223Z --interventions LUFS_match_tts_to_nat
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import (
    STUDY_SAMPLES,
    TARGET_SR,
    load_audio_mono,
    ensure_output_dirs,
)


# ---------------------------------------------------------------------------
# Dynamic module loading (scripts start with digits so normal imports fail)
# ---------------------------------------------------------------------------


def _load_script_module(filename: str):
    """Load a sibling script as a module by file path.

    Registers the module in ``sys.modules`` before execution so that
    ``@dataclass`` (which inspects ``sys.modules[cls.__module__]``) works.
    """
    path = _REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ditto_mod = _load_script_module("03_ditto.py")
_intv_mod = _load_script_module("20_causal_feature_interventions.py")
_gxe_mod = _load_script_module("19_generate_eval_matrix.py")

run_ditto = _ditto_mod.run_ditto
parse_syncnet_output = _gxe_mod.parse_syncnet_output
run_syncnet_pipeline = _gxe_mod.run_syncnet_pipeline
probe_duration = _gxe_mod.probe_duration

apply_lufs_match = _intv_mod.apply_lufs_match
apply_spectral_tilt_match = _intv_mod.apply_spectral_tilt_match
apply_dynamic_transform = _intv_mod.apply_dynamic_transform
compute_lufs = _intv_mod.compute_lufs
compute_spectral_tilt = _intv_mod.compute_spectral_tilt


# ---------------------------------------------------------------------------
# Intervention registry
# ---------------------------------------------------------------------------


@dataclass
class Intervention:
    """Single causal intervention specification."""

    name: str
    source: str  # "tts" or "natural"
    baseline_cond: str  # "tts_raw" or "natural_raw"
    transform_description: str
    expected_sync_direction: str  # "decrease", "increase", or "unknown"
    # transform: (y_source, y_counterpart, sr, sid) -> y_transformed
    transform: callable  # type: ignore[type-arg]
    parameters: dict[str, object] | None = None


def _build_interventions() -> list[Intervention]:
    """Define the 8 causal interventions with stable parameter metadata.

    The metadata is part of the stage cache fingerprint. A transform change
    must therefore invalidate transformed audio and all downstream products.
    """
    return [
        Intervention(
            name="LUFS_match_tts_to_nat",
            source="tts",
            baseline_cond="tts_raw",
            transform_description="Scale TTS loudness to match natural LUFS",
            expected_sync_direction="decrease",
            parameters={"target": "natural_lufs"},
            transform=lambda y_tts, y_nat, sr, sid: apply_lufs_match(
                y_tts, sr, compute_lufs(y_nat, sr)
            ),
        ),
        Intervention(
            name="LUFS_match_nat_to_tts",
            source="natural",
            baseline_cond="natural_raw",
            transform_description="Scale natural loudness to match TTS LUFS",
            expected_sync_direction="increase",
            parameters={"target": "tts_lufs"},
            transform=lambda y_tts, y_nat, sr, sid: apply_lufs_match(
                y_nat, sr, compute_lufs(y_tts, sr)
            ),
        ),
        Intervention(
            name="spectral_tilt_match_tts_to_nat",
            source="tts",
            baseline_cond="tts_raw",
            transform_description="Reshape TTS spectrum to match natural tilt",
            expected_sync_direction="decrease",
            parameters={"target": "natural_spectral_tilt"},
            transform=lambda y_tts, y_nat, sr, sid: apply_spectral_tilt_match(
                y_tts, sr, compute_spectral_tilt(y_nat, sr)
            ),
        ),
        Intervention(
            name="spectral_tilt_match_nat_to_tts",
            source="natural",
            baseline_cond="natural_raw",
            transform_description="Reshape natural spectrum to match TTS tilt",
            expected_sync_direction="increase",
            parameters={"target": "tts_spectral_tilt"},
            transform=lambda y_tts, y_nat, sr, sid: apply_spectral_tilt_match(
                y_nat, sr, compute_spectral_tilt(y_tts, sr)
            ),
        ),
        Intervention(
            name="dynamic_compression_tts_compress",
            source="tts",
            baseline_cond="tts_raw",
            transform_description="Compress TTS dynamic range (exp=0.7)",
            expected_sync_direction="unknown",
            parameters={"exponent": 0.7},
            transform=lambda y_tts, y_nat, sr, sid: apply_dynamic_transform(
                y_tts, sr, 0.7
            ),
        ),
        Intervention(
            name="dynamic_compression_nat_compress",
            source="natural",
            baseline_cond="natural_raw",
            transform_description="Compress natural dynamic range (exp=0.7)",
            expected_sync_direction="unknown",
            parameters={"exponent": 0.7},
            transform=lambda y_tts, y_nat, sr, sid: apply_dynamic_transform(
                y_nat, sr, 0.7
            ),
        ),
        Intervention(
            name="dynamic_expansion_tts_expand",
            source="tts",
            baseline_cond="tts_raw",
            transform_description="Expand TTS dynamic range (exp=1.3)",
            expected_sync_direction="unknown",
            parameters={"exponent": 1.3},
            transform=lambda y_tts, y_nat, sr, sid: apply_dynamic_transform(
                y_tts, sr, 1.3
            ),
        ),
        Intervention(
            name="dynamic_expansion_nat_expand",
            source="natural",
            baseline_cond="natural_raw",
            transform_description="Expand natural dynamic range (exp=1.3)",
            expected_sync_direction="unknown",
            parameters={"exponent": 1.3},
            transform=lambda y_tts, y_nat, sr, sid: apply_dynamic_transform(
                y_nat, sr, 1.3
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Baseline lookup (reuses existing diagonal SyncNet results from 04_eval/)
# ---------------------------------------------------------------------------


def load_baseline_results(eval_base: Path, conditions: list[str]) -> dict[str, dict[int, dict]]:
    """Return {condition: {sample_id: {sync_c, sync_d, av_offset}}}.

    Mirrors script 19's ``find_existing_results`` but normalised for use here.
    """
    baselines: dict[str, dict[int, dict]] = {}
    for cond in conditions:
        cond_dir = eval_base / cond
        if not cond_dir.is_dir():
            continue
        baselines[cond] = {}
        for sid_dir in sorted(cond_dir.iterdir()):
            if not sid_dir.is_dir():
                continue
            try:
                sid = int(sid_dir.name)
            except ValueError:
                continue
            sj = sid_dir / "syncnet.json"
            if sj.exists():
                try:
                    data = json.loads(sj.read_text())
                    baselines[cond][sid] = {
                        k: data[k] for k in ("sync_c", "sync_d", "av_offset") if k in data
                    }
                except (json.JSONDecodeError, KeyError):
                    pass
    return baselines


# ---------------------------------------------------------------------------
# Pipeline stages: transform → save → Ditto → SyncNet
# ---------------------------------------------------------------------------


def save_transformed_audio(y: np.ndarray, sr: int, path: Path) -> None:
    """Write a transformed waveform as 16-bit PCM at the chosen sample rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


def _write_sidecar(path: Path, payload: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cache_matches(path: Path, expected: dict) -> bool:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    integrity = {
        "output_sha256": metadata.get("output_sha256"),
        "output_size": metadata.get("output_size"),
    }
    if integrity["output_sha256"] != _sha256_file(path):
        return False
    if integrity["output_size"] != path.stat().st_size:
        return False
    observed = {
        key: value
        for key, value in metadata.items()
        if key not in {"output_sha256", "output_size"}
    }
    return observed == expected


def _stage_fingerprint(
    *,
    stage: str,
    sid: int,
    run_id: str,
    intervention: Intervention,
    audio_sha256: str | None,
    natural_sha256: str,
    tts_sha256: str,
    image_sha256: str | None,
    cfg: dict,
    seed: int | None,
    upstream_sha256: str | None = None,
    ditto_mode: str | None = None,
    syncnet_model_sha256: str | None = None,
    syncnet_evaluator: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict:
    """Build a deterministic dependency record for one pipeline stage."""
    return {
        "schema_version": 1,
        "stage": stage,
        "sample_id": sid,
        "run_id": run_id,
        "intervention": {
            "name": intervention.name,
            "source": intervention.source,
            "parameters": intervention.parameters or {},
            "description": intervention.transform_description,
        },
        "audio_sha256": audio_sha256,
        "natural_sha256": natural_sha256,
        "tts_sha256": tts_sha256,
        "image_sha256": image_sha256,
        "config": cfg,
        "seed": seed,
        "upstream_sha256": upstream_sha256,
        "ditto_mode": ditto_mode,
        "syncnet_model_sha256": syncnet_model_sha256,
        "syncnet_evaluator": syncnet_evaluator,
        "extra": extra or {},
    }


def _cache_payload(fingerprint: dict, path: Path) -> dict:
    """Attach output integrity data to a dependency fingerprint."""
    return {
        **fingerprint,
        "output_sha256": _sha256_file(path),
        "output_size": path.stat().st_size,
    }


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _valid_syncnet_result(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return _finite_metric(value.get("sync_c")) is not None


def _valid_syncnet_cache(value: object) -> bool:
    return _valid_syncnet_result(value)


def run_intervention_pipeline(
    intervention: Intervention,
    sid: int,
    y_tts: np.ndarray,
    y_nat: np.ndarray,
    sr: int,
    img_path: Path,
    audio_out_dir: Path,
    video_out_dir: Path,
    eval_out_dir: Path,
    repo: Path,
    cfg: dict,
    run_id: str,
    skip_audio: bool = True,
    skip_video: bool = True,
    skip_syncnet: bool = True,
    dry_run: bool = False,
    seed: int | None = None,
) -> tuple[dict | None, str]:
    """Run one (intervention, sample) cell end-to-end.

    Returns (syncnet_result_dict, status_string).
    Each stage is cached independently when its ``skip_*`` flag is true.
    """
    audio_path = audio_out_dir / f"{sid}.wav"
    video_path = video_out_dir / f"{sid}.mp4"
    syncnet_json = eval_out_dir / str(sid) / "syncnet.json"
    natural_sha256 = _sha256_bytes(y_nat)
    tts_sha256 = _sha256_bytes(y_tts)
    image_sha256 = _sha256_file(img_path) if img_path.is_file() else None
    source_fingerprint = _stage_fingerprint(
        stage="audio",
        sid=sid,
        run_id=run_id,
        intervention=intervention,
        audio_sha256=None,
        natural_sha256=natural_sha256,
        tts_sha256=tts_sha256,
        image_sha256=image_sha256,
        cfg=cfg,
        seed=seed,
    )

    # Stage 1 — transform + write audio. Filename-only caches are legacy misses.
    if skip_audio and _cache_matches(audio_path, source_fingerprint):
        audio_status = "cached"
    else:
        if dry_run:
            return None, f"WOULD write transformed audio → {audio_path}"
        try:
            y_out = intervention.transform(y_tts, y_nat, sr, sid)
            save_transformed_audio(y_out, sr, audio_path)
            _write_sidecar(audio_path, _cache_payload(source_fingerprint, audio_path))
            audio_status = "computed"
        except Exception as exc:
            return None, f"transform failed: {exc}"

    audio_sha256 = _sha256_file(audio_path) if audio_path.is_file() else None
    video_fingerprint = _stage_fingerprint(
        stage="video",
        sid=sid,
        run_id=run_id,
        intervention=intervention,
        audio_sha256=audio_sha256,
        natural_sha256=natural_sha256,
        tts_sha256=tts_sha256,
        image_sha256=image_sha256,
        cfg=cfg,
        seed=seed,
        upstream_sha256=audio_sha256,
    )

    # Stage 2 — Ditto inference. Audio changes transitively invalidate video.
    if skip_video and _cache_matches(video_path, video_fingerprint):
        video_status = "cached"
    else:
        if dry_run:
            return None, f"WOULD run Ditto on {audio_path} → {video_path}"
        if not audio_path.exists():
            return None, "audio missing for Ditto"
        ok = run_ditto(
            repo, run_id, cfg, audio_path, img_path, video_path, "trt_online", seed=seed
        )
        ditto_mode = "trt_online"
        if not ok:
            ok = run_ditto(
                repo, run_id, cfg, audio_path, img_path, video_path, "pytorch", seed=seed
            )
            ditto_mode = "pytorch"
        video_status = "computed" if ok else "ditto-failed"
        if not ok:
            return None, "ditto inference failed"
        video_fingerprint["ditto_mode"] = ditto_mode
        _write_sidecar(video_path, _cache_payload(video_fingerprint, video_path))

    video_sha256 = _sha256_file(video_path) if video_path.is_file() else None
    syncnet_dir = repo / cfg["paths"]["syncnet_repo"]
    syncnet_model = syncnet_dir / "data" / "syncnet_v2.model"
    syncnet_fingerprint = _stage_fingerprint(
        stage="syncnet",
        sid=sid,
        run_id=run_id,
        intervention=intervention,
        audio_sha256=audio_sha256,
        natural_sha256=natural_sha256,
        tts_sha256=tts_sha256,
        image_sha256=image_sha256,
        cfg=cfg,
        seed=seed,
        upstream_sha256=video_sha256,
        syncnet_model_sha256=(
            _sha256_file(syncnet_model) if syncnet_model.is_file() else None
        ),
        syncnet_evaluator="19_generate_eval_matrix.parse_syncnet_output",
    )

    # Stage 3 — SyncNet. Require a sidecar and parseable Sync-C; legacy JSON is a miss.
    if skip_syncnet and _cache_matches(syncnet_json, syncnet_fingerprint):
        try:
            cached = json.loads(syncnet_json.read_text())
            if _valid_syncnet_cache(cached):
                return (
                    {k: cached[k] for k in ("sync_c", "sync_d", "av_offset") if k in cached},
                    f"cached ({audio_status}/{video_status}/syncnet-cached)",
                )
        except (OSError, json.JSONDecodeError):
            pass

    if dry_run:
        return None, f"WOULD run SyncNet on {video_path}"

    syncnet_dir = repo / cfg["paths"]["syncnet_repo"]
    syn_env = Path(cfg["paths"]["envs_dir"]) / "syncnet"
    syncnet_python = str(syn_env / "bin" / "python")
    syncnet_bin = str(syn_env / "bin")
    data_dir = eval_out_dir / str(sid) / "syncnet_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    reference = f"{intervention.name}_{sid}"

    for attempt in range(2):
        try:
            stdout, stderr = run_syncnet_pipeline(
                syncnet_dir,
                syncnet_python,
                syncnet_bin,
                video_path,
                data_dir,
                reference,
                syncnet_model,
            )
            if stdout is None:
                if attempt == 1:
                    return None, f"syncnet failed: {stderr}"
                time.sleep(1)
                continue
            parsed = parse_syncnet_output(stdout)
            if parsed:
                parsed_out = {
                    k: parsed[k] for k in ("sync_c", "sync_d", "av_offset") if k in parsed
                }
                parsed_out["sample_id"] = sid
                parsed_out["intervention"] = intervention.name
                syncnet_json.parent.mkdir(parents=True, exist_ok=True)
                syncnet_json.write_text(json.dumps(parsed_out, indent=2))
                _write_sidecar(
                    syncnet_json,
                    _cache_payload(syncnet_fingerprint, syncnet_json),
                )
                return parsed_out, f"{audio_status}/{video_status}/syncnet-computed"
            return None, "syncnet parse failed"
        except Exception as exc:
            if attempt == 1:
                return None, f"syncnet exception: {exc}"
            time.sleep(1)
    return None, "syncnet exhausted retries"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: object) -> float | None:
    if not isinstance(value, (int, float, str, np.number)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None




def aggregate_deltas(
    results: dict[str, dict[int, dict]],
    baselines: dict[str, dict[int, dict]],
    interventions: list[Intervention],
) -> dict:
    """Compute per-intervention ΔSync-C / ΔSync-D vs original diagonal baseline.

    Returns a JSON-serialisable dict with one entry per intervention.
    """
    summary: dict = {}
    for iv in interventions:
        per_sample = results.get(iv.name, {})
        baseline_cond = baselines.get(iv.baseline_cond, {})

        delta_c_list: list[float] = []
        delta_d_list: list[float] = []
        per_sample_records: dict[str, dict] = {}

        for sid in sorted(per_sample.keys()):
            res = per_sample[sid]
            base = baseline_cond.get(sid)
            if base is None or "sync_c" not in res or "sync_c" not in base:
                continue
            res_c = _finite_metric(res.get("sync_c"))
            base_c = _finite_metric(base.get("sync_c"))
            if res_c is None or base_c is None:
                continue
            dc = res_c - base_c
            res_d = _finite_metric(res.get("sync_d"))
            base_d = _finite_metric(base.get("sync_d"))
            dd = res_d - base_d if res_d is not None and base_d is not None else None
            delta_c_list.append(dc)
            if dd is not None:
                delta_d_list.append(dd)
            per_sample_records[str(sid)] = {
                "intervention_sync_c": res["sync_c"],
                "intervention_sync_d": res.get("sync_d"),
                "baseline_sync_c": base["sync_c"],
                "baseline_sync_d": base.get("sync_d"),
                "delta_c": dc,
                "delta_d": dd,
            }

        n = len(delta_c_list)
        summary[iv.name] = {
            "source": iv.source,
            "baseline_cond": iv.baseline_cond,
            "description": iv.transform_description,
            "expected_sync_direction": iv.expected_sync_direction,
            "n_sync_c": len(delta_c_list),
            "n_sync_d": len(delta_d_list),
            "n_samples": len(delta_c_list),
            "mean_delta_c": float(np.mean(delta_c_list)) if n else None,
            "mean_delta_d": float(np.mean(delta_d_list)) if delta_d_list else None,
            "std_delta_c": float(np.std(delta_c_list, ddof=1)) if n > 1 else None,
            "std_delta_d": float(np.std(delta_d_list, ddof=1)) if len(delta_d_list) > 1 else None,
            "per_sample": per_sample_records,
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dose-response SyncNet evaluation of Phase 2 audio interventions"
    )
    ap.add_argument("--run-id", required=True, help="Run identifier (e.g., aishell1_strict_20260707T081223Z)")
    ap.add_argument(
        "--samples",
        default=",".join(str(s) for s in STUDY_SAMPLES),
        help="Comma-separated sample IDs (default: STUDY_SAMPLES)",
    )
    ap.add_argument(
        "--interventions",
        default=None,
        help="Comma-separated intervention names to run (default: all 8)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for dose_response.json (default: data/wav2sem_analysis/metrics)",
    )
    ap.add_argument(
        "--no-cache-audio",
        action="store_true",
        help="Recompute transformed audio even if .wav exists",
    )
    ap.add_argument(
        "--no-cache-video",
        action="store_true",
        help="Re-run Ditto even if .mp4 exists",
    )
    ap.add_argument(
        "--no-cache-syncnet",
        action="store_true",
        help="Re-run SyncNet even if syncnet.json exists",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen, run nothing")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return

    cfg_path = repo / "scripts" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sample_ids = [int(s.strip()) for s in args.samples.split(",") if s.strip()]
    interventions_all = _build_interventions()
    if args.interventions:
        wanted = {n.strip() for n in args.interventions.split(",")}
        interventions = [iv for iv in interventions_all if iv.name in wanted]
        missing = wanted - {iv.name for iv in interventions}
        if missing:
            print(f"ERROR: unknown interventions: {sorted(missing)}")
            return
    else:
        interventions = interventions_all

    # Baselines
    eval_base = run_dir / "04_eval"
    baseline_conds = sorted({iv.baseline_cond for iv in interventions})
    baselines = load_baseline_results(eval_base, baseline_conds)

    # Audio source dirs
    nat_audio_dir = repo / "data" / "data" / "audio"
    tts_audio_dir = run_dir / "02_tts"
    if not tts_audio_dir.is_dir():
        print(f"ERROR: TTS audio dir not found: {tts_audio_dir}")
        return

    img_dir = repo / "data" / "data" / "image"

    # Load audio for all sample IDs once
    natural_audio: dict[int, np.ndarray] = {}
    tts_audio: dict[int, np.ndarray] = {}
    loaded_ids: list[int] = []
    for sid in sample_ids:
        nat_p = nat_audio_dir / f"{sid}.wav"
        tts_p = tts_audio_dir / f"{sid}.wav"
        if nat_p.exists() and tts_p.exists():
            natural_audio[sid], _ = load_audio_mono(nat_p, target_sr=TARGET_SR)
            tts_audio[sid], _ = load_audio_mono(tts_p, target_sr=TARGET_SR)
            loaded_ids.append(sid)
        else:
            print(f"WARN: sample {sid} missing audio (nat={nat_p.exists()}, tts={tts_p.exists()})")

    sr = TARGET_SR

    # Output dirs
    ditto_intv_base = run_dir / "03_ditto" / "interventions"
    eval_intv_base = run_dir / "04_eval" / "interventions"
    out_dir = args.output_dir if args.output_dir else ensure_output_dirs()["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dose] run_id={args.run_id}")
    print(f"[dose] samples loaded: {loaded_ids}")
    print(f"[dose] interventions: {[iv.name for iv in interventions]}")
    print(f"[dose] baselines: {[(c, len(v)) for c, v in baselines.items()]}")
    if args.dry_run:
        print("[dose] DRY-RUN mode — no Ditto/SyncNet will be executed\n")

    results: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
    failures: list[dict] = []
    t0 = time.monotonic()

    for iv in interventions:
        print(f"\n=== {iv.name} ({iv.transform_description}) ===")
        for sid in loaded_ids:
            img_path = img_dir / f"{sid}.png"
            if not img_path.exists():
                failures.append({"intervention": iv.name, "sample_id": sid, "error": "image missing"})
                continue

            audio_out = ditto_intv_base / iv.name
            video_out = ditto_intv_base / iv.name
            eval_out = eval_intv_base / iv.name

            res, status = run_intervention_pipeline(
                intervention=iv,
                sid=sid,
                y_tts=tts_audio[sid],
                y_nat=natural_audio[sid],
                sr=sr,
                img_path=img_path,
                audio_out_dir=audio_out,
                video_out_dir=video_out,
                eval_out_dir=eval_out,
                repo=repo,
                cfg=cfg,
                run_id=args.run_id,
                skip_audio=not args.no_cache_audio,
                skip_video=not args.no_cache_video,
                skip_syncnet=not args.no_cache_syncnet,
                dry_run=args.dry_run,
            )

            if res is None:
                failures.append({"intervention": iv.name, "sample_id": sid, "error": status})
                print(f"  sample {sid}: FAIL — {status}")
            else:
                results[iv.name][sid] = res
                c = _finite_metric(res.get("sync_c"))
                d = _finite_metric(res.get("sync_d"))
                base = baselines.get(iv.baseline_cond, {}).get(sid, {})
                bc = _finite_metric(base.get("sync_c"))
                if c is not None and bc is not None:
                    print(f"  sample {sid}: Sync-C={c:.3f} (base={bc:.3f}, Δ={c - bc:+.3f}) — {status}")
                elif c is not None:
                    print(f"  sample {sid}: Sync-C={c:.3f} (no baseline) — {status}")
                else:
                    print(f"  sample {sid}: Sync-C=n/a (invalid metric) — {status}")

    elapsed = time.monotonic() - t0
    print(f"\n[dose] completed in {elapsed:.0f}s; {sum(len(v) for v in results.values())} ok, {len(failures)} failures")

    if args.dry_run:
        print("[dose] DRY-RUN — no JSON written")
        return

    summary = aggregate_deltas(results, baselines, interventions)

    output = {
        "run_id": args.run_id,
        "samples": loaded_ids,
        "baselines": {
            cond: {str(sid): vals for sid, vals in baselines.get(cond, {}).items()}
            for cond in baseline_conds
        },
        "interventions": summary,
        "failed": failures,
        "elapsed_s": round(elapsed, 1),
    }

    out_path = out_dir / "dose_response.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"[dose] results -> {out_path}")

    # Console summary
    print("\n=== Dose-Response Summary ===")
    print(f"{'intervention':<36} {'n_C':>4} {'n_D':>4} {'ΔSync-C':>10} {'ΔSync-D':>10} {'expected':>10}")
    for iv in interventions:
        s = summary.get(iv.name, {})
        if s.get("n_sync_c"):
            mean_d = s.get("mean_delta_d")
            d_text = f"{mean_d:+10.4f}" if mean_d is not None else f"{'n/a':>10}"
            print(
                f"{iv.name:<36} {s['n_sync_c']:>4} {s['n_sync_d']:>4} "
                f"{s['mean_delta_c']:+10.4f} {d_text} "
                f"{iv.expected_sync_direction:>10}"
            )
        else:
            print(f"{iv.name:<36}    0    0        n/a        n/a {iv.expected_sync_direction:>10}")


if __name__ == "__main__":
    main()