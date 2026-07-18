#!/usr/bin/env python3
"""25_stability_perturbation_syncnet.py — Stability-targeted adversarial perturbation.

Phase 2 stability-targeted intervention: surgically perturb HuBERT layer-11
``segment_stability`` on TTS and natural audio via PGD under a tight L_inf
budget, then re-run the Ditto + SyncNet pipeline used by script 22. A
random-direction noise control at the same epsilon provides a tighter
baseline than script 23's identity control.

Three intervention cells (n=9 AISHELL-1 paired samples, excl. sample 9):

  * ``stability_adj_tts``  — PGD raises the cost (destabilize TTS)        -> expect Sync-C down
  * ``stability_adj_nat``  — PGD lowers  the cost (stabilize natural)    -> expect Sync-C up
  * ``random_noise_tts``   — random ±eps sign-noise on TTS               -> control baseline

Post-hoc verification recomputes ``segment_stability`` on the perturbed
audio and the LUFS / spectral_tilt / dynamic-range drifts; cells whose
metric moves the wrong way or whose acoustic drift exceeds the documented
threshold are flagged in the per-sample JSON.

See ``docs/superpowers/specs/2026-07-18-stability-intervention-design.md``
for the full design.

Usage:
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --dry-run
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --samples 1
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --eps 0.005 --alpha 0.001 --pgd-steps 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import STUDY_SAMPLES, TARGET_SR, load_audio_mono, ensure_output_dirs


# ---------------------------------------------------------------------------
# Dynamic module loading (scripts start with digits so normal imports fail)
# ---------------------------------------------------------------------------


def _load_script_module(filename: str):
    """Load a sibling script as a module by file path (mirrors script 22)."""
    path = _REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_s22 = _load_script_module("22_dose_response_syncnet.py")
_s15 = _load_script_module("15_extract_ssl_embeddings.py")
_s16 = _load_script_module("16_feature_separability.py")
_s20 = _load_script_module("20_causal_feature_interventions.py")

Intervention = _s22.Intervention
run_intervention_pipeline = _s22.run_intervention_pipeline
load_baseline_results = _s22.load_baseline_results
aggregate_deltas = _s22.aggregate_deltas

load_model = _s15.load_model
extract_frame_embeddings = _s15.extract_frame_embeddings
_compute_frame_stride = _s15._compute_frame_stride

boundary_sharpness = _s16.boundary_sharpness

compute_lufs = _s20.compute_lufs
compute_spectral_tilt = _s20.compute_spectral_tilt
compute_energy_env_std = _s20.compute_energy_env_std


# ---------------------------------------------------------------------------
# Locked-in symbol stubs (filled in by later tasks)
# ---------------------------------------------------------------------------


def stability_loss(h_l11, frame_times_np: np.ndarray, boundary_idx_np: np.ndarray):
    """Differentiable `segment_stability` cost matching script 16's metric.

    Parameters
    ----------
    h_l11 : torch.Tensor, shape (1, T, D) — hidden states at HuBERT layer 11
    frame_times_np : np.ndarray, shape (T,) — frame centre times in seconds
                      (only used for shape consistency; indices follow order)
    boundary_idx_np : np.ndarray, shape (B,) — frame *indices* that are
                       token-boundary frames (the "before" frame of each
                       boundary). Pairs (i, i+1) where i is in this set are
                       excluded from the within-segment average, mirroring
                       script 16.

    Returns
    -------
    loss : torch.Tensor scalar
        Mean (1 - cos_sim) over within-segment consecutive frame pairs.
    """
    import torch

    if h_l11.dim() != 3 or h_l11.shape[0] != 1:
        raise ValueError(
            f"expected h_l11 of shape (1, T, D), got {tuple(h_l11.shape)}"
        )
    T = h_l11.shape[1]
    if T < 2:
        raise ValueError("need >= 2 frames to compute stability")

    h = h_l11.squeeze(0)
    norm = h.norm(dim=1, keepdim=True).clamp_min(1e-10)
    h_norm = h / norm

    cos_sim = (h_norm[:-1] * h_norm[1:]).sum(dim=1)
    cost_pair = 1.0 - cos_sim

    if boundary_idx_np.size > 0:
        before_set = set(int(i) for i in boundary_idx_np)
        mask_vals = [0.0 if i in before_set else 1.0 for i in range(T - 1)]
        mask = torch.tensor(mask_vals, dtype=h.dtype, device=h.device)
    else:
        mask = torch.ones(T - 1, dtype=h.dtype, device=h.device)

    masked = cost_pair * mask
    if float(mask.sum().item()) < 1.0:
        return torch.zeros((), dtype=h.dtype, device=h.device)
    return masked.sum() / mask.sum()


FALLBACK_SEGMENT_SEC: float = 0.05


def apply_uniform_fallback_boundaries(audio_duration_s: float) -> np.ndarray:
    """Return uniform-50ms segment-end-times under audio_duration_s.

    Matches script 16's uniform-segmentation fallback (50 ms uniform segments
    used when MFA phoneme alignments are unavailable for a sample).
    """
    if audio_duration_s <= 0.0:
        return np.empty(0, dtype=np.float64)
    end = float(np.floor(audio_duration_s / FALLBACK_SEGMENT_SEC)) * FALLBACK_SEGMENT_SEC
    if end <= 0.0:
        return np.empty(0, dtype=np.float64)
    return np.arange(
        FALLBACK_SEGMENT_SEC, end + FALLBACK_SEGMENT_SEC, FALLBACK_SEGMENT_SEC, dtype=np.float64,
    )


def _load_manifest_samples(manifest_path: Path) -> dict[tuple[int, str], list[float]]:
    """Return {(sample_id, condition): [end_s, end_s, ...]} from manifest JSON."""
    if not manifest_path or not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text())
    samples = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples, list):
        return {}
    out: dict[tuple[int, str], list[float]] = {}
    for entry in samples:
        sid = entry.get("sample_id")
        cond = entry.get("condition")
        tokens = entry.get("tokens", [])
        if sid is None or cond is None or not tokens:
            continue
        ends = [float(t["end_s"]) for t in tokens if "end_s" in t]
        if len(ends) >= 2:
            out[(int(sid), str(cond))] = ends[:-1]
    return out


def load_token_boundaries(
    sample_id: int,
    condition: str,
    repo: Path,
    manifest_path: Path | None = None,
    audio_duration_s: float | None = None,
) -> np.ndarray:
    """Return per-token boundary end-times (seconds) for (sample, condition).

    Looks up `manifest_path` if provided; otherwise falls back to
    `repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"`.

    If no entry exists for this (sample_id, condition), returns uniform 50 ms
    segment boundaries — requires `audio_duration_s` to be provided in seconds
    (caller computes from the loaded audio waveform). Returns an empty array if
    `audio_duration_s` is None — downstream code must handle zero
    within-segment pairs gracefully.
    """
    if manifest_path is None:
        manifest_path = repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"
    table = _load_manifest_samples(manifest_path)
    key = (int(sample_id), str(condition))
    if key in table:
        return np.asarray(table[key], dtype=np.float64)
    if audio_duration_s is None:
        return np.empty(0, dtype=np.float64)
    return apply_uniform_fallback_boundaries(audio_duration_s)


def random_sign_noise_transform(
    y_src: np.ndarray,
    _y_other: np.ndarray | None,
    sr: int,
    sid: int,
    eps: float = 0.005,
) -> np.ndarray:
    """Random ±eps sign-pattern noise, scaled by uniform magnitude in [0, eps].

    Matches the L_inf budget of the PGD transforms. Returns float32 in
    ``[-1, 1]``. The control intervention's purpose is to provide a
    tighter baseline than script 23's identity control: shared ε budget,
    non-targeted direction.

    Parameters
    ----------
    y_src : np.ndarray, shape (n_samples,)
        Source audio waveform (float32, in [-1, 1]).
    sid : int
        Sample id used to seed the RNG (deterministic per sample).
    eps : float
        L_inf bound on the perturbation.

    Returns
    -------
    np.ndarray, same shape/dtype as y_src
    """
    rng = np.random.default_rng(int(sid) * 12345 + 17)
    sign = rng.integers(0, 2, size=y_src.shape, dtype=np.int32) * 2 - 1  # ±1
    magnitude = rng.uniform(0.0, eps, size=y_src.shape).astype(np.float32)
    delta = (sign * magnitude).astype(np.float32)
    out = (y_src.astype(np.float32) + delta).clip(-1.0, 1.0)
    return out


_HUBERT_CACHE: dict[str, object] = {}


def _ensure_hubert_model(device: str = "cuda"):
    """Return a cached HuBERT base model with output_hidden_states=True.

    The model is loaded once per process and reused across the per-sample
    PGD loop. Weights are frozen (eval mode, requires_grad=False on all
    parameters) — only the *input waveform* is a parameter.
    """
    if device in _HUBERT_CACHE:
        return _HUBERT_CACHE[device]
    import torch  # noqa: F401  (delayed import keeps CPU-only tests importing torch lazily)

    model_name = "facebook/hubert-base-ls960"
    model, embedding_dim, frame_stride, num_layers = load_model(
        model_name,
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _HUBERT_CACHE[device] = {
        "model": model,
        "frame_stride": int(frame_stride),
        "embedding_dim": int(embedding_dim),
        "num_layers": int(num_layers),
    }
    return _HUBERT_CACHE[device]


def pgd_stability_transform(
    direction: str,
    eps: float = 0.005,
    alpha: float = 0.001,
    K: int = 50,
    device: str = "cuda",
    repo: Path | None = None,
    manifest_path: Path | None = None,
    condition: str = "tts",
    sr: int = TARGET_SR,
):
    """Return an Intervention-compatible transform closure.

    The Intervention contract from script 22 is:
        transform(y_tts, y_nat, sr, sid) -> np.ndarray (mono waveform)

    Which waveform goes in depends on the Intervention's `.source` field
    ("tts" -> y_tts; "natural" -> y_nat). The dispatch happens inside
    `run_intervention_pipeline`; this closure is called with whichever source
    was selected, paired with the *unused* counterpart (passed as `y_other`).
    """
    if repo is None:
        repo = _REPO
    if manifest_path is None:
        manifest_path = repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"
    if condition not in ("tts", "natural"):
        raise ValueError(f"condition must be tts|natural, got {condition!r}")

    def _impl(y_tts, y_nat, sr_arg, sid):
        if sr_arg != TARGET_SR:
            raise ValueError(
                f"PGD transform requires sr={TARGET_SR}; got sr={sr_arg}"
            )
        if condition == "tts":
            y_src = y_tts
        else:
            y_src = y_nat

        cache = _ensure_hubert_model(device)
        model = cache["model"]
        frame_stride = cache["frame_stride"]
        audio_duration_s = float(y_src.shape[0]) / float(TARGET_SR)
        boundaries = load_token_boundaries(
            sid, condition, repo, manifest_path=manifest_path,
            audio_duration_s=audio_duration_s,
        )
        return pgd_perturb(
            y_src, model, frame_stride, boundaries,
            eps=eps, alpha=alpha, K=K, device=device, direction=direction,
        )

    return _impl


def pgd_perturb(
    y: np.ndarray,
    model,
    frame_stride: int,
    dt_boundaries: np.ndarray,
    eps: float,
    alpha: float,
    K: int,
    device: str,
    direction: str,
) -> np.ndarray:
    """Apply PGD to perturb raw waveform `y` toward a stability target.

    direction: "raise_cost" -> ascend L (less stable);
               "lower_cost" -> descend L (more stable).
    """
    import torch

    if direction not in ("raise_cost", "lower_cost"):
        raise ValueError(f"direction must be raise_cost|lower_cost, got {direction!r}")
    # We always *descend* on (direction_factor * L). raise_cost wants ascent
    # on L -> descend on -L -> direction_factor = -1.
    # lower_cost wants descent on L -> direction_factor = +1.
    direction_factor = -1.0 if direction == "raise_cost" else +1.0

    y_t = torch.from_numpy(np.ascontiguousarray(y)).float().to(device)
    delta = torch.zeros_like(y_t, requires_grad=True)

    with torch.no_grad():
        probe = model(y_t.unsqueeze(0), output_hidden_states=True)
        n_frames = probe.hidden_states[0].shape[1]
    frame_times = np.arange(n_frames, dtype=np.float32) * (
        float(frame_stride) / float(TARGET_SR)
    )

    if dt_boundaries.size > 0:
        before_idx = np.searchsorted(frame_times, dt_boundaries.astype(np.float32)) - 1
        before_idx = before_idx[(before_idx >= 0) & (before_idx < n_frames - 1)]
        boundary_idx_np = before_idx.astype(np.int64)
    else:
        boundary_idx_np = np.empty(0, dtype=np.int64)

    for _ in range(K):
        x_adv = (y_t + delta).clamp(-1.0, 1.0)
        outputs = model(x_adv.unsqueeze(0), output_hidden_states=True)
        h_l11 = outputs.hidden_states[12]
        L = stability_loss(h_l11, frame_times, boundary_idx_np)
        loss = direction_factor * L
        loss.backward()
        with torch.no_grad():
            grad_sign = torch.sign(delta.grad)
            delta -= alpha * grad_sign
            delta.clamp_(-eps, eps)
        delta.grad = None

    with torch.no_grad():
        perturbed = (y_t + delta).clamp(-1.0, 1.0).squeeze()
    return perturbed.cpu().numpy().astype(np.float32)


def _build_interventions(args: argparse.Namespace) -> list[Intervention]:
    """Construct the 3 intervention cells defined by the spec."""
    cells: list[Intervention] = []
    manifest_path = args.manifest
    cells.append(Intervention(
        name="stability_adj_tts",
        source="tts",
        baseline_cond="tts_raw",
        transform_description=(
            "PGD raises HuBERT-L11 segment_stability cost on TTS (destabilize)"
        ),
        expected_sync_direction="decrease",
        transform=pgd_stability_transform(
            direction="raise_cost",
            eps=args.eps, alpha=args.alpha, K=args.pgd_steps,
            device=args.device, repo=args.repo, manifest_path=manifest_path,
            condition="tts",
        ),
    ))
    cells.append(Intervention(
        name="stability_adj_nat",
        source="natural",
        baseline_cond="natural_raw",
        transform_description=(
            "PGD lowers HuBERT-L11 segment_stability cost on natural (stabilize)"
        ),
        expected_sync_direction="increase",
        transform=pgd_stability_transform(
            direction="lower_cost",
            eps=args.eps, alpha=args.alpha, K=args.pgd_steps,
            device=args.device, repo=args.repo, manifest_path=manifest_path,
            condition="natural",
        ),
    ))
    eps_capture = args.eps
    def _random_noise(y_tts, _y_nat, sr, sid, eps=eps_capture):
        return random_sign_noise_transform(y_tts, None, sr, sid, eps=eps)
    cells.append(Intervention(
        name="random_noise_tts",
        source="tts",
        baseline_cond="tts_raw",
        transform_description=(
            f"Random ±eps uniform noise on TTS (eps={eps_capture}, control)"
        ),
        expected_sync_direction="decrease",
        transform=_random_noise,
    ))
    return cells


def post_hoc_verify(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    sr: int,
    sid: int,
    condition: str,
    repo: Path,
    manifest_path: Path | None = None,
    expected_direction: str | None = None,
    device: str = "cuda",
):
    """Recompute stability and acoustic features on pre/post waveforms.

    Returns dict with:
      - stability_metric_pre  / post  / delta
      - expected_direction          (one of raise_cost|lower_cost|None)
      - achieved_direction          (raise_cost|lower_cost|unchanged)
      - achieved_aligns_with_expected (bool|None)
      - lufs_pre/post/delta_lufs, tilt_pre/post/delta_tilt,
        dyn_pre/post/delta_dyn
    """
    if sr != TARGET_SR:
        raise ValueError(f"expected sr={TARGET_SR}; got sr={sr}")

    cache = _ensure_hubert_model(device)
    model = cache["model"]
    frame_stride = cache["frame_stride"]
    layers = [11]

    def _compute_stability(y_audio):
        h, frame_times = extract_frame_embeddings(
            model, y_audio, TARGET_SR, layers, device=device,
        )
        h_l11 = h[0]
        duration_s = float(y_audio.shape[0]) / float(TARGET_SR)
        boundaries = load_token_boundaries(
            sid, condition, repo, manifest_path=manifest_path,
            audio_duration_s=duration_s,
        )
        _bs, ss = boundary_sharpness(frame_times, h_l11, list(boundaries))
        return float(ss)

    ss_pre = _compute_stability(y_pre)
    ss_post = _compute_stability(y_post)
    delta = ss_post - ss_pre

    def _classify_dir(d: float) -> str:
        if abs(d) < 1e-7:
            return "unchanged"
        return "raise_cost" if d > 0 else "lower_cost"

    achieved_direction = _classify_dir(delta)
    if expected_direction is None:
        aligns = None
    elif achieved_direction == "unchanged":
        aligns = None
    else:
        aligns = (achieved_direction == expected_direction)

    lufs_pre = float(compute_lufs(y_pre, sr))
    lufs_post = float(compute_lufs(y_post, sr))
    tilt_pre = float(compute_spectral_tilt(y_pre, sr))
    tilt_post = float(compute_spectral_tilt(y_post, sr))
    dyn_pre = float(compute_energy_env_std(y_pre, sr))
    dyn_post = float(compute_energy_env_std(y_post, sr))

    return {
        "stability_metric_pre": ss_pre,
        "stability_metric_post": ss_post,
        "stability_metric_delta": float(delta),
        "expected_direction": expected_direction,
        "achieved_direction": achieved_direction,
        "achieved_aligns_with_expected": aligns,
        "lufs_pre": lufs_pre,
        "lufs_post": lufs_post,
        "delta_lufs": lufs_post - lufs_pre,
        "tilt_pre": tilt_pre,
        "tilt_post": tilt_post,
        "delta_tilt": tilt_post - tilt_pre,
        "dyn_pre": dyn_pre,
        "dyn_post": dyn_post,
        "delta_dyn": dyn_post - dyn_pre,
    }


def main() -> None:
    """CLI entry point: dispatch the 3 stability interventions over samples.

    Mirrors script 23's structure: load baselines, iterate cells × samples,
    each cell calls run_intervention_pipeline with the cell's transform;
    post-hoc verification is appended to each per-sample record so the
    output JSON gives script 26 (the verdict classifier, arriving later)
    enough material to decide STABILITY_CAUSAL / REJECTED / INCONCLUSIVE.
    """
    ap = argparse.ArgumentParser(
        description="Stability-targeted adversarial perturbation (script 25)"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--samples",
        default=",".join(str(s) for s in STUDY_SAMPLES),
        help="Comma-separated sample IDs (default: STUDY_SAMPLES)",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None,
                   help="Token-boundary manifest JSON (default: data/wav2sem_analysis/manifest/alignment.json)")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--pgd-steps", type=int, default=50)
    ap.add_argument("--pgd-restarts", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda",
                   help="cpu or cuda (default: cuda)")
    ap.add_argument("--no-cache-audio", action="store_true")
    ap.add_argument("--no-cache-video", action="store_true")
    ap.add_argument("--no-cache-syncnet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.repo = _REPO
    if args.output_dir is None:
        args.output_dir = ensure_output_dirs()["metrics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.repo / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return

    cfg_path = args.repo / "scripts" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sample_ids = [int(s.strip()) for s in args.samples.split(",") if s.strip()]

    interventions = _build_interventions(args)
    eval_base = run_dir / "04_eval"
    baseline_conds = sorted({iv.baseline_cond for iv in interventions})
    baselines = load_baseline_results(eval_base, baseline_conds)

    nat_audio_dir = args.repo / "data" / "data" / "audio"
    tts_audio_dir = run_dir / "02_tts"
    img_dir = args.repo / "data" / "data" / "image"

    if not tts_audio_dir.is_dir():
        print(f"ERROR: TTS audio dir not found: {tts_audio_dir}")
        return

    # Preload audio (so dry-run doesn't need HuBERT).
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

    ditto_intv_base = run_dir / "03_ditto" / "interventions"
    eval_intv_base = run_dir / "04_eval" / "interventions"

    print(f"[script25] run_id={args.run_id}")
    print(f"[script25] samples loaded: {loaded_ids}")
    print(f"[script25] cells: {[iv.name for iv in interventions]}")
    print(f"[script25] pgd: eps={args.eps}, alpha={args.alpha}, K={args.pgd_steps}, device={args.device}")
    print(f"[script25] baselines: {[(c, len(v)) for c, v in baselines.items()]}")
    if args.dry_run:
        print("[script25] DRY-RUN — no Ditto/SyncNet invocation\n")

    results: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
    post_hoc_records: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
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

            # Save unperturbed source for post-hoc verification. The intervention
            # transform itself may overwrite the cached audio, so compute
            # verification BEFORE the pipeline invocation using the *pre* source.
            if iv.source == "tts":
                y_pre = tts_audio[sid]
                condition = "tts"
            else:
                y_pre = natural_audio[sid]
                condition = "natural"
            expected_direction = {
                "stability_adj_tts": "raise_cost",
                "stability_adj_nat": "lower_cost",
                "random_noise_tts": None,
            }[iv.name]

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
                repo=args.repo,
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
                continue

            # Load perturbed audio for post-hoc verification. The transformed
            # audio was written to audio_out / f"{sid}.wav" by the pipeline.
            if not args.dry_run:
                perturbed_path = audio_out / f"{sid}.wav"
                if perturbed_path.exists():
                    y_post, _ = load_audio_mono(perturbed_path, target_sr=TARGET_SR)
                else:
                    y_post = y_pre
                post_hoc = post_hoc_verify(
                    y_pre=y_pre, y_post=y_post, sr=sr, sid=sid, condition=condition,
                    repo=args.repo, manifest_path=args.manifest,
                    expected_direction=expected_direction, device=args.device,
                )
                res["post_hoc"] = post_hoc
                post_hoc_records[iv.name][sid] = post_hoc

            results[iv.name][sid] = res

            c = res.get("sync_c", 0.0)
            base = baselines.get(iv.baseline_cond, {}).get(sid, {})
            bc = base.get("sync_c")
            if bc is not None:
                print(
                    f"  sample {sid}: Sync-C={c:.3f} (base={bc:.3f}, Δ={c - bc:+.3f})"
                    f" — {status}"
                )
            else:
                print(f"  sample {sid}: Sync-C={c:.3f} (no baseline) — {status}")

    elapsed = time.monotonic() - t0
    print(
        f"\n[script25] completed in {elapsed:.0f}s; "
        f"{sum(len(v) for v in results.values())} ok, {len(failures)} failures"
    )

    if args.dry_run:
        print("[script25] DRY-RUN — no JSON written")
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
        "post_hoc": post_hoc_records,
        "failed": failures,
        "elapsed_s": round(elapsed, 1),
        "pgd_params": {
            "eps": args.eps,
            "alpha": args.alpha,
            "K": args.pgd_steps,
            "restarts": args.pgd_restarts,
        },
        "verdict": "TBD",  # filled by script 26 downstream
    }

    out_path = args.output_dir / "stability_intervention.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"[script25] results -> {out_path}")

    print("\n=== Stability Intervention Summary ===")
    for iv_name, s in summary.items():
        if s.get("n_samples"):
            print(
                f"{iv_name:<22} n={s['n_samples']:>2}  "
                f"ΔSync-C={s['mean_delta_c']:+.4f}  ΔSync-D={s['mean_delta_d']:+.4f}"
            )


if __name__ == "__main__":
    main()
