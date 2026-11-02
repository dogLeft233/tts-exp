"""Model-aware dispatch for the frozen cross-TFG generation head.

The routing table is intentionally separate from :mod:`pipeline`.  A TFG
renderer chooses its generator-control waveform through ``apply_model_route``;
the caller keeps the original input waveform for scoring and for the final
audio track.  Unknown models and unavailable candidate implementations fail
closed to the fixed RMS control declared by the routing table.

The built-in MuseTalk provider is CPU-only and lazy-loads PyTorch.  It is a
small application wrapper around the already pinned source-conditioned
``SharedRecipientLocalPowerSpectralHeadV1`` implementation.  Importing this
module therefore has no model or audio side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .pipeline import transform


_ROUTING_SCHEMA = "cross_tfg_model_aware_routing_config_v1"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "model_aware_routing_v1.json"
_DEFAULT_TARGET_RMS = 0.0376838172675746
_EXPLICIT11_METHOD = "exact_k15_shape_env_pos025_neg025"
_RMS_METHOD = "rms_fixed"
_EXPLICIT11_SOURCE_PINS = {
    "shared_recipient_local_power_spectral_head_v1.py": "8f22db628ec84f4b67d01e6e9c1bce12edf9280cca88e52cfc55d32d7b94dcfe",
    "shared_recipient_transition_head_v3.py": "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87",
    "baselines.py": "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2",
}


class RoutingConfigError(ValueError):
    """The routing table is malformed or cannot be used safely."""


class CandidateUnavailable(RuntimeError):
    """A selected candidate provider is not usable in the current runtime."""


@dataclass(frozen=True)
class RouteResolution:
    """One immutable route decision, suitable for a generation receipt."""

    requested_model_id: str | None
    normalized_model_id: str | None
    route_key: str
    canonical_model_id: str | None
    match_kind: str
    selected_method_id: str
    selected_method_kind: str
    fallback_method_id: str | None
    route_config_path: str
    route_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteApplication:
    """The selected route plus the waveform application outcome."""

    resolution: RouteResolution
    applied_method_id: str
    applied_method_kind: str
    fallback_used: bool
    fallback_reason: str | None
    input_sample_count: int
    output_sample_count: int
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    scoring_audio_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.resolution.to_dict(),
            # ``selected_method_id`` is the method requested by the table.  A
            # candidate can fail closed at runtime, so receipts also expose
            # the method that actually produced the returned waveform.
            "applied_method_id": self.applied_method_id,
            "applied_method_kind": self.applied_method_kind,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "input_sample_count": self.input_sample_count,
            "output_sample_count": self.output_sample_count,
            "input_sample_rate_hz": self.input_sample_rate_hz,
            "output_sample_rate_hz": self.output_sample_rate_hz,
            "scoring_audio_policy": self.scoring_audio_policy,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_model_id(model_id: str | None) -> str | None:
    if model_id is None:
        return None
    value = str(model_id).strip()
    return value.casefold() if value else None


def _load_config(path: str | Path | None) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path or _DEFAULT_CONFIG).expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingConfigError(f"cannot read routing config: {resolved}") from exc
    if not isinstance(value, dict) or value.get("schema") != _ROUTING_SCHEMA:
        raise RoutingConfigError("routing config schema mismatch")
    if value.get("selection_policy", {}).get("match_case") != "casefold":
        raise RoutingConfigError("routing config must use casefold matching")
    routes = value.get("routes")
    if not isinstance(routes, Mapping) or not isinstance(routes.get("default"), Mapping):
        raise RoutingConfigError("routing config must contain routes.default")
    digest = _sha256_file(resolved)
    return value, resolved, digest


def _iter_ordered_routes(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    routes = config["routes"]
    policy = config.get("selection_policy", {})
    order = policy.get("match_order")
    if not isinstance(order, list) or not order:
        raise RoutingConfigError("selection_policy.match_order is missing")
    result: list[tuple[str, Mapping[str, Any]]] = []
    for key in order:
        if not isinstance(key, str) or key not in routes:
            raise RoutingConfigError(f"match_order references missing route: {key!r}")
        route = routes[key]
        if not isinstance(route, Mapping):
            raise RoutingConfigError(f"route {key!r} is not an object")
        result.append((key, route))
    if "default" not in dict(result):
        raise RoutingConfigError("match_order must include default")
    return result


def _route_model_ids(route: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = route.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
        raise RoutingConfigError(f"route {field} must be a list of non-empty strings")
    return tuple(value.strip() for value in values)


def resolve_model_route(
    model_id: str | None,
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[RouteResolution, Mapping[str, Any], dict[str, Any]]:
    """Resolve exact model ids first, then aliases, then the default route.

    The returned mapping is the route object from the frozen JSON table.  The
    caller should pass it back to :func:`apply_model_route` only through this
    function; no route object is mutated by the dispatcher.
    """

    if config is None:
        loaded, path, digest = _load_config(config_path)
    else:
        loaded = dict(config)
        path = Path(config_path or _DEFAULT_CONFIG).expanduser().resolve()
        digest = hashlib.sha256(
            json.dumps(loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if loaded.get("schema") != _ROUTING_SCHEMA:
            raise RoutingConfigError("routing config schema mismatch")

    ordered = _iter_ordered_routes(loaded)
    normalized = _canonical_model_id(model_id)
    if normalized is not None:
        # Exact model_id matching has priority over aliases, regardless of the
        # order in which the route families appear in the table.
        for key, route in ordered:
            if key == "default":
                continue
            model_ids = _route_model_ids(route, "model_ids")
            for candidate in model_ids:
                if _canonical_model_id(candidate) == normalized:
                    resolution = _make_resolution(
                        model_id, normalized, key, route, "model_id", candidate, path, digest
                    )
                    return resolution, route, loaded
        for key, route in ordered:
            if key == "default":
                continue
            aliases = _route_model_ids(route, "aliases")
            for alias in aliases:
                if _canonical_model_id(alias) == normalized:
                    model_ids = _route_model_ids(route, "model_ids")
                    canonical = model_ids[0] if model_ids else None
                    resolution = _make_resolution(
                        model_id, normalized, key, route, "alias", canonical, path, digest
                    )
                    return resolution, route, loaded

    default = loaded["routes"]["default"]
    resolution = _make_resolution(
        model_id, normalized, "default", default, "default", None, path, digest
    )
    return resolution, default, loaded


def _make_resolution(
    requested: str | None,
    normalized: str | None,
    route_key: str,
    route: Mapping[str, Any],
    match_kind: str,
    canonical: str | None,
    path: Path,
    digest: str,
) -> RouteResolution:
    method_id = route.get("method_id")
    method_kind = route.get("method_kind")
    if not isinstance(method_id, str) or not method_id:
        raise RoutingConfigError(f"route {route_key!r} has no method_id")
    if not isinstance(method_kind, str) or not method_kind:
        raise RoutingConfigError(f"route {route_key!r} has no method_kind")
    fallback = route.get("fallback_method_id")
    if fallback is not None and (not isinstance(fallback, str) or not fallback):
        raise RoutingConfigError(f"route {route_key!r} has an invalid fallback_method_id")
    return RouteResolution(
        requested_model_id=None if requested is None else str(requested),
        normalized_model_id=normalized,
        route_key=route_key,
        canonical_model_id=canonical,
        match_kind=match_kind,
        selected_method_id=method_id,
        selected_method_kind=method_kind,
        fallback_method_id=fallback,
        route_config_path=str(path),
        route_config_sha256=digest,
    )


def _canonical_input(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.dtype != np.float32 or array.ndim != 1 or array.size == 0:
        raise ValueError("routed transform expects a non-empty mono float32 waveform")
    if not np.isfinite(array).all():
        raise ValueError("routed transform input contains NaN or infinite samples")
    return np.ascontiguousarray(array)


def _apply_rms(
    audio: np.ndarray,
    route: Mapping[str, Any],
    *,
    config_root: Path,
    route_config_path: Path,
) -> np.ndarray:
    config_ref = route.get("config_ref")
    if config_ref is None:
        config = {"method": "rms", "sample_rate_hz": 16_000, "target_rms": _DEFAULT_TARGET_RMS}
    else:
        reference = Path(str(config_ref)).expanduser()
        if reference.is_absolute():
            baseline_path = reference.resolve(strict=True)
        else:
            # New external route tables may keep their baseline beside the
            # table.  The shipped table historically used a repository-root
            # path (``scripts/audio_enhancement/configs/...``), so retain that
            # fallback when the table-relative spelling does not exist.
            table_relative = (route_config_path.parent / reference).resolve()
            repository_relative = (config_root / reference).resolve()
            baseline_path = (
                table_relative if table_relative.is_file() else repository_relative
            ).resolve(strict=True)
        config = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (
            not isinstance(config, dict)
            or config.get("method_id") != _RMS_METHOD
            or config.get("method") != "rms"
            or int(config.get("sample_rate_hz", -1)) != 16_000
            or float(config.get("target_rms", float("nan"))) != _DEFAULT_TARGET_RMS
        ):
            raise RoutingConfigError("RMS route config is not the frozen rms_fixed baseline")
    return transform(audio, config)


def _apply_builtin_explicit11(audio: np.ndarray, route: Mapping[str, Any]) -> np.ndarray:
    """Apply the frozen K15 + (+0.25,-0.25) candidate on CPU.

    The source-conditioned head uses the exact source sample count and does
    not resample or change the clock.  All constants are carried by the
    routing JSON so the route receipt is self-contained.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise CandidateUnavailable("PyTorch is unavailable for explicit11 candidate") from exc

    candidate = route.get("candidate")
    if not isinstance(candidate, Mapping):
        raise CandidateUnavailable("explicit11 route has no candidate settings")
    shape = candidate.get("shape_coordinates_db")
    if not isinstance(shape, list) or len(shape) != 9:
        raise CandidateUnavailable("explicit11 route has no nine-dimensional shape seed")
    env = [candidate.get("positive_flux_envelope_db"), candidate.get("negative_flux_envelope_db")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in shape + env):
        raise CandidateUnavailable("explicit11 route contains non-finite coordinates")
    shape_scale = float(candidate.get("shape_coordinate_scale_db", 2.5))
    env_scale = float(candidate.get("envelope_coordinate_scale_db", 1.0))
    if shape_scale <= 0.0 or env_scale <= 0.0:
        raise CandidateUnavailable("explicit11 coordinate scales must be positive")
    declared_shape_digest = candidate.get("shape_coordinates_sha256")
    actual_shape_digest = hashlib.sha256(struct.pack("<9d", *(float(value) for value in shape))).hexdigest()
    if declared_shape_digest != actual_shape_digest:
        raise CandidateUnavailable("explicit11 shape seed digest mismatch")

    package_root = Path(__file__).resolve().parent
    configured_pins = candidate.get("source_pins")
    if not isinstance(configured_pins, Mapping):
        raise CandidateUnavailable("explicit11 route has no source pins")
    for name, expected in _EXPLICIT11_SOURCE_PINS.items():
        if configured_pins.get(name) != expected:
            raise CandidateUnavailable(f"explicit11 route source pin differs: {name}")
        path = package_root / name
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            raise CandidateUnavailable(f"explicit11 source is unavailable: {name}") from exc
        if actual != expected:
            raise CandidateUnavailable(f"explicit11 source pin mismatch: {name}")

    from .shared_recipient_local_power_spectral_head_v1 import SharedRecipientLocalPowerSpectralHeadV1

    v3 = {
        "schema": "cross_tfg_shared_recipient_transition_head_v3",
        "sample_rate_hz": 16000,
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window_kind": "hann",
        "window_periodic": True,
        "stft_center": True,
        "stft_pad_mode": "reflect",
        "stft_normalized": False,
        "stft_onesided": True,
        "stft_return_complex": True,
        "band_edges_hz": [0.0, 500.0, 1500.0, 3500.0, 8000.0],
        "reference_band_index": 3,
        "flux_denominator_floor": 1e-8,
        "activity_floor": 1e-6,
        "flux_normalization_scale": 1.0,
        "positive_flux_quantile": 0.6,
        "negative_flux_quantile": 0.6,
        "gate_temperature": 0.15,
        "gate_smoothing_frames": 3,
        "gate_smoothing_pad_mode": "reflect",
        "mask_smoothing_frequency_bins": 3,
        "mask_smoothing_time_frames": 3,
        "mask_smoothing_pad_mode": "reflect",
        "frequency_common_mode_policy": "remove_per_frame",
        "max_abs_mask_db": 3.0,
        "target_rms": _DEFAULT_TARGET_RMS,
        "rms_epsilon": 1e-12,
        "peak_ceiling": None,
        "baseline_match_atol": 2e-5,
        "video_fps_num": 25,
        "video_fps_den": 1,
        "frame_power_floor": 1e-12,
        "stored_trainable_scalar_count": 9,
        "reference_source_pins": {
            "shared_recipient_transition_head_v2.py": "664938a3aabd15b0a5e091cf88f8dc7ebae6916c545d1c0624ca837cb37e85bc",
            "shared_recipient_transition_head_v1.py": "25098dfcdcea733968010a132d24ff120dc300d75b7c18f0004edfffea127abd",
            "wav2lip_fullcontext_fit_runtime_v4_attempt01.py": "dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458",
            "wav2lip_frontend.py": "3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef",
            "baselines.py": "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2",
            "phase_clock_diagnostics_v1.py": "25ed6f204dea7e5c149f7c897dfccfebe972ad0ef30eda1a2b1dbdc59ae216c5",
        },
    }
    head = SharedRecipientLocalPowerSpectralHeadV1(
        v3_configuration=v3,
        shape_coordinate_scale_db=shape_scale,
        envelope_coordinate_scale_db=env_scale,
        stft_power_activity_floor=float(candidate.get("stft_power_activity_floor", 1e-12)),
        one_sided_power_bin_policy="real_onesided_parseval_bins_dc_and_nyquist_1_interior_2",
        envelope_centering_policy="subtract_arithmetic_mean_on_valid_source_transitions_then_zero_invalid",
    ).to(device="cpu")
    baseline, _ = _rms_baseline_for_candidate(audio)
    shape_tensor = torch.as_tensor(shape, dtype=torch.float32)
    env_tensor = torch.as_tensor(env, dtype=torch.float32)
    with torch.no_grad():
        head.shape_coordinates.copy_(torch.atanh(shape_tensor / shape_scale).reshape(3, 3))
        head.envelope_coordinates.copy_(torch.atanh(env_tensor / env_scale))
        output = head(torch.from_numpy(audio), torch.from_numpy(baseline))
        result = output["cells"]["full"]["output_pcm_float32"]
    return result.detach().cpu().numpy().astype(np.float32, copy=False)


def _rms_baseline_for_candidate(audio: np.ndarray) -> tuple[np.ndarray, Any]:
    from .baselines import apply_global_rms

    return apply_global_rms(audio, target_rms=_DEFAULT_TARGET_RMS, peak_ceiling=None, silence_epsilon=1e-12)


def _validate_output(output: Any, source: np.ndarray) -> np.ndarray:
    result = np.asarray(output)
    if result.dtype != np.float32:
        result = result.astype(np.float32, copy=False)
    if result.ndim != 1 or result.shape != source.shape or not np.isfinite(result).all():
        raise CandidateUnavailable("candidate changed waveform shape or returned non-finite samples")
    return np.ascontiguousarray(result)


def apply_model_route(
    audio_16k_mono_f32: np.ndarray,
    model_id: str | None,
    *,
    config_path: str | Path | None = None,
    candidate_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    fallback_on_candidate_error: bool = True,
) -> tuple[np.ndarray, RouteApplication]:
    """Apply the frozen route and return generator audio plus a receipt.

    ``audio_16k_mono_f32`` is copied before candidate application.  The caller
    must retain the original input for SyncNet and final muxing.  A supplied
    ``candidate_transform`` overrides the built-in explicit11 provider and is
    useful for a separately packaged candidate; it receives a copy of the
    canonical waveform and must return the same sample count.
    """

    source = _canonical_input(audio_16k_mono_f32)
    resolution, route, config = resolve_model_route(model_id, config_path=config_path)
    repo_root = Path(__file__).resolve().parents[2]
    selected = source.copy()
    applied_method_id = resolution.selected_method_id
    applied_method_kind = resolution.selected_method_kind
    fallback_used = False
    fallback_reason: str | None = None

    if resolution.selected_method_id == _RMS_METHOD:
        selected = _apply_rms(
            source,
            route,
            config_root=repo_root,
            route_config_path=Path(resolution.route_config_path),
        )
    elif resolution.selected_method_id == _EXPLICIT11_METHOD:
        provider = candidate_transform or (lambda value: _apply_builtin_explicit11(value, route))
        try:
            selected = _validate_output(provider(source.copy()), source)
        except Exception as exc:
            if not fallback_on_candidate_error or not resolution.fallback_method_id:
                raise
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if resolution.fallback_method_id != _RMS_METHOD:
                raise RoutingConfigError(
                    f"unsupported explicit11 fallback method: {resolution.fallback_method_id}"
                ) from exc
            selected = _apply_rms(
                source,
                route,
                config_root=repo_root,
                route_config_path=Path(resolution.route_config_path),
            )
            applied_method_id = _RMS_METHOD
            applied_method_kind = "fixed_rms_control"
    else:
        raise RoutingConfigError(f"unsupported selected route method: {resolution.selected_method_id}")

    selected = _validate_output(selected, source)
    receipt = RouteApplication(
        resolution=resolution,
        applied_method_id=applied_method_id,
        applied_method_kind=applied_method_kind,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        input_sample_count=int(source.size),
        output_sample_count=int(selected.size),
        input_sample_rate_hz=16_000,
        output_sample_rate_hz=16_000,
        scoring_audio_policy="retain the untouched original natural PCM and original sample clock",
    )
    return selected, receipt


__all__ = [
    "CandidateUnavailable",
    "RouteApplication",
    "RouteResolution",
    "RoutingConfigError",
    "apply_model_route",
    "resolve_model_route",
]
