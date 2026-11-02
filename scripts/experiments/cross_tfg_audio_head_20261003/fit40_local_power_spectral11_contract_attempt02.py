"""Pure contract and SPSA math for the additive 11-coordinate family.

This module deliberately performs no file I/O, model construction, rendering,
or scoring. Callers must validate and pin the profile/helper/source bundle
before opening inputs or starting a process. It does not import or modify the
legacy nine-coordinate selector.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


FAMILY_ID = "local_power_spectral11_attempt01"
METHOD_SCHEMA = "fit40_local_power_spectral11_method_v1"
POINT_SCHEMA = "fit40_local_power_spectral11_point_v1"
PAIR_SCHEMA = "fit40_local_power_spectral11_pair_v1"
UPDATE_SCHEMA = "fit40_local_power_spectral11_update_v1"
ROW_INPUT_SCHEMA = "fit40_local_power_spectral11_source_row_v1"
RMS_REFERENCE_METHOD_ID = "rms_fixed"
RMS_BASELINE_TARGET_RMS = 0.0376838172675746
RMS_BASELINE_SILENCE_EPSILON = 1.0e-12

TEMPORAL_COMPONENTS = ("static", "positive_flux", "negative_flux")
SHAPE_COORDINATE_ORDER = tuple(
    f"shape.{temporal}.free_band_{band_index}"
    for temporal in TEMPORAL_COMPONENTS
    for band_index in range(3)
)
COORDINATE_ORDER = SHAPE_COORDINATE_ORDER + (
    "envelope.positive_flux",
    "envelope.negative_flux",
)
DIMENSION = 11

# These are the ROOT-frozen optimizer-v1 values. The audio profile remains a
# separately pinned input; these constants do not imply that it is available.
SHAPE_SCALE_DB = 2.5
ENVELOPE_SCALE_DB = 1.0
COORDINATE_SCALE_DB = (SHAPE_SCALE_DB,) * 9 + (ENVELOPE_SCALE_DB,) * 2
SHAPE_ABS_BOUND_DB = 2.4
ENVELOPE_ABS_BOUND_DB = 0.95
COORDINATE_ABS_BOUND_DB = (SHAPE_ABS_BOUND_DB,) * 9 + (ENVELOPE_ABS_BOUND_DB,) * 2

SPSA_RNG_SEED = 20261009
RENDERER_SEED = 20261003
SPSA_ITERATIONS = 20
SPSA_BATCH_SPEAKERS = 4
SPSA_CLIPS_PER_SPEAKER = 2
TAU = 0.050
IDENTITY_CBEST_TARGET = 0.0
RMS_CBEST_TARGET = 0.050
GUARD_OUTER_COEFFICIENT = 2.0
REGULARIZATION_LAMBDA = 0.001
MAX_UPDATE_COMPONENT_DB = 0.25
MIN_PERTURBATION_DB = 0.01
NEXT_PERTURBATION_MARGIN_DB = 2.0 * MIN_PERTURBATION_DB
MAX_BACKTRACK_EXPONENT = 40
U_TO_Z_ROUNDTRIP_ATOL_DB = 2.0e-6

METHOD_CONTRACT: Mapping[str, Any] = {
    "schema": METHOD_SCHEMA,
    "family_id": FAMILY_ID,
    "coordinate_space": "u_pre_kappa_db",
    "dimension": DIMENSION,
    "coordinate_order": COORDINATE_ORDER,
    "shape_layout": "3x3_row_major_temporal_by_three_nonreference_bands",
    "temporal_component_order": TEMPORAL_COMPONENTS,
    "envelope_order": ("positive_flux", "negative_flux"),
    "u_to_head_latent": "z=atanh(u_db/coordinate_scale_db)",
    "scale_db": COORDINATE_SCALE_DB,
    "absolute_bound_db": COORDINATE_ABS_BOUND_DB,
    "u_recovery_atol_db": U_TO_Z_ROUNDTRIP_ATOL_DB,
    "seed": {
        "source_step": 15,
        "journal_entry_index": 14,
        "shape_coordinates": "exact_K15_theta9_order",
        "envelope_coordinates_db": (0.0, 0.0),
    },
    "spsa": {
        "iterations": SPSA_ITERATIONS,
        "completed_tell_statuses": ("UPDATED", "TELL_COMPLETED_NO_EFFECTIVE_CHANGE"),
        "no_effective_change": "exact_F64_u_or_identical_actual_helper_recovered_F32_u; tell_and_rng_advance",
        "no_effective_change_flags": (
            "u_f64_unchanged",
            "helper_recovered_f32_unchanged",
        ),
        "preclassification_status": "UPDATE_PROPOSED",
        "completed_tell_classification": "finalize_both_flags_after_pinned_helper_maps_for_before_and_after_u; proposal_is_not_a_completed_tell",
        "boundary_limited": "no_tell_and_no_iteration_advance",
        "rng": "numpy.PCG64",
        "rng_seed": SPSA_RNG_SEED,
        "direction": "independent_Rademacher_pm1_dimension11_stream",
        "a_k": "0.5/(k+5)^0.602",
        "c_k_base": "0.25/(k+1)^0.101",
        "c_min_db": MIN_PERTURBATION_DB,
        "c_actual": "min(c_k_base,0.5*min_j(U_j-abs(u_j)))",
        "gradient": "(Lplus-Lminus)/(2*c_actual*delta_i)",
        "component_cap_db": MAX_UPDATE_COMPONENT_DB,
        "common_scale": "s=min(1,0.25/max_abs(v)); s=1 when v=0",
        "backtracking": "first_beta=2^-m,m=0..40 satisfying abs(u+beta*s*v)<=U-0.02",
        "boundary_status": "BOUNDARY_LIMITED_NO_UPDATE",
    },
    "objective": {
        "tau": TAU,
        "identity_cbest_target": IDENTITY_CBEST_TARGET,
        "rms_cbest_target": RMS_CBEST_TARGET,
        "guard_outer_coefficient": GUARD_OUTER_COEFFICIENT,
        "regularization_lambda": REGULARIZATION_LAMBDA,
        "regularization_anchor": "fixed_K15_seed_u11",
    },
    "batch": {
        "speaker_count": 20,
        "speakers_per_iteration": SPSA_BATCH_SPEAKERS,
        "clips_per_speaker": SPSA_CLIPS_PER_SPEAKER,
        "speaker_order": "sorted_cyclic",
        "start_index": "(iteration*4)%20",
    },
    "initial_design": {
        "seed_full_rows": 120,
        "identity_reference_rows": 120,
        "rms_fixed_reference_rows": 120,
        "total_rows": 360,
        "legacy_initial_720_role": "reference_and_seed_provenance_only",
        "reference_provenance": "inherit_exact_old_240_reference_NPZ_maps_natural_PCM_clock_source_and_runtime_pins",
        "common_support": "recompute_three_method_support_from_full_raw_matrix_and_original_source_masks",
        "forbidden_shortcut": "old_six_method_or_initial720_reduced_scalar_rows",
        "reference_role_identity": "old-source_reference/new-family_identity_arm",
        "reference_role_rms_fixed": "old-source_reference/new-family_rms_fixed_arm_preserving_method_id",
        "seed_role": "new-family_evaluation_initialized_from_pinned_K15_center",
    },
    "support_policy": {
        "spsa_pair_methods": 4,
        "final_selection_methods": 6,
        "final_hard_gate_count": 19,
        "source_clocks_and_roi": "inherit_exact_frozen_runbook_source_clock_and_ROI_contract",
        "original_pcm_and_frame_maps": "exact_pinned_full_body_closure",
        "per_source_projection_diagnostics": ("kappa", "A", "B", "kappa_u", "kappa_active_count"),
        "scoring_audio": "exact_original_natural_PCM",
    },
    "application_interface": {
        "call": "apply_one_sample(z_head_f32,original_pcm_float32,canonical_rms_baseline_pcm_float32)",
        "row_mapping_schema": ROW_INPUT_SCHEMA,
        "row_mapping_required_keys": (
            "schema", "sample_id", "source_pcm_pin", "rms_fixed_reference", "renderer_pcm_equality",
            "source_clock_pin", "frame_map_pin", "roi_policy_pin", "input_plan_pin",
        ),
        "source_pcm_pin_fields": ("sample_id", "path", "bytes", "sha256", "pcm_sha256", "sample_rate_hz", "channels", "sample_count"),
        "rms_fixed_reference_fields": (
            "method_id", "generator_wav_pin", "decoded_pcm_sha256", "sample_rate_hz",
            "channels", "sample_count", "baseline_policy", "gain_info",
        ),
        "renderer_pcm_equality_fields": ("pcm_sha256", "sample_count", "models"),
        "renderer_model_fields": ("model_id", "renderer_wav_pin", "pcm_sha256", "sample_count"),
        "rms_fixed_policy": {
            "target_rms": RMS_BASELINE_TARGET_RMS,
            "peak_ceiling": None,
            "silence_epsilon": RMS_BASELINE_SILENCE_EPSILON,
            "implementation": "scripts/audio_enhancement/baselines.py:apply_global_rms",
        },
        "candidate_waveform_scope": "one_transformed_waveform_shared_in_sequence_by_all_three_model_renderers",
        "preserve_full_input_sample_count": True,
    },
    "selection_design": {
        "pair_count": 20,
        "proposal_points": 40,
        "full_center_count": 4,
        "center_after_iterations": (5, 10, 15, 20),
    },
    "source_status": "SOURCE_ONLY_PROFILE_AND_RUNTIME_BINDINGS_REQUIRED",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


METHOD_CONTRACT_SHA256 = hashlib.sha256(canonical_json_bytes(METHOD_CONTRACT)).hexdigest()


class ContractError(ValueError):
    """Malformed source, profile, coordinate, or evidence contract."""


@dataclass(frozen=True)
class BoundaryLimitedNoUpdate:
    """A typed SPSA stop whose diagnostics remain serializable as evidence."""

    status: str
    phase: str
    iteration: int
    center_u_db: tuple[float, ...]
    direction: tuple[int, ...] | None
    c_base_db: float | None
    c_actual_db: float | None
    available_margin_db: float | None
    gradient: tuple[float, ...] | None
    raw_update_v: tuple[float, ...] | None
    common_scale_s: float | None
    chosen_beta: float | None
    backtrack_exponent: int | None
    proposed_u_db: tuple[float, ...] | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": UPDATE_SCHEMA,
            "status": self.status,
            "phase": self.phase,
            "iteration": self.iteration,
            "center_u_db": list(self.center_u_db),
            "direction": None if self.direction is None else list(self.direction),
            "c_base_db": self.c_base_db,
            "c_actual_db": self.c_actual_db,
            "available_margin_db": self.available_margin_db,
            "gradient": None if self.gradient is None else list(self.gradient),
            "raw_update_v": None if self.raw_update_v is None else list(self.raw_update_v),
            "common_scale_s": self.common_scale_s,
            "chosen_beta": self.chosen_beta,
            "backtrack_exponent": self.backtrack_exponent,
            "proposed_u_db": None if self.proposed_u_db is None else list(self.proposed_u_db),
            "reason": self.reason,
            "method_contract_sha256": METHOD_CONTRACT_SHA256,
        }


@dataclass(frozen=True)
class SpsaPair:
    schema: str
    status: str
    iteration: int
    center_u_db: tuple[float, ...]
    direction: tuple[int, ...]
    a_k: float
    c_base_db: float
    c_actual_db: float | None
    available_margin_db: float
    plus_u_db: tuple[float, ...] | None
    minus_u_db: tuple[float, ...] | None
    boundary_record: BoundaryLimitedNoUpdate | None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "iteration": self.iteration,
            "center_u_db": list(self.center_u_db),
            "direction": list(self.direction),
            "a_k": self.a_k,
            "c_base_db": self.c_base_db,
            "c_actual_db": self.c_actual_db,
            "available_margin_db": self.available_margin_db,
            "plus_u_db": None if self.plus_u_db is None else list(self.plus_u_db),
            "minus_u_db": None if self.minus_u_db is None else list(self.minus_u_db),
            "boundary_record": None if self.boundary_record is None else self.boundary_record.to_record(),
            "method_contract_sha256": METHOD_CONTRACT_SHA256,
        }


@dataclass(frozen=True)
class SpsaUpdate:
    schema: str
    status: str
    iteration: int
    center_before_u_db: tuple[float, ...]
    direction: tuple[int, ...]
    a_k: float
    c_base_db: float
    c_actual_db: float
    loss_plus: float
    loss_minus: float
    gradient: tuple[float, ...]
    raw_update_v: tuple[float, ...]
    common_scale_s: float
    chosen_beta: float | None
    backtrack_exponent: int | None
    center_after_u_db: tuple[float, ...] | None
    boundary_record: BoundaryLimitedNoUpdate | None
    no_effective_change_reason: str | None = None
    u_f64_unchanged: bool | None = None
    helper_recovered_f32_unchanged: bool | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "iteration": self.iteration,
            "center_before_u_db": list(self.center_before_u_db),
            "direction": list(self.direction),
            "a_k": self.a_k,
            "c_base_db": self.c_base_db,
            "c_actual_db": self.c_actual_db,
            "loss_plus": self.loss_plus,
            "loss_minus": self.loss_minus,
            "gradient": list(self.gradient),
            "raw_update_v": list(self.raw_update_v),
            "common_scale_s": self.common_scale_s,
            "chosen_beta": self.chosen_beta,
            "backtrack_exponent": self.backtrack_exponent,
            "center_after_u_db": None if self.center_after_u_db is None else list(self.center_after_u_db),
            "boundary_record": None if self.boundary_record is None else self.boundary_record.to_record(),
            "no_effective_change_reason": self.no_effective_change_reason,
            "u_f64_unchanged": self.u_f64_unchanged,
            "helper_recovered_f32_unchanged": self.helper_recovered_f32_unchanged,
            "effective_change_classification_complete": (
                self.u_f64_unchanged is not None and self.helper_recovered_f32_unchanged is not None
            ),
            "method_contract_sha256": METHOD_CONTRACT_SHA256,
        }


def _finite_vector(values: Sequence[float], *, name: str, size: int = DIMENSION) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != size:
        raise ContractError(f"{name} must contain exactly {size} numeric values")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ContractError(f"{name} cannot contain booleans")
        number = float(value)
        if not math.isfinite(number):
            raise ContractError(f"{name} must contain finite values")
        out.append(number)
    return tuple(out)


def validate_u_db(values: Sequence[float], *, name: str = "u_db") -> tuple[float, ...]:
    vector = _finite_vector(values, name=name)
    if any(abs(value) > bound for value, bound in zip(vector, COORDINATE_ABS_BOUND_DB)):
        raise ContractError(f"{name} violates the frozen coordinate box")
    return vector


def seed_u_from_k15(theta9: Sequence[float], provenance: Mapping[str, Any]) -> tuple[float, ...]:
    """Lift the exact K15 nine shape values and append zero envelope values.

    The caller must supply the journal and terminal raw/self pins. This pure
    function does not read either source and does not claim that K15 is itself
    admitted as an 11-coordinate evaluation.
    """
    shape = _finite_vector(theta9, name="K15 theta", size=9)
    required = {
        "source_step": 15,
        "journal_entry_index": 14,
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ContractError(f"K15 seed provenance must bind {key}={expected}")
    for key in (
        "journal_raw_sha256",
        "journal_self_sha256",
        "center_terminal_raw_sha256",
        "center_terminal_self_sha256",
    ):
        _require_sha256(provenance.get(key), key)
    return validate_u_db(shape + (0.0, 0.0), name="K15-lifted u_db")


def make_spsa_rng() -> np.random.Generator:
    """Create the new family’s independent PCG64 stream, not the old 9-D one."""
    return np.random.Generator(np.random.PCG64(SPSA_RNG_SEED))


def draw_rademacher_direction(rng: np.random.Generator) -> tuple[int, ...]:
    if not isinstance(rng, np.random.Generator) or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ContractError("11-D directions require the family PCG64 generator")
    draw = rng.integers(0, 2, size=DIMENSION, dtype=np.int8)
    direction = tuple(int(value) * 2 - 1 for value in draw)
    if len(direction) != DIMENSION or any(value not in (-1, 1) for value in direction):
        raise ContractError("PCG64 produced an invalid Rademacher direction")
    return direction


def batch_sample_ids(sample_to_speaker: Mapping[str, str], iteration: int) -> tuple[str, ...]:
    if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
        raise ContractError("iteration must be in 0..19")
    if not sample_to_speaker or any(not isinstance(k, str) or not isinstance(v, str) for k, v in sample_to_speaker.items()):
        raise ContractError("sample-to-speaker mapping must be non-empty string data")
    speakers = tuple(sorted(set(sample_to_speaker.values())))
    if len(speakers) != 20:
        raise ContractError("the frozen SPSA batch requires exactly 20 speakers")
    clips_by_speaker = {
        speaker: tuple(sorted(sample for sample, sid in sample_to_speaker.items() if sid == speaker))
        for speaker in speakers
    }
    if any(len(clips_by_speaker[speaker]) != SPSA_CLIPS_PER_SPEAKER for speaker in speakers):
        raise ContractError("each of the 20 speakers must have exactly two clips")
    start = (iteration * SPSA_BATCH_SPEAKERS) % len(speakers)
    selected = tuple(speakers[(start + offset) % len(speakers)] for offset in range(SPSA_BATCH_SPEAKERS))
    result = tuple(sample for speaker in selected for sample in clips_by_speaker[speaker])
    if len(result) != SPSA_BATCH_SPEAKERS * SPSA_CLIPS_PER_SPEAKER:
        raise ContractError("cyclic SPSA batch did not resolve to eight clips")
    return result


def step_sizes(iteration: int) -> tuple[float, float]:
    if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
        raise ContractError("iteration must be in 0..19")
    a_k = 0.5 / ((iteration + 5) ** 0.602)
    c_base = 0.25 / ((iteration + 1) ** 0.101)
    return a_k, c_base


def propose_pair(center_u_db: Sequence[float], iteration: int, direction: Sequence[int]) -> SpsaPair:
    center = validate_u_db(center_u_db, name="center_u_db")
    if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
        raise ContractError("iteration must be in 0..19")
    if isinstance(direction, (str, bytes)) or len(direction) != DIMENSION:
        raise ContractError("direction must contain exactly 11 signs")
    delta = tuple(int(value) for value in direction)
    if any(value not in (-1, 1) for value in delta):
        raise ContractError("direction must be a Rademacher +/-1 vector")
    a_k, c_base = step_sizes(iteration)
    margin = min(bound - abs(value) for value, bound in zip(center, COORDINATE_ABS_BOUND_DB))
    c_actual = min(c_base, 0.5 * margin)
    if c_actual < MIN_PERTURBATION_DB:
        stop = BoundaryLimitedNoUpdate(
            status="BOUNDARY_LIMITED_NO_UPDATE",
            phase="paired_perturbation",
            iteration=iteration,
            center_u_db=center,
            direction=delta,
            c_base_db=c_base,
            c_actual_db=c_actual,
            available_margin_db=margin,
            gradient=None,
            raw_update_v=None,
            common_scale_s=None,
            chosen_beta=None,
            backtrack_exponent=None,
            proposed_u_db=None,
            reason="safe common perturbation fell below c_min_db",
        )
        return SpsaPair(PAIR_SCHEMA, stop.status, iteration, center, delta, a_k, c_base, c_actual, margin, None, None, stop)
    plus = tuple(center[i] + c_actual * delta[i] for i in range(DIMENSION))
    minus = tuple(center[i] - c_actual * delta[i] for i in range(DIMENSION))
    validate_u_db(plus, name="plus_u_db")
    validate_u_db(minus, name="minus_u_db")
    return SpsaPair(PAIR_SCHEMA, "PAIR_READY", iteration, center, delta, a_k, c_base, c_actual, margin, plus, minus, None)


def update_center(pair: SpsaPair, loss_plus: float, loss_minus: float) -> SpsaUpdate:
    if pair.schema != PAIR_SCHEMA:
        raise ContractError("pair schema is not the 11-D family schema")
    if pair.status != "PAIR_READY" or pair.c_actual_db is None:
        raise ContractError("cannot update from a boundary-limited or incomplete pair")
    lp, lm = float(loss_plus), float(loss_minus)
    if not math.isfinite(lp) or not math.isfinite(lm):
        raise ContractError("pair losses must be finite")
    a_k = pair.a_k
    c_actual = pair.c_actual_db
    gradient = tuple((lp - lm) / (2.0 * c_actual * pair.direction[i]) for i in range(DIMENSION))
    raw_v = tuple(-a_k * value for value in gradient)
    if not all(math.isfinite(value) for value in (*gradient, *raw_v)):
        raise ContractError("SPSA gradient/update is non-finite")
    max_abs_v = max(abs(value) for value in raw_v)
    scale_s = 1.0 if max_abs_v == 0.0 else min(1.0, MAX_UPDATE_COMPONENT_DB / max_abs_v)
    for exponent in range(MAX_BACKTRACK_EXPONENT + 1):
        beta = 2.0 ** (-exponent)
        proposed = tuple(pair.center_u_db[i] + beta * scale_s * raw_v[i] for i in range(DIMENSION))
        safe = all(
            abs(value) <= bound - NEXT_PERTURBATION_MARGIN_DB
            for value, bound in zip(proposed, COORDINATE_ABS_BOUND_DB)
        )
        if safe:
            no_effect = all(proposed[i] == pair.center_u_db[i] for i in range(DIMENSION))
            return SpsaUpdate(
                UPDATE_SCHEMA,
                "UPDATE_PROPOSED",
                pair.iteration, pair.center_u_db, pair.direction,
                a_k, pair.c_base_db, c_actual, lp, lm, gradient, raw_v, scale_s, beta, exponent, proposed, None,
                None,
                no_effect,
                None,
            )
    stop = BoundaryLimitedNoUpdate(
        status="BOUNDARY_LIMITED_NO_UPDATE",
        phase="center_update",
        iteration=pair.iteration,
        center_u_db=pair.center_u_db,
        direction=pair.direction,
        c_base_db=pair.c_base_db,
        c_actual_db=c_actual,
        available_margin_db=pair.available_margin_db,
        gradient=gradient,
        raw_update_v=raw_v,
        common_scale_s=scale_s,
        chosen_beta=None,
        backtrack_exponent=None,
        proposed_u_db=None,
        reason="no dyadic beta in 0..40 preserved the 2*c_min margin",
    )
    return SpsaUpdate(
        UPDATE_SCHEMA, stop.status, pair.iteration, pair.center_u_db, pair.direction,
        a_k, pair.c_base_db, c_actual, lp, lm, gradient, raw_v, scale_s, None, None, None, stop,
    )


def classify_effective_update(
    update: SpsaUpdate,
    before_mapping: Mapping[str, Any],
    after_mapping: Mapping[str, Any],
) -> SpsaUpdate:
    """Classify a tell after the pinned helper's real float32 coefficient map.

    Equality here is only equality of the actual recovered F32 coefficients.
    It makes no claim that the rendered waveform changed, since the shared
    kappa gauge can leave audio unchanged while coefficient vectors differ.
    """
    if update.schema != UPDATE_SCHEMA or update.center_after_u_db is None:
        raise ContractError("effective-update classification needs a completed proposal")
    if update.status != "UPDATE_PROPOSED" or update.helper_recovered_f32_unchanged is not None:
        raise ContractError("only an unclassified update proposal can be finalized as a tell")
    before = validate_helper_mapping(update.center_before_u_db, before_mapping)
    after = validate_helper_mapping(update.center_after_u_db, after_mapping)
    before_f32 = tuple(before["recovered_u_db_f32"])
    after_f32 = tuple(after["recovered_u_db_f32"])
    u_f64_unchanged = tuple(update.center_before_u_db) == tuple(update.center_after_u_db)
    helper_recovered_f32_unchanged = before_f32 == after_f32
    reasons = []
    if u_f64_unchanged:
        reasons.append("identical_f64_u")
    if helper_recovered_f32_unchanged:
        reasons.append("identical_actual_recovered_f32_u")
    return replace(
        update,
        status=(
            "TELL_COMPLETED_NO_EFFECTIVE_CHANGE"
            if u_f64_unchanged or helper_recovered_f32_unchanged
            else "UPDATED"
        ),
        no_effective_change_reason="+".join(reasons) if reasons else None,
        u_f64_unchanged=u_f64_unchanged,
        helper_recovered_f32_unchanged=helper_recovered_f32_unchanged,
    )


def is_center_due(iteration: int) -> bool:
    if type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS:
        raise ContractError("iteration must be in 0..19")
    return (iteration + 1) % 5 == 0


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _logmeanexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ContractError("logmeanexp requires finite, non-empty values")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values) / len(values))


def objective_from_effects(
    effects: Mapping[str, Any],
    u_db: Sequence[float],
    u_initial_db: Sequence[float],
    *,
    model_ids: Sequence[str],
    identity_reference_method_id: str,
    rms_reference_method_id: str,
) -> dict[str, float]:
    """Evaluate the frozen Cbest/guard/regularization objective in direct u dB."""
    current = validate_u_db(u_db, name="objective u_db")
    initial = validate_u_db(u_initial_db, name="fixed K15 seed u_initial_db")
    models = tuple(model_ids)
    if len(models) != 3 or len(set(models)) != 3 or any(not isinstance(x, str) or not x for x in models):
        raise ContractError("objective requires three distinct model IDs")
    if identity_reference_method_id != "identity":
        raise ContractError("identity reference ID must remain 'identity'")
    if rms_reference_method_id != RMS_REFERENCE_METHOD_ID:
        raise ContractError("RMS reference ID must retain the frozen 'rms_fixed' identity")
    refs = (identity_reference_method_id, rms_reference_method_id)
    if set(effects) != set(models):
        raise ContractError("effects must contain exactly the three pinned model IDs")
    for model in models:
        if set(effects[model]) != set(refs):
            raise ContractError("each model must contain both reference arms")
        for ref in refs:
            if not {"Cbest", "D0", "Canchor"}.issubset(effects[model][ref]):
                raise ContractError("each model/reference must provide Cbest, D0, and Canchor deltas")
            for metric in ("Cbest", "D0", "Canchor"):
                if not math.isfinite(float(effects[model][ref][metric])):
                    raise ContractError("effect values must be finite")

    c_terms: list[float] = []
    for ref, target in ((identity_reference_method_id, IDENTITY_CBEST_TARGET), (rms_reference_method_id, RMS_CBEST_TARGET)):
        terms = [(target - float(effects[model][ref]["Cbest"])) / TAU for model in models]
        c_terms.append(TAU * _logmeanexp(terms))
    l_c = sum(c_terms) / len(c_terms)

    guard_terms = [
        TAU * _softplus(-float(effects[model][ref][metric]) / TAU)
        for model in models for ref in refs for metric in ("D0", "Canchor")
    ]
    l_guard = GUARD_OUTER_COEFFICIENT * (sum(guard_terms) / len(guard_terms))
    l_reg = REGULARIZATION_LAMBDA * (sum((a - b) ** 2 for a, b in zip(current, initial)) / DIMENSION)
    total = l_c + l_guard + l_reg
    if not all(math.isfinite(value) for value in (l_c, l_guard, l_reg, total)):
        raise ContractError("objective produced a non-finite value")
    return {
        "L_c": l_c,
        "L_guard": l_guard,
        "L_regularization": l_reg,
        "L_total": total,
        "target_identity_Cbest": IDENTITY_CBEST_TARGET,
        "target_rms_Cbest": RMS_CBEST_TARGET,
        "tau": TAU,
        "guard_outer_coefficient": GUARD_OUTER_COEFFICIENT,
        "regularization_lambda": REGULARIZATION_LAMBDA,
        "regularization_anchor": "fixed_K15_seed_u11",
    }


def serialize_f64_vector(values: Sequence[float], *, name: str = "u_db") -> dict[str, Any]:
    vector = validate_u_db(values, name=name)
    raw = struct.pack("<11d", *vector)
    return {
        "dimension": DIMENSION,
        "dtype": "float64",
        "byte_order": "little",
        "values": list(vector),
        "raw_le_hex": raw.hex(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_helper_mapping(u_db: Sequence[float], mapped: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the Dev-owned u→z→F32 mapping response without applying audio."""
    requested = validate_u_db(u_db, name="mapped requested u_db")
    required = {
        "u_db_f64", "z_head_f64", "z_head_f32", "recovered_u_db_f32",
        "roundtrip_max_abs_error_db",
    }
    if not isinstance(mapped, Mapping) or not required.issubset(mapped):
        raise ContractError(f"helper map response is missing {sorted(required - set(mapped if isinstance(mapped, Mapping) else {}))}")
    u_echo = _finite_vector(mapped["u_db_f64"], name="helper u_db_f64")
    z = _finite_vector(mapped["z_head_f64"], name="helper z_head_f64")
    z_head = _finite_vector(mapped["z_head_f32"], name="helper z_head_f32")
    recovered = _finite_vector(mapped["recovered_u_db_f32"], name="helper recovered_u_db_f32")
    if any(a != b for a, b in zip(u_echo, requested)):
        raise ContractError("helper map response changed the requested u vector")
    expected_z = tuple(math.atanh(value / scale) for value, scale in zip(requested, COORDINATE_SCALE_DB))
    if any(abs(actual - expected) > 1.0e-12 for actual, expected in zip(z, expected_z)):
        raise ContractError("helper latent coordinates do not match the frozen inverse-atanh map")
    z_roundtrip = tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in z)
    if any(actual != expected for actual, expected in zip(z_head, z_roundtrip)):
        raise ContractError("helper head coordinates are not the F32 latent vector")
    expected_recovered = tuple(scale * math.tanh(value) for scale, value in zip(COORDINATE_SCALE_DB, z_head))
    if any(abs(actual - expected) > U_TO_Z_ROUNDTRIP_ATOL_DB for actual, expected in zip(recovered, expected_recovered)):
        raise ContractError("helper recovered-u values do not match its F32 head coordinates")
    actual_max_error = max(abs(actual - expected) for actual, expected in zip(recovered, requested))
    if actual_max_error > U_TO_Z_ROUNDTRIP_ATOL_DB:
        raise ContractError("F32 head mapping exceeds the frozen absolute-u roundtrip tolerance")
    declared_max_error = float(mapped["roundtrip_max_abs_error_db"])
    if not math.isfinite(declared_max_error) or abs(declared_max_error - actual_max_error) > 1.0e-12:
        raise ContractError("helper mapping roundtrip summary is inconsistent")
    z_bytes = struct.pack("<11f", *z_head)
    record = {
        "coordinate_order": list(COORDINATE_ORDER),
        "u_db": serialize_f64_vector(requested),
        "z_head_f64": list(z),
        "z_head_f32": list(z_head),
        "z_head_f32_le_hex": z_bytes.hex(),
        "z_head_f32_sha256": hashlib.sha256(z_bytes).hexdigest(),
        "recovered_u_db_f32": list(recovered),
        "roundtrip_max_abs_error_db": actual_max_error,
    }
    for field in ("u_sha256", "z_sha256", "recovered_sha256"):
        if field in mapped:
            record[f"helper_{field}"] = _require_sha256(mapped[field], f"helper_mapping.{field}")
    if "seed_kind" in mapped:
        record["seed_kind"] = str(mapped["seed_kind"])
    return record


def point_record(
    *,
    point_id: str,
    point_kind: str,
    iteration: int | None,
    u_db: Sequence[float],
    helper_mapping: Mapping[str, Any],
    profile_raw_sha256: str,
    profile_self_sha256: str,
    head_source_sha256: str,
    helper_source_sha256: str,
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a serializable point record after callers have resolved source pins.

    This function has no filesystem or process side effects. It requires the
    profile, head, helper, source, seed and reference pins rather than supplying
    defaults for them.
    """
    for name, value in (
        ("point_id", point_id),
        ("point_kind", point_kind),
    ):
        if not isinstance(value, str) or not value:
            raise ContractError(f"{name} must be a non-empty string")
    if iteration is not None and (type(iteration) is not int or not 0 <= iteration < SPSA_ITERATIONS):
        raise ContractError("point iteration must be None or in 0..19")
    profile_sha = _require_sha256(profile_raw_sha256, "profile_raw_sha256")
    profile_self = _require_sha256(profile_self_sha256, "profile_self_sha256")
    head_sha = _require_sha256(head_source_sha256, "head_source_sha256")
    helper_sha = _require_sha256(helper_source_sha256, "helper_source_sha256")
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise ContractError("source_bindings must contain exact role pins")
    normalized_bindings: dict[str, dict[str, Any]] = {}
    for role, pin in sorted(source_bindings.items()):
        if not isinstance(role, str) or not role or not isinstance(pin, Mapping):
            raise ContractError("source binding roles must map to pin objects")
        path = pin.get("path")
        byte_count = pin.get("bytes")
        sha256 = _require_sha256(pin.get("sha256"), f"source_bindings.{role}.sha256")
        if not isinstance(path, str) or not path or type(byte_count) is not int or byte_count <= 0:
            raise ContractError(f"source_bindings.{role} requires an exact path and positive byte count")
        normalized_bindings[role] = {"path": path, "bytes": byte_count, "sha256": sha256}
        if "self_sha256" in pin:
            normalized_bindings[role]["self_sha256"] = _require_sha256(
                pin["self_sha256"], f"source_bindings.{role}.self_sha256"
            )
    mapped = validate_helper_mapping(u_db, helper_mapping)
    return {
        "schema": POINT_SCHEMA,
        "family_id": FAMILY_ID,
        "point_id": point_id,
        "point_kind": point_kind,
        "iteration": iteration,
        "coordinate_space": "u_pre_kappa_db",
        "coordinate_order": list(COORDINATE_ORDER),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "profile_raw_sha256": profile_sha,
        "profile_self_sha256": profile_self,
        "head_source_sha256": head_sha,
        "application_helper_source_sha256": helper_sha,
        "parameter_mapping": mapped,
        "source_bindings": normalized_bindings,
        "outcome_scope": "candidate_waveform_is_shared_across_the_three_model_renderers",
        "status": "POINT_SPECIFIED_NOT_EXECUTED",
    }


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{name} must be a 64-character SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{name} must be hexadecimal") from exc
    return value.lower()


def validate_source_binding_map(
    source_bindings: Mapping[str, Mapping[str, Any]],
    *,
    required_roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate a caller-declared source closure before any input I/O or spawn."""
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        raise ContractError("source_bindings must be a non-empty role-to-pin map")
    roles = tuple(required_roles)
    if not roles or any(not isinstance(role, str) or not role for role in roles) or len(set(roles)) != len(roles):
        raise ContractError("required_roles must explicitly enumerate distinct source roles")
    missing = sorted(set(roles) - set(source_bindings))
    if missing:
        raise ContractError(f"source closure is missing required roles: {missing}")
    normalized: dict[str, dict[str, Any]] = {}
    for role in sorted(source_bindings):
        pin = source_bindings[role]
        if not isinstance(role, str) or not role or not isinstance(pin, Mapping):
            raise ContractError("source bindings must map non-empty roles to pin objects")
        path = pin.get("path")
        byte_count = pin.get("bytes")
        raw_sha256 = _require_sha256(pin.get("sha256"), f"source_bindings.{role}.sha256")
        if not isinstance(path, str) or not path or type(byte_count) is not int or byte_count <= 0:
            raise ContractError(f"source_bindings.{role} needs a pinned path and positive byte count")
        normalized[role] = {"path": path, "bytes": byte_count, "sha256": raw_sha256}
        if "self_sha256" in pin:
            normalized[role]["self_sha256"] = _require_sha256(
                pin["self_sha256"], f"source_bindings.{role}.self_sha256"
            )
    return normalized


def validate_row_input_mapping(
    row_mapping: Mapping[str, Any],
    *,
    expected_model_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate one row's pinned natural/RMS audio and frozen scorer inputs.

    This checks only metadata structure and declared equality. The audio helper
    separately verifies the actual float32 arrays and the apply_global_rms
    relation; the renderer's consumed bytes are bound by its per-model pin.
    """
    required = set(METHOD_CONTRACT["application_interface"]["row_mapping_required_keys"])
    if not isinstance(row_mapping, Mapping) or not required.issubset(row_mapping):
        missing = sorted(required - set(row_mapping if isinstance(row_mapping, Mapping) else {}))
        raise ContractError(f"row input mapping is missing {missing}")
    if row_mapping.get("schema") != ROW_INPUT_SCHEMA:
        raise ContractError("row input mapping has the wrong schema")
    sample_id = row_mapping.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ContractError("row input mapping requires a non-empty sample_id")

    def pcm_file_pin(value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ContractError(f"{name} must be a pinned PCM file record")
        pin = {
            "path": value.get("path"),
            "bytes": value.get("bytes"),
            "sha256": _require_sha256(value.get("sha256"), f"{name}.sha256"),
            "pcm_sha256": _require_sha256(value.get("pcm_sha256"), f"{name}.pcm_sha256"),
            "sample_rate_hz": value.get("sample_rate_hz"),
            "channels": value.get("channels"),
            "sample_count": value.get("sample_count"),
        }
        if not isinstance(pin["path"], str) or not pin["path"]:
            raise ContractError(f"{name}.path is required")
        if type(pin["bytes"]) is not int or pin["bytes"] <= 0:
            raise ContractError(f"{name}.bytes must be a positive integer")
        if pin["sample_rate_hz"] != 16000 or pin["channels"] != 1:
            raise ContractError(f"{name} must describe full mono 16 kHz PCM")
        if type(pin["sample_count"]) is not int or pin["sample_count"] <= 0:
            raise ContractError(f"{name}.sample_count must be a positive integer")
        return pin

    source = pcm_file_pin(row_mapping["source_pcm_pin"], "source_pcm_pin")
    if row_mapping["source_pcm_pin"].get("sample_id") != sample_id:
        raise ContractError("source PCM pin sample_id does not match the row")
    baseline = row_mapping["rms_fixed_reference"]
    if not isinstance(baseline, Mapping) or baseline.get("method_id") != RMS_REFERENCE_METHOD_ID:
        raise ContractError("canonical baseline must retain the exact rms_fixed method ID")
    generator_wav = pcm_file_pin(baseline.get("generator_wav_pin"), "rms_fixed_reference.generator_wav_pin")
    if baseline["generator_wav_pin"].get("sample_id") != sample_id:
        raise ContractError("strict240 RMS-fixed WAV pin sample_id does not match the row")
    if baseline.get("decoded_pcm_sha256") != generator_wav["pcm_sha256"]:
        raise ContractError("RMS-fixed decoded PCM digest differs from its pinned generator WAV")
    if (baseline.get("sample_rate_hz"), baseline.get("channels"), baseline.get("sample_count")) != (
        generator_wav["sample_rate_hz"], generator_wav["channels"], generator_wav["sample_count"]
    ):
        raise ContractError("RMS-fixed decoded format/count differs from its pinned generator WAV")
    if source["sample_count"] != generator_wav["sample_count"]:
        raise ContractError("RMS-fixed baseline changed full source sample count")
    policy = baseline.get("baseline_policy")
    if not isinstance(policy, Mapping):
        raise ContractError("RMS-fixed baseline policy must be explicit")
    policy_pin = policy.get("apply_global_rms_source_pin")
    if not isinstance(policy_pin, Mapping):
        raise ContractError("RMS-fixed policy must pin apply_global_rms source")
    _require_sha256(policy_pin.get("sha256"), "baseline_policy.apply_global_rms_source_pin.sha256")
    if (type(policy_pin.get("bytes")) is not int or policy_pin["bytes"] <= 0
            or not isinstance(policy_pin.get("path"), str) or not policy_pin["path"]):
        raise ContractError("apply_global_rms source pin requires path and positive bytes")
    if policy.get("target_rms") != RMS_BASELINE_TARGET_RMS or policy.get("peak_ceiling") is not None:
        raise ContractError("RMS-fixed baseline target/peak policy differs from the frozen source baseline")
    if policy.get("silence_epsilon") != RMS_BASELINE_SILENCE_EPSILON:
        raise ContractError("RMS-fixed silence policy differs from the pinned source baseline")
    gain = baseline.get("gain_info")
    gain_fields = ("input_rms", "target_rms", "requested_gain", "applied_gain", "output_rms", "peak_limited")
    if not isinstance(gain, Mapping) or not set(gain_fields).issubset(gain):
        raise ContractError("RMS-fixed gain diagnostics are incomplete")
    if any(not math.isfinite(float(gain[field])) for field in gain_fields[:-1]) or type(gain["peak_limited"]) is not bool:
        raise ContractError("RMS-fixed gain diagnostics are malformed")
    if gain["target_rms"] != RMS_BASELINE_TARGET_RMS or gain["peak_limited"]:
        raise ContractError("RMS-fixed gain diagnostics violate the un-limited fixed target")

    equality = row_mapping["renderer_pcm_equality"]
    models = tuple(expected_model_ids)
    if len(models) != 3 or len(set(models)) != 3 or any(not isinstance(m, str) or not m for m in models):
        raise ContractError("expected_model_ids must enumerate the three frozen renderer IDs")
    if not isinstance(equality, Mapping) or equality.get("pcm_sha256") != generator_wav["pcm_sha256"]:
        raise ContractError("three-renderer equality digest does not match the strict240 RMS-fixed PCM")
    if equality.get("sample_count") != generator_wav["sample_count"]:
        raise ContractError("three-renderer equality sample count does not match the RMS-fixed PCM")
    renderer_rows = equality.get("models")
    if not isinstance(renderer_rows, Sequence) or isinstance(renderer_rows, (str, bytes)) or len(renderer_rows) != 3:
        raise ContractError("renderer equality must contain exactly three model rows")
    normalized_renderers: list[dict[str, Any]] = []
    for expected_model, renderer in zip(models, renderer_rows):
        if not isinstance(renderer, Mapping) or renderer.get("model_id") != expected_model:
            raise ContractError("renderer equality model order/identity differs from the frozen model list")
        audio_pin = pcm_file_pin(renderer.get("renderer_wav_pin"), f"renderer_pcm_equality.{expected_model}.renderer_wav_pin")
        if renderer["renderer_wav_pin"].get("sample_id") != sample_id:
            raise ContractError(f"renderer {expected_model} WAV sample_id does not match the row")
        if (renderer.get("pcm_sha256"), renderer.get("sample_count")) != (
            generator_wav["pcm_sha256"], generator_wav["sample_count"]
        ) or (audio_pin["pcm_sha256"], audio_pin["sample_count"]) != (
            generator_wav["pcm_sha256"], generator_wav["sample_count"]
        ):
            raise ContractError(f"renderer {expected_model} did not consume the exact shared RMS-fixed PCM")
        normalized_renderers.append({"model_id": expected_model, "renderer_wav_pin": audio_pin,
                                     "pcm_sha256": audio_pin["pcm_sha256"], "sample_count": audio_pin["sample_count"]})

    references: dict[str, dict[str, Any]] = {}
    for role in ("source_clock_pin", "frame_map_pin", "roi_policy_pin", "input_plan_pin"):
        value = row_mapping[role]
        if not isinstance(value, Mapping):
            raise ContractError(f"{role} must be an exact pinned reference")
        path = value.get("path")
        byte_count = value.get("bytes")
        if not isinstance(path, str) or not path or type(byte_count) is not int or byte_count <= 0:
            raise ContractError(f"{role} needs a pinned path and positive byte count")
        reference = {"path": path, "bytes": byte_count,
                     "sha256": _require_sha256(value.get("sha256"), f"{role}.sha256")}
        if "self_sha256" in value:
            reference["self_sha256"] = _require_sha256(value["self_sha256"], f"{role}.self_sha256")
        references[role] = reference
    return {
        "schema": ROW_INPUT_SCHEMA,
        "sample_id": sample_id,
        "source_pcm_pin": {"sample_id": sample_id, **source},
        "rms_fixed_reference": {
            "method_id": RMS_REFERENCE_METHOD_ID,
            "generator_wav_pin": {"sample_id": sample_id, **generator_wav},
            "decoded_pcm_sha256": baseline["decoded_pcm_sha256"],
            "sample_rate_hz": baseline["sample_rate_hz"],
            "channels": baseline["channels"],
            "sample_count": baseline["sample_count"],
            "baseline_policy": dict(policy),
            "gain_info": dict(gain),
        },
        "renderer_pcm_equality": {"pcm_sha256": equality["pcm_sha256"], "sample_count": equality["sample_count"],
                                  "models": normalized_renderers},
        **references,
    }
