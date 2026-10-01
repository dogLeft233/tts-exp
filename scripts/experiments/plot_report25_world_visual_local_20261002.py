#!/usr/bin/env python3
"""Render three report25 figures from independently checked, sealed summaries.

The script reads no raw experiment artifacts and performs no inferential
analysis. It verifies the saved summary/independent-reconstruction seals,
selects the frozen endpoints, then renders PNG and SVG. All plotted estimates
and intervals are loaded from the sealed JSON/NPZ inputs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/recovered_20260930"
WORLD_DIR = BASE / "independent_validation/world_v6_consumer05_attempt02_root_readback"
LOCAL_DIR = BASE / "independent_validation/world_local_response_attempt08/package02_unpacked"
LOCAL_ROOT = BASE / "independent_validation/world_local_response_attempt08"
VIS_DIR = BASE / "remote/world_visual_analysis_attempt09"
OUT = BASE / "independent_validation/report25_figures_20261002_attempt08"

FILES = {
    "world_statistics": WORLD_DIR / "world_consumer_statistics.json",
    "world_audit": WORLD_DIR / "world_consumer_audit.json",
    "world_protocol": WORLD_DIR / "consumer_protocol.json",
    "world_output_seal": WORLD_DIR / "consumer_output_seal.json",
    "local_summary": LOCAL_DIR / "local_response_summary.json",
    "local_bootstrap": LOCAL_DIR / "local_bootstrap_reconstruction.npz",
    "local_stft": LOCAL_DIR / "local_stft_sensitivity_summary.json",
    "local_independent_reconstruction": LOCAL_ROOT / "local_response_independent_bootstrap_reconstruction_attempt01.json",
    "visual_analysis": VIS_DIR / "world_visual_analysis.json",
    "visual_analysis_seal": VIS_DIR / "world_visual_analysis_seal.json",
    "visual_endpoint_table": VIS_DIR / "world_visual_independent_endpoint_table_attempt01.json",
    "visual_independent_reconstruction": VIS_DIR / "world_visual_independent_reconstruction_attempt01.json",
    "visual_availability_freeze": VIS_DIR / "world_visual_availability_freeze.json",
    "visual_availability_seal": VIS_DIR / "world_visual_availability_freeze.seal.json",
    "visual_metadata_readback": VIS_DIR / "world_visual_metadata_bundle_independent_readback_attempt01.json",
}

# Exact-byte pins from completed independent readbacks.
EXPECTED_SHA256 = {
    "world_statistics": "b11f5530820ac82d654d77240bbb05b93ef2b6ab554a8e2c05e1b34fb9313b88",
    "world_audit": "fa06babb3a106bf1ed3acc337a6dc3bddc2c7df35dbd176f6c545c80cf599e30",
    "world_protocol": "e2e61f18c221123ccd43433333a405de685e3b0d6e77d3e082b21b11aca7aae5",
    "world_output_seal": "5f4f6677b5740a010e89120670f751a16b5e929899545b9be2e7856340290350",
    "local_summary": "0b7fe22c5f13ea3ab1dff1cf7a8a1ad6f1e21e0136e0334d854deae1e82e8ad7",
    "local_bootstrap": "464075d71dc2c53334b116c7120643727412f74f1c81ecbbc345b4397f6a0a11",
    "local_stft": "712265047cf20ab6a649d9ec53b3d36d1bc03eb2e7c94da46db2e033d541ccd2",
    "local_independent_reconstruction": "841210d86e9ae05b3b49ee8a1dc2ea0440973f8cb161b33865e05cc53336bb88",
    "visual_analysis": "5d0b8d18ec4565572292d5dbbdf43d25b92e72fbb3a0120eac0731536f31f98a",
    "visual_analysis_seal": "19e356d4d0392e1ce63270dade850821cfa1c1c8d05b2031a5d1fb20cca7d211",
    "visual_endpoint_table": "b0a8973965d89390319715f0d6c489ad28e94b5b763ec968601a3b9041681d1d",
    "visual_independent_reconstruction": "d7153b8fb64075ed5d6252348c37968664c8cb8f0de7084c03142bbe7285c83e",
    "visual_availability_freeze": "67131b9bd381dd2e393ab739ae51408e5edf15594cd3e9b67b844c5857621ccf",
    "visual_availability_seal": "1e7bab71c2d16689f5f316b543883dec3058f27aa053f7689d9495e95b3b37d2",
    "visual_metadata_readback": "0c2d4338119f53e42f0f7ee2e3090cf71695b88ce0d2b9898f9802cc0fac792c",
}

PREVIOUS_FIGURES = [
    BASE / "independent_validation/mechanism_figures_20261001_attempt05/rms_generator_G_observed_and_excess.png",
    BASE / "independent_validation/mechanism_figures_20261001_attempt05/mel_partition_C_observed_and_excess.png",
    BASE / "independent_validation/mechanism_figures_20261001_attempt05/mel_partition_C_anchor_observed_and_excess.png",
]

PHONE_ORDER = ("a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥", "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ")
PHASE_ORDER = ("whole", "early", "middle", "late")
SUPPORT_ORDER = ("full", "valid", "guard15")
SUPPORT_LABEL = {"full": "FULL", "valid": "VALID", "guard15": "guard 15"}
SUPPORT_COLOR = {"full": "#45556f", "valid": "#9a6bb4", "guard15": "#168b83"}
ARM_PRIMARY = ("EX", "SP", "BOTH", "F0", "AP")
ARM_CONTROL = ("EX_EM", "SP_EM", "E")
ARM_COLORS = {
    "EX": "#176b87", "SP": "#198c8c", "BOTH": "#277da1", "F0": "#568f58", "AP": "#7159a6",
    "EX_EM": "#c47a26", "SP_EM": "#b4525e", "E": "#777e87", "RT": "#334155", "FIXED": "#a7adb6",
}
DIRECTION_LABEL = {"tts_to_natural": "TTS donor → natural recipient", "natural_to_tts": "natural donor → TTS recipient"}
WORLD_CONTRAST_LABEL = {
    "G_q10_minus_q00": "G  (q10−q00)",
    "E_q01_minus_q00": "E  (q01−q00)",
    "I_q11_minus_q10_minus_q01_plus_q00": "I  (q11−q10−q01+q00)",
    "total_q11_minus_q00": "Total  (q11−q00)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_json(name: str) -> dict[str, Any]:
    path = FILES[name].resolve(strict=True)
    actual = sha256(path)
    expected = EXPECTED_SHA256[name]
    if actual != expected or len(expected) != 64:
        raise ValueError(f"{name} input seal mismatch: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def input_receipt() -> dict[str, Any]:
    return {
        name: {"path": str(path.resolve(strict=True)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in FILES.items()
    }


def _finite_ci(row: dict[str, Any], key: str) -> tuple[float, float]:
    ci = row.get(key)
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"missing interval {key} for row {row}")
    lo, hi = float(ci[0]), float(ci[1])
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError(f"invalid interval {key}")
    return lo, hi


def validate_seals(data: dict[str, dict[str, Any]]) -> None:
    wa = data["world_audit"]
    ws = data["world_output_seal"]
    if wa.get("error_count") != 0 or wa.get("all_files_report", {}).get("hash_mismatch_count") != 0:
        raise ValueError("WORLD independent consumer audit is not clean")
    if wa.get("all_files_report", {}).get("inventory_count") != 10887 or wa.get("all_files_report", {}).get("present_file_count") != 10887:
        raise ValueError("WORLD consumer all-files coverage differs from its sealed inventory")
    if ws.get("status") != "SEALED" or ws.get("consumer_source_sha256") != wa.get("consumer_source_sha256"):
        raise ValueError("WORLD output seal and independent audit disagree")

    local = data["local_summary"]
    local_recon = data["local_independent_reconstruction"]
    if local_recon.get("status") != "INDEPENDENT_BOOTSTRAP_RECONSTRUCTION_PASS":
        raise ValueError("local08 independent bootstrap reconstruction did not pass")
    meta = local["metadata"]
    if len(local["endpoints"]) != 16640 or meta.get("complete_estimable_family_endpoint_count") != 14560:
        raise ValueError("local08 endpoint/family count differs from frozen package")
    if local_recon.get("planned_endpoints") != 16640 or local_recon.get("estimable_endpoints") != 14560:
        raise ValueError("local08 independent reconstruction counts differ")
    if meta.get("bootstrap_reconstruction", {}).get("sha256") != EXPECTED_SHA256["local_bootstrap"]:
        raise ValueError("local08 summary does not bind the plotted bootstrap NPZ")
    if local_recon.get("bootstrap_npz_sha256") != EXPECTED_SHA256["local_bootstrap"]:
        raise ValueError("local08 independent reconstruction does not bind the bootstrap NPZ")

    visseal = data["visual_analysis_seal"]
    visrecon = data["visual_independent_reconstruction"]
    visbundle = data["visual_metadata_readback"]
    if visseal.get("status") != "SEALED" or visrecon.get("status") != "PASS" or visrecon.get("mismatches") != 0:
        raise ValueError("visual09 sealed report or independent reconstruction did not pass")
    if visrecon.get("summary_objects_recomputed") != 418 or visrecon.get("summary_fields_compared") != 5016:
        raise ValueError("visual09 independent reconstruction coverage differs")
    if visbundle.get("status") != "PASS" or visbundle.get("track_inventory_entries_bound") != 3202:
        raise ValueError("visual09 metadata bundle readback did not pass")
    if data["visual_availability_seal"].get("status") != "FROZEN_BEFORE_GEOMETRY_READ":
        raise ValueError("visual09 availability freeze is not properly sealed")


def validate_local_arrays(local: dict[str, Any], npz_path: Path) -> dict[tuple[str, ...], int]:
    archive = np.load(npz_path, allow_pickle=False)
    expected_shapes = {
        "endpoint_key_json_unicode": (16640,),
        "estimable_family_mask_bool": (16640,),
        "speaker_values_by_endpoint_float64": (16640, 40),
        "shared_speaker_resample_indices_int16": (20000, 40),
        "shared_speaker_resample_counts_uint8": (20000, 40),
        "family_max_t_draws_float64": (20000,),
    }
    for name, shape in expected_shapes.items():
        if name not in archive or archive[name].shape != shape:
            raise ValueError(f"local bootstrap NPZ shape mismatch for {name}")
    keys = archive["endpoint_key_json_unicode"]
    estimable_mask = archive["estimable_family_mask_bool"]
    speaker_values_by_endpoint = archive["speaker_values_by_endpoint_float64"]
    if len(set(map(str, keys))) != 16640:
        raise ValueError("local bootstrap endpoint identities are not unique")
    key_index: dict[tuple[str, ...], int] = {}
    # The frozen NPZ identity key deliberately omits the derived arm_stratum;
    # that grouping is bound by the summary's arm taxonomy, not repeated in
    # each compact endpoint key.
    identity_fields = ("arm", "direction", "estimand", "geometry", "outcome", "phone", "support")
    for i, encoded in enumerate(keys):
        item = json.loads(str(encoded))
        key = tuple(str(item.get(field, "")) for field in identity_fields)
        if key in key_index:
            raise ValueError(f"duplicate local endpoint identity {key}")
        key_index[key] = i
    if len(local["endpoints"]) != len(key_index):
        raise ValueError("local summary and bootstrap endpoint identities differ in count")
    for row in local["endpoints"]:
        key = tuple(str(row.get(field, "")) for field in identity_fields)
        if key not in key_index:
            raise ValueError(f"local endpoint missing from bootstrap NPZ: {key}")
        index = key_index[key]
        estimable = bool(estimable_mask[index])
        if row.get("status") == "ok":
            if not estimable:
                raise ValueError(f"local summary marks non-estimable NPZ row as ok: {key}")
            values = speaker_values_by_endpoint[index]
            mean = float(np.mean(values, dtype=np.float64))
            if abs(mean - float(row["estimate_speaker_equal_mean"])) > 1e-12:
                raise ValueError(f"local summary/bootstrap estimate mismatch: {key}")
        elif estimable:
            raise ValueError(f"local NPZ marks incomplete endpoint estimable: {key}")
    return key_index


def _unique(rows: list[dict[str, Any]], where: dict[str, Any], label: str) -> dict[str, Any]:
    hits = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(hits) != 1:
        raise ValueError(f"{label} selection has {len(hits)} rows: {where}")
    return hits[0]


def collect_world(world: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = world["bootstrap"]["summaries"]
    base = {"metric": "C_fixed_anchor", "domain": "raw", "support": "guard15"}
    arms = ("SP", "SP_EM", "EX", "F0", "AP", "E")
    gen: list[dict[str, Any]] = []
    sp_paths: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []
    for direction in ("tts_to_natural", "natural_to_tts"):
        for arm in arms:
            gen.append(_unique(rows, {**base, "family": "raw", "condition": arm, "direction": direction, "contrast": "G_q10_minus_q00"}, "WORLD generation"))
        for family in ("raw", "within_world"):
            for contrast in ("G_q10_minus_q00", "E_q01_minus_q00", "I_q11_minus_q10_minus_q01_plus_q00"):
                sp_paths.append(_unique(rows, {**base, "family": family, "condition": "SP", "direction": direction, "contrast": contrast}, "WORLD SP path"))
        for contrast in ("G_q10_minus_q00", "E_q01_minus_q00", "I_q11_minus_q10_minus_q01_plus_q00", "total_q11_minus_q00"):
            bridge.append(_unique(rows, {**base, "family": "bridge", "condition": "RT_bridge", "direction": direction, "contrast": contrast}, "WORLD RT bridge"))
    for row in gen + sp_paths + bridge:
        if row.get("status") != "ok" or row.get("clip_count") != 80 or row.get("speaker_count") != 40:
            raise ValueError(f"WORLD plotted bootstrap row is incomplete: {row.get('family')} {row.get('condition')} {row.get('direction')}")
        _finite_ci(row, "ci99_percentile")
    if (len(gen), len(sp_paths), len(bridge)) != (12, 12, 8):
        raise ValueError("WORLD selected endpoint coverage changed")
    return gen, sp_paths, bridge


def collect_local(local: dict[str, Any], stft: dict[str, Any], key_index: dict[tuple[str, ...], int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = local["endpoints"]
    labels = tuple(local["metadata"].get("phone_labels", []))
    if labels != PHONE_ORDER:
        raise ValueError("local08 frozen phone order differs from the plotting contract")
    atlas: dict[str, list[dict[str, Any]]] = {}
    for stratum, arms in (("primary", ARM_PRIMARY), ("control", ARM_CONTROL)):
        selected = [r for r in rows if r.get("arm_stratum") == stratum and r.get("arm") in arms and
                    r.get("estimand") in ("observed", "time_excess") and r.get("geometry") in ("raw", "unit") and
                    r.get("support") in SUPPORT_ORDER and r.get("outcome") in PHASE_ORDER]
        expected = len(arms) * 2 * 2 * len(SUPPORT_ORDER) * 2 * 4 * 16
        if len(selected) != expected or any(r.get("status") != "ok" for r in selected):
            raise ValueError(f"local08 {stratum} atlas coverage/status mismatch")
        for r in selected:
            _finite_ci(r, "ci99_simultaneous_max_t")
            # The sealed reconstruction NPZ omits the derived arm_stratum field.
            key = tuple(str(r.get(field, "")) for field in
                        ("arm", "direction", "estimand", "geometry", "outcome", "phone", "support"))
            if key not in key_index:
                raise ValueError(f"local plot row lacks independent bootstrap identity: {key}")
        atlas[stratum] = selected

    stft_rows = stft["endpoints"]
    main_whole = {(r["arm"], r["direction"], r["geometry"], r["support"], r["phone"]): r
                  for r in rows if r.get("estimand") == "time_excess" and r.get("outcome") == "whole"}
    boundary = []
    for r in stft_rows:
        if r.get("support") not in SUPPORT_ORDER:
            continue
        if r.get("status") != "descriptive_only":
            raise ValueError("a selected 50-ms STFT sensitivity row is unestimable")
        key = (r["arm"], r["direction"], r["geometry"], r["support"], r["phone"])
        main = main_whole.get(key)
        if main is None or main.get("status") != "ok":
            raise ValueError(f"STFT boundary sensitivity lacks whole-phone center estimate: {key}")
        stratum = "primary" if r["arm"] in ARM_PRIMARY else "control"
        boundary.append({"support": r["support"], "arm_stratum": stratum,
                         "delta": float(r["time_excess_speaker_equal_mean"] - main["estimate_speaker_equal_mean"])})
    if len(boundary) != 1536:
        raise ValueError(f"unexpected paired STFT boundary sensitivity coverage {len(boundary)}")
    return atlas, boundary


def collect_visual(analysis: dict[str, Any], table: dict[str, Any], recon: dict[str, Any]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    if recon.get("analysis_sha256") != EXPECTED_SHA256["visual_analysis"] or table.get("analysis_sha256") != EXPECTED_SHA256["visual_analysis"]:
        raise ValueError("visual09 endpoint table does not bind the sealed full report")
    if table.get("independent_recheck_sha256") != EXPECTED_SHA256["visual_independent_reconstruction"]:
        raise ValueError("visual09 endpoint table does not bind the independent reconstruction")
    contrasts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    levels: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for direction in ("tts_to_natural", "natural_to_tts"):
        block = table["reconstruction"][direction]
        crows = []
        for name, value in block["primary_contrasts"].items():
            cell = value["minus_85ms_phone_separation"]
            if cell.get("intended_clips") != 80 or cell.get("intended_speakers") != 40:
                raise ValueError("visual09 phone contrast has a changed intended denominator")
            if cell.get("complete_clips") not in (52, 53) or cell.get("complete_two_clip_speakers") not in (16, 17):
                raise ValueError("visual09 phone contrast has an unexpected availability denominator")
            ci = cell["ci99"]
            _finite_ci({"ci": ci}, "ci")
            crows.append({"label": name, **cell})
        bridge = [r for r in crows if r["label"] == "V_RT_minus_V_FIXED"]
        arms = [r for r in crows if r["label"] != "V_RT_minus_V_FIXED"]
        if len(arms) != 8 or len(bridge) != 1:
            raise ValueError("visual09 needs eight arm-vs-RT contrasts plus one separate RT−FIXED bridge")
        contrasts[direction] = {"arms": arms, "bridge": bridge}
        levels[direction] = []
        for arm, value in block["arm_levels"].items():
            for metric in ("geometry_aperture", "geometry_absolute_aperture_velocity_per_s"):
                cell = value[metric]
                if cell.get("complete_clips") != 80 or cell.get("complete_two_clip_speakers") != 40:
                    raise ValueError("visual09 aperture/velocity level is not complete on the full 80/40 cohort")
                _finite_ci(cell, "ci99")
                levels[direction].append({"arm": arm, "metric": metric, **cell})
        if len(levels[direction]) != 20:
            raise ValueError("visual09 arm-level aperture/velocity coverage changed")
    return contrasts, levels


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b7c0ca")
    ax.spines["bottom"].set_color("#b7c0ca")
    ax.tick_params(colors="#45515f", labelsize=8)
    ax.grid(axis="x", color="#e7ebf0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def draw_forest(ax: plt.Axes, rows: list[dict[str, Any]], labels: list[str], *, value_key: str, ci_key: str,
                colors: list[str], xlim: tuple[float, float] | None = None, markers: list[str] | None = None) -> None:
    ys = np.arange(len(rows))[::-1]
    for i, (row, y, label, color) in enumerate(zip(rows, ys, labels, colors)):
        estimate = float(row[value_key])
        lo, hi = map(float, row[ci_key])
        marker = markers[i] if markers else "o"
        ax.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]], fmt=marker, markersize=4.2,
                    color=color, ecolor=color, elinewidth=1.5, capsize=2.4, capthick=1.2, zorder=3)
    ax.axvline(0, color="#1f2937", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(ys, labels)
    if xlim:
        ax.set_xlim(*xlim)
    style_axes(ax)
    ax.tick_params(axis="y", length=0)


def render_world(gen: list[dict[str, Any]], sp_paths: list[dict[str, Any]], bridge: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(15.8, 11.2), constrained_layout=False)
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.89, hspace=.49, wspace=.24)
    fig.suptitle("WORLD acoustic arms: fixed-anchor confidence contrasts", fontsize=17, fontweight="bold", y=.97)
    fig.text(.5, .938, "Raw geometry · RAW recipient scoring audio where labeled · guard 15 · anchor k=+3 / column 18 / official offset −3",
             ha="center", fontsize=10.2, color="#465365")
    arms = ("SP", "SP_EM", "EX", "F0", "AP", "E")
    arm_labels = {"SP": "SP", "SP_EM": "SP_EM", "EX": "EX", "F0": "F0", "AP": "AP", "E": "E"}
    arm_color = {"SP": ARM_COLORS["SP"], "SP_EM": ARM_COLORS["SP_EM"], "EX": ARM_COLORS["EX"],
                 "F0": ARM_COLORS["F0"], "AP": ARM_COLORS["AP"], "E": ARM_COLORS["E"]}
    for col, direction in enumerate(("tts_to_natural", "natural_to_tts")):
        s = [r for r in gen if r["direction"] == direction]
        order = {a: i for i, a in enumerate(arms)}
        s.sort(key=lambda r: order[r["condition"]])
        ax = axes[0, col]
        draw_forest(ax, s, [arm_labels[r["condition"]] for r in s], value_key="estimate_speaker_equal_mean", ci_key="ci99_percentile",
                    colors=[arm_color[r["condition"]] for r in s])
        ax.set_title(f"{DIRECTION_LABEL[direction]}\nG = q10−q00; V_X/A_RAW − V_RT/A_RAW", fontsize=10.4, pad=8)
        ax.set_xlabel("Δ C_fixed_anchor = B_median − D18  (positive = higher confidence)", fontsize=8.5)

        s = [r for r in sp_paths if r["direction"] == direction]
        contrast_order = ("G_q10_minus_q00", "E_q01_minus_q00", "I_q11_minus_q10_minus_q01_plus_q00")
        family_order = ("raw", "within_world")
        marker_for = {contrast: marker for contrast, marker in zip(contrast_order, ("o", "s", "D"))}
        sorted_rows = []
        labels = []
        colors = []
        markers = []
        family_label = {"raw": "RAW", "within_world": "RT audio"}
        contrast_short = {"G_q10_minus_q00": "G", "E_q01_minus_q00": "E",
                          "I_q11_minus_q10_minus_q01_plus_q00": "I"}
        for contrast in contrast_order:
            for family in family_order:
                row = _unique(s, {"contrast": contrast, "family": family}, "SP path row")
                sorted_rows.append(row)
                labels.append(f"{contrast_short[contrast]} · {family_label[family]}")
                colors.append("#176b87" if family == "raw" else "#bf6c29")
                markers.append(marker_for[contrast])
        ax = axes[1, col]
        draw_forest(ax, sorted_rows, labels, value_key="estimate_speaker_equal_mean", ci_key="ci99_percentile", colors=colors, markers=markers)
        ax.set_title(f"{DIRECTION_LABEL[direction]}\nSP four-cell decomposition, two score-audio paths", fontsize=10.4, pad=8)
        ax.set_xlabel("Δ C_fixed_anchor = B_median − D18", fontsize=8.5)

        b = [r for r in bridge if r["direction"] == direction]
        b.sort(key=lambda r: ("G_q10_minus_q00", "E_q01_minus_q00", "I_q11_minus_q10_minus_q01_plus_q00", "total_q11_minus_q00").index(r["contrast"]))
        bridge_labels = [{"G_q10_minus_q00": "G", "E_q01_minus_q00": "E",
                          "I_q11_minus_q10_minus_q01_plus_q00": "I", "total_q11_minus_q00": "Total"}[r["contrast"]] for r in b]
        ax = axes[2, col]
        draw_forest(ax, b, bridge_labels, value_key="estimate_speaker_equal_mean", ci_key="ci99_percentile",
                    colors=["#697586"] * len(b), markers=["D", "s", "o", "^"])
        ax.set_title(f"{DIRECTION_LABEL[direction]}\nRT synthesis bridge (separate counterfactual)", fontsize=10.4, pad=8)
        ax.set_xlabel("Δ C_fixed_anchor = B_median − D18", fontsize=8.5)
    fig.text(.5, .071,
             "G=q10−q00; E=q01−q00; I=q11−q10−q01+q00. Pointwise 99% percentile intervals from 20,000 equal-speaker bootstrap draws (40 speakers; two clips averaged within speaker). "
             "Positive ΔC indicates higher SyncNet confidence; negative ΔC indicates lower confidence. The RT bridge is V_FIXED/V_RT × A_RAW/A_RT and is shown separately from arm treatments.",
             ha="center", va="center", fontsize=8.2, color="#374151", wrap=True)
    fig.text(.5, .035,
             "Intervals are pointwise, not familywise adjusted. q contrasts retain their RAW-score and within-WORLD-score families separately.",
             ha="center", va="center", fontsize=8.1, color="#5b6574")
    return fig


def _local_key(row: dict[str, Any]) -> tuple[str, ...]:
    fields = ("arm", "direction", "geometry", "support", "estimand", "phone", "outcome")
    return tuple(str(row.get(f, "")) for f in fields)


def render_local(local_atlas: dict[str, list[dict[str, Any]]], boundary: list[dict[str, Any]], stratum: str) -> plt.Figure:
    arms = ARM_PRIMARY if stratum == "primary" else ARM_CONTROL
    nrows = len(arms)
    title_suffix = "primary interventions" if stratum == "primary" else "prespecified controls"
    selected = local_atlas[stratum]
    by_key = {_local_key(row): row for row in selected}
    all_ci = [abs(float(v)) for row in selected for v in row["ci99_simultaneous_max_t"]]
    half_range = max(all_ci) * 1.08
    fig = plt.figure(figsize=(20.8, 2.55 * nrows + (2.9 if stratum == "primary" else 1.0)), constrained_layout=False)
    add_boundary = stratum == "primary"
    gs = fig.add_gridspec(nrows + (1 if add_boundary else 0), 4,
                          height_ratios=[1.0] * nrows + ([0.53] if add_boundary else []),
                          left=.105, right=.985, top=.86, bottom=.10 if add_boundary else .075,
                          hspace=.28, wspace=.11)
    fig.suptitle(f"Local WORLD response by frozen phone × phase — {title_suffix}", fontsize=17, fontweight="bold", y=.99)
    for ri, arm in enumerate(arms):
        for ci, (direction, geometry) in enumerate((("tts_to_natural", "raw"), ("tts_to_natural", "unit"),
                                                    ("natural_to_tts", "raw"), ("natural_to_tts", "unit"))):
            ax = fig.add_subplot(gs[ri, ci])
            if ri == 0:
                ax.set_title(f"{DIRECTION_LABEL[direction]} · {geometry.upper()}", fontsize=9.2, pad=5)
            # Phone row 0 remains at top; the frozen CAL label order is retained.
            segments = {(support, estimand): [] for support in SUPPORT_ORDER for estimand in ("observed", "time_excess")}
            points = {(support, estimand): ([], []) for support in SUPPORT_ORDER for estimand in ("observed", "time_excess")}
            phase_guides = []
            for pi, phone in enumerate(PHONE_ORDER):
                yy = pi
                if phone in ("n", "ŋ"):
                    ax.axhspan(yy - .5, yy + .5, color="#fff3d7", zorder=0)
                if pi % 2 == 1:
                    ax.axhspan(yy - .5, yy + .5, color="#f8fafc", zorder=-1)
                for phase_i, phase in enumerate(PHASE_ORDER):
                    phase_guides.append([(phase_i, yy - .48), (phase_i, yy + .48)])
                    support_offsets = {"full": -.115, "valid": 0.0, "guard15": .115}
                    for support in SUPPORT_ORDER:
                        for estimand, eoff, linestyle in (("observed", -.032, "-"), ("time_excess", .032, (0, (2, 1)))):
                            key = (arm, direction, geometry, support, estimand, phone, phase)
                            row = by_key.get(key)
                            if row is None:
                                raise ValueError(f"missing local matrix endpoint {key}")
                            estimate = float(row["estimate_speaker_equal_mean"])
                            lo, hi = map(float, row["ci99_simultaneous_max_t"])
                            if max(abs(lo), abs(hi), abs(estimate)) > half_range:
                                raise ValueError("local confidence interval exceeds shared cell scale")
                            y = yy + support_offsets[support] + eoff
                            scale = .405 / half_range
                            xlo, xhi, xp = phase_i + lo * scale, phase_i + hi * scale, phase_i + estimate * scale
                            segments[(support, estimand)].append([(xlo, y), (xhi, y)])
                            points[(support, estimand)][0].append(xp)
                            points[(support, estimand)][1].append(y)
            ax.add_collection(LineCollection(phase_guides, colors="#e5eaf0", linewidths=.35, zorder=0))
            for (support, estimand), line_segments in segments.items():
                linestyle = "solid" if estimand == "observed" else (0, (2, 1))
                ax.add_collection(LineCollection(line_segments, colors=SUPPORT_COLOR[support], linewidths=.72,
                                                 linestyles=linestyle, capstyle="round", zorder=3))
                px, py = points[(support, estimand)]
                ax.scatter(px, py, s=3.6, color=SUPPORT_COLOR[support], linewidths=0, zorder=4)
            ax.set_xlim(-.5, 3.5)
            ax.set_ylim(-.5, len(PHONE_ORDER) - .5)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(4), ["whole", "early", "middle", "late"] if ri == nrows - 1 or (add_boundary and ri == nrows - 1) else [])
            ax.set_yticks(np.arange(16), PHONE_ORDER if ci == 0 else [])
            if ci == 0:
                ax.set_ylabel(f"{arm}  ·  {('primary' if stratum == 'primary' else 'control')}", fontsize=9, fontweight="bold", labelpad=8)
                for label in ax.get_yticklabels():
                    if label.get_text() in ("n", "ŋ"):
                        label.set_color("#a55b00")
                        label.set_fontweight("bold")
                    label.set_fontsize(7.2)
            ax.tick_params(axis="x", labelsize=7, length=0, pad=2)
            ax.tick_params(axis="y", labelsize=7, length=0, pad=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if stratum == "primary" and ri == len(ARM_PRIMARY) - 1:
                ax.spines["bottom"].set_visible(True)
                ax.spines["bottom"].set_color("#b7c0ca")
        if stratum == "primary" and ri == len(ARM_PRIMARY) - 1:
            # A clear boundary between the prespecified primary and control strata.
            for col in range(4):
                fig.axes[ri * 4 + col].spines["top"].set_visible(True)
                fig.axes[ri * 4 + col].spines["top"].set_color("#c47a26")
                fig.axes[ri * 4 + col].spines["top"].set_linewidth(1.3)

    if add_boundary:
        ax = fig.add_subplot(gs[nrows, :])
        ax.axvline(0, color="#273444", linewidth=1, linestyle=(0, (3, 3)))
        categories = [(support, stratum_name) for support in SUPPORT_ORDER for stratum_name in ("primary", "control")]
        positions = np.arange(len(categories))
        for i, (support, stratum_name) in enumerate(categories):
            values = [r["delta"] for r in boundary if r["support"] == support and r["arm_stratum"] == stratum_name]
            expected_rows = (len(ARM_PRIMARY) if stratum_name == "primary" else len(ARM_CONTROL)) * 2 * 2 * 16
            if len(values) != expected_rows:
                raise ValueError(f"boundary sensitivity count mismatch for {support}/{stratum_name}")
            color = SUPPORT_COLOR[support] if stratum_name == "primary" else "#a7adb6"
            ax.boxplot(values, positions=[positions[i]], widths=.55, patch_artist=True,
                       showfliers=False, whis=(10, 90), manage_ticks=False, orientation="horizontal",
                       boxprops={"facecolor": color, "edgecolor": color, "alpha": .42},
                       medianprops={"color": "#202733", "linewidth": 1.1},
                       whiskerprops={"color": color, "linewidth": 1.0},
                       capprops={"color": color, "linewidth": 1.0})
        ax.set_yticks(positions, [f"{SUPPORT_LABEL[s]} · {st}" for s, st in categories])
        ax.set_xlabel("50-ms STFT-overlap time-excess minus mel-center time-excess (descriptive endpoint spread; no CI/p)", fontsize=8)
        ax.set_title("Boundary sensitivity: centered 50-ms STFT overlap weighting", loc="left", fontsize=9.3, pad=4)
        ax.tick_params(axis="y", labelsize=7.6, length=0)
        style_axes(ax)
    support_handles = [Line2D([0], [0], color=SUPPORT_COLOR[s], marker="o", lw=1.5, label=SUPPORT_LABEL[s]) for s in SUPPORT_ORDER]
    estimand_handles = [Line2D([0], [0], color="#303846", lw=1.4, linestyle="-", label="observed"),
                        Line2D([0], [0], color="#303846", lw=1.4, linestyle=(0, (2, 1)), label="observed − mean 100-permutation null")]
    fig.legend(handles=support_handles + estimand_handles, loc="upper center", bbox_to_anchor=(.5, .932),
               ncol=3, frameon=False, fontsize=7.7, handlelength=2.2)
    fig.text(.5, .016,
             f"Cell centers are zero; shared response scale within every phase cell is ±{half_range:.3g}. "
             "Whiskers are shared-family 99% max-|t| intervals. Nasal labels n and ŋ are shaded; exact-label exposure stays on the frozen support denominator. "
             "The 16,640-endpoint family has 14,560 estimable endpoints; 2,080 guard20 endpoints remain incomplete (short T support). "
             "Mel/RF phone weights include mixed-phone and boundary windows; they are not pure-phone effects or per-exposure efficiencies.",
             ha="center", va="bottom", fontsize=7.6, color="#374151", wrap=True)
    return fig


def render_local_sign_counts(local_atlas: dict[str, list[dict[str, Any]]]) -> tuple[plt.Figure, int]:
    """Show every complete-support family row as per-cell signed CI counts.

    Each phone×phase cell pools only across the frozen eight arms and two
    estimands (16 rows). Cell text reports positive point estimates and the
    numbers of common-family simultaneous intervals that are positive,
    negative, or include zero. The fill is positive-CI count minus
    negative-CI count; no endpoint is selected by its observed value.
    """
    rows = local_atlas["primary"] + local_atlas["control"]
    expected_total = 8 * 2 * 2 * len(SUPPORT_ORDER) * 2 * len(PHONE_ORDER) * len(PHASE_ORDER)
    if len(rows) != expected_total:
        raise ValueError(f"local sign-count atlas input coverage changed: {len(rows)} != {expected_total}")
    by_cell: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["direction"], row["geometry"], row["support"], row["phone"], row["outcome"])
        by_cell.setdefault(key, []).append(row)

    fig, axes = plt.subplots(3, 4, figsize=(18.5, 14.2), constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.91, top=.92, bottom=.10, hspace=.34, wspace=.17)
    fig.suptitle("Local response: shared-family simultaneous-CI sign counts", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .956,
             "Each 16×4 cell pools all 8 arms × observed/time-excess (n=16); all complete supports, both directions, RAW/UNIT. "
             "Text: positive estimate count; CI-positive / CI-negative / CI-crosses-zero counts. Fill: CI-positive minus CI-negative.",
             ha="center", va="center", fontsize=8.7, color="#465365")
    total_rows = 0
    for ri, support in enumerate(SUPPORT_ORDER):
        for ci, (direction, geometry) in enumerate((
            ("tts_to_natural", "raw"), ("tts_to_natural", "unit"),
            ("natural_to_tts", "raw"), ("natural_to_tts", "unit"),
        )):
            ax = axes[ri, ci]
            net = np.zeros((len(PHONE_ORDER), len(PHASE_ORDER)), dtype=np.int8)
            for pi, phone in enumerate(PHONE_ORDER):
                for oi, phase in enumerate(PHASE_ORDER):
                    cell = by_cell.get((direction, geometry, support, phone, phase), [])
                    if len(cell) != 16:
                        raise ValueError(f"local shared-family count cell lacks 16 rows: {direction}/{geometry}/{support}/{phone}/{phase}: {len(cell)}")
                    total_rows += len(cell)
                    positive_estimates = sum(float(r["estimate_speaker_equal_mean"]) > 0.0 for r in cell)
                    positive_ci = sum(float(r["ci99_simultaneous_max_t"][0]) > 0.0 for r in cell)
                    negative_ci = sum(float(r["ci99_simultaneous_max_t"][1]) < 0.0 for r in cell)
                    zero_ci = len(cell) - positive_ci - negative_ci
                    net[pi, oi] = positive_ci - negative_ci
                    ax.text(oi, pi, f"P+{positive_estimates}\n+{positive_ci}/−{negative_ci}/0{zero_ci}",
                            ha="center", va="center", fontsize=5.2,
                            color="#17212b" if abs(int(net[pi, oi])) < 11 else "white", linespacing=1.05)
                if phone in ("n", "ŋ"):
                    ax.add_patch(Rectangle((-.48, pi - .48), len(PHASE_ORDER) - .04, .96,
                                           fill=False, edgecolor="#b56500", linewidth=1.1, zorder=4))
            image = ax.imshow(net, cmap="RdBu_r", vmin=-16, vmax=16, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(PHASE_ORDER)), PHASE_ORDER)
            ax.set_yticks(np.arange(len(PHONE_ORDER)), PHONE_ORDER)
            ax.tick_params(axis="both", length=0, labelsize=7.2)
            for tick in ax.get_yticklabels():
                if tick.get_text() in ("n", "ŋ"):
                    tick.set_color("#a55b00")
                    tick.set_fontweight("bold")
            ax.set_xticks(np.arange(-.5, len(PHASE_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(PHONE_ORDER), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.75)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_title(f"{SUPPORT_LABEL[support]} · {DIRECTION_LABEL[direction]} · {geometry.upper()}", fontsize=8.1, pad=5)
    cax = fig.add_axes([.925, .26, .018, .48])
    cbar = fig.colorbar(image, cax=cax, ticks=(-16, -8, 0, 8, 16))
    cbar.set_label("CI-positive count − CI-negative count (of 16)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(.5, .055,
             f"Complete-support rows included: {total_rows:,} (= 12,288 planned cells). Shared 99% max-|t| intervals retain the full 16,640-endpoint family; 2,080 guard20 endpoints remain incomplete and are shown in the endpoint atlas note. "
             "CI-crosses-zero is the third count. Positive-estimate counts are descriptive; the heatmap reports no p-values. Nasal n/ŋ rows are outlined.",
             ha="center", va="center", fontsize=8.2, color="#374151", wrap=True)
    return fig, total_rows


def _visual_cell(source: dict[str, Any], estimate: str, ci: str) -> tuple[float, tuple[float, float]]:
    return float(source[estimate]), _finite_ci(source, ci)


def render_visual(contrasts: dict[str, dict[str, list[dict[str, Any]]]], levels: dict[str, dict[str, list[dict[str, Any]]]]) -> plt.Figure:
    fig = plt.figure(figsize=(15.8, 16.0), constrained_layout=False)
    gs = fig.add_gridspec(4, 2, height_ratios=[2.65, .95, 2.45, 2.45], left=.16, right=.98, top=.90, bottom=.08, hspace=.6, wspace=.25)
    fig.suptitle("Visual mouth response on the recipient clock", fontsize=17, fontweight="bold", y=.99)
    fig.text(.5, .948, "Frozen two-class phone separation at −85 ms · open-vowel minus bilabial normalized aperture",
             ha="center", fontsize=10.3, color="#465365")
    arm_order = ("EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
    contrasts_for_arm = {"EX": "V_EX_minus_V_RT", "SP": "V_SP_minus_V_RT", "BOTH": "V_BOTH_minus_V_RT",
                         "F0": "V_F0_minus_V_RT", "AP": "V_AP_minus_V_RT", "EX_EM": "V_EX_EM_minus_V_RT",
                         "SP_EM": "V_SP_EM_minus_V_RT", "E": "V_E_minus_V_RT"}
    # Shared limits across directions for each endpoint family.
    top_limits = []
    for direction in ("tts_to_natural", "natural_to_tts"):
        for row in contrasts[direction]["arms"]:
            if row["label"] in contrasts_for_arm.values():
                top_limits.extend([*map(float, row["ci99"])])
    bridge_limits = [v for direction in ("tts_to_natural", "natural_to_tts") for row in contrasts[direction]["bridge"] for v in map(float, row["ci99"])]
    for col, direction in enumerate(("tts_to_natural", "natural_to_tts")):
        arm_rows = {r["label"]: r for r in contrasts[direction]["arms"]}
        selected = [arm_rows[contrasts_for_arm[a]] for a in arm_order]
        nclip = selected[0]["complete_clips"]
        nsp = selected[0]["complete_two_clip_speakers"]
        if any((r["complete_clips"], r["complete_two_clip_speakers"]) != (nclip, nsp) for r in selected):
            raise ValueError("visual09 arm contrasts do not share the frozen endpoint availability denominator")
        ax = fig.add_subplot(gs[0, col])
        draw_forest(ax, selected, [f"{a} − RT" for a in arm_order], value_key="equal_speaker_mean", ci_key="ci99",
                    colors=[ARM_COLORS[a] for a in arm_order], xlim=(min(top_limits) * 1.08, max(top_limits) * 1.08))
        ax.set_title(f"{DIRECTION_LABEL[direction]}\n{nclip}/80 eligible clips · {nsp}/40 complete speakers", fontsize=10, pad=7)
        ax.set_xlabel("Change in −85-ms phone-class separation (aperture units)", fontsize=8.3)

        b = contrasts[direction]["bridge"][0]
        ax = fig.add_subplot(gs[1, col])
        bridge_lo, bridge_hi = min(bridge_limits), max(bridge_limits)
        bridge_span = max(bridge_hi - bridge_lo, abs(bridge_lo), abs(bridge_hi), 1e-6)
        draw_forest(ax, [b], ["RT − FIXED bridge"], value_key="equal_speaker_mean", ci_key="ci99",
                    colors=["#697586"], xlim=(min(bridge_lo, 0.0) - .08 * bridge_span,
                                               max(bridge_hi, 0.0) + .08 * bridge_span), markers=["D"])
        ax.set_title(f"{DIRECTION_LABEL[direction]} · baseline bridge", fontsize=9, pad=5)
        ax.set_xlabel("−85-ms phone-class separation contrast", fontsize=8.0)

        for row_index, metric, title, xlabel in (
            (2, "geometry_aperture", "Full-cohort mean aperture", "Normalized mouth aperture"),
            (3, "geometry_absolute_aperture_velocity_per_s", "Full-cohort absolute aperture velocity", "Absolute aperture velocity per second"),
        ):
            vals = [r for r in levels[direction] if r["metric"] == metric]
            order = ("V_FIXED", "V_RT", "V_EX", "V_SP", "V_BOTH", "V_F0", "V_AP", "V_EX_EM", "V_SP_EM", "V_E")
            vals.sort(key=lambda r: order.index(r["arm"]))
            if len(vals) != 10:
                raise ValueError("visual09 arm-level chart needs all ten generated visual IDs")
            ax = fig.add_subplot(gs[row_index, col])
            xlim = (min(float(r["ci99"][0]) for r in vals), max(float(r["ci99"][1]) for r in vals))
            span = xlim[1] - xlim[0]
            xlim = (xlim[0] - .12 * span, xlim[1] + .12 * span)
            draw_forest(ax, vals, [r["arm"].replace("V_", "") for r in vals], value_key="equal_speaker_mean", ci_key="ci99",
                        colors=[ARM_COLORS.get(r["arm"].replace("V_", ""), ARM_COLORS["RT"]) for r in vals], xlim=xlim)
            if any(r["intended_clips"] != 80 or r["intended_speakers"] != 40 for r in vals):
                raise ValueError("visual09 full-cohort arm-level denominator changed")
            ax.set_title(f"{DIRECTION_LABEL[direction]} · 80 clips / 40 speakers", fontsize=9.5, pad=6)
            ax.set_xlabel(xlabel, fontsize=8.3)
    fig.text(.5, .046,
             "Top: eight intervention arms versus RT. The RT−FIXED phone-class bridge is isolated below and is not an arm effect. "
             "Bottom rows show absolute geometry summaries on all 80/40 contexts (descriptive levels, not contrasts). "
             "Intervals are rowwise 99% speaker-bootstrap percentiles (pointwise, not familywise adjusted) from the independently reconstructed CAL-v2 endpoint table; the −85-ms endpoint uses the recipient TextGrid clock.",
             ha="center", va="center", fontsize=8.0, color="#374151", wrap=True)
    return fig


def save_figure(fig: plt.Figure, out: Path, stem: str) -> dict[str, Any]:
    png = out / f"{stem}.png"
    svg = out / f"{stem}.svg"
    if png.exists() or svg.exists():
        raise FileExistsError(f"figure output already exists: {stem}")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white", metadata={"Software": "Matplotlib"})
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Creator": "report25 sealed-summary figure script"})
    plt.close(fig)
    from PIL import Image
    with Image.open(png) as im:
        dimensions = list(im.size)
    return {"png": {"path": str(png.resolve()), "bytes": png.stat().st_size, "sha256": sha256(png), "pixels": dimensions},
            "svg": {"path": str(svg.resolve()), "bytes": svg.stat().st_size, "sha256": sha256(svg)}}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty figure directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "svg.fonttype": "none", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    data = {name: load_pinned_json(name) for name in FILES if name != "local_bootstrap"}
    validate_seals(data)
    previous_hashes = {str(p.resolve()): sha256(p) for p in PREVIOUS_FIGURES}
    if len(previous_hashes) != 3:
        raise ValueError("expected the three pre-existing RMS/MEL plots")

    local_npz_path = FILES["local_bootstrap"].resolve(strict=True)
    if sha256(local_npz_path) != EXPECTED_SHA256["local_bootstrap"]:
        raise ValueError("local08 bootstrap NPZ SHA mismatch")
    local_npz_index = validate_local_arrays(data["local_summary"], local_npz_path)
    world_gen, world_sp_paths, world_bridge = collect_world(data["world_statistics"])
    local_atlas, boundary = collect_local(data["local_summary"], data["local_stft"], local_npz_index)
    visual_contrasts, visual_levels = collect_visual(data["visual_analysis"], data["visual_endpoint_table"], data["visual_independent_reconstruction"])

    figures = {}
    figures["figure25a_world_fixed_anchor"] = save_figure(render_world(world_gen, world_sp_paths, world_bridge), OUT, "figure25a_world_fixed_anchor")
    figures["figure25b_local_phone_phase"] = save_figure(render_local(local_atlas, boundary, "primary"), OUT, "figure25b_local_phone_phase")
    sign_fig, sign_rows = render_local_sign_counts(local_atlas)
    figures["figure25b_local_sign_counts"] = save_figure(sign_fig, OUT, "figure25b_local_sign_counts")
    # Controls are reported in a separately labeled stratum rather than mixed into primary-arm facets.
    figures["figure25b_supp_local_controls"] = save_figure(render_local(local_atlas, boundary, "control"), OUT, "figure25b_supp_local_controls")
    figures["figure25c_visual_mouth"] = save_figure(render_visual(visual_contrasts, visual_levels), OUT, "figure25c_visual_mouth")

    if {str(p.resolve()): sha256(p) for p in PREVIOUS_FIGURES} != previous_hashes:
        raise ValueError("an existing RMS/MEL plot changed while rendering report25 figures")

    manifest = {
        "schema_version": 1,
        "status": "RENDERED_FROM_SEALED_SUMMARIES",
        "figure_source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "inputs": input_receipt(),
        "input_validation": {
            "world_independent_consumer_errors": data["world_audit"].get("error_count"),
            "world_files_verified": data["world_audit"].get("all_files_report", {}).get("present_file_count"),
            "local_planned_endpoints": len(data["local_summary"]["endpoints"]),
            "local_estimable_endpoints": data["local_independent_reconstruction"].get("estimable_endpoints"),
            "local_bootstrap_estimate_identity_checks": len(data["local_summary"]["endpoints"]),
            "local_independent_bootstrap_reconstruction": data["local_independent_reconstruction"].get("status"),
            "visual_independent_summary_objects_recomputed": data["visual_independent_reconstruction"].get("summary_objects_recomputed"),
            "visual_independent_field_comparisons": data["visual_independent_reconstruction"].get("summary_fields_compared"),
            "visual_metadata_track_refs_verified": data["visual_metadata_readback"].get("track_inventory_entries_bound"),
        },
        "selection_contract": {
            "world": "bootstrap summaries; metric=C_fixed_anchor; raw geometry/domain; guard15; generation G for six requested arms; SP G/E/I for raw and within-WORLD score families; RT bridge G/E/I/total isolated",
            "world_interval": "20,000 paired/equal-speaker percentile bootstrap draws; pointwise 99% intervals as sealed in consumer statistics",
            "local": "all 16 frozen CAL labels × whole/early/middle/late × observed and time_excess × full/valid/guard15 × raw/unit × both directions; primary and control strata visually separated",
            "local_interval": "shared 20,000×40 speaker bootstrap with simultaneous max-|t| 99% intervals over the frozen planned family; unestimable guard20 rows remain in family mask",
            "local_boundary": "descriptive paired 50-ms STFT-overlap minus mel-center time_excess for whole-phone endpoint; no CI or p-value; endpoint rows are dependent",
            "local_sign_counts": "all eight arms × observed/time_excess endpoints aggregated per phone×phase (n=16), for each direction×geometry×full/valid/guard15; text gives positive point estimates and CI-positive/CI-negative/CI-crosses-zero counts, fill gives CI-positive minus CI-negative",
            "local_sign_counts_scope": "12,288 complete-support endpoint rows from the 16,640 planned common max-|t| family; 2,080 incomplete guard20 rows remain in the family mask and are identified in the endpoint atlas",
            "visual": "all eight arm−RT phone-separation contrasts plus isolated RT−FIXED bridge at −85 ms; absolute aperture and absolute aperture velocity for all ten visual IDs on full 80/40 cohort",
            "visual_interval": "rowwise 99% speaker-bootstrap percentile intervals, pointwise rather than familywise-adjusted; independently reconstructed from the sealed report and frozen availability table",
        },
        "row_counts": {
            "world_generation_G": len(world_gen), "world_SP_two_paths_G_E_I": len(world_sp_paths), "world_RT_bridge_G_E_I_total": len(world_bridge),
            "local_primary_rows": len(local_atlas["primary"]), "local_control_rows": len(local_atlas["control"]),
            "local_shared_family_sign_count_rows": sign_rows,
            "local_paired_STFT_boundary_rows": len(boundary),
            "visual_phone_arm_contrasts": sum(len(v["arms"]) for v in visual_contrasts.values()),
            "visual_RT_FIXED_bridges": sum(len(v["bridge"]) for v in visual_contrasts.values()),
            "visual_full_cohort_level_endpoints": sum(len(v) for v in visual_levels.values()),
        },
        "preserved_previous_RMS_MEL_figures": [{"path": path, "sha256": digest} for path, digest in previous_hashes.items()],
        "figures": figures,
        "notes": [
            "No effect values were typed into source; plotted estimates and intervals are selected from pinned report rows.",
            "The local phone-weighted response is a receptive-field localization measure, not an isolated-phone causal effect or per-exposure efficiency.",
            "The local 50-ms STFT boundary panel is descriptive and carries no inferential interval.",
            "No future mean/dynamic factorial values are included; add that panel only after its independent consumer completes.",
        ],
    }
    manifest_path = OUT / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(OUT.resolve()), "figure_source_sha256": manifest["figure_source"]["sha256"],
                      "figures": figures, "manifest_sha256": sha256(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
