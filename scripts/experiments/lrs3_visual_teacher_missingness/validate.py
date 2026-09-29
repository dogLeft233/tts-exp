from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    finite_array,
    load_self_hashed,
    tree_manifest,
    write_json,
)

METRICS = ("b_dynamic", "q_natural", "q_candidate")
BOOTSTRAP_COUNT = 20_000
BOOTSTRAP_SEED = 20260910
THRESHOLD = 0.02


def _step(t: np.ndarray, fallback: float) -> float:
    if t.size <= 1:
        return float(fallback)
    d = np.diff(t)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ProtocolError("timestamps are not strictly increasing")
    return float(np.median(d))


def _feature(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            mouth = np.asarray(z["canonical_mouth"], dtype=np.float64)
            valid = np.asarray(z["valid"], dtype=bool)
            times = np.asarray(z["timestamps_s"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ProtocolError(f"invalid feature: {path}") from exc
    if (
        mouth.ndim != 3
        or mouth.shape[1:] != (31, 2)
        or valid.shape != (len(mouth),)
        or times.shape != valid.shape
    ):
        raise ProtocolError(f"feature shape mismatch: {path}")
    finite_array(mouth, "canonical_mouth")
    finite_array(times, "timestamps")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ProtocolError(f"timestamps are not strictly increasing: {path}")
    if np.any(~valid & (np.abs(mouth).sum(axis=(1, 2)) > 0)):
        raise ProtocolError(f"invalid feature rows are not zero: {path}")
    return {"mouth": mouth, "valid": valid, "timestamps": times}


def _matches(
    g: Mapping[str, np.ndarray], x: Mapping[str, np.ndarray]
) -> list[tuple[int, int, bool]]:
    gt, xt = g["timestamps"], x["timestamps"]
    tol = (
        min(_step(gt, 0.02), _step(xt, _step(gt, 0.02) if len(xt) <= 1 else 0.02)) * 0.5
    )
    i = j = 0
    out = []
    while i < len(gt) and j < len(xt):
        delta = float(xt[j] - gt[i])
        if abs(delta) <= tol:
            out.append((i, j, bool(g["valid"][i] and x["valid"][j])))
            i += 1
            j += 1
        elif delta < -tol:
            j += 1
        else:
            i += 1
    return out


def _metric(g: Mapping[str, np.ndarray], x: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ds = [
        float(np.sqrt(np.mean((g["mouth"][i] - x["mouth"][j]) ** 2, dtype=np.float64)))
        for i, j, ok in _matches(g, x)
        if ok
    ]
    den = max(len(g["valid"]), len(x["valid"]))
    return {
        "distance": float(np.median(np.asarray(ds, dtype=np.float64))) if ds else None,
        "coverage": float(len(ds) / den) if den else 0.0,
        "matched_count": len(ds),
        "match_count": len(_matches(g, x)),
    }


def _ratio(a: float, b: float) -> float:
    return float(a / (b + 1e-12)) if b else 0.0


def _raw(record: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        a: Path(str(record["feature_paths"][a]))
        for a in ("real", "natural", "candidate")
    }
    fs = {a: _feature(p) for a, p in paths.items()}
    exact = {a: _metric(fs["real"], fs[a]) for a in ("natural", "candidate")}
    fractions = {a: float(np.mean(fs[a]["valid"])) for a in paths}
    checks = {
        "real_valid_fraction": fractions["real"] >= config.VALID_FRACTION_MIN,
        "natural_valid_fraction": fractions["natural"] >= config.VALID_FRACTION_MIN,
        "candidate_valid_fraction": fractions["candidate"] >= config.VALID_FRACTION_MIN,
        "natural_coverage": exact["natural"]["coverage"] >= config.COVERAGE_MIN,
        "candidate_coverage": exact["candidate"]["coverage"] >= config.COVERAGE_MIN,
        "primary_finite": exact["natural"]["distance"] is not None
        and exact["candidate"]["distance"] is not None,
    }
    eligible = bool(all(checks.values()))
    gn = {i: j for i, j, ok in _matches(fs["real"], fs["natural"]) if ok}
    gm = {i: j for i, j, ok in _matches(fs["real"], fs["candidate"]) if ok}
    gi = np.asarray(sorted(set(gn) & set(gm)), dtype=np.int64)
    ni = np.asarray([gn[i] for i in gi], dtype=np.int64)
    mi = np.asarray([gm[i] for i in gi], dtype=np.int64)
    den = max(*(len(fs[a]["valid"]) for a in paths))
    observed = bool(
        record.get("eligibility", {}).get("eligible", False)
        and eligible
        and len(gi) / den >= config.COVERAGE_MIN
        and len(gi) > 0
    )
    reasons = []
    if not record.get("eligibility", {}).get("eligible", False) or not eligible:
        reasons.append("parent_eligibility_false")
    if len(gi) / den < config.COVERAGE_MIN:
        reasons.append("common_coverage_below_0_85")
    if not len(gi):
        reasons.append("empty_common_support")
    row = {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "parent_eligibility": bool(
            record.get("eligibility", {}).get("eligible", False)
        ),
        "recomputed_eligibility": eligible,
        "eligibility_checks": checks,
        "exact_time": exact,
        "common_support": {
            "real_indices": gi.tolist(),
            "natural_indices": ni.tolist(),
            "candidate_indices": mi.tolist(),
            "count": len(gi),
            "coverage": float(len(gi) / den) if den else 0.0,
        },
        "observed": observed,
        "missing_reasons": reasons,
        "feature_hashes": {a: file_sha256(p) for a, p in paths.items()},
    }
    vals = {m: None for m in METRICS}
    if len(gi):
        g, n, m = (
            fs[a]["mouth"][idx].astype(np.float64, copy=False)
            for a, idx in (("real", gi), ("natural", ni), ("candidate", mi))
        )
        mg, mn, mm = (np.mean(v, axis=0, dtype=np.float64) for v in (g, n, m))
        gc, nc, mc = g - mg, n - mn, m - mm
        tn, tm = (
            float(np.mean((g - n) ** 2, dtype=np.float64)),
            float(np.mean((g - m) ** 2, dtype=np.float64)),
        )
        sn, sm = (
            float(np.mean((mg - mn) ** 2, dtype=np.float64)),
            float(np.mean((mg - mm) ** 2, dtype=np.float64)),
        )
        dn, dm = (
            float(np.mean((gc - nc) ** 2, dtype=np.float64)),
            float(np.mean((gc - mc) ** 2, dtype=np.float64)),
        )
        rn, rm = (
            float(np.mean((gc - nc[::-1]) ** 2, dtype=np.float64)),
            float(np.mean((gc - mc[::-1]) ** 2, dtype=np.float64)),
        )
        if abs(tn - sn - dn) > 1e-10 * max(1.0, tn) or abs(tm - sm - dm) > 1e-10 * max(
            1.0, tm
        ):
            raise ProtocolError(
                f"MSE decomposition identity failed: {record['sample_id']}"
            )
        vals = {
            "b_dynamic": _ratio(dn - dm, dn + dm),
            "q_natural": _ratio(rn - dn, rn + dn),
            "q_candidate": _ratio(rm - dm, rm + dm),
        }
        common_n = float(
            np.median(
                np.sqrt(
                    np.mean(
                        (fs["real"]["mouth"][gi] - fs["natural"]["mouth"][ni]) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
        )
        common_m = float(
            np.median(
                np.sqrt(
                    np.mean(
                        (fs["real"]["mouth"][gi] - fs["candidate"]["mouth"][mi]) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
        )
        row.update(
            {
                "e_total_natural": tn,
                "e_total_candidate": tm,
                "e_static_natural": sn,
                "e_static_candidate": sm,
                "e_dynamic_natural": dn,
                "e_dynamic_candidate": dm,
                "e_reverse_natural": rn,
                "e_reverse_candidate": rm,
                "identity_error_natural": tn - sn - dn,
                "identity_error_candidate": tm - sm - dm,
                "max_original_time_gap_s": float(
                    np.max(np.diff(fs["real"]["timestamps"][gi]))
                )
                if len(gi) > 1
                else 0.0,
                "d_n": common_n,
                "d_m": common_m,
                "b": _ratio(common_n - common_m, common_n + common_m)
                if observed
                else None,
            }
        )
    else:
        row.update(
            {
                k: None
                for k in (
                    "d_n",
                    "d_m",
                    "b",
                    "e_total_natural",
                    "e_total_candidate",
                    "e_static_natural",
                    "e_static_candidate",
                    "e_dynamic_natural",
                    "e_dynamic_candidate",
                    "e_reverse_natural",
                    "e_reverse_candidate",
                    "identity_error_natural",
                    "identity_error_candidate",
                )
            },
            max_original_time_gap_s=0.0,
        )
    if not observed:
        vals = {m: None for m in METRICS}
    row.update(vals)
    return row


def _bounds(
    rows: list[dict[str, Any]], groups: list[str], metric: str
) -> tuple[list[dict[str, Any]], float, float]:
    result, ls, us = [], [], []
    for group in groups:
        values = [
            float(r[metric])
            for r in rows
            if r["observed"]
            and r["source_group"] == group
            and r.get(metric) is not None
        ]
        n = sum(r["source_group"] == group for r in rows)
        missing = n - len(values)
        total = float(np.sum(values)) if values else 0.0
        lo, hi = ((total - missing) / n, (total + missing) / n) if n else (-1.0, 1.0)
        result.append(
            {
                "source_group": group,
                "n": n,
                "observed": len(values),
                "missing": missing,
                "sum_observed": total,
                "L_g": float(lo),
                "U_g": float(hi),
            }
        )
        ls.append(lo)
        us.append(hi)
    return result, float(np.mean(ls)), float(np.mean(us))


def _stats(rows: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    active = [
        g for g in groups if any(r["observed"] and r["source_group"] == g for r in rows)
    ]
    if not active:
        raise ProtocolError("observed group denominator is empty")
    values = np.asarray(
        [
            [
                float(
                    np.mean(
                        [
                            r[m]
                            for r in rows
                            if r["observed"] and r["source_group"] == g
                        ],
                        dtype=np.float64,
                    )
                )
                for m in METRICS
            ]
            for g in active
        ],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    idx = rng.integers(
        0, len(active), size=(BOOTSTRAP_COUNT, len(active)), dtype=np.int64
    )
    boot = values[idx].mean(axis=1)
    out = {
        "label": "observed_subset",
        "group_count": len(active),
        "record_count": sum(r["observed"] for r in rows),
        "excluded_groups": [g for g in groups if g not in active],
        "bootstrap_count": BOOTSTRAP_COUNT,
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "groups": [],
    }
    for g in active:
        rr = [r for r in rows if r["observed"] and r["source_group"] == g]
        out["groups"].append(
            {
                "source_group": g,
                "n": len(rr),
                **{
                    m: float(np.mean([r[m] for r in rr], dtype=np.float64))
                    for m in METRICS
                },
            }
        )
    for i, m in enumerate(METRICS):
        ci = [
            float(np.quantile(boot[:, i], 0.005, method="linear")),
            float(np.quantile(boot[:, i], 0.995, method="linear")),
        ]
        out[m] = {
            "mean": float(np.mean(values[:, i])),
            "positive_groups": int(np.sum(values[:, i] > 0)),
            "total_groups": len(active),
            "ci99": ci,
            "passes": bool(
                ci[0] > 0
                and np.mean(values[:, i]) > THRESHOLD
                and np.sum(values[:, i] > 0) >= 17
            ),
        }
    return out


def _verify_feature_bindings(
    records: list[dict[str, Any]], audit: Mapping[str, Any]
) -> None:
    bound = {str(row["path"]): str(row["sha256"]) for row in audit.get("features", [])}
    if len(bound) != 399:
        raise ProtocolError("cached H feature audit is incomplete")
    for record in records:
        for value in record["feature_paths"].values():
            path = Path(str(value))
            try:
                rel = path.resolve().relative_to(config.REPO.resolve()).as_posix()
            except ValueError as exc:
                raise ProtocolError(f"feature path outside repository: {path}") from exc
            if rel not in bound or file_sha256(path) != bound[rel]:
                raise ProtocolError(f"feature binding changed: {path}")


def _assert_frozen_cohort(
    records: list[dict[str, Any]], groups: list[str], cached: Mapping[str, Any]
) -> None:
    if cached.get("records") != records or cached.get("groups") != groups:
        raise ProtocolError("local cohort differs from frozen H records.json")


def _verify_fixed_inputs(records: list[dict[str, Any]], groups: list[str]) -> None:
    for name, digest in config.PARENT_HASHES.items():
        if file_sha256(config.PARENT / name) != digest:
            raise ProtocolError(f"parent hash changed: {config.PARENT / name}")
    for name, digest in config.LOCK_HASHES.items():
        if file_sha256(config.LOCK / name) != digest:
            raise ProtocolError(f"lock hash changed: {config.LOCK / name}")
    if file_sha256(config.REVIEW) != config.REVIEW_HASH:
        raise ProtocolError("artifact review hash changed")
    files, _ = tree_manifest(config.PARENT / "records")
    feature_files, _ = tree_manifest(config.PARENT / "features")
    combined = sorted(files + feature_files, key=lambda row: row["path"])
    digest = hashlib.sha256(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in combined).encode()
    ).hexdigest()
    if len(files) != 133 or len(feature_files) != 399 or digest != config.TREE_HASH:
        raise ProtocolError("record/feature tree manifest changed")
    for name, digest in config.H_HASHES.items():
        if file_sha256(config.H / name) != digest:
            raise ProtocolError(f"cached H hash changed: {config.H / name}")
    audit = load_self_hashed(config.H / "input_audit.json")
    cached_records = load_self_hashed(config.H / "records.json")
    _assert_frozen_cohort(records, groups, cached_records)
    _verify_feature_bindings(records, audit)


def _recompute(records: list[dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    rows = [_raw(r) for r in records]
    group_output, bounds = {}, {}
    for m in METRICS:
        gr, lo, hi = _bounds(rows, groups, m)
        group_output[m] = gr
        bounds[m] = {
            "L": lo,
            "U": hi,
            "interpretation": "all 23 groups equally weighted; missing values bounded in [-1,1]",
        }
    stats = _stats(rows, groups)
    timing = stats["q_natural"]["passes"] and stats["q_candidate"]["passes"]
    dynamic = stats["b_dynamic"]["passes"]
    if not timing:
        decision = "TIMING_SPECIFICITY_NOT_ESTABLISHED"
    elif not dynamic:
        decision = "NO_OBSERVED_DYNAMIC_ADVANTAGE_ESTABLISHED"
    elif all(bounds[m]["L"] > 1e-12 for m in METRICS):
        decision = "CACHED_DYNAMIC_SIGNAL_ROBUST_TO_MISSINGNESS"
    else:
        decision = "OBSERVED_DYNAMIC_SIGNAL_REQUIRES_NEW_COHORT"
    counts: dict[str, int] = {}
    for r in rows:
        for reason in r["missing_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    old_groups, old_l, old_u = _bounds(rows, groups, "b")
    history = load_self_hashed(config.H / "analysis.json")
    legacy = {
        "observed_count": sum(r["observed"] for r in rows),
        "record_count": len(rows),
        "group_count": len(groups),
        "L": old_l,
        "U": old_u,
        "group_rows": old_groups,
        "history_observed_count": int(history["observed_count"]),
        "history_L": float(history["bounds"]["L"]),
        "history_U": float(history["bounds"]["U"]),
        "max_abs_LU_error": max(
            abs(old_l - float(history["bounds"]["L"])),
            abs(old_u - float(history["bounds"]["U"])),
        ),
    }
    return {
        "schema_version": 2,
        "record_count": len(rows),
        "group_count": len(groups),
        "observed_count": sum(r["observed"] for r in rows),
        "missing_reason_counts": counts,
        "records": rows,
        "groups": group_output,
        "bounds": bounds,
        "observed_subset": stats,
        "legacy_reproduction": legacy,
        "diagnostic_decision": decision,
        "engineering_decision": "GO",
        "parent_gate_repaired": False,
        "stage02_authorized": False,
        "replacement_confirmed": False,
        "waveform_head_authorized": False,
        "generalization_established": False,
        "model_calls": 0,
        "new_videos": 0,
        "new_scores": 0,
    }


def validate(root: Path) -> dict[str, Any]:
    paths = config.RunPaths(root)
    protocol = load_self_hashed(paths.protocol)
    payload = load_self_hashed(paths.records)
    actual = load_self_hashed(paths.analysis)
    _verify_fixed_inputs(payload["records"], payload.get("groups", []))
    if protocol.get("groups") != payload.get("groups") or protocol.get(
        "split"
    ) != payload.get("split"):
        raise ProtocolError(
            "protocol cohort identity differs from frozen H records.json"
        )
    expected = _recompute(payload["records"], protocol["groups"])
    for field in expected:
        if field == "schema_version":
            continue
        if expected[field] != actual.get(field):
            raise ProtocolError(f"independent analysis mismatch: {field}")
    if actual.get("support_indices_sha256") != file_sha256(paths.support_indices):
        raise ProtocolError("support index hash mismatch")
    with (
        np.load(paths.support_indices, allow_pickle=False) as current,
        np.load(config.H / "support_indices.npz", allow_pickle=False) as cached,
    ):
        if set(current.files) != {"offsets", "real", "natural", "candidate"} or any(
            not np.array_equal(current[key], cached[key])
            for key in ("offsets", "real", "natural", "candidate")
        ):
            raise ProtocolError("support indices differ from cached H")
    for name, digest in config.H_HASHES.items():
        if file_sha256(config.H / name) != digest:
            raise ProtocolError("cached H changed after analysis")
    if expected["observed_count"] != 115 or expected["group_count"] != 23:
        raise ProtocolError("cached denominator mismatch")
    return write_json(
        paths.validation,
        {
            "schema_version": 1,
            "status": "PASS",
            "engineering_decision": "GO",
            "independent": True,
            "record_count": expected["record_count"],
            "observed_count": expected["observed_count"],
            "diagnostic_decision": expected["diagnostic_decision"],
            "analysis_sha256": actual["artifact_sha256"],
            "checks": [
                "independent support/decomposition",
                "independent bootstrap and bounds",
                "cached H hashes rechecked",
                "zero model/video/score budget",
            ],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate(args.run_root)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            args.run_root / "validation.json",
            {
                "schema_version": 1,
                "status": "FAIL",
                "engineering_decision": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
