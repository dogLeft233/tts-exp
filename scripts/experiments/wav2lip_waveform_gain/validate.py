from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt
from scripts.experiments import wav2lip_probe_validator_common as vc

from . import config


def _independent_gains(pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(pcm, dtype=np.int16)
    peak = float(np.max(np.abs(values.astype(np.float64))))
    gain_plus = min(float(10.0 ** (3.0 / 20.0)), float(0.98 * 32767.0 / peak))
    plus = np.rint(values.astype(np.float64) * gain_plus)
    minus = np.rint(values.astype(np.float64) / gain_plus)
    if np.any(plus < -32768.0) or np.any(plus > 32767.0) or np.any(minus < -32768.0) or np.any(minus > 32767.0):
        raise rt.ProtocolError("independent PCM construction would clip")
    return plus.astype(np.int16), minus.astype(np.int16), gain_plus


def _worker_metrics(score_row: dict, label: str) -> dict:
    visual, audio, cached = rt.load_worker_arrays(score_row)
    rebuilt = vc.matrix_from_embeddings(visual, audio)
    if rebuilt.shape != cached.shape or float(np.max(np.abs(rebuilt.astype(np.float64) - cached.astype(np.float64)))) > 1e-4:
        raise rt.ProtocolError(f"independent SyncNet matrix mismatch: {label}")
    # Score the verified cache so the independent result binds exactly to the
    # matrix that the producer attached to the SyncNet row.
    return vc.score_metrics(cached)


def validate(root: Path) -> dict:
    paths = config.RunPaths(root)
    protocol = rt.load_self(paths.protocol)
    drivers = rt.load_self(paths.drivers)
    scores = rt.load_self(paths.scores)
    if protocol.get("record_count") != 16 or len(drivers.get("rows", [])) != 16 or len(scores.get("rows", [])) != 36:
        raise rt.ProtocolError("E2 record or score count mismatch")
    candidate_rows = [row for row in scores["rows"] if str(row["video_arm"]) in {"GAIN_PLUS", "GAIN_MINUS"}]
    if len(candidate_rows) != 32 or len(scores["rows"]) - len(candidate_rows) != 4: raise rt.ProtocolError("candidate/control score split mismatch")
    score_map = {(str(row["sample_id"]), str(row["video_arm"])): row for row in candidate_rows}
    if len(score_map) != 32:
        raise rt.ProtocolError("duplicate or missing candidate score cell")
    control = rt.load_self(config.P_CONTROL)
    fresh_rows = [row for row in scores["rows"] if str(row["video_arm"]).startswith("FRESH_")]
    vc.validate_fresh_controls(fresh_rows, control["rows"])
    records = []
    for row in drivers["rows"]:
        sid = str(row["sample_id"])
        raw = rt.source_pcm16(Path(str(row["natural_audio"]["path"])))
        pcm = rt.pcm_array(raw)
        plus, minus, gain_plus = _independent_gains(pcm)
        for arm, expected in (("GAIN_PLUS", plus), ("GAIN_MINUS", minus)):
            audio_path = Path(str(row["arms"][arm]["audio"]))
            actual = rt.pcm_array(rt.source_pcm16(audio_path))
            if not np.array_equal(actual, expected):
                raise rt.ProtocolError(f"PCM construction mismatch: {sid}/{arm}")
            if rt.bytes_sha256(rt.source_pcm16(audio_path)) != str(row["arms"][arm]["pcm_sha256"]):
                raise rt.ProtocolError(f"candidate PCM binding mismatch: {sid}/{arm}")
            mel = rt.load_mel(Path(str(row["arms"][arm]["path"])))
            expected_mel = rt.natural_mel_from_pcm(rt.source_pcm16(audio_path))
            if float(np.max(np.abs(mel.astype(np.float64) - expected_mel.astype(np.float64)))) > 1e-6:
                raise rt.ProtocolError(f"candidate mel mismatch: {sid}/{arm}")
        natural_row = next(item for item in control["rows"] if str(item["sample_id"]) == sid and str(item["video_arm"]) == "N" and str(item["audio_arm"]) == "N")
        record = {"sample_id": sid, "source_group": str(row["source_group"]), "N": _worker_metrics(natural_row, f"{sid}/N")}
        for arm in ("GAIN_PLUS", "GAIN_MINUS"):
            score_row = score_map[(sid, arm)]
            record[arm] = _worker_metrics(score_row, f"{sid}/{arm}")
            if max(abs(float(record[arm]["curve"][i]) - float(score_row["score"]["full"]["curve"][i])) for i in range(31)) > 1e-6:
                raise rt.ProtocolError(f"score metric binding mismatch: {sid}/{arm}")
        records.append(record)
    expected = {arm: vc.contrast_summary(records, arm) for arm in ("GAIN_PLUS", "GAIN_MINUS")}
    analysis = rt.load_self(paths.analysis)
    for arm in expected:
        if analysis.get("contrasts", {}).get(arm) != expected[arm]:
            raise rt.ProtocolError(f"analysis mismatch: {arm}")
    result = rt.write_json(paths.validation, {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "status": "PASS", "independent": True, "engineering_decision": "GO", "scientific_decision": analysis["scientific_decision"], "contrasts": expected, "candidate_authorized": False, "checks": ["independent ties-to-even PCM rebuild", "original PCM bound to scoring", "independent matrix metrics", "shared grouped 99% bootstrap"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate(args.run_root)
        return 0
    except Exception as exc:
        rt.write_json(args.run_root / "validation.json", {"schema_version": 1, "protocol_id": "wav2lip_waveform_gain", "status": "FAIL", "engineering_decision": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
