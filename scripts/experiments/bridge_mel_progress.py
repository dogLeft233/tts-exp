"""Measure bridge movement in the official Wav2Lip mel space.

This is a score-independent diagnostic.  It deliberately reads the frozen
waveforms from the natural-video bridge sweep and never reads SyncNet scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = REPO / "runs/natural_video_bridge_sweep_20260913_v2"
DEFAULT_STATIC_RUN = REPO / "runs/static_image_bridge_lowalpha_20260913_v2"
DEFAULT_INPUTS = REPO / "runs/static_image_bridge_20260913/inputs.json"
DEFAULT_COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
DEFAULT_OUTPUT = REPO / "runs/bridge_mel_progress_official_20260913"
SAMPLE_RATE = 16_000
BRIDGE_ARMS = ("B025", "B050")
CHECK_ARMS = ("N", "B025", "B050", "B075", "MFA")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def official_wav2lip_audio_module():
    wav2lip_root = REPO / "third_party/Wav2Lip"
    sys.path.insert(0, str(wav2lip_root))
    import audio  # type: ignore[import-not-found]

    return audio


def load_official_mel(audio_module: Any, path: Path) -> np.ndarray:
    waveform = audio_module.load_wav(str(path), SAMPLE_RATE)
    mel = np.asarray(audio_module.melspectrogram(waveform), dtype=np.float64)
    if mel.ndim != 2 or not np.isfinite(mel).all() or mel.size == 0:
        raise ValueError(f"invalid official Wav2Lip mel: {path}")
    return mel


def progress(natural: np.ndarray, mfa: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if natural.shape != mfa.shape or natural.shape != candidate.shape:
        raise ValueError(
            f"mel shape mismatch: N={natural.shape}, M={mfa.shape}, B={candidate.shape}"
        )
    direction = mfa - natural
    displacement = candidate - natural
    denominator = float(np.sum(direction * direction))
    if denominator <= 1e-12:
        raise ValueError("natural-to-MFA mel direction is degenerate")
    value = float(np.sum(displacement * direction) / denominator)
    residual = displacement - value * direction
    return {
        "progress_to_M_minus_N": value,
        "mae_to_N": float(np.mean(np.abs(displacement))),
        "mae_to_M": float(np.mean(np.abs(candidate - mfa))),
        "orthogonal_ratio": float(
            np.linalg.norm(residual) / max(np.linalg.norm(displacement), 1e-12)
        ),
        "direction_norm": float(np.linalg.norm(direction)),
        "displacement_norm": float(np.linalg.norm(displacement)),
    }


def bootstrap_mean(values: np.ndarray, *, seed: int = 20260913, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def summarize(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    selected = rows if subset == "all" else [row for row in rows if row["subset"] == subset]
    result: dict[str, Any] = {"record_count": len(selected), "arms": {}}
    for arm in BRIDGE_ARMS:
        values = np.asarray(
            [row["arms"][arm]["progress_to_M_minus_N"] for row in selected],
            dtype=np.float64,
        )
        result["arms"][arm] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std_population": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "bootstrap_mean_ci95": bootstrap_mean(values),
            "records_ge_0_15": int(np.sum(values >= 0.15)),
            "records_ge_0_25": int(np.sum(values >= 0.25)),
            "records_ge_0_50": int(np.sum(values >= 0.50)),
            "records_ge_0_75": int(np.sum(values >= 0.75)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--static-run", type=Path, default=DEFAULT_STATIC_RUN)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sweep = args.sweep.resolve()
    static_run = args.static_run.resolve()
    inputs_path = args.inputs.resolve()
    cohort_path = args.cohort.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sweep_protocol_path = sweep / "protocol.json"
    static_scope_path = static_run / "analysis_scope.json"
    sweep_protocol = load_json(sweep_protocol_path)
    static_scope = load_json(static_scope_path)
    inputs = load_json(inputs_path)
    cohort = load_json(cohort_path)
    if list(sweep_protocol.get("arms", [])) != [
        "N",
        "RT",
        "B025",
        "B050",
        "B075",
        "B100",
        "MFA",
        "SWAP",
        "DELAY200",
    ]:
        raise ValueError("unexpected score-free bridge sweep protocol arms")
    static_ids = {str(sample_id) for sample_id in static_scope["sample_ids"]}
    cohort_by_id = {str(row["sample_id"]): row for row in cohort["records"]}
    audio_module = official_wav2lip_audio_module()

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for input_row in inputs["records"]:
        sample_id = str(input_row["sample_id"])
        cohort_row = cohort_by_id.get(sample_id)
        if cohort_row is None:
            raise ValueError(f"sample missing from score-free cohort: {sample_id}")
        sample_dir = sweep / sample_id
        paths = {arm: sample_dir / f"{arm}.wav" for arm in CHECK_ARMS}
        try:
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing)
            actual_hashes = {arm: sha256_file(path) for arm, path in paths.items()}
            if actual_hashes["N"] != str(input_row["audio"]["N"]["container_sha256"]):
                raise ValueError(f"natural audio hash mismatch for {sample_id}")
            if actual_hashes["MFA"] != str(cohort_row["mfa_linear_audio"]["sha256"]):
                raise ValueError(f"MFA-linear audio hash mismatch for {sample_id}")
            mels = {arm: load_official_mel(audio_module, path) for arm, path in paths.items()}
            n_mel = mels["N"]
            m_mel = mels["MFA"]
            arm_results = {
                arm: progress(n_mel, m_mel, mels[arm])
                for arm in CHECK_ARMS
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_group": str(input_row["source_group"]),
                    "subset": "static_11" if sample_id in static_ids else "full_22_only",
                    "paths": {arm: str(path) for arm, path in paths.items()},
                    "sha256": actual_hashes,
                    "mel_shapes": {arm: [int(value) for value in mel.shape] for arm, mel in mels.items()},
                    "arms": arm_results,
                }
            )
        except (OSError, ValueError) as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})

    if failures:
        raise RuntimeError(f"failed records: {failures}")
    if len(rows) != len(inputs["records"]):
        raise RuntimeError(f"expected {len(inputs['records'])} rows, got {len(rows)}")

    all_ids = {row["sample_id"] for row in rows}
    if not static_ids.issubset(all_ids):
        raise RuntimeError("static 11 IDs are not all present in the bridge sweep")
    summary = {
        "full_22": summarize(rows, "all"),
        "static_11": {
            "record_count": len(static_ids),
            "arms": {},
        },
    }
    static_rows = [dict(row, subset="static_11") for row in rows if row["sample_id"] in static_ids]
    summary["static_11"] = summarize(static_rows, "static_11")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "metric": "official_wav2lip_mel_projection_progress",
        "formula": "dot(mel(B)-mel(N), mel(M)-mel(N)) / max(dot(mel(M)-mel(N), mel(M)-mel(N)), 1e-12)",
        "interpretation": "0=N, 1=MFA-linear target direction endpoint in flattened official Wav2Lip mel space",
        "official_wav2lip_audio": str((REPO / "third_party/Wav2Lip/audio.py").resolve()),
        "official_wav2lip_audio_sha256": sha256_file(REPO / "third_party/Wav2Lip/audio.py"),
        "sample_rate": SAMPLE_RATE,
        "bridge_arms": list(BRIDGE_ARMS),
        "check_arms": list(CHECK_ARMS),
        "source_sweep": str(sweep),
        "source_sweep_protocol": str(sweep_protocol_path),
        "source_sweep_protocol_sha256": sha256_file(sweep_protocol_path),
        "source_inputs": str(inputs_path),
        "source_inputs_sha256": sha256_file(inputs_path),
        "source_cohort": str(cohort_path),
        "source_cohort_sha256": sha256_file(cohort_path),
        "static_analysis_scope": str(static_scope_path),
        "static_analysis_scope_sha256": sha256_file(static_scope_path),
        "record_count": len(rows),
        "rows": rows,
        "summary": summary,
    }
    write_json(output / "analysis.json", payload)
    lines = [
        "# Official Wav2Lip mel progress for bridge audio",
        "",
        "使用冻结的 natural `N`、MFA-linear `MFA`、bridge `B025/B050` 音频，通过官方 `third_party/Wav2Lip/audio.py:melspectrogram` 计算。未读取 SyncNet 分数。",
        "",
        "公式：`progress = dot(mel(B)-mel(N), mel(MFA)-mel(N)) / dot(mel(MFA)-mel(N), mel(MFA)-mel(N))`。",
        "",
        "| cohort | arm | n | mean | median | bootstrap 95% CI for mean | min | max | >=0.15 | >=0.25 | >=0.50 | >=0.75 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort_name, cohort_summary in (("full_22", summary["full_22"]), ("static_11", summary["static_11"])):
        for arm in BRIDGE_ARMS:
            values = cohort_summary["arms"][arm]
            ci = values["bootstrap_mean_ci95"]
            lines.append(
                f"| {cohort_name} | {arm} | {cohort_summary['record_count']} | {values['mean']:.4f} | {values['median']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {values['min']:.4f} | {values['max']:.4f} | {values['records_ge_0_15']} | {values['records_ge_0_25']} | {values['records_ge_0_50']} | {values['records_ge_0_75']} |"
            )
    lines.extend(
        [
            "",
            "校验臂中 `N` 的 progress 应为 0，`MFA` 应为 1；完整逐样本数值见 `analysis.json`。",
            "",
            "- `full_22`：自然视频 bridge 扫描的全部 22 条记录。",
            "- `static_11`：静态图低强度实验实际使用的前 11 条记录。",
            "- `B025/B050` 的 alpha 是 STFT log-magnitude 插值强度，不是 mel 空间中的理论 progress；progress 由实际落盘 WAV 经官方 Wav2Lip mel 重新计算。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
