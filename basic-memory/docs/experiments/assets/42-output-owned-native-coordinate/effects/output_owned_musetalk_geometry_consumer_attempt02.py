#!/usr/bin/env python3
"""SOURCE_ONLY six-row MuseTalk adapter for the qualified geometry consumer.

The statistical/phone/frame algorithm is left in geometry_effect_consumer_attempt03.py.
This adapter changes only the frozen protocol/cohort binding and accepts the
six-row output-owned readout manifest. It fails closed if the direct pre-codec
non-owned RGB control did not pass.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "output_owned_musetalk_protocol_attempt01.json"
PROTOCOL_SHA256 = "561e81a747b62c02af7a2465eac9457436013ffba45286be15d8c6cf40c9123e"
QUALIFIED_CONSUMER = HERE / "geometry_effect_consumer_attempt03.py"
QUALIFIED_CONSUMER_SHA256 = "2e419a0e511771f2117bedb5efeffe57dad0a4c92b5e444afc0a63ecdee814f9"
SAMPLES = ("BAC009S0022W0123", "BAC009S0024W0122")
ARMS = ("native_height_zero", "native_height_plus", "native_height_minus")
SEED = 20261003
OUTPUT_PROTOCOL_ARMS = {"zero", "plus", "minus"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("qualified_native_height_geometry_consumer", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import qualified geometry source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_consumer(module: Any) -> None:
    if sha_file(QUALIFIED_CONSUMER) != QUALIFIED_CONSUMER_SHA256:
        raise ValueError("qualified geometry consumer source pin mismatch")
    if sha_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("frozen output-owned MuseTalk protocol SHA mismatch")
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.PROTOCOL_SHA256 = PROTOCOL_SHA256
    module.SAMPLE_IDS = SAMPLES
    module.MODELS = ("musetalk",)
    module.MODEL_ALIASES = {
        "wav2lip": "wav2lip", "Wav2Lip": "wav2lip", "wav2lip_gan_bac9a81e": "wav2lip",
        "musetalk": "musetalk", "MuseTalk": "musetalk", "musetalk_fixed_portrait_attempt16": "musetalk",
        "ditto": "ditto", "Ditto": "ditto", "ditto_hubert_fixed_portrait_attempt18": "ditto",
    }
    module.MODEL_LABELS = {"musetalk": "MuseTalk"}

    def checked_frozen_inputs():
        protocol = module.read_json(PROTOCOL_PATH)
        if (protocol.get("schema") != "output_owned_musetalk_articulatory_probe_v1"
                or protocol.get("status") != "FROZEN_SOURCE_AND_CPU_PREPARATION_ONLY"
                or protocol.get("cohort") != list(SAMPLES) or protocol.get("model") != "musetalk"
                or protocol.get("seed") != SEED
                or set(protocol.get("future_gpu_cap", {}).get("arms", [])) != OUTPUT_PROTOCOL_ARMS
                or protocol.get("p_by_sample") != {sample: 4 for sample in SAMPLES}):
            raise ValueError("output-owned protocol scope or fixed p=4 differs")
        for path, expected, label in (
            (module.S17_PATH, module.S17_SHA256, "S17 support"),
            (module.LAG_EFFECT_PATH, module.LAG_EFFECT_SHA256, "old-RMS lag effect"),
            (module.FUNCTIONAL03_SOURCE, module.FUNCTIONAL03_SOURCE_SHA256, "functional endpoint source"),
        ):
            if sha_file(path) != expected:
                raise ValueError(f"frozen {label} source pin mismatch")
        support = module.read_json(module.S17_PATH)
        if (support.get("schema") != "fixed_functional_lag_s17_support_v1"
                or support.get("fps") != module.FPS
                or support.get("lags") != list(module.ABSOLUTE_LAGS)
                or support.get("lag_definition") !=
                "At original core time j/25, read geometry at j+lag; no clock edit or interpolation."):
            raise ValueError("frozen S17 support or original absolute-lag clock differs")
        lag_effect = module.read_json(module.LAG_EFFECT_PATH)
        if (lag_effect.get("schema") != "fixed_functional_lag_effect_v1"
                or lag_effect.get("status") != "COMPLETE_SEEN_DEV_EXPLORATORY_NOT_HUMAN_ACCURACY"):
            raise ValueError("old-RMS phase source identity differs")
        phases = frozen_musetalk_phase(module, lag_effect)
        if any(phases[("musetalk", sample)] != 4 for sample in SAMPLES):
            raise ValueError("protocol fixed p differs from original MuseTalk RMS phase")
        return protocol, support, lag_effect

    def frozen_musetalk_phase(base: Any, lag_effect: dict[str, Any]):
        output: dict[tuple[str, str], int] = {}
        expected_seeds = {20261003, 20261004, 20261005}
        for item in lag_effect.get("utterance_rms_offset_reference", []):
            sample = item.get("sample_id")
            model = base.MODEL_ALIASES.get(str(item.get("model")))
            if sample not in SAMPLES or model != "musetalk" or item.get("attribute_id") != base.ATTRIBUTE:
                continue
            if item.get("status") != "USABLE":
                raise ValueError(f"frozen MuseTalk old-RMS phase is unusable: {sample}")
            seeds = item.get("rms_seed_lag_fields")
            if not isinstance(seeds, list) or len(seeds) != 3:
                raise ValueError("old-RMS phase must bind the original three seeds")
            offsets, seen = [], set()
            for seed_row in seeds:
                seed = int(seed_row.get("seed", -1))
                if seed not in expected_seeds or "_rms_fixed" not in str(seed_row.get("row_id", "")):
                    raise ValueError("old-RMS phase contains a non-RMS or unexpected seed")
                seen.add(seed)
                offsets.append(int(seed_row["full_official_av_offset_frames"]))
            if seen != expected_seeds:
                raise ValueError("old-RMS phase seed roster differs")
            phase = int(item["full_offset_median_integer"])
            if phase != int(np.median(np.asarray(offsets, dtype=np.int64))):
                raise ValueError("fixed MuseTalk p is not the frozen three-seed RMS median")
            if phase not in base.ABSOLUTE_LAGS or phase - 4 not in base.ABSOLUTE_LAGS or phase + 4 not in base.ABSOLUTE_LAGS:
                raise ValueError(f"fixed p and p±4 must lie in absolute S17: {phase}")
            key = ("musetalk", str(sample))
            if key in output:
                raise ValueError(f"duplicate old-RMS MuseTalk phase: {sample}")
            output[key] = phase
        expected = {("musetalk", sample) for sample in SAMPLES}
        if set(output) != expected:
            raise ValueError(f"old-RMS MuseTalk phase roster mismatch: {sorted(set(output) ^ expected)}")
        return output

    def index_six_readout_rows(
        manifest_path: Path, remote_root_raw: str, readback_root: Path,
        contracts: dict[str, Any],
    ):
        manifest = module.read_json(manifest_path)
        if (manifest.get("schema") != "output_owned_musetalk_readout_manifest_v1"
                or manifest.get("status") != "PASS" or manifest.get("row_count") != 6):
            raise ValueError("pre-codec readout did not PASS its direct non-owned RGB control or six-row roster")
        control = manifest.get("precodec_nonowned_rgb_control", {})
        if control.get("status") != "PASS":
            raise ValueError("pre-codec non-owned RGB control is not PASS")
        entries = manifest.get("rows")
        if not isinstance(entries, list) or len(entries) != 6:
            raise ValueError("output-owned readout manifest must expose exactly six rows")
        remote_root = PurePosixPath(remote_root_raw.replace("\\", "/"))
        if not remote_root.is_absolute():
            raise ValueError("--remote-readout-root must be an absolute POSIX path")
        output: dict[tuple[str, str, str], dict[str, Any]] = {}
        natural_hashes: dict[str, set[str]] = {sample: set() for sample in SAMPLES}
        for entry in entries:
            remote_receipt = entry.get("row_receipt_path")
            expected_sha = entry.get("row_receipt_sha256")
            if not isinstance(remote_receipt, str) or not isinstance(expected_sha, str):
                raise ValueError("readout manifest row lacks receipt path/SHA")
            receipt_path = module.map_remote_path(remote_receipt, remote_root, readback_root)
            actual_receipt_sha = sha_file(receipt_path)
            if actual_receipt_sha != expected_sha:
                raise ValueError(f"row receipt SHA differs from readout manifest: {receipt_path}")
            row = module.selected_json_fields(receipt_path, {
                "row_id", "sample_id", "speaker_id", "model", "arm", "seed",
                "natural_pcm_float32_sha256", "mouth_motion",
            })
            sample = str(row.get("sample_id", ""))
            model = str(row.get("model", ""))
            arm = str(row.get("arm", ""))
            if sample not in SAMPLES or model != "musetalk" or arm not in ARMS or row.get("seed") != SEED:
                raise ValueError(f"unexpected output-owned row identity: {receipt_path}")
            if not isinstance(row.get("speaker_id"), str) or not row["speaker_id"]:
                raise ValueError(f"row receipt lacks speaker_id: {receipt_path}")
            mouth = row.get("mouth_motion")
            if not isinstance(mouth, dict) or mouth.get("status") != "PASS":
                raise ValueError(f"pre-codec visual extraction did not PASS: {receipt_path}")
            remote_mouth_receipt = mouth.get("receipt")
            mouth_receipt_sha = mouth.get("receipt_sha256")
            remote_npz = mouth.get("arrays_npz")
            npz_sha = mouth.get("arrays_npz_sha256")
            if not all(isinstance(value, str) for value in
                       (remote_mouth_receipt, mouth_receipt_sha, remote_npz, npz_sha)):
                raise ValueError(f"pre-codec visual row lacks receipt/NPZ pins: {receipt_path}")
            mouth_path = module.map_remote_path(remote_mouth_receipt, remote_root, readback_root)
            if sha_file(mouth_path) != mouth_receipt_sha:
                raise ValueError(f"pre-codec mouth receipt SHA differs: {mouth_path}")
            mouth_meta = module.selected_json_fields(mouth_path, {"schema", "status", "arrays_npz"})
            if mouth_meta.get("status") != "PASS" or not isinstance(mouth_meta.get("arrays_npz"), dict):
                raise ValueError(f"pre-codec mouth receipt is not PASS: {mouth_path}")
            npz_meta = mouth_meta["arrays_npz"]
            npz_path_from_meta = module.map_remote_path(str(npz_meta.get("path", "")), remote_root, readback_root)
            npz_path_from_row = module.map_remote_path(remote_npz, remote_root, readback_root)
            if npz_path_from_meta != npz_path_from_row or npz_meta.get("sha256") != npz_sha:
                raise ValueError(f"row/mouth geometry NPZ bindings differ: {receipt_path}")
            if sha_file(npz_path_from_row) != npz_sha:
                raise ValueError(f"pre-codec mouth NPZ SHA differs: {npz_path_from_row}")
            natural_sha = row.get("natural_pcm_float32_sha256")
            contract_sha = contracts[sample].get("natural_pcm_float32_sha256")
            if not isinstance(natural_sha, str) or natural_sha != contract_sha:
                raise ValueError(f"original natural PCM binding differs from renderer source: {sample}")
            natural_hashes[sample].add(natural_sha)
            key = (model, sample, arm)
            if key in output:
                raise ValueError(f"duplicate response row: {key}")
            output[key] = {
                "row_id": str(row.get("row_id", "")), "speaker_id": row["speaker_id"],
                "row_receipt_path": str(receipt_path), "row_receipt_sha256": actual_receipt_sha,
                "mouth_receipt_path": str(mouth_path), "mouth_receipt_sha256": mouth_receipt_sha,
                "npz_path": npz_path_from_row, "npz_sha256": npz_sha,
                "arrays": npz_meta.get("arrays"), "natural_pcm_float32_sha256": natural_sha,
            }
        expected_rows = {("musetalk", sample, arm) for sample in SAMPLES for arm in ARMS}
        if set(output) != expected_rows:
            raise ValueError(f"six-row readout roster mismatch: missing={sorted(expected_rows-set(output))}, extra={sorted(set(output)-expected_rows)}")
        if any(len(values) != 1 for values in natural_hashes.values()):
            raise ValueError("original natural PCM hash changed across MuseTalk arms")
        return output, {
            "path": str(manifest_path.resolve(strict=True)), "sha256": sha_file(manifest_path),
            "rows": len(output), "remote_root": str(remote_root),
            "readback_root": str(readback_root.resolve(strict=True)),
        }

    module.checked_frozen_inputs = checked_frozen_inputs
    module.old_rms_phase = lambda lag: frozen_musetalk_phase(module, lag)
    module.index_readout_rows = index_six_readout_rows


def main(argv: list[str] | None = None) -> int:
    module = load_module(QUALIFIED_CONSUMER)
    bind_consumer(module)
    return module.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
