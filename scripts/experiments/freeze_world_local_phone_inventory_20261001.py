#!/usr/bin/env python3
"""Create the CAL-only exact-phone inventory for local WORLD response analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CAL_QC_SHA256 = "5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5"
CAL_MANIFEST_SHA256 = "297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc"
EXPECTED_LABELS = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def create_inventory(qc_path: Path, cal_manifest_path: Path) -> dict[str, Any]:
    if file_sha256(qc_path) != CAL_QC_SHA256:
        raise ValueError("CAL conditional-acoustic QC SHA differs from the reviewed file")
    if file_sha256(cal_manifest_path) != CAL_MANIFEST_SHA256:
        raise ValueError("CAL physical manifest SHA differs from the reviewed file")

    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    cal_manifest = json.loads(cal_manifest_path.read_text(encoding="utf-8"))
    if qc.get("status") != "CAL_ONLY_READ_ONLY_DIAGNOSTIC" or qc.get("evaluation_outcomes_read") is not False:
        raise ValueError("phone selection requires the sealed CAL-only diagnostic")
    if qc.get("world_manifest_file_sha256") != CAL_MANIFEST_SHA256:
        raise ValueError("CAL diagnostic does not bind the frozen WORLD manifest")
    if cal_manifest.get("split") != "calibration" or len(cal_manifest.get("records", [])) != 20:
        raise ValueError("expected exactly 20 calibration records")

    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"sample_ids": set(), "speaker_ids": set(), "matched_tokens_across_directions": 0}
    )
    for record in qc["records"]:
        sample_id, speaker_id = str(record["id"]), str(record["speaker"])
        for direction in record["directions"].values():
            for category in direction.get("matched_phone_categories", []):
                label = str(category["phone_category"])
                counts[label]["sample_ids"].add(sample_id)
                counts[label]["speaker_ids"].add(speaker_id)
                counts[label]["matched_tokens_across_directions"] += int(category["matched_phone_token_count"])

    inventory = []
    for label in sorted(counts):
        sample_ids = sorted(counts[label]["sample_ids"])
        speaker_ids = sorted(counts[label]["speaker_ids"])
        inventory.append({
            "label": label,
            "calibration_utterance_count": len(sample_ids),
            "calibration_speaker_count": len(speaker_ids),
            "matched_tokens_across_both_directions": counts[label]["matched_tokens_across_directions"],
            "calibration_sample_ids": sample_ids,
            "calibration_speaker_ids": speaker_ids,
            "selected": len(sample_ids) >= 10 and len(speaker_ids) >= 5,
        })

    selected = [item["label"] for item in inventory if item["selected"]]
    if tuple(selected) != EXPECTED_LABELS:
        raise ValueError(f"CAL-only deterministic phone inventory changed: {selected!r}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_CAL_ONLY_PHONE_INVENTORY",
        "selection_split": "calibration",
        "evaluation_textgrids_or_endpoints_read": False,
        "source": {
            "calibration_qc_path": str(qc_path.resolve()),
            "calibration_qc_sha256": CAL_QC_SHA256,
            "calibration_manifest_path": str(cal_manifest_path.resolve()),
            "calibration_manifest_file_sha256": CAL_MANIFEST_SHA256,
            "calibration_manifest_embedded_sha256": cal_manifest.get("manifest_sha256"),
            "calibration_record_count": len(cal_manifest["records"]),
            "calibration_speaker_count": len({str(record["speaker"]) for record in cal_manifest["records"]}),
            "phone_normalization": "world-phone-identity-v1; exact NFC normalized labels; tone/length/aspiration retained",
        },
        "rule": {
            "unit": "distinct CAL utterance containing a matched phone token in either frozen transfer direction",
            "minimum_distinct_calibration_utterances": 10,
            "minimum_distinct_calibration_speakers": 5,
            "deduplicate_directions_for_utterance_and_speaker_counts": True,
            "tie_break": "Unicode code-point ascending order",
        },
        "selected_phone_labels": selected,
        "selected_phone_count": len(selected),
        "inventory_all_observed_labels": inventory,
    }
    result["inventory_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-qc", type=Path, required=True)
    parser.add_argument("--cal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = create_inventory(args.cal_qc.resolve(strict=True), args.cal_manifest.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "labels": result["selected_phone_count"], "inventory_sha256": result["inventory_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
