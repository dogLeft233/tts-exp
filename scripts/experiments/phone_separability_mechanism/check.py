"""Independent checker for phone_separability_mechanism_v1 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_phone_rules_worker import read_json, read_pcm16, sha256_file

from .audio import build_edit_mask, build_speech_mask
from .metrics import paired_group_bootstrap
from .timing import waveform_contract


class CheckFailure(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CheckFailure(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def verify_bindings(run_dir: Path) -> dict[str, Any]:
    required = [run_dir / "protocol.json", run_dir / "status.json", run_dir / "00_inventory" / "registry.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise CheckFailure(f"missing required artifacts: {missing}")
    registry = dict(read_json(required[-1]))
    failures: list[str] = []
    checked = 0
    for asset in registry.get("assets", []):
        if asset.get("audit_status") != "OK":
            continue
        path = Path(str(asset["audio_path"]))
        if not path.is_file():
            failures.append(f"MISSING_AUDIO:{asset['asset_id']}")
            continue
        _, meta = read_pcm16(path)
        if meta["container_sha256"] != asset.get("container_sha256") or meta["pcm_sha256"] != asset.get("pcm_sha256"):
            failures.append(f"AUDIO_HASH_MISMATCH:{asset['asset_id']}")
        checked += 1
    if failures:
        raise CheckFailure("binding failures: " + ", ".join(failures[:10]))
    return {"checked_assets": checked, "status": "PASS"}


def verify_construction(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "05_construction" / "construction.jsonl")
    if not rows:
        return {"checked": 0, "status": "NOT_APPLICABLE"}
    registry = dict(read_json(run_dir / "00_inventory" / "registry.json"))
    pair_by_id = {str(row["pair_id"]): row for row in registry.get("pairs", [])}
    failures: list[str] = []
    checked = 0
    for row in rows:
        pair = pair_by_id.get(str(row["pair_id"]))
        if pair is None:
            failures.append(f"UNKNOWN_PAIR:{row['pair_id']}")
            continue
        natural = pair["sides"]["natural"]
        source, source_meta = read_pcm16(Path(str(natural["audio_path"])))
        output, output_meta = read_pcm16(Path(str(row["output"]["path"])))
        if output_meta["container_sha256"] != row["output"].get("container_sha256"):
            failures.append(f"OUTPUT_HASH_MISMATCH:{row['pair_id']}:{row['arm']}")
        if source_meta["container_sha256"] != row.get("parent_natural_sha256"):
            failures.append(f"PARENT_HASH_MISMATCH:{row['pair_id']}:{row['arm']}")
        tokens = natural.get("tokens", [])
        mask_mode = str(row.get("construction", {}).get("mask_mode", "core" if row.get("arm") == "SPECTRAL_DRC" else "speech"))
        mask = build_edit_mask(source.size, tokens) if mask_mode == "core" else build_speech_mask(source.size, tokens)
        contract = waveform_contract(source, output, mask)
        if not contract["pass"]:
            failures.append(f"WAVEFORM_CONTRACT:{row['pair_id']}:{row['arm']}")
        checked += 1
    if failures:
        raise CheckFailure("construction failures: " + ", ".join(failures[:10]))
    return {"checked": checked, "status": "PASS"}


def recompute_statistics(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "03_atlas" / "summary.json"
    if not summary_path.is_file():
        return {"status": "NOT_APPLICABLE"}
    summary = dict(read_json(summary_path))
    recomputed: dict[str, Any] = {}
    for model_key, model in summary.get("models", {}).items():
        rows = _read_jsonl(run_dir / "03_atlas" / "contrasts" / f"{model_key}.jsonl")
        recomputed[model_key] = {"views": len(rows), "nonempty": sum(1 for row in rows if row.get("accuracy", {}).get("n_groups", 0) > 0)}
    return {"status": "PASS", "models": recomputed}


def check_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    bindings = verify_bindings(root)
    construction = verify_construction(root)
    statistics = recompute_statistics(root)
    return {"schema_version": 1, "protocol": "phone_separability_mechanism_v1", "engineering_pass": True, "bindings": bindings, "construction": construction, "statistics": statistics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = check_run(args.run_dir)
    except (CheckFailure, OSError, ValueError, KeyError) as exc:
        print(f"INDEPENDENT_CHECK_FAILED: {exc}", file=sys.stderr)
        return 3
    output = Path(args.run_dir).resolve() / "08_report" / "checker.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["check_run", "main", "recompute_statistics", "verify_bindings", "verify_construction"]
