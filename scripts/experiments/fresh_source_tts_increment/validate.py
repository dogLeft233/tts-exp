from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiments.fresh_source_inputs.protocol import (
    ProtocolError,
    load_self_hashed,
    read_json,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_root.resolve() / "run" / "D"
    errors: list[str] = []
    for name in ("tts_inputs.json", "final.json", "validation.json"):
        try:
            load_self_hashed(root / name)
        except (OSError, ProtocolError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    final = read_json(root / "final.json")
    inputs = read_json(root / "tts_inputs.json")
    allowed = {"DEFERRED_MEASUREMENT", "BLOCKED_TTS_ASSET", "DEFERRED_BUDGET"}
    if inputs.get("status") not in allowed or final.get("status") not in allowed:
        errors.append("deferred TTS branch has an invalid terminal status")
    if inputs.get("provider_invoked") is not False or final.get("provider_invoked") is not False:
        errors.append("deferred TTS branch must not invoke a provider")
    if inputs.get("cloud_fallback") is not False:
        errors.append("deferred TTS branch must not use a cloud fallback")
    if final.get("replacement_confirmed") is not False:
        errors.append("deferred TTS branch cannot confirm replacement")
    if final.get("formal_cells") != 0:
        errors.append("deferred TTS branch contains scientific cells")
    result = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors}
    write_json(root / "validation_independent.json", result)
    print(result)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
