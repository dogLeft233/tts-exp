"""Run the frozen complete-final-center ingest and seal its Fit-only receipt.

This entry point adds no selector logic.  It calls the pinned final-center
ingestor, which replays the sealed ask/tell journal, verifies the initial
720-row family and all 40 proposal plus four center point families, and runs
the canonical Torch CPU selector.  The output remains Fit-only evidence; it
does not admit a Dev or Confirm candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_final_center_stagea_ingest_attempt05 as final_ingest,
)


def finalize(*, final_center_index: str | Path,
             output_receipt: str | Path) -> dict[str, object]:
    """Verify the full final-center index and create one selector receipt."""
    index_path = Path(final_center_index)
    output_path = Path(output_receipt)
    if (not index_path.is_absolute() or index_path.is_symlink()
            or not index_path.is_file()):
        raise ValueError("final-center index must be an absolute regular non-symlink file")
    if (not output_path.is_absolute() or output_path.exists()
            or output_path.is_symlink()):
        raise FileExistsError("selector receipt destination must be absent and absolute")
    if (output_path.parent.is_symlink() or not output_path.parent.is_dir()
            or output_path.resolve(strict=False) == index_path.resolve(strict=True)):
        raise ValueError("selector receipt parent must exist and destination must differ from index")

    # The frozen ingestor performs the complete 720-row, source-pin, journal,
    # terminal, NPZ, frame-map, PCM, and selector checks.  Do not accept a
    # caller-provided method ID or theta at this boundary.
    verified = final_ingest.ingest_final_center_family(
        index_path, metric_backend="torch_cpu")
    receipt = final_ingest.write_selection_evidence_exclusive(output_path, verified)
    return {
        "status": verified.final_selection.status,
        "selector_receipt": receipt,
        "selection_status": verified.final_selection.status,
        "selected_method_id": verified.final_selection.selected_method_id,
        "center_specs": len(verified.center_specs),
        "verified_initial_rows": verified.verified_initial_rows,
        "verified_proposal_rows": verified.verified_proposal_rows,
        "verified_center_rows": verified.verified_center_rows,
        "verified_final_rows": verified.verified_final_rows,
        "metric_backend": verified.final_selection.metric_backend,
        "final_dev_or_confirm_admission_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-center-index", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(final_center_index=args.final_center_index,
                          output_receipt=args.output_receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED",
                          "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
