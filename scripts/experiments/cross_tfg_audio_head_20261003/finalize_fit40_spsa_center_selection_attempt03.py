"""Run the frozen complete-final-center ingest and seal its Fit-only receipt.

This entry point adds no selector logic.  It calls the pinned final-center
ingestor, which replays the sealed ask/tell journal, verifies the initial
720-row family and all 40 proposal plus four center point families, and runs
the canonical Torch CPU selector.  The output remains Fit-only evidence; it
does not admit a Dev or Confirm candidate.

Invoke from the checkout root as
``python -m scripts.experiments.cross_tfg_audio_head_20261003.finalize_fit40_spsa_center_selection_attempt02``
and pass ``--repo-root`` explicitly so namespace-package resolution is bound
to the checkout whose frozen sources are pinned below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

FINAL_INGEST_MODULE = (
    "scripts.experiments.cross_tfg_audio_head_20261003."
    "fit_only_final_center_stagea_ingest_attempt07"
)
FINAL_INGEST_PIN = {
    "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_final_center_stagea_ingest_attempt07.py",
    "bytes": 110499,
    "sha256": "fea1467893fc1bb611672895e8490e3681be83504b5fb7b9dbb36cd3d0c33ef3",
}
INGEST_DEPENDENCY_PINS = {
    "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06": {
        "bytes": 66767,
        "sha256": "dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2",
    },
    "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt03": {
        "bytes": 53184,
        "sha256": "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95",
    },
    "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt05": {
        "bytes": 47748,
        "sha256": "cd5e2b80ab0bdd4f599d863468c687724898b2e6baad451a5acfe2123648bdae",
    },
    "scripts.experiments.cross_tfg_audio_head_20261003.fit40_native_point_score_adapter_attempt02": {
        "bytes": 20147,
        "sha256": "b4132a55da20f46338452885ea5ba7dc80ac94d4c6478756ead30cc333bc72c6",
    },
}


class FinalizerError(RuntimeError):
    """The frozen final-ingest source closure is not the pinned closure."""


def _source_bytes(repo_root: Path, relative: str, expected_bytes: int,
                  expected_sha256: str) -> tuple[Path, bytes]:
    relative_path = Path(*relative.split("/"))
    if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
        raise FinalizerError("invalid relative path in frozen source pin")
    path = repo_root.joinpath(relative_path)
    cursor = repo_root
    for component in relative_path.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise FinalizerError(f"pinned source path traverses a symlink: {relative}")
    if not path.is_file():
        raise FinalizerError(f"pinned source is not a regular file: {relative}")
    info = path.stat(follow_symlinks=False)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != info.st_size or (len(raw), digest) != (expected_bytes, expected_sha256):
        raise FinalizerError(f"pinned source bytes/SHA mismatch: {relative}")
    return path.resolve(strict=True), raw


def _verify_module_source(module: object, expected_path: Path, expected_raw: bytes,
                          label: str) -> None:
    module_path_text = getattr(module, "__file__", None)
    if not isinstance(module_path_text, str):
        raise FinalizerError(f"{label} has no filesystem __file__")
    module_path = Path(module_path_text)
    if module_path.is_symlink() or module_path.resolve(strict=True) != expected_path:
        raise FinalizerError(f"{label} loaded from an unexpected __file__")
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or Path(origin).resolve(strict=True) != expected_path:
        raise FinalizerError(f"{label} import spec origin is unexpected")
    actual = module_path.read_bytes()
    if actual != expected_raw:
        raise FinalizerError(f"{label} source changed after the pre-import pin check")


def _check_namespace_package(repo_root: Path, package_name: str,
                             relative_dir: str) -> None:
    package = sys.modules.get(package_name)
    if package is None:
        return
    expected = repo_root.joinpath(*relative_dir.split("/")).resolve(strict=True)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise FinalizerError(f"{package_name} is not the expected namespace package")
    actual = {Path(str(item)).resolve(strict=True) for item in package_path}
    if actual != {expected}:
        raise FinalizerError(f"{package_name} resolves outside the explicit repo root")


def _load_pinned_final_ingest(repo_root: Path):
    """Pin every source file and package origin before importing the frozen ingestor."""
    if not repo_root.is_absolute() or repo_root.is_symlink() or not repo_root.is_dir():
        raise FinalizerError("--repo-root must name an absolute, existing, non-symlink directory")
    repo_root = repo_root.resolve(strict=True)
    expected_packages = (
        ("scripts", "scripts"),
        ("scripts.experiments", "scripts/experiments"),
        ("scripts.experiments.cross_tfg_audio_head_20261003",
         "scripts/experiments/cross_tfg_audio_head_20261003"),
    )
    for package_name, relative_dir in expected_packages:
        package_path = repo_root.joinpath(*relative_dir.split("/"))
        if package_path.is_symlink() or not package_path.is_dir():
            raise FinalizerError(f"pinned package directory is unavailable: {relative_dir}")
        _check_namespace_package(repo_root, package_name, relative_dir)

    pinned: dict[str, tuple[Path, bytes]] = {}
    ingest_path, ingest_raw = _source_bytes(
        repo_root, FINAL_INGEST_PIN["path"],
        FINAL_INGEST_PIN["bytes"], FINAL_INGEST_PIN["sha256"])
    pinned[FINAL_INGEST_MODULE] = (ingest_path, ingest_raw)
    for module_name, pin in INGEST_DEPENDENCY_PINS.items():
        relative = module_name.replace(".", "/") + ".py"
        pinned[module_name] = _source_bytes(
            repo_root, relative, pin["bytes"], pin["sha256"])

    # Fail before imports if an earlier caller populated a different module or
    # left a stale package attribute that a ``from package import module`` can
    # resolve without consulting the requested source root.
    package_name = FINAL_INGEST_MODULE.rpartition(".")[0]
    parent = sys.modules.get(package_name)
    if parent is not None:
        for module_name, (expected_path, expected_raw) in pinned.items():
            short_name = module_name.rpartition(".")[2]
            existing = sys.modules.get(module_name)
            attribute = getattr(parent, short_name, None)
            if existing is not None:
                _verify_module_source(existing, expected_path, expected_raw, module_name)
                if attribute is not None and attribute is not existing:
                    raise FinalizerError(f"package attribute for {module_name} is stale")
            elif attribute is not None:
                raise FinalizerError(f"package attribute for {module_name} is not source-pinned")

    root_text = str(repo_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    import_order = (
        "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06",
        "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt03",
        "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt05",
        "scripts.experiments.cross_tfg_audio_head_20261003.fit40_native_point_score_adapter_attempt02",
        FINAL_INGEST_MODULE,
    )
    loaded = {}
    for module_name in import_order:
        expected_path, expected_raw = pinned[module_name]
        module = importlib.import_module(module_name)
        _verify_module_source(module, expected_path, expected_raw, module_name)
        loaded[module_name] = module

    ingestor = loaded[FINAL_INGEST_MODULE]
    bound_attributes = {
        "selector": import_order[0],
        "initial_ingest": import_order[1],
        "packet_builder": import_order[2],
        "native_point_score_adapter": import_order[3],
    }
    for attribute, module_name in bound_attributes.items():
        dependency = getattr(ingestor, attribute, None)
        if dependency is not loaded[module_name]:
            raise FinalizerError(f"frozen ingestor did not bind the pinned {attribute} module")
        expected_path, expected_raw = pinned[module_name]
        _verify_module_source(dependency, expected_path, expected_raw, f"ingestor.{attribute}")
    return ingestor, repo_root, pinned


def _verify_pinned_sources(pinned: dict[str, tuple[Path, bytes]]) -> None:
    for module_name, (path, expected_raw) in pinned.items():
        module = sys.modules.get(module_name)
        if module is None:
            raise FinalizerError(f"pinned source module disappeared from sys.modules: {module_name}")
        _verify_module_source(module, path, expected_raw, module_name)
        actual = path.read_bytes()
        if actual != expected_raw:
            raise FinalizerError(f"source changed during finalization: {module_name}")


def finalize(*, final_center_index: str | Path,
             output_receipt: str | Path,
             repo_root: str | Path) -> dict[str, object]:
    """Verify the full final-center index and create one selector receipt."""
    source_root = Path(repo_root)
    final_ingest, source_root, pinned = _load_pinned_final_ingest(source_root)
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
    _verify_pinned_sources(pinned)
    receipt = final_ingest.write_selection_evidence_exclusive(output_path, verified)
    _verify_pinned_sources(pinned)
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
        "source_root": str(source_root),
        "final_ingest_module": FINAL_INGEST_MODULE,
        "final_ingest_source_pin": {
            "path": str(pinned[FINAL_INGEST_MODULE][0]),
            "bytes": len(pinned[FINAL_INGEST_MODULE][1]),
            "sha256": hashlib.sha256(pinned[FINAL_INGEST_MODULE][1]).hexdigest(),
        },
        "dependency_source_pins": {
            name: {"path": str(path), "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}
            for name, (path, raw) in pinned.items() if name != FINAL_INGEST_MODULE
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True,
                        help="absolute checkout root containing the frozen pinned source closure")
    parser.add_argument("--final-center-index", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(final_center_index=args.final_center_index,
                          output_receipt=args.output_receipt,
                          repo_root=args.repo_root)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED",
                          "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
