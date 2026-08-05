# TTS Artifact Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task.

**Goal:** Add a local, searchable catalog for generated TTS audio without changing the existing `runs/{run_id}/02_tts/` pipeline layout.

**Architecture:** Keep `runs/` as the raw execution workspace and publish selected runs into `data/tts/`. A single standard-library Python CLI owns artifact path generation, manifest creation, catalog updates, lookup, and checksum verification. Existing fixed paths remain readable during migration.

**Tech Stack:** Python 3.10, `argparse`, `dataclasses`, `datetime`, `hashlib`, `json`, `pathlib`, `shutil`, `subprocess`, `wave`, pytest, Markdown/YAML documentation.

---

## File Map

Create these files:

- `scripts/tts_catalog.py` — catalog data model, import operation, lookup commands, and verification.
- `tests/test_tts_catalog.py` — unit and integration tests using temporary directories and synthetic PCM WAV files.
- `data/tts/catalog.json` — tracked catalog index initialized with schema version 1 and no artifacts.
- `data/tts/README.md` — short daily-use guide for finding and importing TTS audio.
- `docs/reference/tts-artifacts.md` — complete artifact layout and manifest reference.

Modify these files:

- `.gitignore` — ignore generated WAV files under `data/tts/**/audio/` while keeping manifests and documentation trackable.
- `docs/reference/pipeline.md` — document the post-download import step and the new local lookup path.
- `CONTEXT.md` — add the `data/tts/` responsibility and the faster Qwen entry point without exceeding the existing 100-line limit.

Do not modify in the first implementation:

- `scripts/02_tts.py` default output path.
- `scripts/03_ditto.py` historical run-path behavior.
- `data/data/` source paths.
- Existing `runs/`, `results/`, and analysis manifests.

## Task 1: Define the Catalog Data Model

**Files:**

- Create: `tests/test_tts_catalog.py`
- Create: `scripts/tts_catalog.py`

- [ ] **Step 1: Write failing path and ID tests**

Add this test setup and test class to `tests/test_tts_catalog.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tts_catalog import (
    ArtifactSpec,
    artifact_id,
    artifact_relpath,
    load_catalog,
    save_catalog,
    upsert_catalog_entry,
)


def test_artifact_path_uses_stable_field_order(tmp_path):
    spec = ArtifactSpec(
        provider="faster_qwen3",
        model="qwen3-tts-12hz-0.6b-base",
        language="zh",
        dataset="aishell1",
        condition="self_clone",
        run_id="r1_faster_qwen3_20260707T104138Z",
    )

    assert artifact_relpath(spec) == Path(
        "faster_qwen3/qwen3-tts-12hz-0.6b-base/zh/aishell1/"
        "self_clone/r1_faster_qwen3_20260707T104138Z"
    )
    assert artifact_id(spec) == (
        "faster_qwen3__qwen3-tts-12hz-0.6b-base__zh__aishell1__"
        "self_clone__r1_faster_qwen3_20260707T104138Z"
    )


def test_legacy_artifact_has_null_run_id():
    spec = ArtifactSpec(
        provider="dashscope_vc",
        model="qwen3-tts-vc-2026-01-22",
        language="en",
        dataset="librispeech",
        condition="self_clone",
        run_id=None,
        legacy_name="audio_en_qwen3_tts",
    )

    assert artifact_relpath(spec).name == "legacy_audio_en_qwen3_tts"
    assert artifact_id(spec).endswith("__legacy_audio_en_qwen3_tts")


@pytest.mark.parametrize("field", ["provider", "model", "language", "dataset", "condition"])
def test_empty_identity_field_is_rejected(field):
    values = {
        "provider": "faster_qwen3",
        "model": "qwen3-tts-12hz-0.6b-base",
        "language": "zh",
        "dataset": "aishell1",
        "condition": "self_clone",
        "run_id": "run-1",
    }
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        ArtifactSpec(**values)


def test_path_separator_in_identity_field_is_rejected():
    with pytest.raises(ValueError, match="model"):
        ArtifactSpec(
            provider="faster_qwen3",
            model="../outside",
            language="zh",
            dataset="aishell1",
            condition="self_clone",
            run_id="run-1",
        )
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: FAIL because `scripts/tts_catalog.py` and its public symbols do not exist yet.

- [ ] **Step 3: Implement the minimal data model**

Define in `scripts/tts_catalog.py`:

```python
@dataclass(frozen=True)
class ArtifactSpec:
    provider: str
    model: str
    language: str
    dataset: str
    condition: str
    run_id: str | None
    legacy_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "model", "language", "dataset", "condition"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"{name} must not be empty")
            if Path(value).name != value or "/" in value or "\\" in value:
                raise ValueError(f"{name} must be a single path segment")
        if not self.run_id and not self.legacy_name:
            raise ValueError("run_id or legacy_name is required")
        if self.run_id and self.legacy_name:
            raise ValueError("run_id and legacy_name are mutually exclusive")


def artifact_relpath(spec: ArtifactSpec) -> Path:
    leaf = spec.run_id or f"legacy_{spec.legacy_name}"
    return Path(spec.provider, spec.model, spec.language, spec.dataset, spec.condition, leaf)


def artifact_id(spec: ArtifactSpec) -> str:
    leaf = spec.run_id or f"legacy_{spec.legacy_name}"
    return "__".join((spec.provider, spec.model, spec.language, spec.dataset, spec.condition, leaf))
```

Use `from __future__ import annotations`, `dataclasses.dataclass`, and `pathlib.Path`. Keep the public functions at module scope so tests and later scripts can use them without invoking the CLI.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: PASS for the three path-validation cases.

## Task 2: Implement Manifest and Catalog Storage

**Files:**

- Modify: `scripts/tts_catalog.py`
- Modify: `tests/test_tts_catalog.py`

- [ ] **Step 1: Write failing JSON storage tests**

Add tests for these exact behaviors:

```python
from tts_catalog import load_catalog, save_catalog, upsert_catalog_entry


def test_missing_catalog_loads_empty_schema(tmp_path):
    path = tmp_path / "data" / "tts" / "catalog.json"
    assert load_catalog(path) == {"schema_version": 1, "artifacts": []}


def test_upsert_replaces_same_artifact_id_without_duplicates(tmp_path):
    path = tmp_path / "catalog.json"
    entry = {
        "artifact_id": "a",
        "path": "a",
        "provider": "faster_qwen3",
        "model": "model",
        "language": "zh",
        "dataset": "aishell1",
        "condition": "self_clone",
        "preferred": False,
    }
    save_catalog(path, {"schema_version": 1, "artifacts": [entry]})

    upsert_catalog_entry(path, {**entry, "preferred": True})
    data = load_catalog(path)

    assert data["artifacts"] == [{"artifact_id": "a", "path": "a", "preferred": True}]


def test_preferred_is_unique_for_same_selection(tmp_path):
    path = tmp_path / "catalog.json"
    first = {
        "artifact_id": "a",
        "path": "a",
        "provider": "faster_qwen3",
        "model": "m",
        "language": "zh",
        "dataset": "aishell1",
        "condition": "self_clone",
        "preferred": True,
    }
    second = {**first, "artifact_id": "b", "path": "b"}
    save_catalog(path, {"schema_version": 1, "artifacts": [first]})

    upsert_catalog_entry(path, second)
    data = load_catalog(path)

    assert [item["preferred"] for item in data["artifacts"]] == [False, True]
```

- [ ] **Step 2: Run the tests and confirm the storage tests fail**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: FAIL with missing catalog storage functions.

- [ ] **Step 3: Implement catalog storage**

Implement these functions in `scripts/tts_catalog.py`:

```python
CATALOG_SCHEMA_VERSION = 1


def load_catalog(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": CATALOG_SCHEMA_VERSION, "artifacts": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported catalog schema_version")
    if not isinstance(data.get("artifacts"), list):
        raise ValueError("catalog artifacts must be a list")
    return data


def save_catalog(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_catalog_entry(path: Path, entry: dict[str, object]) -> None:
    data = load_catalog(path)
    artifacts = [item for item in data["artifacts"] if item["artifact_id"] != entry["artifact_id"]]
    selection = (entry["provider"], entry["model"], entry["language"], entry["dataset"], entry["condition"])
    if entry.get("preferred"):
        for item in artifacts:
            old_selection = tuple(item.get(key) for key in ("provider", "model", "language", "dataset", "condition"))
            if old_selection == selection:
                item["preferred"] = False
    artifacts.append(entry)
    save_catalog(path, {"schema_version": CATALOG_SCHEMA_VERSION, "artifacts": artifacts})
```

Keep catalog writes deterministic: preserve insertion order, use UTF-8, and end the file with a newline.

- [ ] **Step 4: Run the storage tests**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: PASS for path, validation, and catalog-storage tests.

## Task 3: Add Safe Artifact Import and Verification

**Files:**

- Modify: `scripts/tts_catalog.py`
- Modify: `tests/test_tts_catalog.py`

- [ ] **Step 1: Add synthetic WAV and import tests**

Use the standard-library `wave` module in tests so no real model or audio file is required:

```python
import hashlib
import json
import wave

from tts_catalog import import_artifact, verify_artifact


def write_test_wav(path: Path, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


def test_import_copies_wavs_and_writes_manifest_and_catalog(tmp_path):
    source = tmp_path / "runs" / "run-1" / "02_tts"
    write_test_wav(source / "1.wav")
    (source / "tts_meta.json").write_text(json.dumps({
        "provider": "faster_qwen3",
        "language": "Chinese",
        "seed": 42,
        "results": {
            "1": {
                "sample_id": 1,
                "duration_s": 0.1,
                "sample_rate": 16000,
                "text": "测试文本",
                "backend": "faster_qwen3",
                "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            }
        }
    }, ensure_ascii=False), encoding="utf-8")

    artifact = import_artifact(
        source=source,
        data_tts_root=tmp_path / "data" / "tts",
        spec=ArtifactSpec("faster_qwen3", "qwen3-tts-12hz-0.6b-base", "zh", "aishell1", "self_clone", "run-1"),
        repo_root=tmp_path,
        preferred=True,
    )

    audio = artifact / "audio" / "1.wav"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((tmp_path / "data" / "tts" / "catalog.json").read_text(encoding="utf-8"))

    assert audio.exists()
    assert manifest["files"][0]["path"] == "audio/1.wav"
    assert manifest["files"][0]["sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert catalog["artifacts"][0]["preferred"] is True
    assert verify_artifact(artifact, manifest) == []


def test_import_refuses_existing_artifact_without_replace(tmp_path):
    source = tmp_path / "source"
    write_test_wav(source / "1.wav")
    spec = ArtifactSpec("faster_qwen3", "model", "zh", "aishell1", "self_clone", "run-1")
    import_artifact(source, tmp_path / "data" / "tts", spec, tmp_path, False)

    with pytest.raises(FileExistsError):
        import_artifact(source, tmp_path / "data" / "tts", spec, tmp_path, False)


def test_verify_reports_modified_audio(tmp_path):
    source = tmp_path / "source"
    write_test_wav(source / "1.wav")
    spec = ArtifactSpec("faster_qwen3", "model", "zh", "aishell1", "self_clone", "run-1")
    artifact = import_artifact(source, tmp_path / "data" / "tts", spec, tmp_path, False)
    (artifact / "audio" / "1.wav").write_bytes(b"changed")

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    errors = verify_artifact(artifact, manifest)
    assert any("sha256" in error for error in errors)
```

- [ ] **Step 2: Run the tests and confirm import tests fail**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: FAIL because `import_artifact`, `verify_artifact`, and manifest construction do not exist.

- [ ] **Step 3: Implement manifest construction and safe copying**

Add `from datetime import datetime, timezone` and `import subprocess` to the script imports, then implement the following functions in `scripts/tts_catalog.py`:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as handle:
        return {
            "sample_rate_hz": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "frames": handle.getnframes(),
        }


def import_artifact(
    source: Path,
    data_tts_root: Path,
    spec: ArtifactSpec,
    repo_root: Path,
    preferred: bool,
    replace: bool = False,
    source_manifest: Path | None = None,
    config_snapshot: Path | None = None,
) -> Path:
    wav_paths = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix == ".wav"),
        key=lambda path: (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem),
    )
    if not wav_paths:
        raise ValueError(f"no WAV files found in {source}")
    if any(not path.stem.isdigit() for path in wav_paths):
        raise ValueError("all imported WAV files must use numeric sample IDs")

    target = data_tts_root / artifact_relpath(spec)
    if target.exists() and not replace:
        raise FileExistsError(f"artifact already exists: {target}")
    staging = target.parent / f".{target.name}.importing"
    if staging.exists():
        shutil.rmtree(staging)
    audio_dir = staging / "audio"
    audio_dir.mkdir(parents=True)

    metadata = load_source_metadata(source, source_manifest)
    file_entries: list[dict[str, object]] = []
    for source_wav in wav_paths:
        info = inspect_wav(source_wav)
        destination = audio_dir / source_wav.name
        shutil.copy2(source_wav, destination)
        sample_id = int(source_wav.stem)
        sample_meta = metadata.get(str(sample_id), {})
        file_entries.append({
            "sample_id": sample_id,
            "path": str(Path("audio") / source_wav.name),
            "reference_audio": sample_meta.get("reference_audio"),
            "text": sample_meta.get("text"),
            "status": "ok",
            "duration_s": round(info["frames"] / info["sample_rate_hz"], 3),
            "sample_rate_hz": info["sample_rate_hz"],
            "channels": info["channels"],
            "sha256": sha256_file(destination),
        })

    manifest = build_manifest(repo_root, spec, source, file_entries, source_manifest, config_snapshot)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if config_snapshot is not None:
        shutil.copy2(config_snapshot, staging / "config.yaml")

    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)
    entry = catalog_entry(spec, target, repo_root, file_entries, preferred)
    upsert_catalog_entry(data_tts_root / "catalog.json", entry)
    return target


def verify_artifact(artifact_dir: Path, manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    root = artifact_dir.resolve()
    for item in manifest.get("files", []):
        relative = Path(str(item["path"]))
        path = (artifact_dir / relative).resolve()
        if path != root and root not in path.parents:
            errors.append(f"path escapes artifact: {relative}")
            continue
        if not path.exists():
            errors.append(f"missing file: {relative}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != item.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
        try:
            inspect_wav(path)
        except (OSError, EOFError, wave.Error) as exc:
            errors.append(f"invalid WAV {relative}: {exc}")
    return errors


def load_source_metadata(source: Path, source_manifest: Path | None) -> dict[str, dict[str, object]]:
    candidates = []
    if source_manifest is not None:
        candidates.append(source_manifest)
    else:
        candidates.extend((source / "tts_meta.json", source / "manifest.json"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("results", [])
        if isinstance(records, dict):
            records = list(records.values())
        return {
            str(record["sample_id"]): record
            for record in records
            if isinstance(record, dict) and "sample_id" in record
        }
    return {}


def build_manifest(
    repo_root: Path,
    spec: ArtifactSpec,
    source: Path,
    file_entries: list[dict[str, object]],
    source_manifest: Path | None,
    config_snapshot: Path | None,
) -> dict[str, object]:
    try:
        source_run_dir = source.parent.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        source_run_dir = None
    try:
        config_name = config_snapshot.resolve().relative_to(repo_root.resolve()).as_posix() if config_snapshot else None
    except ValueError:
        config_name = config_snapshot.name if config_snapshot else None
    try:
        manifest_name = source_manifest.resolve().relative_to(repo_root.resolve()).as_posix() if source_manifest else None
    except ValueError:
        manifest_name = source_manifest.name if source_manifest else None
    return {
        "schema_version": 1,
        "artifact_type": "tts_audio",
        "artifact_id": artifact_id(spec),
        "run_id": spec.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": spec.dataset,
            "language": spec.language,
            "source_manifest": manifest_name,
        },
        "tts": {"provider": spec.provider, "model": spec.model},
        "provenance": {
            "repo_commit": current_git_commit(repo_root),
            "source_run_dir": source_run_dir,
            "config_snapshot": config_name,
        },
        "files": file_entries,
    }


def current_git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def catalog_entry(
    spec: ArtifactSpec,
    artifact_dir: Path,
    repo_root: Path,
    file_entries: list[dict[str, object]],
    preferred: bool,
) -> dict[str, object]:
    relative_path = artifact_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    return {
        "artifact_id": artifact_id(spec),
        "path": relative_path,
        "provider": spec.provider,
        "model": spec.model,
        "language": spec.language,
        "dataset": spec.dataset,
        "condition": spec.condition,
        "sample_count": len(file_entries),
        "status": "complete",
        "preferred": preferred,
    }
```

The implementation must:

- accept only direct `*.wav` children of `source`;
- reject an empty source directory;
- open every WAV with `wave.open` before copying;
- copy into `artifact_dir/audio/` with `shutil.copy2`;
- preserve numeric filenames such as `1.wav`;
- compute duration as `frames / sample_rate_hz`;
- merge matching per-sample metadata from `tts_meta.json` or an explicit source manifest;
- write `manifest.json` with relative paths and `run_id: null` for legacy imports;
- copy an explicit config file to `artifact_dir/config.yaml` when provided;
- reject an existing target unless `replace=True`;
- update `catalog.json` only after all files and manifest writes succeed;
- return verification errors instead of raising for missing or modified files.

Use a temporary sibling directory during import and rename it into place after all validation succeeds, so an interrupted copy cannot leave a catalog entry pointing to a partial artifact.

- [ ] **Step 4: Add legacy import coverage**

Add a test that imports a directory with `manifest.json` but no `run_id` using:

```python
spec = ArtifactSpec(
    provider="dashscope_vc",
    model="qwen3-tts-vc-2026-01-22",
    language="en",
    dataset="librispeech",
    condition="self_clone",
    run_id=None,
    legacy_name="audio_en_qwen3_tts",
)
```

Assert that the output directory is named `legacy_audio_en_qwen3_tts`, the manifest stores `run_id` as `None`, and the catalog provider remains `dashscope_vc`.

- [ ] **Step 5: Run import and verification tests**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: PASS for normal import, legacy import, collision rejection, checksum generation, and checksum failure detection.

## Task 4: Add CLI Commands

**Files:**

- Modify: `scripts/tts_catalog.py`
- Modify: `tests/test_tts_catalog.py`

- [ ] **Step 1: Write CLI lookup tests**

Add tests that call `main` with explicit argument lists and a temporary `--data-tts-root`, then assert:

```python
def test_list_filters_provider(capsys, tmp_path):
    save_catalog(tmp_path / "catalog.json", {"schema_version": 1, "artifacts": [
        {"artifact_id": "fast", "path": "faster/model/zh/a/self_clone/r1", "provider": "faster_qwen3", "model": "model", "language": "zh", "dataset": "a", "condition": "self_clone", "sample_count": 1, "preferred": True},
        {"artifact_id": "cloud", "path": "cloud/model/en/b/self_clone/r2", "provider": "dashscope_vc", "model": "model", "language": "en", "dataset": "b", "condition": "self_clone", "sample_count": 1, "preferred": False},
    ]})
    main(["list", "--data-tts-root", str(tmp_path), "--provider", "faster_qwen3"])
    captured = capsys.readouterr()
    assert "faster_qwen3" in captured.out
    assert "dashscope_vc" not in captured.out


def test_locate_returns_audio_directory(capsys, tmp_path):
    relative = Path("faster_qwen3/model/zh/aishell1/self_clone/run-1")
    (tmp_path / relative / "audio").mkdir(parents=True)
    save_catalog(tmp_path / "catalog.json", {"schema_version": 1, "artifacts": [
        {"artifact_id": "fast", "path": relative.as_posix(), "provider": "faster_qwen3", "model": "model", "language": "zh", "dataset": "aishell1", "condition": "self_clone", "sample_count": 1, "preferred": True},
    ]})
    main(["locate", "--data-tts-root", str(tmp_path), "--provider", "faster_qwen3", "--dataset", "aishell1"])
    assert capsys.readouterr().out.strip() == str((tmp_path / relative / "audio").resolve())


def test_verify_command_returns_nonzero_for_bad_artifact(tmp_path):
    result = main(["verify", "--data-tts-root", str(tmp_path), "--artifact", "missing"])
    assert result == 1
```

Use separate `captured = capsys.readouterr()` assignments so the test does not consume stdout twice.

- [ ] **Step 2: Run the CLI tests and confirm they fail**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: FAIL because `main` and the argparse subcommands do not exist.

- [ ] **Step 3: Implement the argparse interface**

Implement these commands and options:

```text
python scripts/tts_catalog.py list [--data-tts-root PATH]
    [--provider VALUE] [--model VALUE] [--language VALUE]
    [--dataset VALUE] [--condition VALUE]

python scripts/tts_catalog.py locate [--data-tts-root PATH]
    [--provider VALUE] [--model VALUE] [--language VALUE]
    [--dataset VALUE] [--condition VALUE] [--artifact VALUE]

python scripts/tts_catalog.py import --source PATH
    --provider VALUE --model VALUE --language VALUE
    --dataset VALUE --condition VALUE
    [--run-id VALUE | --legacy]
    [--manifest PATH] [--config PATH] [--preferred] [--replace]
    [--data-tts-root PATH]

python scripts/tts_catalog.py verify --artifact VALUE
    [--data-tts-root PATH]
```

Behavior:

- `list` prints one line per matching artifact with ID, provider, model, language, dataset, sample count, and preferred marker.
- `locate` prints the absolute `audio/` path for exactly one match; it returns exit code 1 for no match and exit code 2 for multiple non-preferred matches.
- `import` requires `--run-id` for normal runs and requires `--legacy` for fixed directories; it never infers a run ID from a directory name.
- `verify` prints every error and returns exit code 1 if any error exists.
- Default `--data-tts-root` is `Path(__file__).resolve().parent.parent / "data" / "tts"`.

Keep CLI output plain text so it works over SSH and in shell scripts. Do not implement a second output format in this plan.

- [ ] **Step 4: Run CLI tests and smoke-test help**

Run:

```bash
pytest tests/test_tts_catalog.py -q
python scripts/tts_catalog.py --help
python scripts/tts_catalog.py list --data-tts-root /tmp/tts-catalog-empty
```

Expected: all tests PASS, help lists `list`, `locate`, `import`, and `verify`, and an empty catalog prints no artifact rows without a traceback.

## Task 5: Initialize the Local Catalog and Migrate Existing Audio

**Files:**

- Create: `data/tts/catalog.json`
- Create: `data/tts/README.md`
- Modify: `.gitignore`
- Test manually: `data/data/audio_en_qwen3_tts/manifest.json`

- [ ] **Step 1: Add the empty tracked catalog**

Create `data/tts/catalog.json` with exactly:

```json
{
  "schema_version": 1,
  "artifacts": []
}
```

- [ ] **Step 2: Add the local quick-reference README**

Document the following commands in `data/tts/README.md`:

```bash
python scripts/tts_catalog.py list --provider faster_qwen3
python scripts/tts_catalog.py locate --provider faster_qwen3 --dataset aishell1 --language zh
python scripts/tts_catalog.py verify --artifact dashscope_vc__qwen3-tts-vc-2026-01-22__en__librispeech__self_clone__legacy_audio_en_qwen3_tts
```

State explicitly that `data/data/audio_en_qwen3_tts/` is a legacy DashScope VC directory, not a faster Qwen directory.

- [ ] **Step 3: Add the audio ignore rule**

Append this rule to `.gitignore` without changing existing unrelated rules:

```gitignore
# Local published TTS audio; manifests remain trackable.
data/tts/**/audio/*.wav
```

- [ ] **Step 4: Import the existing English DashScope artifact**

Run from the repository root:

```bash
python scripts/tts_catalog.py import \
  --source data/data/audio_en_qwen3_tts \
  --manifest data/data/audio_en_qwen3_tts/manifest.json \
  --provider dashscope_vc \
  --model qwen3-tts-vc-2026-01-22 \
  --language en \
  --dataset librispeech \
  --condition self_clone \
  --legacy \
  --preferred
```

Expected: a new artifact under `data/tts/dashscope_vc/qwen3-tts-vc-2026-01-22/en/librispeech/self_clone/legacy_audio_en_qwen3_tts/`, with 13 WAV files and a catalog entry. If the source manifest does not prove `self_clone`, stop and pass the condition explicitly only after confirming it from the experiment record; the importer must not infer it from the directory name.

- [ ] **Step 5: Import available local faster Qwen runs**

For every local faster Qwen run that has been downloaded from the server, use the same command shape with the actual run ID and source metadata:

```bash
RUN_ID=r1_faster_qwen3_20260707T104138Z
python scripts/tts_catalog.py import \
  --source "runs/${RUN_ID}/02_tts" \
  --run-id "${RUN_ID}" \
  --provider faster_qwen3 \
  --model qwen3-tts-12hz-0.6b-base \
  --language zh \
  --dataset aishell1 \
  --condition self_clone \
  --preferred
```

Do not create a faster Qwen entry if the source directory is unavailable locally. The catalog must represent local files, not server-only paths.

- [ ] **Step 6: Verify the catalog and migrated files**

Run:

```bash
python scripts/tts_catalog.py list --provider dashscope_vc
python scripts/tts_catalog.py verify --artifact dashscope_vc__qwen3-tts-vc-2026-01-22__en__librispeech__self_clone__legacy_audio_en_qwen3_tts
pytest tests/test_tts_catalog.py -q
```

Expected: the DashScope artifact is listed, verification returns success, and no old source directory has been deleted or modified.

## Task 6: Document the Artifact Lifecycle

**Files:**

- Create: `docs/reference/tts-artifacts.md`
- Modify: `docs/reference/pipeline.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Write the reference document**

`docs/reference/tts-artifacts.md` must include:

- the distinction between `runs/` and `data/tts/`;
- the full path layout and field naming rules;
- the manifest fields and the no-secrets rule;
- the four CLI commands with real command syntax;
- the legacy import rule;
- the difference between faster Qwen and DashScope VC;
- the rule that existing paths are retained until active consumers are migrated.

- [ ] **Step 2: Update the pipeline reference**

Add a section after `02_tts — TTS 生成` to document:

```text
Remote run template: runs/{run_id}/02_tts/*.wav
Local archive template: data/tts/{provider}/{model}/{language}/{dataset}/{condition}/{run_id}/audio/*.wav
Import command template: python scripts/tts_catalog.py import --source runs/{run_id}/02_tts --run-id {run_id} --provider {provider} --model {model} --language {language} --dataset {dataset} --condition {condition}
Lookup command template: python scripts/tts_catalog.py locate --provider {provider} --model {model} --language {language} --dataset {dataset} --condition {condition}
```

Keep the existing pipeline commands unchanged.

- [ ] **Step 3: Update CONTEXT.md without exceeding 100 lines**

Add `data/tts/` to the documented active data layout and state that it is the local published archive for generated TTS audio. Do not add server passwords, transient run IDs, or detailed command reference to `CONTEXT.md`; keep those in `docs/reference/tts-artifacts.md`.

- [ ] **Step 4: Check documentation paths**

Run:

```bash
rg "data/tts|tts_catalog\.py" docs/reference CONTEXT.md data/tts/README.md
wc -l CONTEXT.md
```

Expected: the new entry points appear in the documentation and `CONTEXT.md` remains below 100 lines.

## Task 7: Add Regression Coverage and Final Verification

**Files:**

- Modify: `tests/test_tts_catalog.py` to add the named invalid-input and lookup-conflict cases.
- No production changes are expected in this task.

- [ ] **Step 1: Test invalid and unsafe inputs**

Add tests for:

- source directory containing no WAV files;
- source directory containing a malformed WAV;
- an absolute path in a manifest being rejected or normalized to a relative path;
- a missing artifact ID during `verify` returning exit code 1;
- two matching non-preferred artifacts causing `locate` to return exit code 2;
- a second preferred artifact clearing the previous preferred flag only within the same provider/model/language/dataset/condition selection.

- [ ] **Step 2: Run the focused suite**

Run:

```bash
pytest tests/test_tts_catalog.py -q
```

Expected: PASS with coverage for path generation, manifest creation, import safety, catalog uniqueness, lookup, and verification.

- [ ] **Step 3: Run the existing full suite**

Run:

```bash
pytest -q
```

Expected: existing tests remain green. If unrelated pre-existing failures occur, record their exact test names and do not alter unrelated files.

- [ ] **Step 4: Run a final repository-local smoke test**

Run:

```bash
python scripts/tts_catalog.py list --provider faster_qwen3
python scripts/tts_catalog.py list --provider dashscope_vc
git diff --check
git status --short
```

Expected: both provider queries run without traceback, `git diff --check` reports no whitespace errors, and only intended catalog/doc/script/test files are changed in addition to pre-existing worktree changes.

## Self-Review Checklist

- Spec coverage: directory separation, manifest, catalog, CLI, legacy import, migration, documentation, Git policy, and verification each have a task.
- Compatibility: `runs/{run_id}/02_tts/` remains the default pipeline output and old data paths are not deleted.
- Provider correctness: the existing English artifact is imported as `dashscope_vc`; faster Qwen requires an explicit source and metadata.
- No silent guessing: normal imports require `--run-id`; fixed directories require `--legacy`.
- No secret leakage: manifests and docs contain no credentials.
- No placeholder implementation work: every CLI command, required argument, failure code, and test command is specified.
- Worktree safety: do not revert or modify unrelated pre-existing changes, and do not commit unless the user explicitly requests a commit.
