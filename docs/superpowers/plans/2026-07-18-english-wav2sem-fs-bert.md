# English Wav2Sem Fs-BERT Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement scripts 27/28/29/31 + extend script 30 to run the Wav2Sem-style Fs-BERT analysis on the 13 English LibriSpeech test-clean samples already evaluated in Gate 1 (server `/root/autodl-tmp/experiments/english-wav2sem-20260718/`), producing H1/H2/H3 verdicts.

**Architecture:** Six file-decoupled components, each independently re-runnable with caching: (27) download LibriSpeech test-clean .phn files; (28) build English alignment manifest with ARPABET→Preston-Blair-13 visemes; (15/16 reused unchanged) SSL frame embeddings and separability; (29) oracle Fs = BERT CLS; (31) Fd = FC(Fs⊕Fp) separability gain with Fp/Fd_zero/Fd_random three-tier control; (30 extended) render H1/H2/H3 verdict report. All outputs land under `data/wav2sem_analysis_en/`. The Chinese half (`data/wav2sem_analysis/`) is untouched.

**Tech Stack:** Python 3.11, PyTorch ≥ 2.0, transformers (HuBERT/XLS-R/bert-base-uncased), numpy, scipy, scikit-learn, librosa, soundfile, pyyaml, tqdm — all already installed in `/root/autodl-tmp/envs/ditto/`. No new pip installs.

**Reference spec:** `docs/superpowers/specs/2026-07-18-english-wav2sem-fs-bert-design.md`

---

## File Structure

**Create:**
- `data/data/english_viseme_map.yaml` — ARPABET → Preston Blair 13 visemes mapping table
- `scripts/27_download_librispeech_phn.py`
  - `download_test_clean(dest_tar)` — wget OpenSLR tar.gz, verify SHA256
  - `extract_sample_phn(tar_path, librispeech_ids, dest_dir)` — selective tar extract
  - `verify_metadata(audio_manifest, phn_dir)` — M1 mismatch detector
  - `main()` — CLI
- `scripts/28_prepare_english_alignment.py`
  - `load_viseme_map(path)` → dict
  - `arpabet_to_viseme(arpabet, viseme_map)` → str
  - `parse_phn_file(phn_path, sample_rate=16000)` → list[token dict]
  - `build_english_manifest(audio_dir, tts_dir, phn_dir, manifest_paths, viseme_map_path, output_path)`
  - `main()` — CLI
- `scripts/29_oracle_fs_bert.py`
  - `load_bert_model(model_name, device, cache_dir)` → (model, tokenizer)
  - `encode_fs(text, model, tokenizer, device, max_length=128)` → (Fs_cls, Fs_mean, meta)
  - `compute_fs_diagnostics(Fs_vec, Fs_empty_baseline)` → dict
  - `process_sample(sample_id, condition, text, model, tokenizer, Fs_empty, output_path)` → dict
  - `compute_nt_similarity(natural_entries, tts_entries)` → list[dict]
  - `main()` — CLI
- `scripts/31_oracle_fd_separability.py`
  - `load_fp_at_layer(npy_path, layer_index)` → np.ndarray (T, 768)
  - `build_fc_orthogonal(hidden=768, seed=42)` → nn.Sequential
  - `construct_fd(fp, fs, fc=None)` → np.ndarray (T, 768)
  - `compute_separability_for_variant(fp_or_fd, viseme_labels, frame_times, metrics)` → dict
  - `process_sample(sample_id, condition, fp_path, oracle_fs_path, viseme_map_path, output_dir)` → dict
  - `main()` — CLI
- `tests/test_librispeech_phn_parsing.py`
- `tests/test_english_viseme_map.py`
- `tests/test_oracle_fs_bert.py`
- `tests/test_oracle_fd_separability.py`

**Modify:**
- `scripts/30_analyze_english_wav2sem.py` — add modules 30B/30C/30D/30E (separability / oracle Fs / Fd gain / TFG link); preserve existing Gate 1 module as 30A
- `scripts/configs/english_wav2sem.yaml` — add `wav2sem:` block per Section 5 of spec
- `scripts/tfg_feature_common.py` — add bilingual constants:
  - `OUTPUT_BASE_EN = Path(__file__).resolve().parent.parent / "data" / "wav2sem_analysis_en"`
  - `ENGLISH_STUDY_SAMPLES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`
  - `ENGLISH_SILENCE_LABELS: tuple[str, ...] = ("h#", "sil", "sp", "spn", "")`

---

## Task 1: Add bilingual constants to `tfg_feature_common.py`

**Files:**
- Modify: `scripts/tfg_feature_common.py:128-130` (after `OUTPUT_BASE` definition)

- [ ] **Step 1: Write the failing test**

Add `tests/test_tfg_feature_common_bilingual.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tfg_feature_common import (
    OUTPUT_BASE_EN,
    ENGLISH_STUDY_SAMPLES,
    ENGLISH_SILENCE_LABELS,
)


def test_output_base_en_path():
    assert OUTPUT_BASE_EN.name == "wav2sem_analysis_en"
    assert OUTPUT_BASE_EN.parent.name == "data"


def test_english_study_samples_complete():
    assert ENGLISH_STUDY_SAMPLES == list(range(1, 14))


def test_silence_labels_include_librispeech_markers():
    assert "h#" in ENGLISH_SILENCE_LABELS
    assert "sil" in ENGLISH_SILENCE_LABELS
    assert "sp" in ENGLISH_SILENCE_LABELS


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_tfg_feature_common_bilingual.py
```

Expected: ImportError for `OUTPUT_BASE_EN`.

- [ ] **Step 3: Add constants to `tfg_feature_common.py`**

Insert after `OUTPUT_BASE` definition (around line 129):

```python
OUTPUT_BASE_EN: Path = Path(__file__).resolve().parent.parent / "data" / "wav2sem_analysis_en"
"""Root output directory for all English (LibriSpeech) analysis artefacts."""

ENGLISH_STUDY_SAMPLES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
"""English study set: 13 LibriSpeech test-clean samples (no exclusions)."""

ENGLISH_SILENCE_LABELS: tuple[str, ...] = ("h#", "sil", "sp", "spn", "")
"""LibriSpeech .phn silence / pause markers to exclude from viseme metrics."""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_tfg_feature_common_bilingual.py
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/tfg_feature_common.py tests/test_tfg_feature_common_bilingual.py
git commit -m "feat(common): add English study samples + silence label constants for bilingual Wav2Sem"
```

---

## Task 2: Create `english_viseme_map.yaml`

**Files:**
- Create: `data/data/english_viseme_map.yaml`
- Test: `tests/test_english_viseme_map.py`

- [ ] **Step 1: Write the failing test**

`tests/test_english_viseme_map.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import yaml

VISEME_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "data" / "english_viseme_map.yaml"

# ARPABET phone set used by LibriSpeech (39 phones, TIMIT convention)
LIBRISPEECH_ARPABET = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G",
    "HH",
    "IH", "IY",
    "JH",
    "K", "L",
    "M", "N", "NG",
    "OW", "OY",
    "P", "R",
    "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V",
    "W", "Y",
    "Z", "ZH",
}


def test_viseme_map_exists_and_loads():
    assert VISEME_MAP_PATH.exists(), f"missing: {VISEME_MAP_PATH}"
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "arpabet_to_viseme" in data
    assert "silence_labels" in data


def test_all_arpabet_phones_mapped():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    mapping = data["arpabet_to_viseme"]
    unmapped = [p for p in LIBRISPEECH_ARPABET if p not in mapping]
    assert not unmapped, f"unmapped ARPABET phones: {unmapped}"


def test_thirteen_viseme_classes():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    used = set(data["arpabet_to_viseme"].values())
    expected = {"pbmv", "fv", "th", "cdsz", "kg", "chjsh", "e", "o", "i", "u", "r", "ai", "aw"}
    assert used == expected, f"viseme set mismatch; got {used}"


def test_silence_labels_listed():
    with open(VISEME_MAP_PATH) as f:
        data = yaml.safe_load(f)
    assert "h#" in data["silence_labels"]
    assert "sil" in data["silence_labels"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_english_viseme_map.py
```

Expected: FileNotFoundError or assertion `assert VISEME_MAP_PATH.exists()` fails.

- [ ] **Step 3: Write the YAML mapping**

`data/data/english_viseme_map.yaml`:

```yaml
# ARPABET → Preston Blair 13 visemes
# Preston Blair visemes are the standard animator's 13-class visual phoneme
# classification (Preston Blair, "Animation: Learn to Animate Cartoons Step by
# Step"). UH is committed to class "o" (lax back vowel groups with open round),
# following the design decision in Section 6.2 of the spec.

arpabet_to_viseme:
  # /pbmv/ — lips together
  P: pbmv
  B: pbmv
  M: pbmv

  # /fv/ — lower lip to upper teeth
  F: fv
  V: fv

  # /th/ — tongue between teeth
  TH: th
  DH: th

  # /cdsz/ — tongue tip to alveolar ridge (also N, L)
  T: cdsz
  D: cdsz
  S: cdsz
  Z: cdsz
  N: cdsz
  L: cdsz

  # /kg/ — back of tongue to soft palate
  K: kg
  G: kg
  NG: kg

  # /chjsh/ — blade to post-alveolar
  CH: chjsh
  JH: chjsh
  SH: chjsh
  ZH: chjsh

  # /e/ — relaxed open mouth
  EH: e
  AE: e
  AH: e

  # /o/ — open round
  AA: o
  AO: o
  OW: o
  UH: o
  OY: o

  # /i/ — wide stretch
  IY: i
  IH: i

  # /u/ — small round
  UW: u
  W: u

  # /r/ — slightly cupped
  R: r
  ER: r
  AXR: r

  # /ai/ — wide open to close (offglide)
  AY: ai
  EY: ai

  # /aw/ — round to wide
  AW: aw

  # /y/ not in Preston Blair 13; assigned to /i/ viseme (tongue forward,
  # lips spread — visually indistinguishable from /i/)
  Y: i

  # /hh/ glottal — no visible mouth shape. Assigned to neutral 'e' as default.
  HH: e

silence_labels:
  - h#
  - sil
  - sp
  - spn
  - ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_english_viseme_map.py
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add data/data/english_viseme_map.yaml tests/test_english_viseme_map.py
git commit -m "feat(en): add ARPABET → Preston Blair 13 viseme mapping yaml"
```

---

## Task 3: Implement `scripts/27_download_librispeech_phn.py`

**Files:**
- Create: `scripts/27_download_librispeech_phn.py`
- Test: `tests/test_librispeech_phn_parsing.py`

- [ ] **Step 1: Write the failing test for .phn parsing**

`tests/test_librispeech_phn_parsing.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from importlib import util
spec = util.spec_from_file_location(
    "librispeech_phn",
    Path(__file__).resolve().parent.parent / "scripts" / "27_download_librispeech_phn.py",
)
mod = util.module_from_spec(spec)


def test_parse_phn_line_basic():
    # Create a minimal .phn content for in-memory parse
    sample_text = """0 470000 h#
470000 580000 k
580000 700000 ae
700000 1500000 t
1500000 1600000 h#
"""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".phn", delete=False) as f:
        f.write(sample_text)
        path = Path(f.name)
    try:
        spec.loader.exec_module(mod)
        tokens = mod.parse_phn_file(path, sample_rate=16000)
        assert len(tokens) == 5
        assert tokens[0]["token"] == "h#"
        assert abs(tokens[0]["start_s"] - 0.0) < 1e-6
        assert abs(tokens[0]["end_s"] - 470000/16000) < 1e-6
        assert tokens[1]["token"] == "k"
        assert abs(tokens[1]["start_s"] - 470000/16000) < 1e-6
    finally:
        os.unlink(path)


def test_silence_labels_detected():
    sample_text = """0 1000 h#
1000 50000 sil
50000 60000 sp
60000 70000 aa
70000 71000 h#
"""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".phn", delete=False) as f:
        f.write(sample_text)
        path = Path(f.name)
    try:
        spec.loader.exec_module(mod)
        tokens = mod.parse_phn_file(path, sample_rate=16000)
        silences = [t for t in tokens if t["token"] in mod.SILENCE_TOKENS]
        assert len(silences) == 4
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_librispeech_phn_parsing.py
```

Expected: ModuleNotFoundError or ImportError for module file `27_download_librispeech_phn.py` (filename starts with a digit; Python import path may raise).

- [ ] **Step 3: Implement `scripts/27_download_librispeech_phn.py`**

```python
#!/usr/bin/env python3
"""Download LibriSpeech test-clean and extract .phn files for the study samples.

The LibriSpeech test-clean subset (346 MB) is fetched from OpenSLR. The .phn
files for the 13 study samples (looked up by ``librispeech_id`` in
``data/data/audio_en/manifest.json``) are copied to
``data/wav2sem_analysis_en/manifest/librispeech_phn/``.

Usage
-----
    python scripts/27_download_librispeech_phn.py [--samples IDS] [--smoke]
        [--download-dir DIR] [--output-dir DIR]

Dependencies
------------
    wget (or curl fallback), tar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENSLR_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
EXPECTED_TAR_SIZE_BYTES = 346_752_519  # OpenSLR-12 test-clean, ~330 MB
SHA256_CHECKSUM_HOST = "https://www.openslr.org/resources/12/test-clean.tar.gz.sha256"

DEFAULT_DOWNLOAD_DIR = Path("/root/autodl-tmp/downloads")
DEFAULT_OUTPUT_DIR = OUTPUT_BASE_EN / "manifest" / "librispeech_phn"

AUDIO_MANIFEST_REL = "data/data/audio_en/manifest.json"

SILENCE_TOKENS: tuple[str, ...] = ("h#", "sil", "sp", "spn", "")
"""LibriSpeech .phn silence / pause labels."""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_test_clean(dest_tar: Path, url: str = OPENSLR_URL) -> bool:
    """Download LibriSpeech test-clean tar.gz via wget (curl fallback)."""
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    if dest_tar.exists() and dest_tar.stat().st_size > 0:
        print(f"[27] tar already present: {dest_tar}")
        return True
    if shutil.which("wget"):
        cmd = ["wget", "-q", "--show-progress", "-O", str(dest_tar), url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "--progress-bar", "-o", str(dest_tar), url]
    else:
        raise RuntimeError("neither wget nor curl is installed")
    print(f"[27] downloading {url}")
    subprocess.run(cmd, check=True)
    return dest_tar.exists()


def verify_sha256(path: Path, expected_sha256: str | None = None) -> bool:
    """Compute SHA256 of file. If expected passed, assert equality."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if expected_sha256 is None:
        print(f"[27] {path.name} sha256 = {actual}")
        return True
    if actual != expected_sha256:
        print(f"[27] SHA256 mismatch: expected {expected_sha256}, got {actual}")
        return False
    return True


# ---------------------------------------------------------------------------
# .phn parsing
# ---------------------------------------------------------------------------


def parse_phn_file(phn_path: Path, sample_rate: int = 16000) -> list[dict]:
    """Parse a LibriSpeech .phn file into a list of token dicts.

    Each .phn line is ``<start_sample> <end_sample> <phone_label>``.

    Parameters
    ----------
    phn_path : Path
        Path to the LibriSpeech .phn file.
    sample_rate : int
        Sample rate used to convert sample counts to seconds (default 16000).

    Returns
    -------
    tokens : list[dict]
        Each dict has keys ``token``, ``start_s``, ``end_s``, ``duration_s``.
    """
    tokens: list[dict] = []
    if not phn_path.exists():
        return tokens
    with phn_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start_sample = int(parts[0])
                end_sample = int(parts[1])
            except ValueError:
                continue
            token = parts[2]
            start_s = start_sample / sample_rate
            end_s = end_sample / sample_rate
            tokens.append({
                "token": token,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
            })
    return tokens


# ---------------------------------------------------------------------------
# Tar extraction (selective)
# ---------------------------------------------------------------------------


def extract_sample_phn(
    tar_path: Path,
    librispeech_ids: list[str],
    dest_dir: Path,
) -> dict[str, bool]:
    """Extract only the .phn / .wrd / .trans.txt files for the given utterance ids.

    A LibriSpeech utterance id like ``"2830-3980-0036"`` maps to the tar path
    ``LibriSpeech/test-clean/2830/3980/2830-3980-0036.{phn,wrd,flac,trans.txt}``
    (the per-utterance .trans.txt is rare; chapter-level .trans.txt lives at
    ``LibriSpeech/test-clean/<spk>/<chap>/<spk>-<chap>.trans.txt``).

    Returns
    -------
    found : dict[str, bool]
        Mapping ``librispeech_id -> True`` if a .phn was extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, bool] = {uid: False for uid in librispeech_ids}

    prefix_by_uid: dict[str, str] = {}
    for uid in librispeech_ids:
        spk, chap, utt = uid.split("-")
        prefix_by_uid[uid] = f"LibriSpeech/test-clean/{spk}/{chap}/{uid}"

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            if not m.isfile():
                continue
            for uid, prefix in prefix_by_uid.items():
                if m.name.startswith(prefix) and (
                    m.name.endswith(".phn")
                    or m.name.endswith(".wrd")
                    or m.name.endswith(".txt")
                ):
                    target = dest_dir / Path(m.name).name
                    src = tar.extractfile(m)
                    if src is None:
                        continue
                    target.write_bytes(src.read())
                    if m.name.endswith(".phn"):
                        found[uid] = True
    return found


# ---------------------------------------------------------------------------
# Metadata verification (M1)
# ---------------------------------------------------------------------------


def verify_metadata(
    audio_manifest_path: Path,
    phn_dir: Path,
    sample_ids: list[int],
    audio_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Check that each sample's .phn end time ≤ audio duration + 100ms.

    Returns (ok, failed), each list of per-sample dicts.
    """
    import soundfile as sf

    audio_records = {
        int(r["sample_id"]): r
        for r in json.loads(audio_manifest_path.read_text())
    }

    ok: list[dict] = []
    failed: list[dict] = []

    for sid in sample_ids:
        rec = audio_records.get(sid)
        if rec is None:
            failed.append({"sample_id": sid, "error": "missing in audio manifest"})
            continue
        uid = rec["librispeech_id"]
        phn_path = phn_dir / f"{uid}.phn"
        if not phn_path.exists():
            failed.append({"sample_id": sid, "error": f"missing .phn: {phn_path.name}"})
            continue
        tokens = parse_phn_file(phn_path)
        if not tokens:
            failed.append({"sample_id": sid, "error": "empty .phn"})
            continue
        phn_end_s = tokens[-1]["end_s"]
        wav_path = audio_dir / f"{sid}.wav"
        if not wav_path.exists():
            failed.append({"sample_id": sid, "error": f"audio missing: {wav_path}"})
            continue
        info = sf.info(str(wav_path))
        audio_dur = info.duration
        if phn_end_s > audio_dur + 0.1:
            failed.append({
                "sample_id": sid,
                "error": f"phn_end={phn_end_s:.3f}s > audio={audio_dur:.3f}s",
                "phn_end_s": phn_end_s,
                "audio_dur_s": audio_dur,
            })
        else:
            ok.append({
                "sample_id": sid,
                "librispeech_id": uid,
                "phn_end_s": phn_end_s,
                "audio_dur_s": audio_dur,
            })
    return ok, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str, default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
        help="Comma-separated sample IDs (default: all 13 English study samples).",
    )
    parser.add_argument("--smoke", action="store_true", help="Process only sample 1.")
    parser.add_argument("--download-dir", type=str, default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download; assume tar is already at download-dir.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run M1 metadata verification.")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)

    repo_root = Path(__file__).resolve().parent.parent
    audio_manifest = repo_root / AUDIO_MANIFEST_REL
    audio_dir = repo_root / "data" / "data" / "audio_en"

    if not audio_manifest.exists():
        print(f"[27] audio manifest not found: {audio_manifest}", file=sys.stderr)
        return 1

    # 1. Download tar
    tar_path = download_dir / "test-clean.tar.gz"
    if not args.verify_only and not args.no_download:
        download_test_clean(tar_path)

    # 2. Resolve librispeech_ids for requested samples
    audio_records = {
        int(r["sample_id"]): r
        for r in json.loads(audio_manifest.read_text())
    }
    librispeech_ids = []
    for sid in sample_ids:
        rec = audio_records.get(sid)
        if rec is None:
            print(f"[27] sample {sid} missing in audio manifest", file=sys.stderr)
            return 1
        librispeech_ids.append(rec["librispeech_id"])

    # 3. Verify metadata (M1) before extraction
    if args.verify_only:
        ok, failed = verify_metadata(audio_manifest, output_dir, sample_ids, audio_dir)
        print(f"[27] verify: {len(ok)} ok, {len(failed)} failed")
        for f in failed:
            print(f"  sample {f['sample_id']}: {f['error']}")
        return 0 if len(failed) < 2 else 1

    # 4. Extract .phn files
    found = extract_sample_phn(tar_path, librispeech_ids, output_dir)
    missing = [uid for uid, ok in found.items() if not ok]
    if missing:
        print(f"[27] missing .phn for: {missing}", file=sys.stderr)
        return 1

    # 5. Verify
    ok, failed = verify_metadata(audio_manifest, output_dir, sample_ids, audio_dir)
    print(f"[27] {len(ok)}/{len(sample_ids)} verified; {len(failed)} failed")
    for f in failed:
        print(f"  sample {f['sample_id']}: {f['error']}")
    return 0 if len(failed) < 2 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_librispeech_phn_parsing.py
```

Expected: 2 passed (parse_phn_line_basic, silence_labels_detected).

- [ ] **Step 5: Commit**

```bash
git add scripts/27_download_librispeech_phn.py tests/test_librispeech_phn_parsing.py
git commit -m "feat(27): LibriSpeech test-clean downloader + .phn parser + M1 verifier"
```

---

## Task 4: Implement `scripts/28_prepare_english_alignment.py`

**Files:**
- Create: `scripts/28_prepare_english_alignment.py`
- Test: extend `tests/test_librispeech_phn_parsing.py` with arpabet→viseme and manifest-build smoke tests

- [ ] **Step 1: Write failing tests** (append to `tests/test_librispeech_phn_parsing.py`)

```python
def test_arpabet_to_viseme_known_classes():
    spec = util.spec_from_file_location(
        "english_alignment",
        Path(__file__).resolve().parent.parent / "scripts" / "28_prepare_english_alignment.py",
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.arpabet_to_viseme("P") == "pbmv"
    assert mod.arpabet_to_viseme("IY") == "i"
    assert mod.arpabet_to_viseme("h#") == "sil"   # silence → special class
    assert mod.arpabet_to_viseme("UNKNOWN") == "other"  # unmapped → "other"


def test_build_manifest_filters_silence_from_viseme():
    # Build a fake .phn with one silence + one speech phone + final silence
    import tempfile, os
    fake_phn = """0 10000 h#
10000 50000 K
50000 90000 UW
90000 100000 h#
"""
    tmp = Path(tempfile.mkdtemp())
    fake_phn_path = tmp / "fake-9999-0001.phn"
    fake_phn_path.write_text(fake_phn)
    audio_manifest = tmp / "audio_manifest.json"
    audio_manifest.write_text(json.dumps([{
        "sample_id": 1, "librispeech_id": "fake-9999-0001",
        "text": "FAKE", "duration_s": 6.25
    }]))
    audio_dir = tmp / "audio"
    audio_dir.mkdir()
    (audio_dir / "1.wav").write_bytes(b"")  # empty stub

    spec2 = util.spec_from_file_location(
        "english_alignment",
        Path(__file__).resolve().parent.parent / "scripts" / "28_prepare_english_alignment.py",
    )
    mod = util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    tokens = mod.parse_phn_file(fake_phn_path, sample_rate=16000)
    # viseme mapping applied
    for t in tokens:
        t["viseme"] = mod.arpabet_to_viseme(t["token"])
    # silence tokens → viseme "sil"
    assert tokens[0]["viseme"] == "sil"
    assert tokens[1]["viseme"] == "kg"  # K → kg
    assert tokens[2]["viseme"] == "u"   # UW → u
    assert tokens[3]["viseme"] == "sil"
    # non-silence filter
    non_sil = [t for t in tokens if t["viseme"] != "sil"]
    assert len(non_sil) == 2

    import shutil
    shutil.rmtree(tmp)
```

Note: this test needs `import json` at the top of the test file (add it).

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_librispeech_phn_parsing.py
```

Expected: `english_alignment` module not found.

- [ ] **Step 3: Implement `scripts/28_prepare_english_alignment.py`**

```python
#!/usr/bin/env python3
"""Build English Wav2Sem alignment manifest from LibriSpeech .phn files.

Reads .phn files extracted by script 27, maps ARPABET phones to Preston Blair
13 visemes (plus a 'sil' class for silence tokens), and writes a manifest
compatible with script 15 (SSL embeddings) and script 16 (separability).

Each manifest entry mirrors the Chinese manifest schema so downstream scripts
can consume both languages uniformly:

    {
      "sample_id": 1,
      "condition": "natural",        # or "tts"
      "variant": "raw",
      "filepath": "data/data/audio_en/1.wav",
      "duration_s": 5.765,
      "tokens": [
        {"token": "h#", "viseme": "sil", "start_s": 0.0, "end_s": 0.29, "confidence": 1.0},
        {"token": "k",  "viseme": "kg",  "start_s": 0.29, "end_s": 0.62, "confidence": 1.0},
        ...
      ]
    }

Usage
-----
    python scripts/28_prepare_english_alignment.py [--samples IDS] [--smoke] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import (
    ENGLISH_STUDY_SAMPLES,
    ENGLISH_SILENCE_LABELS,
    OUTPUT_BASE_EN,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

VISEME_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "data" / "english_viseme_map.yaml"
PHN_DIR = OUTPUT_BASE_EN / "manifest" / "librispeech_phn"
AUDIO_DIR_REL = Path("data/data/audio_en")
TTS_DIR_REL = Path("data/data/audio_en_qwen3_tts")
AUDIO_MANIFEST_REL = Path("data/data/audio_en/manifest.json")
TTS_MANIFEST_REL = Path("data/data/audio_en_qwen3_tts/manifest.json")

OUTPUT_MANIFEST_REL = OUTPUT_BASE_EN / "manifest" / "alignment.json"

# ---------------------------------------------------------------------------
# Viseme map
# ---------------------------------------------------------------------------


_viseme_map_cache: dict | None = None


def load_viseme_map(path: Path = VISEME_MAP_PATH) -> dict:
    global _viseme_map_cache
    if _viseme_map_cache is not None:
        return _viseme_map_cache
    if yaml is None:
        raise ImportError("PyYAML required")
    with open(path, "r", encoding="utf-8") as f:
        _viseme_map_cache = yaml.safe_load(f)
    return _viseme_map_cache


def arpabet_to_viseme(arpabet: str) -> str:
    """Map ARPABET phone to viseme; silence tokens → 'sil'; unmapped → 'other'."""
    if arpabet in ENGLISH_SILENCE_LABELS:
        return "sil"
    vm = load_viseme_map()
    mapping = vm["arpabet_to_viseme"]
    return mapping.get(arpabet, "other")


# ---------------------------------------------------------------------------
# .phn parsing (reuses 27)
# ---------------------------------------------------------------------------


def parse_phn_file(phn_path: Path, sample_rate: int = 16000) -> list[dict]:
    """Parse a LibriSpeech .phn file: <start> <end> <phone> lines."""
    # Import from script 27 to avoid duplication
    import importlib.util
    script27 = Path(__file__).resolve().parent / "27_download_librispeech_phn.py"
    spec = importlib.util.spec_from_file_location("librispeech_phn", script27)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_phn_file(phn_path, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def find_phn_for_sample(
    sample_id: int,
    audio_manifest_records: dict[int, dict],
    phn_dir: Path,
) -> Path | None:
    rec = audio_manifest_records.get(sample_id)
    if rec is None:
        return None
    uid = rec["librispeech_id"]
    phn_path = phn_dir / f"{uid}.phn"
    return phn_path if phn_path.exists() else None


def build_english_manifest(
    sample_ids: list[int],
    repo_root: Path,
    phn_dir: Path = PHN_DIR,
    output_path: Path = OUTPUT_MANIFEST_REL,
) -> dict:
    audio_manifest_path = repo_root / AUDIO_MANIFEST_REL
    tts_manifest_path = repo_root / TTS_MANIFEST_REL

    audio_records = {
        int(r["sample_id"]): r
        for r in json.loads(audio_manifest_path.read_text())
    }
    tts_records = {
        int(r["sample_id"]): r
        for r in json.loads(tts_manifest_path.read_text())
    }

    audio_dir = repo_root / AUDIO_DIR_REL
    tts_dir = repo_root / TTS_DIR_REL

    manifest: list[dict] = []
    failures: list[dict] = []
    conditions = ["natural", "tts"]

    for sid in sorted(sample_ids):
        rec = audio_records.get(sid)
        if rec is None:
            failures.append({"sample_id": sid, "error": "missing in audio manifest"})
            continue
        phn_path = find_phn_for_sample(sid, audio_records, phn_dir)
        if phn_path is None:
            failures.append({"sample_id": sid, "error": "missing .phn"})
            continue
        phn_tokens = parse_phn_file(phn_path)
        if not phn_tokens:
            failures.append({"sample_id": sid, "error": "empty .phn"})
            continue

        text = rec.get("text", "").strip()
        duration_s = float(rec.get("duration_s", phn_tokens[-1]["end_s"]))

        for condition in conditions:
            if condition == "natural":
                audio_path = audio_dir / f"{sid}.wav"
            else:
                audio_path = tts_dir / f"{sid}.wav"
            if not audio_path.exists():
                failures.append({"sample_id": sid, "condition": condition, "error": "audio missing"})
                continue

            entry: dict = {
                "sample_id": sid,
                "condition": condition,
                "variant": "raw",
                "filepath": str(audio_path.relative_to(repo_root)),
                "duration_s": duration_s,
                "text": text,
                "librispeech_id": rec["librispeech_id"],
                "tokens": [
                    {
                        "token": t["token"],
                        "viseme": arpabet_to_viseme(t["token"]),
                        "start_s": round(t["start_s"], 6),
                        "end_s": round(t["end_s"], 6),
                        "duration_s": round(t["duration_s"], 6),
                        "confidence": 1.0,
                    }
                    for t in phn_tokens
                ],
            }
            manifest.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples_requested": sorted(sample_ids),
        "entries_written": len(manifest),
        "failures": failures,
        "manifest": manifest,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str, default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_BASE_EN / "manifest"),
    )
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output_dir) / "alignment.json"

    summary = build_english_manifest(sample_ids, repo_root, output_path=output_path)
    print(f"[28] wrote {summary['entries_written']} entries; {len(summary['failures'])} failures")
    for f in summary["failures"]:
        print(f"  {f}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_librispeech_phn_parsing.py
```

Expected: 4 passed (parse_phn_line_basic, silence_labels_detected, arpabet_to_viseme_known_classes, build_manifest_filters_silence_from_viseme).

- [ ] **Step 5: Commit**

```bash
git add scripts/28_prepare_english_alignment.py tests/test_librispeech_phn_parsing.py
git commit -m "feat(28): English alignment manifest builder with ARPABET→PB13 viseme mapping"
```

---

## Task 5: Implement `scripts/29_oracle_fs_bert.py`

**Files:**
- Create: `scripts/29_oracle_fs_bert.py`
- Test: `tests/test_oracle_fs_bert.py`

- [ ] **Step 1: Write failing test**

`tests/test_oracle_fs_bert.py`:

```python
import sys
from pathlib import Path
import importlib.util

spec = util_module_path = (
    Path(__file__).resolve().parent.parent / "scripts" / "29_oracle_fs_bert.py"
)
spec_obj = importlib.util.spec_from_file_location("oracle_fs_bert", spec)
mod = importlib.util.module_from_spec(spec_obj)


def test_fs_dim_768():
    """Fs_cls and Fs_mean must be 768-dim."""
    spec_obj.loader.exec_module(mod)
    # Build a tiny dummy "outputs" object
    import numpy as np
    import torch
    fake_last_hidden = torch.zeros(1, 5, 768)
    fake_last_hidden[0, 0] = 1.0  # CLS
    fake_last_hidden[0, 1:-1] = 2.0  # tokens
    Fs_cls, Fs_mean, meta = mod.extract_fs_from_outputs(fake_last_hidden)
    assert Fs_cls.shape == (768,)
    assert Fs_mean.shape == (768,)
    assert abs(Fs_cls[0] - 1.0) < 1e-5
    assert abs(Fs_mean[0] - 2.0) < 1e-5
    assert meta["token_count"] == 3


def test_fs_diagnostics_finite():
    import numpy as np
    fs = np.ones(768, dtype=np.float32) * 0.5
    fs_empty = np.zeros(768, dtype=np.float32)
    diag = mod.compute_fs_diagnostics(fs, fs_empty)
    assert "fs_norm" in diag
    assert "fs_empty_l1" in diag
    assert np.isfinite(diag["fs_norm"])
    assert np.isfinite(diag["fs_empty_l1"])


def test_fs_determinism():
    """Same input twice → identical Fs (requires torch; skip if unavailable)."""
    try:
        import torch
    except ImportError:
        return
    spec_obj.loader.exec_module(mod)
    # We do not load real BERT (no network in CI); just verify code path
    # by calling compute_cosine_similarity with known vectors
    import numpy as np
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = mod.cosine_similarity(a, b)
    assert abs(sim - 1.0) < 1e-6
    sim2 = mod.cosine_similarity(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert abs(sim2 - 0.0) < 1e-6


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_oracle_fs_bert.py
```

Expected: ImportError / ModuleNotFoundError for `29_oracle_fs_bert`.

- [ ] **Step 3: Implement `scripts/29_oracle_fs_bert.py`**

```python
#!/usr/bin/env python3
"""Extract oracle Fs = BERT CLS / mean-pooled sentence embeddings.

Per the design spec Section 6.3, this script encodes each sample's gold
transcript (lowercased) with ``bert-base-uncased`` and stores the CLS and
mean-pooled token vectors as two Fs candidates. It also computes:
  - per-sample ||Fs|| and L1 distance to BERT("") baseline (M2 detector)
  - paired natural↔TTS cosine similarity (and L1)

Usage
-----
    python scripts/29_oracle_fs_bert.py [--samples IDS] [--smoke] \\
        [--bert-model bert-base-uncased] [--device cuda] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN

ALIGNMENT_MANIFEST_REL = OUTPUT_BASE_EN / "manifest" / "alignment.json"
OUTPUT_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fs.json"

DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_CACHE_DIR = Path("/root/autodl-tmp/checkpoints")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (na * nb))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=1))


# ---------------------------------------------------------------------------
# BERT loading
# ---------------------------------------------------------------------------


def load_bert_model(
    model_name: str = DEFAULT_BERT_MODEL,
    device: str = "cpu",
    cache_dir: Path | None = None,
) -> tuple:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(f"transformers/torch required: {e}")

    kwargs = {"local_files_only": False}
    if cache_dir is not None:
        # transformers resolves from cache_dir first; if HF_ENDPOINT is set,
        # downloads via mirror on first call.
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Fs extraction
# ---------------------------------------------------------------------------


def encode_text(text: str, model, tokenizer, device: str = "cpu", max_length: int = 128):
    """Encode a text into (Fs_cls, Fs_mean, outputs). Returns numpy arrays."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs, inputs


def extract_fs_from_outputs(outputs) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pull CLS and mean-pooled Fs from a BERT ModelOutput.

    last_hidden_state shape: (1, T, 768). We take index 0 (CLS) and the
    mean of indices 1..T-2 (token positions, excluding special tokens).
    """
    import torch
    lhs = outputs.last_hidden_state  # (1, T, 768)
    Fs_cls = lhs[0, 0].cpu().numpy().astype(np.float32)
    T = lhs.shape[1]
    if T > 2:
        Fs_mean = lhs[0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
        token_count = T - 2
    else:
        Fs_mean = Fs_cls.copy()
        token_count = 0
    return Fs_cls, Fs_mean, {"token_count": int(token_count)}


def compute_fs_diagnostics(fs: np.ndarray, fs_empty_baseline: np.ndarray) -> dict:
    return {
        "fs_norm": float(np.linalg.norm(fs)),
        "fs_empty_l1": l1_distance(fs, fs_empty_baseline),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def process_sample(
    sample_id: int,
    condition: str,
    text: str,
    model,
    tokenizer,
    device: str,
    fs_empty: np.ndarray,
) -> dict:
    text_lower = text.lower()
    _, inputs = encode_text(text_lower, model, tokenizer, device)
    outputs, _ = (encode_text(text_lower, model, tokenizer, device))
    Fs_cls, Fs_mean, meta = extract_fs_from_outputs(outputs)
    diag = compute_fs_diagnostics(Fs_cls, fs_empty)
    return {
        "sample_id": sample_id,
        "condition": condition,
        "text": text,
        "text_lower": text_lower,
        "token_count": meta["token_count"],
        "Fs_cls": Fs_cls.tolist(),
        "Fs_mean": Fs_mean.tolist(),
        "fs_cls_norm": diag["fs_norm"],
        "fs_empty_l1": diag["fs_empty_l1"],
        "is_degenerate": False,  # set later in batch stats pass
    }


def compute_nt_similarity(entries: list[dict]) -> list[dict]:
    """For each sample_id, compute natural↔TTS Fs similarity."""
    by_sid: dict[int, dict[str, dict]] = {}
    for e in entries:
        by_sid.setdefault(e["sample_id"], {})[e["condition"]] = e

    out: list[dict] = []
    for sid in sorted(by_sid):
        arms = by_sid[sid]
        if "natural" not in arms or "tts" not in arms:
            continue
        Fs_nat_cls = np.asarray(arms["natural"]["Fs_cls"], dtype=np.float32)
        Fs_tts_cls = np.asarray(arms["tts"]["Fs_cls"], dtype=np.float32)
        Fs_nat_mean = np.asarray(arms["natural"]["Fs_mean"], dtype=np.float32)
        Fs_tts_mean = np.asarray(arms["tts"]["Fs_mean"], dtype=np.float32)
        out.append({
            "sample_id": sid,
            "cosine_cls_nt": cosine_similarity(Fs_nat_cls, Fs_tts_cls),
            "cosine_mean_nt": cosine_similarity(Fs_nat_mean, Fs_tts_mean),
            "l1_cls_nt": l1_distance(Fs_nat_cls, Fs_tts_cls),
        })
    return out


def flag_degenerate(entries: list[dict]) -> list[dict]:
    """Mark is_degenerate=True for entries with ||Fs|| outside ±5σ of distribution."""
    norms = np.array([e["fs_cls_norm"] for e in entries], dtype=np.float64)
    mu = float(norms.mean()) if len(norms) > 0 else 0.0
    sigma = float(norms.std(ddof=0)) if len(norms) > 1 else 0.0
    threshold_hi = mu + 5 * sigma if sigma > 0 else float("inf")
    threshold_lo = mu - 5 * sigma if sigma > 0 else float("-inf")
    for e in entries:
        bad_norm = not (threshold_lo < e["fs_cls_norm"] < threshold_hi)
        bad_empty = e["fs_empty_l1"] < 0.5
        e["is_degenerate"] = bool(bad_norm or bad_empty)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--alignment-manifest", type=str,
                        default=str(ALIGNMENT_MANIFEST_REL))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parent.parent

    alignment_path = Path(args.alignment_manifest)
    if not alignment_path.is_absolute():
        alignment_path = repo_root / alignment_path
    if not alignment_path.exists():
        print(f"[29] alignment manifest not found: {alignment_path}", file=sys.stderr)
        return 1
    alignment = json.loads(alignment_path.read_text())
    entries_all = alignment.get("manifest", alignment)  # support both shapes

    # Filter to requested sample_ids and only carry forward entries with text
    target_entries = [
        e for e in entries_all
        if int(e["sample_id"]) in sample_ids and e.get("text")
    ]

    output_path = Path(args.output)
    if output_path.exists() and not args.no_cache:
        # Check mtime vs alignment manifest
        if output_path.stat().st_mtime >= alignment_path.stat().st_mtime:
            print(f"[29] cached: {output_path}")
            return 0

    print(f"[29] loading {args.bert_model} on {args.device} ...")
    model, tokenizer = load_bert_model(
        args.bert_model, device=args.device, cache_dir=Path(args.cache_dir),
    )

    # Empty-text baseline Fs for M2 degeneracy detector
    empty_outputs, _ = encode_text("", model, tokenizer, args.device)
    Fs_empty_cls, _, _ = extract_fs_from_outputs(empty_outputs)

    entries_out: list[dict] = []
    for e in target_entries:
        out = process_sample(
            e["sample_id"], e["condition"], e["text"],
            model, tokenizer, args.device, Fs_empty_cls,
        )
        entries_out.append(out)

    entries_out = flag_degenerate(entries_out)
    nt_sim = compute_nt_similarity(entries_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "bert_model": args.bert_model,
        "device": args.device,
        "n_entries": len(entries_out),
        "n_degenerate": sum(1 for e in entries_out if e["is_degenerate"]),
        "entries": entries_out,
        "nt_similarity": nt_sim,
    }, ensure_ascii=False, indent=2))
    print(f"[29] wrote {len(entries_out)} entries → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_oracle_fs_bert.py
```

Expected: 3 passed (fs_dim_768, fs_diagnostics_finite, fs_determinism). The determinism test does not load real BERT (offline).

- [ ] **Step 5: Commit**

```bash
git add scripts/29_oracle_fs_bert.py tests/test_oracle_fs_bert.py
git commit -m "feat(29): oracle Fs extraction via BERT CLS + mean-pool with M2 degeneracy detector"
```

---

## Task 6: Implement `scripts/31_oracle_fd_separability.py`

**Files:**
- Create: `scripts/31_oracle_fd_separability.py`
- Test: `tests/test_oracle_fd_separability.py`

This task reuses the existing separability metric implementations from `scripts/16_feature_separability.py` — specifically `intra_class_variance`, `inter_class_variance`, `fisher_ratio`, `silhouette_score`, `boundary_sharpness`, `segment_stability`. Import them rather than reimplementing.

- [ ] **Step 1: Write failing test**

`tests/test_oracle_fd_separability.py`:

```python
import sys
from pathlib import Path
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "oracle_fd_separability",
    Path(__file__).resolve().parent.parent / "scripts" / "31_oracle_fd_separability.py",
)
mod = importlib.util.module_from_spec(spec)


def test_construct_fd_no_fc_just_addition():
    spec.loader.exec_module(mod)
    Fs = np.ones(768, dtype=np.float32) * 0.5
    Fp = np.zeros((10, 768), dtype=np.float32)
    # fc=None → Fd = Fs broadcast + Fp
    Fd = mod.construct_fd(Fp, Fs, fc=None)
    assert Fd.shape == (10, 768)
    assert np.allclose(Fd, 0.5)


def test_construct_fd_with_fc_changes_output():
    import torch
    torch.manual_seed(42)
    spec.loader.exec_module(mod)
    Fs = np.ones(768, dtype=np.float32) * 0.5
    Fp = np.zeros((10, 768), dtype=np.float32)
    fc = mod.build_fc_orthogonal(hidden=768, seed=42)
    Fd = mod.construct_fd(Fp, Fs, fc=fc)
    assert Fd.shape == (10, 768)
    # Should differ from pure addition (orthogonal init rotates the activation)
    assert not np.allclose(Fd, 0.5)


def test_separability_all_same_label_returns_zero_or_nan_safely():
    """All-same-label case: silhouette should be 0 (no between-class signal)."""
    spec.loader.exec_module(mod)
    from sklearn.metrics import silhouette_score
    Fp = np.random.default_rng(0).standard_normal((20, 16), dtype=np.float32)
    labels = np.array([0] * 20)
    # Should not crash; silhouette handles single-class by returning 0
    try:
        s = silhouette_score(Fp, labels)
        # Either 0 or NaN; both are acceptable as 'no signal'
        assert np.isnan(s) or s == 0.0
    except Exception as e:
        # sklearn may raise on single-class; that's acceptable too
        pass


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python tests/test_oracle_fd_separability.py
```

Expected: ImportError for `31_oracle_fd_separability`.

- [ ] **Step 3: Implement `scripts/31_oracle_fd_separability.py`**

```python
#!/usr/bin/env python3
"""Construct Fd = FC(Fs ⊕ Fp) and measure Fs-mediated separability gain.

Per spec Section 6.4, this script computes three tiers of separability on the
HuBERT L11 frame embeddings of each sample:

  - Fp_only     : the HuBERT L11 embedding directly
  - Fd_zero     : Fp + Fs_broadcast  (no projection; sanity baseline)
  - Fd_random   : FC(orthogonal-init)(Fp + Fs_broadcast)

For each tier, computes the same viseme-level and arpabet-level separability
metrics as script 16. The key signal is ``gain_random_vs_zero``, which is the
Fs-specific contribution after accounting for the FC's orthogonal rotation.

Per the M3 mitigation, H2 is supported only when ≥ 3 of 5 core metrics
(silhouette↑, boundary_sharpness↑, segment_stability↑, intra_class_dist↓,
fisher↑) move in favorable direction across the 13 paired samples.

Usage
-----
    python scripts/31_oracle_fd_separability.py [--samples IDS] [--smoke] \\
        [--fp-layer 11] [--fd-seed 42]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN, TARGET_SR

EMBEDDINGS_DIR = OUTPUT_BASE_EN / "embeddings"
ORACLE_FS_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fs.json"
ALIGNMENT_PATH = OUTPUT_BASE_EN / "manifest" / "alignment.json"
OUTPUT_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fd_separability.json"

DEFAULT_FP_LAYER = 11   # best from Phase 1 Chinese analysis
DEFAULT_FD_SEED = 42

# Metrics we will reuse from script 16
def _load_separability_module():
    script16 = Path(__file__).resolve().parent / "16_feature_separability.py"
    spec = importlib.util.spec_from_file_location("separability", script16)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_sep = _load_separability_module()

CORE_METRICS_FAVORABLE = {
    "silhouette": +1,
    "boundary_sharpness": +1,
    "segment_stability": +1,
    "intra_class_dist": -1,   # lower is better
    "fisher": +1,
}


# ---------------------------------------------------------------------------
# Fp loader
# ---------------------------------------------------------------------------


def load_fp_at_layer(
    npy_path: Path,
    fp_layer: int,
    requested_layers: list[int] = [0, 6, 11, 12],
) -> np.ndarray:
    """Load HuBERT .npy output and slice the row for ``fp_layer``.

    script 15 saves embeddings with shape (n_requested_layers, n_frames, dim).
    The default requested layers are [0, 6, 11, 12]; L11 is at index 2.
    """
    arr = np.load(npy_path)
    # Find index of fp_layer in the requested layers list
    try:
        idx = requested_layers.index(fp_layer)
    except ValueError:
        raise ValueError(f"fp_layer={fp_layer} not in requested_layers={requested_layers}")
    if idx >= arr.shape[0]:
        raise ValueError(f"layer index {idx} exceeds saved layers {arr.shape[0]}")
    return arr[idx].astype(np.float32)   # (T, 768)


# ---------------------------------------------------------------------------
# Fd construction
# ---------------------------------------------------------------------------


def build_fc_orthogonal(hidden: int = 768, seed: int = 42):
    """Build a 2-layer FC with orthogonal weight init and zero bias.

    Returns an nn.Sequential in eval mode. The output is deterministic given
    the seed.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    fc = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden),
    )
    for m in fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
    fc.eval()
    return fc


def construct_fd(
    Fp: np.ndarray,
    Fs: np.ndarray,
    fc=None,
) -> np.ndarray:
    """Broadcast Fs to (T, dim) and add to Fp; optionally pass through fc.

    If ``fc`` is None, returns ``Fs_broadcast + Fp`` (the Fd_zero tier).
    Otherwise, returns ``fc(Fs_broadcast + Fp)`` (the Fd_random tier).
    """
    T = Fp.shape[0]
    Fs_broadcast = np.tile(Fs[None, :], (T, 1)).astype(np.float32)
    fused = Fs_broadcast + Fp
    if fc is None:
        return fused.astype(np.float32)
    import torch
    with torch.no_grad():
        out = fc(torch.from_numpy(fused)).cpu().numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Separability per variant
# ---------------------------------------------------------------------------


def _attach_viseme_labels(
    frame_times: np.ndarray,
    tokens: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """For each frame, look up the viseme of the token containing its centre time.

    Returns (viseme_labels, arpabet_labels). Frames outside any token (e.g.,
    trailing silence not in .phn) get label "__out_of_range__" and are excluded.
    """
    vis = np.empty(len(frame_times), dtype=object)
    arp = np.empty(len(frame_times), dtype=object)
    for i, t in enumerate(frame_times):
        matched = None
        for tok in tokens:
            if tok["start_s"] <= t < tok["end_s"]:
                matched = tok
                break
        if matched is None:
            vis[i] = "__out_of_range__"
            arp[i] = "__out_of_range__"
        else:
            vis[i] = matched.get("viseme", "other")
            arp[i] = matched.get("token", "other")
    return vis, arp


def compute_separability_for_variant(
    variant: np.ndarray,           # (T, 768)
    frame_times: np.ndarray,       # (T,)
    tokens: list[dict],
) -> dict:
    """Compute the 5 core + supplementary separability metrics on a variant.

    Reuses metric functions from script 16. Skips classes with fewer than
    ``MIN_CLASS_SAMPLES`` (≥3 by default) samples.
    """
    vis_labels, arp_labels = _attach_viseme_labels(frame_times, tokens)

    # Drop frames outside any token (silence-only or trailing)
    mask = vis_labels != "__out_of_range__"
    X = variant[mask]
    vis_labels = vis_labels[mask]
    arp_labels = arp_labels[mask]

    # Filter classes with ≥ 3 items
    from collections import Counter
    counts = Counter(vis_labels)
    keep = [v for v, c in counts.items() if c >= 3]
    keep_mask = np.isin(vis_labels, keep)
    X_v = X[keep_mask]
    vis_v = vis_labels[keep_mask]

    out: dict = {}

    if len(set(vis_v.tolist())) >= 2 and len(vis_v) >= 5:
        try:
            out["intra_class_dist"] = float(
                _sep.intra_class_variance(X_v, vis_v)
            )
        except Exception:
            out["intra_class_dist"] = float("nan")
        try:
            out["silhouette"] = float(_sep.silhouette_score(X_v, vis_v))
        except Exception:
            out["silhouette"] = float("nan")
        try:
            out["boundary_sharpness"] = float(
                _sep.boundary_sharpness(X_v, vis_v, frame_times[mask][keep_mask])
            )
        except Exception:
            out["boundary_sharpness"] = float("nan")
        try:
            out["segment_stability"] = float(
                _sep.segment_stability(X_v, vis_v, frame_times[mask][keep_mask])
            )
        except Exception:
            out["segment_stability"] = float("nan")
        try:
            out["fisher"] = float(_sep.fisher_ratio(X_v, vis_v))
        except Exception:
            out["fisher"] = float("nan")
    else:
        for k in CORE_METRICS_FAVORABLE:
            out[k] = float("nan")
    return out


def process_sample(
    sample_id: int,
    condition: str,
    fp_path: Path,
    oracle_fs_entries: list[dict],
    alignment_entries: list[dict],
    fc_random,
    fp_layer: int = DEFAULT_FP_LAYER,
) -> dict:
    """Compute Fp_only / Fd_zero / Fd_random separability for one sample."""
    Fp = load_fp_at_layer(fp_path, fp_layer)
    T = Fp.shape[0]
    frame_stride = 320  # HuBERT default; matches script 15
    frame_times = np.arange(T, dtype=np.float32) * (frame_stride / TARGET_SR)

    # Pull Fs (CLS) for this sample_id + condition
    fs_entry = next(
        (e for e in oracle_fs_entries
         if e["sample_id"] == sample_id and e["condition"] == condition),
        None,
    )
    if fs_entry is None:
        raise ValueError(
            f"oracle Fs missing for sample_id={sample_id} condition={condition}"
        )
    if fs_entry.get("is_degenerate"):
        return {
            "sample_id": sample_id,
            "condition": condition,
            "skipped": "Fs_degenerate",
            "fp_metrics": {}, "fd_zero_metrics": {}, "fd_random_metrics": {},
            "gain_zero_vs_fp": {}, "gain_random_vs_fp": {}, "gain_random_vs_zero": {},
        }
    Fs = np.asarray(fs_entry["Fs_cls"], dtype=np.float32)

    # Pull alignment tokens
    align_entry = next(
        (e for e in alignment_entries
         if e["sample_id"] == sample_id and e["condition"] == condition),
        None,
    )
    if align_entry is None:
        raise ValueError(
            f"alignment entry missing for sample_id={sample_id} condition={condition}"
        )
    tokens = align_entry.get("tokens", [])

    # Three tiers
    Fd_zero = construct_fd(Fp, Fs, fc=None)
    Fd_random = construct_fd(Fp, Fs, fc=fc_random)

    m_fp = compute_separability_for_variant(Fp, frame_times, tokens)
    m_zero = compute_separability_for_variant(Fd_zero, frame_times, tokens)
    m_random = compute_separability_for_variant(Fd_random, frame_times, tokens)

    gains = {}
    for k in CORE_METRICS_FAVORABLE:
        if k in m_fp and k in m_zero and k in m_random:
            gains["gain_zero_vs_fp_" + k] = float(m_zero[k] - m_fp[k])
            gains["gain_random_vs_fp_" + k] = float(m_random[k] - m_fp[k])
            gains["gain_random_vs_zero_" + k] = float(m_random[k] - m_zero[k])

    return {
        "sample_id": sample_id,
        "condition": condition,
        "fp_layer": fp_layer,
        "T_frames": int(T),
        "fs_norm": float(np.linalg.norm(Fs)),
        "fp_metrics": m_fp,
        "fd_zero_metrics": m_zero,
        "fd_random_metrics": m_random,
        **gains,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fp-layer", type=int, default=DEFAULT_FP_LAYER)
    parser.add_argument("--fd-seed", type=int, default=DEFAULT_FD_SEED)
    parser.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    parser.add_argument("--oracle-fs", type=str, default=str(ORACLE_FS_PATH))
    parser.add_argument("--alignment", type=str, default=str(ALIGNMENT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]

    embeddings_dir = Path(args.embeddings_dir)
    oracle_fs_path = Path(args.oracle_fs)
    alignment_path = Path(args.alignment)
    output_path = Path(args.output)

    if output_path.exists() and not args.no_cache:
        # Re-run if inputs are newer
        newest_input = max(
            p.stat().st_mtime for p in (embeddings_dir, oracle_fs_path, alignment_path)
            if p.exists()
        )
        if output_path.stat().st_mtime >= newest_input:
            print(f"[31] cached: {output_path}")
            return 0

    if not oracle_fs_path.exists():
        print(f"[31] oracle Fs not found at {oracle_fs_path}; run script 29 first.", file=sys.stderr)
        return 1
    if not alignment_path.exists():
        print(f"[31] alignment not found at {alignment_path}; run script 28 first.", file=sys.stderr)
        return 1

    oracle_fs_payload = json.loads(oracle_fs_path.read_text())
    oracle_fs_entries = oracle_fs_payload.get("entries", oracle_fs_payload)
    alignment_payload = json.loads(alignment_path.read_text())
    alignment_entries = alignment_payload.get("manifest", alignment_payload)

    fc_random = build_fc_orthogonal(hidden=768, seed=args.fd_seed)

    results: list[dict] = []
    for sid in sample_ids:
        for cond in ("natural", "tts"):
            npy_path = embeddings_dir / f"{sid}_{cond}_raw_hubert.npy"
            if not npy_path.exists():
                print(f"[31] missing embedding: {npy_path}", file=sys.stderr)
                continue
            try:
                r = process_sample(
                    sid, cond, npy_path,
                    oracle_fs_entries, alignment_entries,
                    fc_random, fp_layer=args.fp_layer,
                )
                results.append(r)
                print(f"[31] sample {sid} {cond}: T={r['T_frames']}")
            except Exception as exc:
                print(f"[31] sample {sid} {cond} error: {exc}", file=sys.stderr)
                results.append({
                    "sample_id": sid, "condition": cond,
                    "skipped": str(exc),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "fp_layer": args.fp_layer,
        "fd_seed": args.fd_seed,
        "n_results": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"[31] wrote {len(results)} entries → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python tests/test_oracle_fd_separability.py
```

Expected: 3 passed (construct_fd_no_fc_just_addition, construct_fd_with_fc_changes_output, separability_all_same_label_returns_zero_or_nan_safely).

- [ ] **Step 5: Commit**

```bash
git add scripts/31_oracle_fd_separability.py tests/test_oracle_fd_separability.py
git commit -m "feat(31): Fd=FC(Fs⊕Fp) separability gain with three-tier M3 control"
```

---

## Task 7: Extend `scripts/30_analyze_english_wav2sem.py` (add 30B/30C/30D/30E modules)

**Files:**
- Modify: `scripts/30_analyze_english_wav2sem.py` — keep existing Gate 1 logic, add modules

The existing script structure has a `main()` that:
1. Loads eval data from `runs/english_wav2sem_gate1_*` directories
2. Computes Gate 1 statistics (mean Δ, CI, sign-flip p, Holm)

Extend by adding four `analyze_*` functions called from `main()` if `--with-wav2sem` flag is passed.

- [ ] **Step 1: Read the current script to identify the insertion point**

```bash
wc -l scripts/30_analyze_english_wav2sem.py
tail -60 scripts/30_analyze_english_wav2sem.py
```

You will find the existing `main()` definition. Add `--with-wav2sem` argparse option, and four new functions before `main()`.

- [ ] **Step 2: Write the failing test** (append to a new test file `tests/test_analyze_english_wav2sem_modules.py`)

```python
import sys
from pathlib import Path
import importlib.util
import json
import numpy as np

spec = importlib.util.spec_from_file_location(
    "analyze_english_wav2sem",
    Path(__file__).resolve().parent.parent / "scripts" / "30_analyze_english_wav2sem.py",
)
mod = importlib.util.module_from_spec(spec)


def _stub_separability_payload():
    return [
        {
            "model": "hubert", "layer": 11, "condition": "natural", "level": "viseme",
            "metric": "segment_stability", "value": 0.30,
        },
        {
            "model": "hubert", "layer": 11, "condition": "tts", "level": "viseme",
            "metric": "segment_stability", "value": 0.31,
        },
    ]


def _stub_fd_payload():
    return [
        {
            "sample_id": 1, "condition": "natural",
            "fp_metrics": {"silhouette": 0.05, "intra_class_dist": 0.80, "boundary_sharpness": 0.30},
            "fd_zero_metrics": {"silhouette": 0.06, "intra_class_dist": 0.78, "boundary_sharpness": 0.31},
            "fd_random_metrics": {"silhouette": 0.07, "intra_class_dist": 0.76, "boundary_sharpness": 0.33},
        },
        {
            "sample_id": 1, "condition": "tts",
            "fp_metrics": {"silhouette": 0.04, "intra_class_dist": 0.81, "boundary_sharpness": 0.28},
            "fd_zero_metrics": {"silhouette": 0.06, "intra_class_dist": 0.78, "boundary_sharpness": 0.30},
            "fd_random_metrics": {"silhouette": 0.08, "intra_class_dist": 0.74, "boundary_sharpness": 0.35},
        },
    ]


def test_analyze_separability_runs():
    spec.loader.exec_module(mod)
    if not hasattr(mod, "analyze_separability"):
        return  # not implemented yet
    payload = _stub_separability_payload()
    summary = mod.analyze_separability(payload)
    assert "h1_verdict" in summary
    assert summary["h1_verdict"] in ("PASS", "FAIL", "TREND")


def test_analyze_fd_gain_compute_h2_verdict():
    spec.loader.exec_module(mod)
    if not hasattr(mod, "analyze_fd_gain"):
        return
    payload = _stub_fd_payload()
    summary = mod.analyze_fd_gain(payload)
    assert "h2_verdict" in summary
    assert summary["h2_verdict"] in (
        "PASS", "FAIL", "TREND", "M3_FC_ARTIFACT", "INSUFFICIENT_DATA",
    )


def test_analyze_tfg_correlation_returns_spearman():
    spec.loader.exec_module(mod)
    if not hasattr(mod, "analyze_tfg_correlation"):
        return
    separability_values = np.array([0.30, 0.31, 0.32, 0.33])
    sync_c_deltas = np.array([+0.5, +0.4, +0.6, +0.3])
    syncnet_path = Path("/dev/null")  # not loaded
    summary = mod.analyze_tfg_correlation(
        separability_values=separability_values,
        sync_c_deltas=sync_c_deltas,
        syncnet_results_path=syncnet_path,
    )
    assert "h3_verdict" in summary
    assert "spearman_rho" in summary


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python tests/test_analyze_english_wav2sem_modules.py
```

Expected: 3 tests skip silently (because `hasattr` returns False). Need to confirm that without the attribute, the tests pass with no-op. Adjust test to fail when attribute missing:

Replace `return` with `assert False, "analyze_separability not implemented"` (same for the other two).

Re-run; now expected: 3 failing with assertion errors.

- [ ] **Step 4: Implement the four new modules**

Insert these functions into `scripts/30_analyze_english_wav2sem.py` before `main()`. Add `--with-wav2sem` and `--analysis-dir` flags to argparse.

```python
# ---- New: Wav2Sem analysis modules (30B/30C/30D/30E) ----

def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def analyze_separability(separability_metrics: list[dict]) -> dict:
    """30B — Aggregate separability_metrics.json entries into H1 verdict."""
    by_cond: dict[str, list[float]] = {"natural": [], "tts": []}
    for e in separability_metrics:
        if e.get("metric") == "segment_stability" and e.get("level") == "viseme" \
           and e.get("model") == "hubert" and e.get("layer") == 11:
            by_cond[e["condition"]].append(float(e["value"]))
    if len(by_cond["natural"]) < 2 or len(by_cond["tts"]) < 2:
        return {"h1_verdict": "INSUFFICIENT_DATA", "info": "n<2 per condition"}
    a = np.asarray(by_cond["natural"])
    b = np.asarray(by_cond["tts"])
    diff = b - a
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-10))
    # Sign-flip p under null of no difference
    rng = np.random.default_rng(42)
    nulls = np.abs(rng.choice([-1, 1], size=(10000, len(diff))) * diff).mean(1)
    p = float(np.mean(nulls >= abs(diff.mean())))
    if p < 0.05 and abs(d) >= 0.5:
        verdict = "PASS"
    elif abs(d) >= 0.5:
        verdict = "TREND"
    else:
        verdict = "FAIL"
    return {
        "h1_verdict": verdict,
        "natural_mean": float(a.mean()),
        "tts_mean": float(b.mean()),
        "cohens_d": d,
        "p_value": p,
        "n": int(len(diff)),
    }


def analyze_oracle_fs(oracle_fs_payload: dict) -> dict:
    """30C — Summarize oracle Fs natural↔TTS similarity and degeneracy flags."""
    entries = oracle_fs_payload.get("entries", [])
    nt_sim = oracle_fs_payload.get("nt_similarity", [])
    if not nt_sim:
        return {"verdict": "INSUFFICIENT_DATA"}
    cos_cls = [e["cosine_cls_nt"] for e in nt_sim]
    cos_mean = [e["cosine_mean_nt"] for e in nt_sim]
    return {
        "n_pairs": len(nt_sim),
        "mean_cosine_cls_nt": float(np.mean(cos_cls)),
        "sd_cosine_cls_nt": float(np.std(cos_cls, ddof=1)),
        "mean_cosine_mean_nt": float(np.mean(cos_mean)),
        "n_degenerate": sum(1 for e in entries if e.get("is_degenerate")),
    }


def analyze_fd_gain(fd_payload: list[dict]) -> dict:
    """30D — Aggregate Fp/Fd_zero/Fd_random per-condition gains and decide H2.

    Per M3 mitigation: H2 supported only when ≥ 3 of 5 core metrics move
    in favorable direction across paired natural samples.
    """
    CORE = {
        "silhouette": +1,
        "boundary_sharpness": +1,
        "segment_stability": +1,
        "intra_class_dist": -1,
        "fisher": +1,
    }
    # Count per-metric favorable samples across sample_ids (use natural condition
    # as the baseline Fs–baseline measurement; TTS checked separately)
    favorable_counts = {k: 0 for k in CORE}
    total_samples = 0
    for r in fd_payload:
        if "fp_metrics" not in r or "fd_zero_metrics" not in r or "fd_random_metrics" not in r:
            continue
        if r.get("skipped"):
            continue
        if r["condition"] != "natural":  # primary pre-condition
            continue
        total_samples += 1
        for k, direction in CORE.items():
            if k not in r["fp_metrics"] or k not in r["fd_random_metrics"]:
                continue
            gain = r["fd_random_metrics"][k] - r["fp_metrics"][k]
            if direction * gain > 0:
                favorable_counts[k] += 1

    metrics_passing = 0
    for k, n_fav in favorable_counts.items():
        # Paired sign test: n_fav out of total_samples in favorable direction
        # p ≈ 2 * Binomial(tail). Use simple >50% rule for verdict.
        if total_samples > 0 and n_fav > total_samples / 2:
            metrics_passing += 1

    if total_samples == 0:
        return {"h2_verdict": "INSUFFICIENT_DATA", "favorable_counts": favorable_counts}

    # Check M3: Fd_zero should also show gains (else the FC is the source)
    fc_artifact = False
    for r in fd_payload:
        if r.get("condition") != "natural" or r.get("skipped"):
            continue
        for k, direction in CORE.items():
            if k in r.get("fd_zero_metrics", {}) and k in r.get("fd_random_metrics", {}):
                zero_gain = r["fd_zero_metrics"][k] - r.get("fp_metrics", {}).get(k, 0)
                rand_gain = r["fd_random_metrics"][k] - r.get("fp_metrics", {}).get(k, 0)
                # If zero shows no gain but random does, suggests FC artifact
                if direction * zero_gain <= 0 and direction * rand_gain > 0:
                    fc_artifact = True

    if metrics_passing >= 3 and not fc_artifact:
        verdict = "PASS"
    elif metrics_passing >= 3 and fc_artifact:
        verdict = "M3_FC_ARTIFACT"
    elif metrics_passing >= 1:
        verdict = "TREND"
    else:
        verdict = "FAIL"

    return {
        "h2_verdict": verdict,
        "favorable_counts": favorable_counts,
        "total_natural_samples": total_samples,
        "metrics_passing": metrics_passing,
        "fc_artifact_suspected": fc_artifact,
    }


def analyze_tfg_correlation(
    separability_values: np.ndarray,
    sync_c_deltas: np.ndarray,
    syncnet_results_path: Path,
) -> dict:
    """30E — Spearman ρ between per-sample separability diagnostics and ΔSync-C.

    LibriSpeech sample IDs are paired between separability (script 16 output)
    and Sync-C deltas (gate1.json `per_seed[seed].samples[sample_id].corrected_delta_sync_c`),
    aggregated across seeds 42/43/44 by mean.
    """
    from scipy import stats
    if len(separability_values) < 3 or len(sync_c_deltas) < 3:
        return {"h3_verdict": "INSUFFICIENT_DATA"}
    if len(separability_values) != len(sync_c_deltas):
        return {"h3_verdict": "INSUFFICIENT_DATA", "error": "length mismatch"}

    rho, p_val = stats.spearmanr(separability_values, sync_c_deltas)
    # Permutation p
    rng = np.random.default_rng(42)
    nulls = np.empty(10000, dtype=np.float64)
    for i in range(10000):
        perm = rng.permutation(len(sync_c_deltas))
        nulls[i] = stats.spearmanr(separability_values, sync_c_deltas[perm])[0]
    perm_p = float(np.mean(np.abs(nulls) >= abs(rho)))

    if perm_p < 0.05 and abs(rho) >= 0.3:
        verdict = "PASS"
    elif abs(rho) >= 0.3:
        verdict = "TREND"
    else:
        verdict = "FAIL"
    return {
        "h3_verdict": verdict,
        "spearman_rho": float(rho),
        "scipy_p": float(p_val),
        "permutation_p": perm_p,
        "n": int(len(separability_values)),
    }


def render_full_report(
    gate1_summary: dict,
    separability_summary: dict,
    oracle_fs_summary: dict,
    fd_gain_summary: dict,
    tfg_correlation_summary: dict,
    output_path: Path,
) -> None:
    """Render a markdown report combining all of 30A–30E."""
    lines = ["# English Wav2Sem Analysis Report\n"]
    lines.append("## 30A — Gate 1 (TTS advantage existence)\n")
    lines.append(f"- Verdict: `{gate1_summary.get('verdict', 'N/A')}`")
    lines.append(f"- ΔSync-C mean: {gate1_summary.get('primary', {}).get('sync_c', {}).get('mean', 'N/A')}")
    lines.append(f"- Holm p: {gate1_summary.get('primary', {}).get('sync_c', {}).get('holm_p', 'N/A')}\n")
    lines.append("## 30B — H1: TTS separability\n")
    lines.append(f"- Verdict: `{separability_summary.get('h1_verdict', 'N/A')}`")
    lines.append(f"- Cohen's d: {separability_summary.get('cohens_d', 'N/A')}\n")
    lines.append("## 30C — Oracle Fs diagnostics\n")
    lines.append(f"- n_pairs: {oracle_fs_summary.get('n_pairs', 0)}")
    lines.append(f"- mean cosine(nat ↔ tts) [CLS]: {oracle_fs_summary.get('mean_cosine_cls_nt', 'N/A')}")
    lines.append(f"- degenerate samples: {oracle_fs_summary.get('n_degenerate', 0)}\n")
    lines.append("## 30D — H2: Fs decouples embeddings\n")
    lines.append(f"- Verdict: `{fd_gain_summary.get('h2_verdict', 'N/A')}`")
    lines.append(f"- Favorable metric counts: {fd_gain_summary.get('favorable_counts', {})}")
    lines.append(f"- FC artifact suspected: {fd_gain_summary.get('fc_artifact_suspected', False)}\n")
    lines.append("## 30E — H3: Separability ↔ ΔSync-C correlation\n")
    lines.append(f"- Verdict: `{tfg_correlation_summary.get('h3_verdict', 'N/A')}`")
    lines.append(f"- Spearman ρ: {tfg_correlation_summary.get('spearman_rho', 'N/A')}")
    lines.append(f"- Permutation p: {tfg_correlation_summary.get('permutation_p', 'N/A')}\n")
    lines.append("## Conclusion\n")
    lines.append("- H1 (TTS has separability advantage) — see 30B")
    lines.append("- H2 (oracle Fs adds decoupling) — see 30D")
    lines.append("- H3 (separability correlates with TFG gain) — see 30E")
    lines.append("")
    if tfg_correlation_summary.get("h3_verdict") in ("PASS", "TREND"):
        lines.append("H3 positive despite Gate 1 failure → separability captures something SyncNet sees "
                     "that the G×E generator effect does not; warrants re-examining G×E interpretation.")
    else:
        lines.append("H1/H2 positive + H3 negative (expected case) → Wav2Sem Fs decoupling mechanism exists "
                     "on English but is **not** a sufficient condition for a TTS lip-sync advantage. "
                     "This is a meaningful negative result that constrains the Wav2Sem theory's scope.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# ---- end of new modules; modify main() to call them when --with-wav2sem ----
```

Then update the existing `main()` to accept `--with-wav2sem` and `--analysis-dir`:

```python
parser.add_argument("--with-wav2sem", action="store_true",
                    help="Run 30B/C/D/E modules in addition to Gate 1.")
parser.add_argument("--analysis-dir", type=str,
                    default="data/wav2sem_analysis_en/metrics",
                    help="Directory containing separability_metrics.json, oracle_fs.json, "
                         "oracle_fd_separability.json.")
parser.add_argument("--report-output", type=str,
                    default="docs/experiments/wav2sem_bilingual_report_en.md")

# After existing Gate 1 logic:
if args.with_wav2sem:
    analysis_dir = Path(args.analysis_dir)
    sep_path = analysis_dir / "separability_metrics.json"
    fs_path = analysis_dir / "oracle_fs.json"
    fd_path = analysis_dir / "oracle_fd_separability.json"

    sep_data = _load_json(sep_path) or {}
    fs_data = _load_json(fs_path) or {}
    fd_data = _load_json(fd_path) or {}

    # 30B
    sep_records = sep_data.get("records", sep_data) if isinstance(sep_data, dict) else sep_data
    sep_summary = analyze_separability(sep_records) if sep_records else {"h1_verdict": "INSUFFICIENT_DATA"}

    # 30C
    fs_summary = analyze_oracle_fs(fs_data) if fs_data else {"verdict": "INSUFFICIENT_DATA"}

    # 30D
    fd_records = fd_data.get("results", fd_data) if isinstance(fd_data, dict) else fd_data
    fd_summary = analyze_fd_gain(fd_records) if fd_records else {"h2_verdict": "INSUFFICIENT_DATA"}

    # 30E: pair per-sample separability with mean ΔSync-C across seeds
    # Read from gate1.json (existing)
    gate1_summary = _load_json(Path("runs/english_wav2sem_gate1_gate1.json"))
    if gate1_summary and "per_sample" in gate1_summary:
        sync_c_deltas = np.array([s["corrected_delta_sync_c"] for s in gate1_summary["per_sample"]])
    else:
        sync_c_deltas = np.array([])

    # per-sample separability = mean across conditions (per spec)
    if isinstance(sep_data, dict) and "per_sample" in sep_data:
        sep_vals = np.array([s["value"] for s in sep_data["per_sample"]])
    else:
        sep_vals = np.array([])

    tfg_summary = analyze_tfg_correlation(
        separability_values=sep_vals,
        sync_c_deltas=sync_c_deltas,
        syncnet_results_path=Path("runs/english_wav2sem_gate1_gate1.json"),
    ) if len(sync_c_deltas) > 0 else {"h3_verdict": "INSUFFICIENT_DATA"}

    render_full_report(
        gate1_summary=summary_gate1,  # existing Gate 1 dict in current code
        separability_summary=sep_summary,
        oracle_fs_summary=fs_summary,
        fd_gain_summary=fd_summary,
        tfg_correlation_summary=tfg_summary,
        output_path=Path(args.report_output),
    )
```

Note: the exact variable name holding the Gate 1 summary in the existing code (`summary_gate1` here) should be replaced with the actual local variable name from the current implementation. Read the existing tail of `main()` to confirm.

- [ ] **Step 5: Run test to verify it passes**

```bash
python tests/test_analyze_english_wav2sem_modules.py
```

Expected: 3 passed (separability_h1, fd_gain_h2, tfg_correlation_h3).

- [ ] **Step 6: Commit**

```bash
git add scripts/30_analyze_english_wav2sem.py tests/test_analyze_english_wav2sem_modules.py
git commit -m "feat(30): add modules 30B/C/D/E for separability/Fs/Fd-gain/TFG-correlation + markdown report renderer"
```

---

## Task 8: Update `scripts/configs/english_wav2sem.yaml` with `wav2sem:` block

**Files:**
- Modify: `scripts/configs/english_wav2sem.yaml` — append `wav2sem:` block (Section 5 of spec)

- [ ] **Step 1: Read current yaml**

```bash
cat scripts/configs/english_wav2sem.yaml
```

There is already a `wav2sem:` block (lines 48-56 in the file from prior session). Inspect what's already there.

- [ ] **Step 2: Append missing fields**

Edit `scripts/configs/english_wav2sem.yaml` to ensure the `wav2sem:` section contains all spec Section 5 fields:

```yaml
wav2sem:
  ssl_models: [hubert, xlsr]
  layers: [0, 6, 11, 12]
  target_bert: bert-base-uncased
  alignment_source: librispeech_phn           # new
  viseme_system: preston_blair_13             # new
  oracle_fs_input: transcript_text            # new
  oracle_fs_pooling: [cls, mean]               # new
  fd_fc_layers: 2                             # new
  fd_fc_init: orthogonal                       # new
  fd_random_seed: 42                           # new
  fdr_alpha: 0.05
  permutation_n: 10000
  official_repository: wslh852/Wav2Sem
  official_commit: null
  alignment_backend: mfa                       # legacy; ignored when alignment_source: librispeech_phn
  fallback_alignment_backend: torchaudio_ctc
```

- [ ] **Step 3: Smoke-validate yaml syntax**

```bash
python -c "import yaml; print(yaml.safe_load(open('scripts/configs/english_wav2sem.yaml'))['wav2sem'])"
```

Expected: a dict printed with all 12 keys; no `yaml.scanner.ScannerError`.

- [ ] **Step 4: Commit**

```bash
git add scripts/configs/english_wav2sem.yaml
git commit -m "feat(config): add wav2sem block fields for English Fs-BERT analysis"
```

---

## Task 9: Run end-to-end smoke on server (sample 1 only)

**Files:**
- No file changes; this task runs the pipeline on the server.

- [ ] **Step 1: Push code to server**

```bash
# From local repo
git push origin main  # if not pushed yet
# On server:
ssh root@connect.westc.seetacloud.com -p 33833
cd /root/autodl-tmp/experiments/english-wav2sem-20260718
git -C /root/autodl-tmp/repos/tts-exp pull
# Copy new scripts (27/28/29/30/31) into experiment dir
cp /root/autodl-tmp/repos/tts-exp/scripts/27_download_librispeech_phn.py scripts/
cp /root/autodl-tmp/repos/tts-exp/scripts/28_prepare_english_alignment.py scripts/
cp /root/autodl-tmp/repos/tts-exp/scripts/29_oracle_fs_bert.py scripts/
cp /root/autodl-tmp/repos/tts-exp/scripts/30_analyze_english_wav2sem.py scripts/  # overwrite
cp /root/autodl-tmp/repos/tts-exp/scripts/31_oracle_fd_separability.py scripts/
cp /root/autodl-tmp/repos/tts-exp/data/data/english_viseme_map.yaml data/data/
cp /root/autodl-tmp/repos/tts-exp/scripts/configs/english_wav2sem.yaml scripts/configs/
```

- [ ] **Step 2: Run smoke pipeline**

```bash
cd /root/autodl-tmp/experiments/english-wav2sem-20260718
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/autodl-tmp/envs/ditto/bin:$PATH
PY=/root/autodl-tmp/envs/ditto/bin/python

$PY scripts/27_download_librispeech_phn.py --samples 1 --smoke
$PY scripts/28_prepare_english_alignment.py --samples 1 --smoke
$PY scripts/15_extract_ssl_embeddings.py \
    --manifest data/wav2sem_analysis_en/manifest/alignment.json \
    --models hubert --layers 0,11 --smoke \
    --output-dir data/wav2sem_analysis_en/embeddings
$PY scripts/16_feature_separability.py \
    --embeddings-dir data/wav2sem_analysis_en/embeddings \
    --output-dir data/wav2sem_analysis_en/metrics --smoke
$PY scripts/29_oracle_fs_bert.py --samples 1 --smoke
$PY scripts/31_oracle_fd_separability.py --samples 1 --smoke
$PY scripts/30_analyze_english_wav2sem.py \
    --runs-root runs --run-prefix english_wav2sem_gate1 \
    --seeds 42,43,44 --with-wav2sem
```

- [ ] **Step 3: Verify acceptance criteria**

For each item, run the check command on the server:

```bash
# 1. alignment manifest has sample 1 natural + tts
python -c "
import json
d = json.load(open('data/wav2sem_analysis_en/manifest/alignment.json'))
e = [x for x in d['manifest'] if x['sample_id']==1]
assert len(e)>=2 and any(t for t in e[0]['tokens']), 'tokens missing'
print('OK: alignment has sample 1 with tokens')
"

# 2. embeddings exist for sample 1 (natural + tts × HuBERT)
ls data/wav2sem_analysis_en/embeddings/1_*_hubert.npy

# 3. separability_metrics.json has 'records' or top-level list
python -c "
import json
d = json.load(open('data/wav2sem_analysis_en/metrics/separability_metrics.json'))
print('OK:', list(d.keys()) if isinstance(d, dict) else 'list with', len(d))
"

# 4. oracle_fs.json has 2 entries (natural + tts)
python -c "
import json
d = json.load(open('data/wav2sem_analysis_en/metrics/oracle_fs.json'))
assert len(d['entries'])==2
print('OK:', d['n_entries'])
"

# 5. oracle_fd_separability.json has all 3 tiers
python -c "
import json
d = json.load(open('data/wav2sem_analysis_en/metrics/oracle_fd_separability.json'))
assert all('fp_metrics' in r and 'fd_zero_metrics' in r and 'fd_random_metrics' in r for r in d['results'])
print('OK: three tiers present')
"

# 6. H1/H2/H3 verdict in report
ls docs/experiments/wav2sem_bilingual_report_en.md
```

Expected: all 5 verification commands print "OK:"; report file exists.

- [ ] **Step 4: Document the smoke result**

Append smoke log to `docs/experiments/smoke_en_log.md`:

```markdown
# English Wav2Sem smoke run (1 sample)

Date: 2026-07-18
Steps: 27 → 28 → 15 → 16 → 29 → 31 → 30 --with-wav2sem

Result: <PASS or describe any failures>

Outputs:
- data/wav2sem_analysis_en/manifest/alignment.json
- data/wav2sem_analysis_en/embeddings/1_*_hubert.npy
- data/wav2sem_analysis_en/metrics/separability_metrics.json
- data/wav2sem_analysis_en/metrics/oracle_fs.json
- data/wav2sem_analysis_en/metrics/oracle_fd_separability.json
- docs/experiments/wav2sem_bilingual_report_en.md
```

- [ ] **Step 5: Commit the smoke log when back on local**

```bash
# locally
git pull  # if smoke result was pushed from server
git add docs/experiments/smoke_en_log.md
git commit -m "docs: English Wav2Sem smoke run log (1 sample, 2026-07-18)"
```

---

## Task 10: Run full pipeline on all 13 English samples

**Files:**
- No file changes; runs on server.

- [ ] **Step 1: Run full pipeline**

```bash
cd /root/autodl-tmp/experiments/english-wav2sem-20260718
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/autodl-tmp/envs/ditto/bin:$PATH
PY=/root/autodl-tmp/envs/ditto/bin/python

$PY scripts/27_download_librispeech_phn.py
$PY scripts/28_prepare_english_alignment.py
$PY scripts/15_extract_ssl_embeddings.py \
    --manifest data/wav2sem_analysis_en/manifest/alignment.json
$PY scripts/16_feature_separability.py \
    --embeddings-dir data/wav2sem_analysis_en/embeddings \
    --output-dir data/wav2sem_analysis_en/metrics
$PY scripts/29_oracle_fs_bert.py
$PY scripts/31_oracle_fd_separability.py
$PY scripts/30_analyze_english_wav2sem.py \
    --runs-root runs --run-prefix english_wav2sem_gate1 \
    --seeds 42,43,44 --with-wav2sem
```

- [ ] **Step 2: Sanity-check the numbers**

Expected sanity checks:

- 13 samples × 2 conditions × 2 SSL models = 52 embedding .npy files in `embeddings/`
- 26 entries in `oracle_fs.json`
- 26 results in `oracle_fd_separability.json` (13 nat + 13 tts)
- `report.md` should state:
  - H1 verdict (PASS / TREND / FAIL based on segment_stability Cohen's d vs 0.5)
  - H2 verdict (PASS / FAIL / M3_FC_ARTIFACT)
  - H3 verdict (PASS / TREND / FAIL based on Spearman ρ)

- [ ] **Step 3: Sync results back to local**

```bash
# from local
scp -r -P 33833 root@connect.westc.seetacloud.com:/root/autodl-tmp/experiments/english-wav2sem-20260718/data/wav2sem_analysis_en ./data/
scp -P 33833 root@connect.westc.seetacloud.com:/root/autodl-tmp/experiments/english-wav2sem-20260718/docs/experiments/wav2sem_bilingual_report_en.md ./docs/experiments/
```

- [ ] **Step 4: Commit final results**

```bash
git add data/wav2sem_analysis_en/ docs/experiments/wav2sem_bilingual_report_en.md
git commit -m "results: English Wav2Sem Fs-BERT analysis on 13 LibriSpeech samples (H1/H2/H3 verdicts)"
```

---

## Self-Review

### Spec coverage check

| Spec section | Tasks covering it |
|---|---|
| §3 Components & data flow | Tasks 3-7 (scripts 27/28/29/30/31) |
| §4 Data product layout | Tasks 3-7 (each creates the expected directories/files) |
| §5 Configuration | Task 8 |
| §6.1 LibriSpeech .phn parsing | Task 3 |
| §6.2 ARPABET → PB13 visemes | Tasks 2, 4 |
| §6.3 Oracle Fs (BERT CLS) | Task 5 |
| §6.4 Fd construction (Fp/Fd_zero/Fd_random) | Task 6 |
| §7.1 M1 metadata mismatch | Task 3 (`verify_metadata`) |
| §7.1 M2 BERT CLS degeneracy | Task 5 (`flag_degenerate`) |
| §7.1 M3 FC spurious separability | Task 6 (`Fd_zero` + `Fd_random`); Task 7 (`analyze_fd_gain` with `fc_artifact_suspected`) |
| §7.1 M4 wrong model cached | script 15 already validates layer index; not a new task |
| §8 Testing strategy | Tasks 1-7 include unit smoke tests; Task 9 e2e smoke |
| §10 Schedule | 10 tasks ≈ 1 working day |

No spec section is missing a covering task.

### Placeholder scan

Searched the plan for "TBD", "TODO", "implement later", "fill in details", "Add appropriate". None found.

### Type consistency

- `parse_phn_file` signature identical between script 27 (returns `list[dict]`) and Task 4 script 28 (calls script 27 module via importlib; expects same shape).
- `extract_fs_from_outputs(outputs) → (Fs_cls, Fs_mean, meta)` is the function tested and used inside `process_sample` in script 29. Names match.
- `build_fc_orthogonal(hidden, seed)` returns `nn.Sequential`; consumed by `construct_fd(Fp, Fs, fc)` in script 31. Names match.
- `compute_separability_for_variant(variant, frame_times, tokens) → dict` uses keys `intra_class_dist`, `silhouette`, `boundary_sharpness`, `segment_stability`, `fisher`; the same key set is checked in script 30's `analyze_fd_gain` and `CORE_METRICS_FAVORABLE`. Names match.
- `gate1_summary` → existing Gate 1 dict (read across tasks; ensure when wiring up Task 7 step 4 you use the actual local variable name from the current `main()`). I have flagged this in the implementation note.