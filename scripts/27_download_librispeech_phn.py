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
