#!/usr/bin/env python3
"""Prepare the frozen AISHELL-1 speaker holdout without model inference or scoring."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/native_mechanism_holdout_20260926"
OUT = ROOT / "runs/tts_native_holdout_inventory_20260926"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def pcm_hash(audio: np.ndarray, sr: int) -> str:
    """Exact decoded samples, including rate/shape; no resampling or tolerance."""
    audio = np.asarray(audio, dtype="<f8")
    header = json.dumps({"sample_rate": sr, "shape": list(audio.shape)}, sort_keys=True).encode()
    return sha(header + b"\n" + audio.tobytes())


def download(speaker: str, revision: str) -> dict:
    rel = f"data_aishell/wav/{speaker}.tar.gz"
    url = f"https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/{revision}/{rel}"
    path = DATA / "archives" / f"{speaker}.tar.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    if not path.exists():
        for attempt in range(1, 5):
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with path.with_suffix(".partial").open("wb") as handle:
                        for chunk in response.iter_content(1 << 20):
                            handle.write(chunk)
                path.with_suffix(".partial").rename(path)
                attempts.append({"attempt": attempt, "status": "downloaded"})
                break
            except requests.RequestException as exc:
                # Never persist redirected signed URLs from exceptions.
                attempts.append({"attempt": attempt, "error_type": type(exc).__name__})
                save(DATA / "provenance" / f"{speaker}_download_attempts.json", attempts)
                if attempt == 4:
                    raise RuntimeError(f"Archive download failed for {speaker}") from None
                time.sleep(min(attempt * 2, 6))
    record = {"speaker_id": speaker, "publisher_url": url, "revision": revision,
              "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
              "sha256": sha(path.read_bytes()), "attempts": attempts}
    publisher_tree = json.loads((DATA / "provenance/hf_archive_tree.json").read_text())
    expected = next(row for row in publisher_tree if row["path"] == rel)
    if record["sha256"] != expected["lfs"]["oid"] or record["bytes"] != expected["size"]:
        raise ValueError(f"Publisher archive checksum/size mismatch for {speaker}")
    record["publisher_lfs_sha256_verified"] = True
    save(DATA / "provenance" / f"{speaker}_archive.json", record)
    return record


def select(speaker: str, transcript: dict[str, str], historic_hashes: set[str]) -> tuple[list, list]:
    decisions, picks = [], []
    with tarfile.open(DATA / "archives" / f"{speaker}.tar.gz", "r:gz") as archive:
        members = sorted([m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith(".wav")],
                         key=lambda m: (Path(m.name).stem, m.name))
        seen_ids = set()
        for member in members:
            uid = Path(member.name).stem
            row = {"speaker_id": speaker, "utterance_id": uid, "archive_member": member.name}
            if uid in seen_ids:
                raise ValueError(f"Duplicate archive utterance ID: {uid}")
            seen_ids.add(uid)
            if not re.fullmatch(rf"BAC009{speaker}W\d{{4}}", uid):
                row["decision"] = "invalid_utterance_speaker_identity"
            elif len(picks) == 2:
                row["decision"] = "not_evaluated_after_first_two_eligible"
            elif not transcript.get(uid, "").strip():
                row["decision"] = "missing_or_empty_manual_transcript"
            else:
                raw = archive.extractfile(member).read()
                try:
                    audio, sr = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
                    info = sf.info(io.BytesIO(raw))
                except (RuntimeError, ValueError):
                    row["decision"] = "audio_decode_failure"
                    decisions.append(row)
                    continue
                duration = len(audio) / sr
                row.update({"sample_rate": sr, "channels": audio.shape[1], "frames": len(audio),
                            "duration_seconds": duration, "subtype": info.subtype,
                            "file_sha256": sha(raw), "pcm_sha256": pcm_hash(audio, sr)})
                if not info.subtype.startswith("PCM_"):
                    row["decision"] = "not_pcm"
                elif not np.isfinite(audio).all():
                    row["decision"] = "nonfinite_audio"
                elif not 3 <= duration <= 8:
                    row["decision"] = "duration_outside_3_to_8_seconds"
                elif row["pcm_sha256"] in historic_hashes:
                    row["decision"] = "duplicate_of_historical_pcm"
                else:
                    row["decision"] = "selected"
                    target = DATA / "audio" / f"{uid}.wav"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    textpath = DATA / "transcripts" / f"{uid}.txt"
                    textpath.parent.mkdir(parents=True, exist_ok=True)
                    textpath.write_text(transcript[uid] + "\n")
                    row.update({"audio_path": str(target.relative_to(ROOT)),
                                "transcript_path": str(textpath.relative_to(ROOT)),
                                "manual_transcript": transcript[uid],
                                "transcript_sha256": sha(textpath.read_bytes())})
                    picks.append(row.copy())
            decisions.append(row)
    return picks, decisions


def main() -> None:
    protocol = (OUT / "protocol.json").read_bytes()
    assert sha(protocol) == (OUT / "protocol.sha256").read_text().split()[0]
    revision_info = json.loads((DATA / "provenance/hf_dataset_revision.json").read_text())
    revision = revision_info["sha"]
    available = sorted(Path(x["rfilename"]).name[:-7] for x in revision_info["siblings"]
                       if re.fullmatch(r"data_aishell/wav/S\d{4}\.tar\.gz", x["rfilename"]))
    history = json.loads((OUT / "history_registry.json").read_text())
    excluded = set(history["excluded_speakers"])
    candidates = [s for s in available if s not in excluded]
    save(OUT / "candidate_speakers.json", {"revision": revision, "available": available,
         "excluded_historical": sorted(set(available) & excluded), "ordered_candidates": candidates})
    if len(candidates) < 40:
        raise RuntimeError(f"Only {len(candidates)} available candidate speakers; report before full-corpus download")
    lines = (DATA / "provenance/aishell_transcript_v0.8.txt").read_text().splitlines()
    transcript = dict(line.split(maxsplit=1) for line in lines if len(line.split(maxsplit=1)) == 2)
    hashes = {r["pcm_sha256"] for r in history["available_audio_hashes"] if "pcm_sha256" in r}
    selected, decisions, archives, speaker_decisions = [], [], [], []
    # First 40 can be fetched concurrently; replacement speakers follow canonical order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        archives.extend(pool.map(lambda s: download(s, revision), candidates[:40]))
    for i, speaker in enumerate(candidates):
        if len(selected) == 80:
            speaker_decisions.append({"speaker_id": speaker, "decision": "not_needed_after_40_speakers"})
            continue
        if i >= 40:
            archives.append(download(speaker, revision))
        picks, audit = select(speaker, transcript, hashes)
        decisions.extend(audit)
        if len(picks) == 2:
            selected.extend(picks)
            speaker_decisions.append({"speaker_id": speaker, "decision": "selected", "utterance_ids": [x["utterance_id"] for x in picks]})
        else:
            speaker_decisions.append({"speaker_id": speaker, "decision": "fewer_than_two_eligible", "eligible_count": len(picks)})
        save(OUT / "utterance_decisions.json", decisions)
        save(OUT / "speaker_decisions.json", speaker_decisions)
    assert len(selected) == 80 and len({x["speaker_id"] for x in selected}) == 40
    assert len({x["pcm_sha256"] for x in selected}) == 80, "Duplicate selected PCM: stop; do not silently substitute"
    manifest = {"schema_version": 1, "protocol_sha256": sha(protocol), "source_revision": revision,
                "transcript_source_sha256": sha((DATA / "provenance/aishell_transcript_v0.8.txt").read_bytes()),
                "n_speakers": 40, "n_utterances": 80,
                "total_duration_seconds": sum(x["duration_seconds"] for x in selected),
                "archives_downloaded_bytes": sum(x["bytes"] for x in archives), "records": selected}
    save(OUT / "manifest.json", manifest)
    save(OUT / "archives.json", archives)
    save(OUT / "speaker_decisions.json", speaker_decisions)
    (OUT / "manifest.sha256").write_text(sha((OUT / "manifest.json").read_bytes()) + "  manifest.json\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
