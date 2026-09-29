"""Capture the final source and artifact fingerprints for follow-up experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

from scripts.experiments.tts_independent_visual import ROOT, read, sha, write


def main(name):
    base = ROOT / "runs" / name
    assert read(base / "validation.json")["status"] == "PASS"
    names = [
        "tts_window_clock.py",
        "check_tts_window_clock.py",
        "tts_prepost_mel.py",
        "check_tts_prepost_mel.py",
        "tts_prepost_controls.py",
        "tts_followup_provenance.py",
        "tts_independent_visual.py",
        "tts_pcm_residual.py",
        "tts_raw_video_transfer.py",
        "tts_pcm_timing.py",
        "check_tts_pcm_residual.py",
        "tts_native_gain_attribution/syncnet.py",
        "tts_native_gain_attribution/common.py",
        "static_image_bridge/score_worker.py",
        "masked_tts_tfg_probe/direct_mel.py",
    ]
    files = [ROOT / "scripts/experiments" / n for n in names]
    files += [
        ROOT / "third_party/Wav2Lip" / n
        for n in (
            "audio.py",
            "hparams.py",
            "inference.py",
            "models/wav2lip.py",
            "models/conv.py",
        )
    ]
    artifacts = [
        p for p in base.rglob("*.json") if p.name != "execution_provenance.json"
    ]
    artifacts += [
        p for p in base.iterdir() if p.suffix in (".md", ".csv", ".png", ".pdf", ".py")
    ]
    result = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "python": sys.version,
        "scripts_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "artifacts_sha256": {str(p.relative_to(base)): sha(p) for p in artifacts},
        "limits": "All input/weights/NPZ/video hashes in frozen protocol and per-cell receipts; verified by validation.json. Working tree preserved, no commit or reset.",
    }
    write(base / "execution_provenance.json", result)
    for path, digest in result["scripts_sha256"].items():
        assert sha(ROOT / path) == digest
    for path, digest in result["artifacts_sha256"].items():
        assert sha(base / path) == digest
    print(
        "final manifest PASS",
        name,
        len(files),
        "sources",
        len(artifacts),
        "artifacts",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "run", choices=["tts_window_clock_20260926", "tts_prepost_mel_20260926"]
    )
    main(p.parse_args().run)
