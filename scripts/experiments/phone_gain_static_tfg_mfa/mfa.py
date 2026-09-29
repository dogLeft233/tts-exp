"""Hash-bound, isolated MFA attempts and TextGrid parsing."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .assets import write_pcm16
from .config import canonical_hash, file_sha256, write_json
from scripts.experiments.lrs3_phone_rules_metrics import is_speech_label, normalize_phone


_MFA_BINDING_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_textgrid(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    intervals = []
    xmin = xmax = None
    tier_name: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name ="):
            tier_name = stripped.split("=", 1)[1].strip().strip('"').lower()
        elif stripped.startswith("xmin ="):
            xmin = float(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("xmax ="):
            xmax = float(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("text ="):
            label = stripped.split("=", 1)[1].strip().strip('"')
            if xmin is None or xmax is None:
                continue
            # MFA exports words and phones in separate tiers.  The phone tier
            # is the only one comparable to the experiment's phone labels.
            if tier_name in {"phone", "phones", "phone tier", "phones tier"} or (tier_name is not None and "phone" in tier_name):
                if not np.isfinite([xmin, xmax]).all() or xmax <= xmin:
                    raise ValueError(f"invalid phone interval in TextGrid: {path}")
                normalized = normalize_phone(label)
                speech = is_speech_label(normalized, silence_labels=("sil", "sp", "spn", "<eps>", ""))
                intervals.append({"start_s": float(xmin), "end_s": float(xmax), "label": normalized, "speech": speech, "silence": not speech})
            xmin = xmax = None
    # Keep occurrence identity explicit. Calibration and matched timing
    # controls must map a boundary back to the same phone occurrence rather
    # than guessing from the position in a speech-only list.
    for index, token in enumerate(intervals):
        token["token_index"] = int(index)
        token["token_id"] = f":{index}"
    return intervals


def mfa_batch_attempt(config: Mapping[str, Any], run_dir: str | Path, *, rows: list[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    """Align a whole condition in one MFA invocation.

    One invocation is materially cheaper and more reproducible than launching
    MFA once per utterance.  ``rows`` contain pair_id, pcm, and transcript;
    output keys are ``pair_id|condition``.
    """
    executable = Path(str(config["quality"]["mfa_executable"]))
    if not executable.is_file():
        return {"status": "DEPENDENCY_BLOCKED", "reason": "MFA_EXECUTABLE_MISSING", "rows": []}
    if not rows:
        return {"status": "ENGINEERING_FAILURE", "reason": "NO_INPUT_ROWS", "condition": condition, "rows": []}
    root = Path(run_dir).resolve()
    binding = _mfa_binding(config, executable)
    digest = hashlib.sha256()
    digest.update(canonical_hash({"condition": condition, "binding": binding}).encode())
    for row in sorted(rows, key=lambda value: str(value["pair_id"])):
        pcm = np.asarray(row["pcm"], dtype=np.int16).reshape(-1)
        digest.update(str(row["pair_id"]).encode())
        digest.update(hashlib.sha256(pcm.tobytes()).digest())
        digest.update(str(row.get("transcript", "")).encode())
    attempt_key = digest.hexdigest()[:16]
    attempt = root / "06_quality/mfa_batches" / f"{_safe(condition)}__{attempt_key}"
    corpus = attempt / "corpus"
    output = attempt / "output"
    result_path = attempt / "result.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("cache_binding") == binding:
            return cached
        attempt = root / "06_quality/mfa_batches" / f"{_safe(condition)}__{attempt_key}__rebind_{canonical_hash(binding)[:8]}"
        corpus = attempt / "corpus"
        output = attempt / "output"
        result_path = attempt / "result.json"
    corpus.mkdir(parents=True, exist_ok=True)
    cell_names: dict[str, str] = {}
    for row in rows:
        key = _safe(f"{row['pair_id']}__{condition}")
        cell_names[str(row["pair_id"])] = key
        write_pcm16(corpus / f"{key}.wav", row["pcm"])
        (corpus / f"{key}.lab").write_text(str(row.get("transcript", "")).strip() + "\n", encoding="utf-8")
    command = [str(executable), "align", str(corpus.resolve()), str(config["quality"]["mfa_dictionary"]), str(config["quality"]["mfa_acoustic_model"]), str(output.resolve()), "--clean"]
    process = subprocess.run(command, cwd=str(attempt), text=True, capture_output=True, check=False)
    textgrids = {path.stem: path for path in output.rglob("*.TextGrid")} if output.is_dir() else {}
    aligned: list[dict[str, Any]] = []
    for row in rows:
        key = cell_names[str(row["pair_id"])]
        grid = textgrids.get(key)
        item = {"pair_id": str(row["pair_id"]), "condition": condition, "status": "COMPLETE" if process.returncode == 0 and grid else "ENGINEERING_FAILURE", "textgrid": str(grid.resolve()) if grid else None}
        if grid:
            item["tokens"] = parse_textgrid(grid)
            if not item["tokens"] or not any(bool(token.get("speech")) for token in item["tokens"]):
                item["status"] = "ENGINEERING_FAILURE"
                item["reason"] = "EMPTY_PHONE_TIER"
        aligned.append(item)
    payload = {"status": "COMPLETE" if process.returncode == 0 and len(aligned) == len(rows) and all(item["status"] == "COMPLETE" for item in aligned) else "ENGINEERING_FAILURE", "condition": condition, "attempt": str(attempt.resolve()), "command": command, "returncode": process.returncode, "executable_sha256": file_sha256(executable), "dictionary": str(config["quality"]["mfa_dictionary"]), "acoustic_model": str(config["quality"]["mfa_acoustic_model"]), "cache_binding": binding, "rows": aligned, "stdout_tail": process.stdout[-2000:], "stderr_tail": process.stderr[-2000:]}
    write_json(result_path, payload)
    return payload


def _mfa_binding(config: Mapping[str, Any], executable: Path) -> dict[str, Any]:
    """Capture executable, version, and installed model inspection bindings."""

    cache_key = (str(executable.resolve()), str(config["quality"]["mfa_dictionary"]), str(config["quality"]["mfa_acoustic_model"]))
    if cache_key in _MFA_BINDING_CACHE:
        return _MFA_BINDING_CACHE[cache_key]
    version = subprocess.run([str(executable), "version"], capture_output=True, text=True, check=False, timeout=60)
    inspections: dict[str, Any] = {}
    for kind, name in (("dictionary", config["quality"]["mfa_dictionary"]), ("acoustic", config["quality"]["mfa_acoustic_model"])):
        probe = subprocess.run([str(executable), "model", "inspect", kind, str(name)], capture_output=True, text=True, check=False, timeout=120)
        output = (probe.stdout or "") + "\n[stderr]\n" + (probe.stderr or "")
        inspections[kind] = {"name": str(name), "returncode": int(probe.returncode), "output_sha256": hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()}
    binding = {"executable": str(executable.resolve()), "executable_sha256": file_sha256(executable), "version": (version.stdout or version.stderr)[-2000:], "dictionary": str(config["quality"]["mfa_dictionary"]), "acoustic_model": str(config["quality"]["mfa_acoustic_model"]), "model_inspections": inspections}
    _MFA_BINDING_CACHE[cache_key] = binding
    return binding


def mfa_attempt(config: Mapping[str, Any], run_dir: str | Path, *, pair_id: str, pcm: Any, transcript: str, condition: str) -> dict[str, Any]:
    root = Path(run_dir)
    executable = Path(str(config["quality"]["mfa_executable"]))
    if not executable.is_file():
        return {"status": "DEPENDENCY_BLOCKED", "reason": "MFA_EXECUTABLE_MISSING", "path": str(executable)}
    attempt_key = hashlib.sha256(f"{pair_id}|{condition}|{hashlib.sha256(bytes(pcm)).hexdigest()}|{transcript}|{file_sha256(executable)}".encode()).hexdigest()[:16]
    attempt = root / "06_quality/mfa_attempts" / f"{_safe(pair_id)}__{condition}__{attempt_key}"
    attempt.mkdir(parents=True, exist_ok=True)
    wav = attempt / "audio.wav"
    lab = attempt / "audio.lab"
    output = attempt / "output"
    if not wav.is_file():
        write_pcm16(wav, pcm)
    lab.write_text(transcript.strip() + "\n", encoding="utf-8")
    corpus = attempt / "corpus"
    corpus.mkdir(exist_ok=True)
    corpus_wav = corpus / "audio.wav"
    corpus_lab = corpus / "audio.lab"
    if not corpus_wav.is_file():
        corpus_wav.write_bytes(wav.read_bytes())
    corpus_lab.write_text(lab.read_text(encoding="utf-8"), encoding="utf-8")
    # MFA is launched with ``cwd=attempt``; pass absolute corpus/output paths
    # so a relative run directory is not accidentally prefixed twice.
    command = [str(executable), "align", str(corpus.resolve()), str(config["quality"]["mfa_dictionary"]), str(config["quality"]["mfa_acoustic_model"]), str(output.resolve()), "--clean"]
    result = subprocess.run(command, cwd=str(attempt), text=True, capture_output=True, check=False)
    textgrids = sorted(output.rglob("*.TextGrid"))
    payload = {"status": "COMPLETE" if result.returncode == 0 and textgrids else "ENGINEERING_FAILURE", "pair_id": pair_id, "condition": condition, "attempt": str(attempt.resolve()), "command": command, "returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:], "executable_sha256": file_sha256(executable), "textgrid": str(textgrids[0].resolve()) if textgrids else None}
    if textgrids:
        payload["tokens"] = parse_textgrid(textgrids[0])
    write_json(attempt / "result.json", payload)
    return payload


__all__ = ["mfa_attempt", "mfa_batch_attempt", "parse_textgrid"]
