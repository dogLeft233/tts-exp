"""Batched, isolated MFA phones for the controlled Chinese-LiPS experiment.

Phones are approximate audio-aligned labels, not visual synchronization truth.
No SyncNet outputs are used for text segmentation, alignment, or eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
import zipfile


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    return obj if isinstance(obj, list) else obj["rows"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dictionary_words(path):
    words = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            words.add(line.split()[0])
    return words


def compatible_dictionary(dictionary, acoustic, output):
    """Filter existing pronunciations before tokenization; never invent phone maps.

    Local MFA inventories can have dictionaries newer than their acoustic model.
    MFA silently drops those pronunciations, turning apparently in-vocabulary
    transcript words into spn. Segment only with the actual surviving vocabulary.
    """
    with zipfile.ZipFile(acoustic) as archive:
        names = [n for n in archive.namelist() if n.endswith("/meta.json") or n == "meta.json"]
        if len(names) != 1:
            raise ValueError("Ambiguous acoustic model metadata")
        metadata = json.loads(archive.read(names[0]))
    inventory = set(metadata["phones"]) | {"sil", "spn"}
    accepted, rejected = [], 0
    for line in Path(dictionary).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        # MFA probability lexicon: word, pronunciation probability, silence
        # probability, two silence-correction values, then phones.
        index = 1
        while index < min(5, len(fields)):
            try:
                float(fields[index])
                index += 1
            except ValueError:
                break
        phones = fields[index:]
        if phones and set(phones).issubset(inventory):
            accepted.append(line)
        else:
            rejected += 1
    if not accepted:
        raise ValueError("No pronunciations compatible with acoustic inventory")
    Path(output).write_text("\n".join(accepted) + "\n", encoding="utf-8")
    return {"accepted_pronunciations": len(accepted), "rejected_pronunciations": rejected,
            "rule": "unchanged pronunciations whose complete inventory is in acoustic meta phones"}


def segment(text, words):
    """Deterministic dictionary maximum matching, with explicit OOV reporting."""
    text = unicodedata.normalize("NFKC", text).strip()
    max_len = max(map(len, words), default=1)
    result, unknown = [], []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace() or unicodedata.category(c).startswith(("P", "S")):
            i += 1
            continue
        hit = next((text[i:i+n] for n in range(min(max_len, len(text)-i), 0, -1)
                    if text[i:i+n] in words), None)
        if hit is None:
            unknown.append(c)
            result.append("<unk>")
            i += 1
        else:
            result.append(hit)
            i += len(hit)
    return result, unknown


def parse_textgrid(path):
    text = Path(path).read_text(encoding="utf-8")
    # MFA long TextGrid; select a whole item, never accidentally merge word tier.
    tiers = re.split(r"(?m)^\s*item\s*\[\d+\]:", text)
    tier = next((t for t in tiers if re.search(r'name\s*=\s*"phones?"', t)), None)
    if tier is None:
        raise ValueError(f"No phone tier: {path}")
    pattern = (r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([-+0-9.eE]+)\s*"
               r"xmax\s*=\s*([-+0-9.eE]+)\s*text\s*=\s*\"((?:\"\"|[^\"])*)\"")
    spans, previous = [], 0.0
    for match in re.finditer(pattern, tier):
        start, end = float(match[1]), float(match[2])
        raw = match[3].replace('""', '"').strip()
        if end <= start:
            continue
        if start < previous - 1e-6:
            raise ValueError("Overlapping/nonmonotonic phone intervals")
        # Strip lexical tones consistently, preserve consonants/vowels.
        phone = re.sub(r"[˥˦˧˨˩]", "", raw)
        phone = phone if phone else "sil"
        spans.append({"start": start, "end": end, "phone": phone,
                      "raw_phone": raw, "unknown": phone in {"spn", "<unk>"}})
        previous = end
    if not spans or not any(s["phone"] not in {"sil", "sp", "spn", "<unk>"} for s in spans):
        raise ValueError("No alignable speech phones")
    return spans


def row_text(row):
    for key in ("text", "transcript"):
        if row.get(key):
            return str(row[key])
    return Path(row["text_path"]).read_text(encoding="utf-8").strip()


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    filtered_dictionary = out / "compatible_mandarin.dict"
    lexicon_audit = compatible_dictionary(args.dictionary, args.acoustic, filtered_dictionary)
    words = dictionary_words(filtered_dictionary)
    selected, exclusions = [], []
    for row in rows:
        tokens, oovs = segment(row_text(row), words)
        if not tokens or oovs:
            exclusions.append({"id": row["id"], "reason": "empty_or_dictionary_oov", "oov": oovs})
            continue
        selected.append((row, tokens))
    write_json(out / "eligibility.json", {"total": len(rows), "selected": len(selected),
               "exclusions": exclusions, "rule": "exclude any dictionary OOV before model outputs"})
    write_json(out / "provenance.json", {"manifest_sha256": digest(args.manifest),
               "dictionary_sha256": digest(args.dictionary), "acoustic_sha256": digest(args.acoustic),
               "effective_dictionary_sha256": digest(filtered_dictionary), "lexicon_audit": lexicon_audit,
               "source_sha256": digest(__file__), "phone_labels": "MFA approximate, no visual truth",
               "tokenizer": "dictionary longest match; NFKC; punctuation removed; any OOV excluded",
               "tone_handling": "strip tone marks after alignment", "started": time.time()})
    completed = []
    for batch_start in range(0, len(selected), args.batch_clips):
        batch = selected[batch_start:batch_start + args.batch_clips]
        pending = [(r, t) for r, t in batch if not (out / "phones" / f"{r['id']}.json").exists()]
        if pending:
            work = out / "batches" / f"b{batch_start:06d}"
            corpus, grids = work / "corpus", work / "grids"
            for row, tokens in pending:
                target = corpus / str(row["speaker"]) / str(row["id"])
                target.parent.mkdir(parents=True, exist_ok=True)
                source = Path(row["audio_path"]).resolve()
                if source.suffix.lower() != ".wav":
                    raise ValueError("MFA phase expects canonical PCM WAV; do not silently transcode clock")
                audio_link = target.with_suffix(".wav")
                if not audio_link.exists():
                    audio_link.symlink_to(source)
                target.with_suffix(".lab").write_text(" ".join(tokens), encoding="utf-8")
            env = dict(os.environ, PYTHONNOUSERSITE="1", MFA_ROOT_DIR=str(work / "runtime"))
            cmd = [args.mfa, "align", str(corpus), str(filtered_dictionary.resolve()),
                   str(Path(args.acoustic).resolve()), str(grids), "--num_jobs", str(args.jobs),
                   "--temporary_directory", str(work / "temporary"), "--output_format", "long_textgrid"]
            write_json(work / "command.json", {"argv": cmd, "timeout_seconds": args.timeout})
            print("MFA_BATCH", batch_start, len(pending), flush=True)
            with (work / "mfa.log").open("w", encoding="utf-8") as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=args.timeout)
            for row, _ in pending:
                matches = list(grids.rglob(f"{row['id']}.TextGrid"))
                if len(matches) != 1:
                    raise ValueError(f"Missing/ambiguous MFA result {row['id']}: {matches}")
                spans = parse_textgrid(matches[0])
                write_json(out / "phones" / f"{row['id']}.json", {
                    "id": row["id"], "audio_sha256": digest(row["audio_path"]),
                    "textgrid_sha256": digest(matches[0]), "phone_spans": spans,
                    "provenance": "MFA mandarin_mfa; approximate audio-only forced alignment"})
            # Only remove this script's temporary feature/DB state; retain grids/logs.
            for name in ("temporary", "runtime", "corpus"):
                candidate = (work / name).resolve()
                if work.resolve() not in candidate.parents:
                    raise ValueError("Unsafe cleanup target")
                if candidate.exists():
                    shutil.rmtree(candidate)
        for row, _ in batch:
            phone_path = out / "phones" / f"{row['id']}.json"
            if phone_path.exists():
                completed.append(dict(row, phone_spans_path=str(phone_path)))
        write_json(out / "manifest.json", {"rows": completed, "status": "RUNNING",
                   "selected_total": len(selected)})
        print("MFA_DONE", len(completed), "/", len(selected), flush=True)
    write_json(out / "manifest.json", {"rows": completed, "status": "COMPLETE",
               "selected_total": len(selected), "excluded": len(exclusions)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mfa", default="/home/wjj/.venvs/mfa/bin/mfa")
    p.add_argument("--dictionary", default="/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict")
    p.add_argument("--acoustic", default="/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--batch-clips", type=int, default=500)
    p.add_argument("--timeout", type=int, default=7200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
