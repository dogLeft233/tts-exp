"""CPU-only, read-only audio smoke for pinned FunASR ASR and text alignment.

Input rows: source, sha256, transcript, id, model. All timestamps are milliseconds.
This validates machine output structure, not human alignment accuracy.
"""
import argparse
import difflib
import hashlib
import importlib.metadata
import json
import pathlib
import sys
import time
import unicodedata
import wave


def normalize(text):
    mapping = []
    chars = []
    for index, char in enumerate(text):
        if not char.isspace() and not unicodedata.category(char).startswith("P"):
            chars.append(char.lower())
            mapping.append(index)
    return {"original": text, "normalized": "".join(chars), "original_indices": mapping}


def validate(timestamps, tokens, duration_ms):
    problems = []
    if not timestamps:
        problems.append("empty_timestamps")
    if len(timestamps) != len(tokens):
        problems.append("token_timestamp_count_mismatch")
    previous = None
    for pair in timestamps:
        if len(pair) != 2:
            problems.append("invalid_pair")
            continue
        start, end = pair
        if not 0 <= start <= end <= duration_ms:
            problems.append("out_of_audio_range")
        if previous is not None and start < previous:
            problems.append("nonmonotonic_or_overlap")
        previous = end
    return sorted(set(problems))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))["audio"]
    if len(rows) > 8:
        raise ValueError("Deployment smoke is capped at eight files")
    import torch
    from funasr import AutoModel
    torch.set_num_threads(2)
    root = pathlib.Path(args.model_root)
    started = time.perf_counter()
    asr = AutoModel(model=str(root / "paraformer-zh"), device="cpu", disable_update=True)
    aligner = AutoModel(model=str(root / "fa-zh"), device="cpu", disable_update=True)
    result = {"device": "cpu", "timestamp_unit": "ms", "argv": sys.argv,
              "versions": {n: importlib.metadata.version(n) for n in
                           ["funasr", "torch", "torchaudio", "modelscope", "numpy"]},
              "model_load_s": time.perf_counter() - started, "audio": []}
    for row in rows:
        path = pathlib.Path(row["source"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
        with wave.open(str(path)) as audio:
            assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
            duration_ms = audio.getnframes() / 16
        given = normalize(row["transcript"])
        t0 = time.perf_counter()
        recognized = asr.generate(input=str(path), output_timestamp=True)
        asr_s = time.perf_counter() - t0
        # Official fa-zh API accepts an audio path plus literal text with this data_type.
        t0 = time.perf_counter()
        aligned = aligner.generate(input=(str(path), given["normalized"]), data_type=("sound", "text"))
        fa_s = time.perf_counter() - t0
        predicted = normalize(recognized[0]["text"])
        operations = [list(x) for x in difflib.SequenceMatcher(
            a=given["normalized"], b=predicted["normalized"], autojunk=False).get_opcodes()]
        entry = dict(row, duration_ms=duration_ms, given_text=given, asr_text=predicted,
                     asr=recognized, alignment=aligned, asr_seconds=asr_s, fa_seconds=fa_s,
                     transcript_exact_match=given["normalized"] == predicted["normalized"],
                     transcript_diff_opcodes=operations)
        entry["asr_issues"] = validate(recognized[0].get("timestamp", []), list(predicted["normalized"]), duration_ms)
        returned = normalize(aligned[0].get("text", ""))
        entry["fa_returned_text"] = returned
        entry["fa_tokens"] = list(returned["normalized"])
        entry["fa_issues"] = validate(aligned[0].get("timestamp", []), entry["fa_tokens"], duration_ms)
        if returned["normalized"] != given["normalized"]:
            entry["fa_issues"].append("returned_text_differs_from_given_text")
        result["audio"].append(entry)
        print(json.dumps({"id": row["id"], "model": row["model"], "asr": predicted["normalized"],
                          "match": entry["transcript_exact_match"], "fa_issues": entry["fa_issues"]}, ensure_ascii=False), flush=True)
    result["total_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
