"""Fresh parent raw-byte/PCM/split audit. Standard library only; no models."""
import argparse
from array import array
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import unicodedata
import wave


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


def audit(root, side, plan, anchors):
    started = time.time()
    mpath = root / "selected600_natural_manifest.json"
    freeze = load(root / "source_split_freeze.json")
    assert digest(mpath) == freeze["source_manifest_sha256"]
    assert digest(plan) == freeze["candidate_plan_sha256"]
    rows = load(mpath)["rows"]
    assert len(rows) == len({r["source_id"] for r in rows}) == 600
    roster = {r["speaker_id"]: r for r in load(plan)["rows"]}
    assert len(roster) == 200
    groups, pcm_groups, texts = defaultdict(list), defaultdict(list), []
    total_bytes, total_seconds = 0, 0.
    for r in rows:
        speaker = r["speaker_id"]
        assert r["split"] == roster[speaker]["split"]
        assert r["ref_source_id"] == r["source_id"]
        assert r["source_id"][6:11] == speaker
        assert r["member"].startswith("train/"+speaker+"/")
        path = Path(r[side+"_path"])
        raw = path.read_bytes()
        assert len(raw) == r["bytes"]
        assert hashlib.sha256(raw).hexdigest() == r["sha256"], r["source_id"]
        with wave.open(io.BytesIO(raw), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            assert w.getnframes() == r["frames"]
            pcm = w.readframes(w.getnframes())
        assert hashlib.sha256(pcm).hexdigest() == r["pcm_sha256"]
        samples = array("h", pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        duration = len(samples)/16000
        assert 3 <= duration <= 12 and duration == r["duration_s"]
        rms = (sum(x*x for x in samples)/len(samples))**.5/32768
        assert rms > 0 and abs(rms-r["rms"]) < 1e-12
        assert max(abs(x) for x in samples)/32768 == r["peak"]
        assert sum(x in (-32768, 32767) for x in samples) == r["clip_samples"] == 0
        text = normalized(r["text"])
        assert text and text == normalized(r["normalized_text"])
        groups[speaker].append(r)
        pcm_groups[r["pcm_sha256"]].append(r)
        texts.append(text)
        total_bytes += len(raw)
        total_seconds += duration
    assert set(groups) == set(roster)
    assert all(len(g) == 3 for g in groups.values())
    assert len(pcm_groups) == 600
    assert Counter(g[0]["split"] for g in groups.values()) == {"fit": 100, "development": 40, "confirmation": 60}
    near = []
    for i, a in enumerate(texts):
        for j in range(i):
            b = texts[j]
            if a == b or (min(len(a), len(b)) >= 8 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= .9):
                near.append({"a": rows[i]["source_id"], "b": rows[j]["source_id"],
                             "cross_split": rows[i]["split"] != rows[j]["split"]})
    assert not near
    old_ids = {r["sample_id"] for r in load(anchors)["rows"] if r["arm"] == "N"}
    assert len(old_ids) == 40
    assert old_ids <= {r["source_id"] for r in rows if r["split"] == "fit"}
    for role, rec in freeze["role_manifests"].items():
        assert digest(root / rec["path"]) == rec["sha256"]
        role_manifest = load(root / rec["path"])
        role_rows = role_manifest if isinstance(role_manifest, list) else role_manifest["rows"]
        assert {r["source_id"] for r in role_rows} == {r["source_id"] for r in rows if r["split"] == role}
    return {"schema": "parent_audio200_readback_v1", "status": "PASS_RAW_PCM_AND_SOURCE_SPLIT",
            "side": side, "speakers": len(groups), "sources": len(rows),
            "split_speakers": dict(Counter(g[0]["split"] for g in groups.values())),
            "split_sources": dict(Counter(r["split"] for r in rows)),
            "files_rehashed": len(rows), "PCM_rehashed": len(rows), "all_near_text_pairs_checked": 179700,
            "near_text_edges": near, "PCM_duplicates": 0, "self_reference_identity_pass": True,
            "old_N40_preserved_in_fit": True, "raw_bytes": total_bytes, "duration_s": total_seconds,
            "source_manifest_sha256": digest(mpath), "code_sha256": digest(__file__),
            "training": False, "models_run": False, "FA_truth": "UNVERIFIED",
            "source_originality_independently_validated_all600": False,
            "media4200_production": "HOLD", "elapsed_s": time.time()-started}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--side", choices=("local", "remote"), required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    result = audit(args.root, args.side, args.plan, args.anchors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
