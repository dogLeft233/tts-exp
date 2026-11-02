"""Independently check saved RF clocks and every proposed phrase-patch index."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

root = Path(sys.argv[1])
def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))
def sha(name):
    return hashlib.sha256((root / name).read_bytes()).hexdigest()

rows = load("input_manifest.json")["rows"]
align = {x["id"]: x for x in load("alignment.json")["audio"]}
windows = {x["id"]: x for x in load("window_clock_audit.json")["rows"]}
patches = load("patch_index.json")["patches"]
assert len(rows) == len(align) == len(windows) == 140
centers = {}
replays = {}
variant_checks = 0
valid_by_id = defaultdict(set)
all_variants_valid = defaultdict(set)
for item in rows:
    key = item["id"]
    rec = load(f"items/{key}/replay.json")
    replays[key] = rec
    seq = windows[key]["rows"]
    assert len(seq) == rec["windows"] == item["score_windows"]
    assert align[key]["audio_sha256"] == item["audio_sha256"]
    assert len(align[key]["tokens"]) == len(align[key]["timestamps_ms"])
    last_end = 0
    for lo, hi in align[key]["timestamps_ms"]:
        assert 0 <= last_end <= lo < hi <= item["features"]["audio_samples"] / 16
        last_end = hi
    expected_centers = []
    for i, w in enumerate(seq):
        lo, hi = max(0, i*640-1), i*640+3439
        expected = (lo+hi)/32000
        assert w["row"] == i
        assert w["raw_samples_inclusive"] == [lo, hi]
        assert w["mfcc_columns_halfopen"] == [4*i, 4*i+20]
        assert w["video_frames_halfopen"] == [i, i+5]
        assert abs(w["score_audio_center_s"]-expected) < 1e-12
        assert hi < rec["audio_samples"]
        expected_centers.append(expected)
    centers[key] = expected_centers
for p in patches:
    key, char = p["id"], p["character_index"]
    a = align[key]
    assert p["character"] == a["tokens"][char]
    interval = a["timestamps_ms"][char]
    assert p["canonical_interval_ms"] == interval
    center = sum(interval)/2000
    assert abs(center-p["canonical_center_s"]) < 1e-12
    delay = replays[key]["clock"]["lag_samples_scoring_minus_canonical"] / 16000
    assert abs(delay-p["score_delay_s"]) < 1e-12
    assert p["fa_truth_verified"] is False
    assert len(p["variants"]) == 3
    statuses = []
    for shift, v in zip((-.04, 0., .04), p["variants"]):
        assert shift == v["shift_s"]
        anchor = center + delay + shift
        idx = min(range(len(centers[key])), key=lambda i: abs(centers[key][i]-anchor))
        assert v["nearest_row"] == idx
        assert abs(v["mapped_anchor_s"] - anchor) < 1e-12
        assert abs(v["nearest_center_s"]-centers[key][idx]) < 1e-12
        assert abs(v["center_error_ms"]-(centers[key][idx]-anchor)*1000) < 1e-8
        good = idx >= 5 and idx+5 < len(centers[key])
        assert v["support_ok"] == good
        assert v["patch_rows"] == (list(range(idx-5, idx+6)) if good else None)
        expected = [max(0, (idx-5)*640-1), (idx+5)*640+3439] if good else None
        assert v["patch_raw_samples_inclusive"] == expected
        if good and shift == 0:
            valid_by_id[key].add(char)
        statuses.append(good)
        variant_checks += 1
    if all(statuses):
        all_variants_valid[key].add(char)
by_source = defaultdict(list)
for item in rows:
    by_source[item["sample_id"]].append(item["id"])
qualification = []
for source, ids in by_source.items():
    assert len(ids) == 7
    token_lists = [align[k]["tokens"] for k in ids]
    same_tokens = all(x == token_lists[0] for x in token_lists)
    common = set.intersection(*(valid_by_id[k] for k in ids)) if same_tokens else set()
    robust = set.intersection(*(all_variants_valid[k] for k in ids)) if same_tokens else set()
    denom = len(token_lists[0])
    qualification.append({"source": source, "same_tokens": same_tokens,
                          "common_7arm_anchors": len(common), "characters": denom,
                          "common_coverage": len(common)/denom,
                          "all_shifts_common_anchors": len(robust),
                          "temporary_5anchor_80percent_gate": len(common)>=5 and len(common)/denom>=.8})
result = {"status":"PASS_INDEX_ARITHMETIC_ONLY", "audio_truth_verified":False,
          "official_rows":sum(len(x) for x in centers.values()),
          "anchors":len(patches), "variants_checked":variant_checks,
          "source_qualification":qualification,
          "source_gate_pass":sum(x["temporary_5anchor_80percent_gate"] for x in qualification),
          "inputs":{n:sha(n) for n in ("input_manifest.json","alignment.json","window_clock_audit.json","patch_index.json")}}
output = root/"parent_independent_window_audit.json"
assert not output.exists()
output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
