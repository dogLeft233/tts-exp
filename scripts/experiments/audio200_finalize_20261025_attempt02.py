"""CPU-only, manifest-driven final acceptance of source200/4200 data.

No fitting, scoring, generation, rank/effect summaries, or implicit retries.
All frozen producer batches and their independently reviewed parent audits must
already exist. Only metadata/pointers are materialized; large caches stay put.
"""
from __future__ import annotations
import argparse
import copy
from collections import Counter, defaultdict
import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import time

SOURCE_SHA="3ef28c8076aba23e64e2bc1a22963e24dba1e9a89ed9e2aeaf7fdaae6821e608"
OLD_SHA="046b4cbb1a53fc11b2455f90eb943a7cfbac57c80e6ec34039a4f00b0b865acf"
SPLIT_SHA="3ffb6d516470d5185a71e18d5bd8390b3180de7d6178640a2726c5eaef6fc749"
ARMS={("N",0)}|{(a,d) for a in ("L06","L17") for d in (1,2,3)}
FOLDERS={"fit":"fit","development":"development","confirmation":"confirmation_sealed"}
EXPECTED_UNITS={"fit":2100,"development":840,"confirmation":1260}
EXPECTED_PAIRS={"fit":4500,"development":1800,"confirmation":2700}
AUDITOR_SHA="fa66be744b1a35e01d76749f6fd227d950b79232d61d4483766c9bf775ed198d"
CANONICAL_LABEL_FIELD="official_syncnet_score_float32"
ORIGINAL_PRODUCER_PROTOCOL_SHA="f2d591056ec61f67ebe6bbf2b26a5260bc0b95eb24257de32e2882e34c61dc5e"


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class FreshPins:
    """Read each referenced file at least once in this finalizer invocation."""
    def __init__(self):self.observed={}
    def pin(self,path,expected=None):
        p=Path(path);key=str(p)
        if key not in self.observed:
            self.observed[key]=dict(sha256=sha(p),bytes=p.stat().st_size)
        digest=self.observed[key]["sha256"]
        assert expected is None or digest==expected,(key,"SHA mismatch")
        return digest


def under_material_batches(path,roots):
    p=Path(path)
    assert p.is_absolute()
    assert any(p.resolve().is_relative_to((Path(root)/"batches").resolve()) for root in roots), (str(p),"outside exact declared material roots")


def assert_route_only_delta(protocol,expected):
    additions={"original_protocol_path","original_protocol_sha256","material_roots","reviewed_recovery_path","reviewed_recovery_sha256"}
    assert set(protocol)==set(expected)|additions,"unexpected recovered protocol keys"
    for key,value in expected.items():
        assert protocol[key]==value,(key,"recovery may only change run and reviewed batch003 N/media routes")


def validate_recovery_protocol(protocol,run,pins):
    """Allow exactly the reviewed routing repair, never a scientific change."""
    run=Path(run)
    original_path=Path(protocol["original_protocol_path"])
    assert protocol["original_protocol_sha256"]==ORIGINAL_PRODUCER_PROTOCOL_SHA
    pins.pin(original_path,ORIGINAL_PRODUCER_PROTOCOL_SHA)
    original=read(original_path);old=Path(original["run"])
    assert original_path==old/"protocol.json" and old!=run
    roots=protocol["material_roots"]
    assert roots==[str(old),str(run)] and all(Path(p).is_absolute() for p in roots)
    assert old.parent==run.parent==Path(protocol["project_root"])/"runs"
    review_path=Path(protocol["reviewed_recovery_path"])
    assert review_path==run/"reviewed_recovery.json"
    pins.pin(review_path,protocol["reviewed_recovery_sha256"])
    review=read(review_path)
    assert review["schema"]=="audio200_reviewed_gpu_guard_recovery_v1"
    assert review["status"]=="PREPARED_CPU_REUSE_VERIFIED_PENDING_PARENT_RELEASE"
    assert review["old_run"]==str(old) and review["new_run"]==str(run)
    assert review["material_roots"]==roots
    assert review["failed_batch"]=="batch003" and review["failed_arm"]=="N"
    assert len(review["group_routes"])==1
    route=review["group_routes"][0]
    assert route["batch_id"]=="batch003" and route["arm"]=="N"
    expected=copy.deepcopy(original);expected["run"]=str(run)
    batch=next(b for b in expected["batches"] if b["id"]=="batch003")
    group=next(g for g in batch["groups"] if g["arm"]=="N")
    old_group=Path(group["path"]);new_group=run/old_group.relative_to(old)
    old_media=Path(batch["media_manifest_path"]);new_media=run/old_media.relative_to(old)
    assert route["old_group_path"]==str(old_group) and route["new_group_path"]==str(new_group)
    assert route["old_media_manifest_path"]==str(old_media) and route["new_media_manifest_path"]==str(new_media)
    assert route["protocol_sha256"]==group["protocol_sha256"]
    group["path"]=str(new_group);batch["media_manifest_path"]=str(new_media)
    assert_route_only_delta(protocol,expected)
    copy_names=("protocol.json","raw_manifest.json","audio_manifest.json","video_manifest.json")
    assert set(review["input_copy_pins"])=={str(new_group/name) for name in copy_names}
    for name in copy_names:
        digest=pins.pin(old_group/name)
        assert review["input_copy_pins"][str(new_group/name)]==digest
        pins.pin(new_group/name,digest)
    failed_stage=old/"stage_receipts/batch003_N_official.json"
    failed_summary=old_group/"official/summary.json"
    assert {str(failed_stage),str(failed_summary)}<=set(review["failure_pins"])
    for path,digest in review["failure_pins"].items():pins.pin(path,digest)
    assert read(failed_stage)["status"]=="FAILED"
    failed=read(failed_summary)
    assert failed["status"]!="COMPLETE" and failed["completed"]==0 and failed["failed"]==0 and failed["pending"]==10
    # The official wrapper pre-populates NOT_RUN records while remuxing, before
    # its GPU guard. Their existence is not evidence that scoring occurred.
    assert len(failed.get("records",[]))==10
    assert all(r["status"]=="NOT_RUN" and r.get("sync_c") is None and r.get("sync_d") is None and not r.get("tracks") for r in failed["records"]),"guard exit must not hide completed scores"
    assert not list((old_group/"official/scores").glob("*.json")),"old failed attempt must not contain accepted score JSON"
    stages={f'{b["id"]}_{g["arm"]}_{s}' for b in original["batches"] for g in b["groups"] for s in ("synthesize","canonical","render","official")}
    assert len(stages)==57*3*4 and set(review["stage_receipt_map"])==stages
    copied={f'batch{i:03d}_{a}_{s}' for i in range(3) for a in ("N","L06","L17") for s in ("synthesize","canonical","render","official")}
    copied|={f'batch003_N_{s}' for s in ("synthesize","canonical","render")}
    for key,entry in review["stage_receipt_map"].items():
        target=run/"stage_receipts"/(key+".json")
        assert entry["path"]==str(target)
        has_origin="origin_path" in entry
        assert has_origin==(key in copied),(key,"only already completed stages may be inherited")
        if has_origin:
            origin=old/"stage_receipts"/(key+".json")
            assert entry["origin_path"]==str(origin)
            digest=entry["origin_sha256"]
            pins.pin(origin,digest);pins.pin(target,digest)
            assert read(origin)["status"]=="COMPLETE"
        else:assert "origin_sha256" not in entry
    consumers=review["consumer_receipt_map"]
    assert set(consumers)=={f"batch{i:03d}" for i in range(57)}
    for key,entry in consumers.items():
        target=run/"consumer_receipts"/(key+".json")
        assert entry["path"]==str(target)
        has_origin="origin_path" in entry
        assert has_origin==(key in {"batch000","batch001","batch002"})
        if has_origin:
            origin=old/"consumer_receipts"/(key+".json")
            assert entry["origin_path"]==str(origin)
            pins.pin(origin,entry["origin_sha256"]);pins.pin(target,entry["origin_sha256"])
            assert read(origin)["status"]=="COMPLETE"
        else:assert "origin_sha256" not in entry
    for b in protocol["batches"]:
        under_material_batches(b["media_manifest_path"],roots)
        for g in b["groups"]:under_material_batches(g["path"],roots)
    return review


def add_source_fields(row,source,reused):
    """Keep every historical key/value, including original_cache_reused."""
    out=dict(row)
    additions=dict(split=source["split"],ref_source_id=source["ref_source_id"],
        natural_source_sha256=source["sha256"],natural_source_pcm_sha256=source["pcm_sha256"],
        reused_in_source200_preparation=reused,status="COMPLETE_DATA_PREPARATION_ONLY")
    for k,v in additions.items():
        if k in out:assert out[k]==v,(row["id"],k,"cannot rewrite historical metadata")
        else:out[k]=v
    for k,v in row.items():assert out[k]==v
    return out


def validate_reuse_binding(reuse,old_rows):
    """Producer names the old feature registry its source manifest here.

    This field is intentionally OLD_SHA, whereas the separate natural600
    manifest is independently bound to SOURCE_SHA by the producer protocol.
    """
    assert reuse["rows"]==old_rows
    assert reuse["source_manifest_sha256"]==OLD_SHA
    assert len(old_rows)==len({r["id"] for r in old_rows})==280


def raw_generation_bindings(group,pins):
    """Read real raw/canonical manifests and preserve EOS's unknown state."""
    folder=Path(group["path"]);gp=read(folder/"protocol.json")
    pins.pin(folder/"protocol.json",group["protocol_sha256"])
    raw_path=folder/"raw_manifest.json";raw_sha=pins.pin(raw_path);raw=read(raw_path)
    canonical_path=folder/"audio_manifest.json";pins.pin(canonical_path);canonical=read(canonical_path)
    assert raw["count"]==len(raw["records"])==gp["expected_cells"]
    assert canonical["count"]==len(canonical["records"])==gp["expected_cells"]
    assert canonical["protocol_sha256"]==group["protocol_sha256"]
    if "protocol_sha256" in raw:assert raw["protocol_sha256"]==group["protocol_sha256"]
    can={(r["sample_id"],r["role"]):r for r in canonical["records"]}
    samples={s["sample_id"]:s for s in gp["samples"]}
    expected={(sid,role) for sid in samples for role in gp["roles"]}
    assert set(can)==expected and len(can)==len(canonical["records"])
    observed={};arm=group["arm"]
    for index,r in enumerate(raw["records"]):
        sid,role=r["sample_id"],r["role"];key=(sid,role)
        assert key in expected and key not in observed
        c=can[key];s=samples[sid]
        assert r["speaker_id"]==c["speaker_id"]==s["speaker_id"]
        assert c["source_path"]==r["path"] and c["source_sha256"]==r["sha256"]
        pins.pin(r["path"],r["sha256"]);pins.pin(c["path"],c["sha256"])
        draw=0 if arm=="N" else int(role[-1]);uid=f"{sid}_{arm}_{draw}"
        info=dict(canonical_audio_path=c["path"],canonical_audio_sha256=c["sha256"],
            raw_audio_path=r["path"],raw_audio_sha256=r["sha256"],
            raw_generation_manifest_path=str(raw_path),raw_generation_manifest_sha256=raw_sha,
            raw_generation_record=dict(index=index,sample_id=sid,role=role))
        if arm=="N":
            assert role=="N"
            info.update(generation_qa_status="not_applicable_natural",eos_status="not_applicable_natural")
        else:
            assert arm in ("L06","L17") and r["model"]==arm and r["draw_index"]==draw
            assert r["seed"]==[2026102201,2026102202,2026102203][draw-1]+100*int(s["speaker_id"][1:])
            assert r["target_text"]==r["reference_text"]=="".join(s["transcript"].split())
            assert r["reference_audio_sha256"]==s["input24_sha256"]
            eos=r.get("eos_in_talker_sequences")
            assert eos is None or type(eos) is bool
            info.update(generation_qa_status="record_bound_not_a_perceptual_quality_label",
                eos_status="unknown" if eos is None else ("present" if eos else "absent"),
                eos_in_talker_sequences=eos,duration_ratio=r["duration_ratio"],seed=r["seed"])
            record_path=folder/"generation"/sid/f"draw{draw}.json"
            pins.pin(record_path);assert read(record_path)==r
            info.update(raw_generation_record_path=str(record_path),raw_generation_record_sha256=pins.pin(record_path))
            for field in ("reference_codes","new_codes","decode_concat_codes","full_decoded","talker_sequences"):
                desc=r[field];pins.pin(desc["path"],desc["sha256"])
        observed[key]=info
    assert set(observed)==expected
    return {f"{sid}_{arm}_{0 if arm=='N' else int(role[-1])}":info for (sid,role),info in observed.items()}


def attach_fields(row,extra):
    result=dict(row)
    for k,v in extra.items():
        if k in result:assert result[k]==v,(row["id"],k,"binding drift")
        else:result[k]=v
    return result


def attach_canonical_score(row,official,expected_legacy):
    """Retain the historical column; give future consumers one label field."""
    assert row["score_float32"]==expected_legacy,(row["id"],"legacy reduction identity changed")
    provenance="numpy_float32_curve_reduction" if row["reused_in_source200_preparation"] else "torch_float32_curve_reduction"
    return attach_fields(row,{CANONICAL_LABEL_FIELD:official["C"],
        "score_float32_provenance":provenance,
        "official_syncnet_score_provenance":"torch_float32_original_activesd_mean0_median_minus_min",
        "legacy_vs_official_score_abs_difference":abs(row["score_float32"]-official["C"])})


def canonical_score_row(row):
    """Exact CPU replay of frozen engine.scalar(original), no model forward."""
    import pickle
    import numpy as np
    import torch
    assert torch.__version__=="2.5.1+cu124", "Use the frozen official scoring environment"
    torch.set_num_threads(2)
    original=np.asarray(pickle.loads(Path(row["activesd_path"]).read_bytes())[0])
    assert original.dtype==np.float32 and original.shape==(row["rows"],31)
    assert np.isfinite(original).all()
    x=torch.from_numpy(original);mu=x.mean(0);minimum,idx=mu.min(0);median=torch.median(mu)
    official=dict(C=float(median-minimum),D=float(minimum),B=float(median),offset=int(15-idx))
    replay=read(row["replay_path"])
    for field,value in official.items():assert replay["original"][field]==value,(row["id"],field,"official scalar mismatch")
    if row["reused_in_source200_preparation"]:
        legacy_mu=original.mean(0)
        legacy=float(np.median(legacy_mu)-legacy_mu.min())
    else:legacy=official["C"]
    return attach_canonical_score(row,official,legacy)


def validate_full_roster(rows,sources):
    assert len(rows)==len({r["id"] for r in rows})==4200
    groups=defaultdict(list);speakers=defaultdict(set);counts=Counter()
    for r in rows:
        sid=r["sample_id"];s=sources[sid]
        assert r["id"]==f'{sid}_{r["arm"]}_{r["draw"]}'
        assert r["speaker_id"]==s["speaker_id"] and r["split"]==s["split"]
        assert "".join(r["text"].split())==s["normalized_text"]
        assert r["ref_source_id"]==s["source_id"]==sid
        assert r["natural_source_sha256"]==s["sha256"]
        groups[sid].append(r);speakers[r["speaker_id"]].add(r["split"]);counts[r["split"]]+=1
    assert len(groups)==600 and len(speakers)==200 and all(len(x)==1 for x in speakers.values())
    assert dict(counts)==EXPECTED_UNITS
    for group in groups.values():assert len(group)==7 and {(r["arm"],r["draw"]) for r in group}==ARMS
    for split,expected in (("fit",100),("development",40),("confirmation",60)):
        assert sum(split in v for v in speakers.values())==expected
    return groups


def expected_pairs(group):
    ids={(r["arm"],r["draw"]):r["id"] for r in group};pairs=[]
    assert len(group)==len(ids)==7 and set(ids)==ARMS
    for d in (1,2,3):
        for f,a,b in (("N-L06",("N",0),("L06",d)),("N-L17",("N",0),("L17",d)),("L17-L06",("L17",d),("L06",d))):
            pairs.append((f,ids[a],ids[b]))
    for a in ("L06","L17"):
        for x,y in combinations((1,2,3),2):pairs.append((a+"-within",ids[a,x],ids[a,y]))
    return pairs


def verify_index_part(index,rows):
    """Check exact fixed pair identities and derive true common support."""
    byid={r["id"]:r for r in rows};groups=defaultdict(list)
    for r in rows:groups[r["sample_id"]].append(r)
    valid={uid:set() for uid in byid};robust={uid:set() for uid in byid}
    seen=set()
    for p in index["patches"]:
        uid,ci=p["id"],p["character_index"]
        assert uid in byid and (uid,ci) not in seen;seen.add((uid,ci))
        variants=p["variants"];assert len(variants)==3
        assert [v["shift_s"] for v in variants]==[-.04,0.,.04]
        for v in variants:
            if v["support_ok"]:
                j=v["nearest_row"];n=byid[uid]["rows"]
                assert j>=5 and j+5<n and v["patch_rows"]==list(range(j-5,j+6))
        if variants[1]["support_ok"]:
            text="".join(byid[uid]["text"].split())
            assert 0<=ci<len(text) and p["character"]==text[ci]
            valid[uid].add(ci)
        if all(v["support_ok"] for v in variants):robust[uid].add(ci)
    actual={(x["family"],x["a_id"],x["b_id"]):x for x in index["pair_coverage"]}
    assert len(actual)==len(index["pair_coverage"]),"duplicate pair index"
    expected={p for g in groups.values() for p in expected_pairs(g)}
    assert set(actual)==expected,"must use 9 matched-draw primary +6 within-model pairs"
    pairs=[];support=[]
    for source,g in groups.items():
        den=len("".join(g[0]["text"].split()))
        common=set.intersection(*(valid[r["id"]] for r in g))
        common_robust=set.intersection(*(robust[r["id"]] for r in g))
        individual=[len(valid[r["id"]])>=5 and len(valid[r["id"]])/max(den,1)>=.8 for r in g]
        support.append(dict(source=source,characters=den,all_clips_individually_pass=all(individual),
            common_7arm_anchors=len(common),common_character_indices=sorted(common),
            common_7arm_coverage=len(common)/max(den,1),common_7arm_gate=len(common)>=5 and len(common)/max(den,1)>=.8,
            all_shifts_common_anchors=len(common_robust)))
        for family,a,b in expected_pairs(g):
            cc=sorted(valid[a]&valid[b]);frozen=actual[family,a,b]
            assert frozen["common_character_indices"]==cc
            gate=len(cc)>=5 and len(cc)/max(den,1)>=.8
            assert frozen["temporary_measurability_gate"]==gate
            pairs.append(dict(source=source,family=family,a_id=a,b_id=b,common_character_indices=cc,
                text_characters=den,coverage=len(cc)/max(den,1),temporary_measurability_gate=gate))
    return pairs,support


def check_parent_audit(audit,batch,manifest_path,manifest_sha,completion_path,completion_sha):
    assert audit["status"]=="PASS_PARENT_BATCH_REPLAY_CLOCK_PATCH",(batch["id"],"parent independent audit not PASS")
    # Parent controller schema binds exact batch and immutable artifacts. Generic
    # PASS files without these identities do not qualify.
    assert audit["batch_id"]==batch["id"]
    assert len(audit["source_ids"])==len(set(audit["source_ids"])) and set(audit["source_ids"])==set(batch["source_ids"])
    assert Path(audit["manifest_path"])==Path(manifest_path) and audit["manifest_sha256"]==manifest_sha
    assert Path(audit["clock_path"])==Path(manifest_path).parent/"window_clock.json"
    assert Path(audit["patch_path"])==Path(manifest_path).parent/"patch_index.json"
    assert Path(audit["consumer_completion_path"])==Path(completion_path)
    assert audit["consumer_completion_sha256"]==completion_sha
    assert audit["auditor_source_sha256"]==AUDITOR_SHA
    assert Path(audit["auditor_source_path"]).is_absolute()


def finalize(args):
    run,audit_root,out=Path(args.producer_run),Path(args.parent_audit_root),Path(args.output_root)
    assert not out.exists(),"finalizer outputs are immutable; choose a new reviewed attempt"
    out.mkdir(parents=True)
    put(out/"launch.json",dict(utc=utc(),command=sys.argv,source_sha256=sha(__file__),no_training=True))
    started=time.time();pins=FreshPins()
    try:
        assert shutil.disk_usage(out).free>=5*1024**3
        protocol_path=run/"protocol.json";protocol=read(protocol_path);pins.pin(protocol_path)
        assert protocol["run"]==str(run) and protocol["no_training"] is True
        recovery=validate_recovery_protocol(protocol,run,pins)
        assert protocol["expected_units"]==4200 and protocol["reused_units"]==280 and protocol["new_units"]==3920
        batches=protocol["batches"]
        assert len(batches)==len({b["id"] for b in batches})==57
        assert [b["id"] for b in batches]==[f"batch{i:03d}" for i in range(57)]
        for p,h in protocol["source_pins"].items():pins.pin(p,h)
        for bundle in protocol["bundles"].values():
            for key in ("source_pins","model_pins"):
                for p,h in bundle[key].items():pins.pin(p,h)
        source_path=Path(protocol["source_manifest_path"]);pins.pin(source_path,SOURCE_SHA)
        sources={r["source_id"]:r for r in read(source_path)["rows"]};assert len(sources)==600
        split_path=source_path.parent/"source_split_freeze.json";pins.pin(split_path,SPLIT_SHA)
        assert len({s["pcm_sha256"] for s in sources.values()})==600
        source_by_speaker=defaultdict(list)
        for sid,s in sources.items():
            assert sid==s["source_id"]==s["ref_source_id"]
            source_by_speaker[s["speaker_id"]].append(s)
            pins.pin(s["remote_path"],s["sha256"])
        assert len(source_by_speaker)==200 and all(len(g)==3 and len({s["split"] for s in g})==1 for g in source_by_speaker.values())
        old_path=Path(protocol["reuse_manifest_path"]);pins.pin(old_path,OLD_SHA)
        old_rows=read(old_path)["rows"]
        reuse_path=run/"reuse280.json";pins.pin(reuse_path)
        reuse=read(reuse_path)
        validate_reuse_binding(reuse,old_rows)
        all_rows=[attach_fields(add_source_fields(r,sources[r["sample_id"]],True),dict(
            generation_qa_status="legacy_format_not_reaggregated",legacy_input_registry_path=str(old_path),
            legacy_input_registry_sha256=OLD_SHA,legacy_source_registry_path=str(reuse_path),
            legacy_source_registry_sha256=pins.pin(reuse_path))) for r in old_rows]
        old_sources={r["sample_id"] for r in old_rows}
        assert len(old_rows)==280 and len(old_sources)==40
        batch_sources=[sid for b in batches for sid in b["source_ids"]]
        assert len(batch_sources)==len(set(batch_sources))==560
        assert not old_sources & set(batch_sources) and old_sources|set(batch_sources)==set(sources)
        index_parts=[];batch_receipts=[]
        old_base=old_path.parent
        old_index=old_base/"dataset280_patch_index.json";old_clock=old_base/"dataset280_window_clock_audit.json"
        pins.pin(old_index);pins.pin(old_clock)
        index_parts.append(dict(split="fit",batch_id="legacy40",rows=all_rows[:],patch_path=str(old_index),clock_path=str(old_clock),alignment_paths=sorted({r["FA"]["path"] for r in old_rows})))
        for b in batches:
            media_path=Path(b["media_manifest_path"])
            under_material_batches(media_path,protocol["material_roots"])
            media_sha=pins.pin(media_path);media=read(media_path)["rows"]
            assert len(media)==len(b["source_ids"])*7
            assert {r["sample_id"] for r in media}==set(b["source_ids"])
            assert all(r["split"]==b["split"] and r["status"]=="COMPLETE" for r in media)
            media_byid={r["id"]:r for r in media};assert len(media_byid)==len(media)
            generation_bindings={}
            for g in b["groups"]:
                pins.pin(Path(g["path"])/"protocol.json",g["protocol_sha256"])
                for stage in ("synthesize","canonical","render","official"):
                    stage_key=f'{b["id"]}_{g["arm"]}_{stage}'
                    stage_path=Path(recovery["stage_receipt_map"][stage_key]["path"])
                    pins.pin(stage_path);stage_rec=read(stage_path)
                    assert stage_rec["status"]=="COMPLETE" and stage_rec["returncode"]==0 and stage_rec["no_training"] is True
                    for path,digest in stage_rec["artifacts"].items():pins.pin(path,digest)
                binding=raw_generation_bindings(g,pins)
                assert not set(generation_bindings)&set(binding)
                generation_bindings.update(binding)
            assert set(generation_bindings)==set(media_byid)
            features=media_path.parent/"features";completion_path=features/"completion.json"
            completion_sha=pins.pin(completion_path);completion=read(completion_path)
            assert completion["status"]=="COMPLETE_FEATURE_FA_WINDOW_DATA_ONLY" and completion["no_training"] is True
            assert completion["media_manifest_sha256"]==media_sha
            assert completion["units"]==completion["new_forward"]==len(media) and completion["reused"]==0
            for f in completion["files"]:
                p=features/f["path"];assert p.resolve().is_relative_to(features.resolve())
                pins.pin(p,f["sha256"]);assert p.stat().st_size==f["bytes"]
            receipt_path=Path(recovery["consumer_receipt_map"][b["id"]]["path"])
            pins.pin(receipt_path);receipt=read(receipt_path);assert receipt["status"]=="COMPLETE"
            for p,h in receipt["artifacts"].items():pins.pin(p,h)
            command=receipt["command"]
            contract_path=Path(command[command.index("--contract")+1]);contract=read(contract_path)
            pins.pin(contract_path,completion["consumer_contract_sha256"])
            pins.pin(command[1],completion["consumer_source_sha256"])
            assert contract["consumer_source_sha256"]==completion["consumer_source_sha256"]
            for p,h in contract["engine_source_pins"].items():pins.pin(p,h)
            for p,h in contract["fa_pins"].items():pins.pin(p,h)
            pins.pin(contract["engine_path"],contract["engine_sha256"])
            pins.pin(contract["legacy_parser_path"],contract["legacy_parser_sha256"])
            audit_path=audit_root/(b["id"]+".json");pins.pin(audit_path)
            audit=read(audit_path)
            assert len(completion["split_manifests"])==1,"producer batches must be single split"
            ms=completion["split_manifests"][0]
            check_parent_audit(audit,b,ms["path"],ms["sha256"],completion_path,completion_sha)
            pins.pin(audit["manifest_path"],audit["manifest_sha256"])
            pins.pin(audit["clock_path"],audit["clock_sha256"])
            pins.pin(audit["patch_path"],audit["patch_sha256"])
            auditor_path=Path(audit["auditor_source_path"])
            auditor_root=Path(protocol["project_root"])/"scripts/experiments"
            assert auditor_path.resolve().is_relative_to(auditor_root.resolve()),"parent auditor must be a pinned repository experiment script"
            pins.pin(auditor_path,AUDITOR_SHA)
            if "source_pins" in audit:
                for p,h in audit["source_pins"].items():pins.pin(p,h)
            batch_receipts.append(dict(batch_id=b["id"],sources=len(b["source_ids"]),units=len(media),
                media_manifest_path=str(media_path),media_manifest_sha256=media_sha,
                consumer_completion_path=str(completion_path),consumer_completion_sha256=completion_sha,
                parent_audit_path=str(audit_path),parent_audit_sha256=pins.pin(audit_path)))
            seen=[]
            for split_entry in completion["split_manifests"]:
                split=split_entry["split"];manifest_path=Path(split_entry["path"])
                assert split==b["split"]
                pins.pin(manifest_path,split_entry["sha256"])
                rr=read(manifest_path)["rows"]
                for r in rr:
                    m=media_byid[r["id"]]
                    assert r["score_path"]==m["score_path"] and r["score_sha256"]==m["score_sha256"]
                    all_rows.append(attach_fields(add_source_fields(r,sources[r["sample_id"]],False),generation_bindings[r["id"]]));seen.append(r["id"])
                folder=manifest_path.parent
                for name in ("patch_index.json","window_clock.json","alignment.json"):pins.pin(folder/name)
                index_parts.append(dict(split=split,batch_id=b["id"],rows=rr,
                    patch_path=str(folder/"patch_index.json"),clock_path=str(folder/"window_clock.json"),alignment_paths=[str(folder/"alignment.json")]))
            assert len(seen)==len(media) and set(seen)==set(media_byid)
        groups=validate_full_roster(all_rows,sources)
        assert sum(r["reused_in_source200_preparation"] for r in all_rows)==280
        # Re-read every actual media/cache/FA binding, not only the cached receipts.
        for r in all_rows:
            for prefix in ("sequence","scoring_audio","replay","canonical_audio","video","score","activesd"):
                pins.pin(r[prefix+"_path"],r[prefix+"_sha256"])
            pins.pin(r["FA"]["path"],r["FA"]["sha256"])
        # Every row uses the same official Torch reduction for future labels;
        # historical score_float32 is retained with its actual provenance.
        all_rows=[canonical_score_row(r) for r in all_rows]
        per_split={};total_pairs=0;technical_summary={}
        for split in FOLDERS:
            rr=sorted((r for r in all_rows if r["split"]==split),key=lambda r:r["id"])
            parts=[p for p in index_parts if p["split"]==split]
            pairs=[];supports=[];pointers=[]
            for part in parts:
                p,s=verify_index_part(read(part["patch_path"]),part["rows"])
                pairs.extend(p);supports.extend(s)
                pointers.append(dict(batch_id=part["batch_id"],unit_ids=[r["id"] for r in part["rows"]],
                    patch_index_path=part["patch_path"],patch_index_sha256=pins.pin(part["patch_path"]),
                    window_clock_path=part["clock_path"],window_clock_sha256=pins.pin(part["clock_path"]),
                    alignment_files=[dict(path=p,sha256=pins.pin(p)) for p in part["alignment_paths"]]))
            assert len(rr)==EXPECTED_UNITS[split] and len(pairs)==EXPECTED_PAIRS[split]
            assert len(supports)==len({r["sample_id"] for r in rr})
            assert len({(p["source"],p["family"],p["a_id"],p["b_id"]) for p in pairs})==len(pairs)
            total_pairs+=len(pairs)
            folder=out/FOLDERS[split]
            put(folder/"manifest.json",dict(schema="audio200_dataset4200_split_v1",split=split,rows=rr,
                original_source_manifest_sha256=SOURCE_SHA,allow_training=False,FA_truth="UNVERIFIED",
                canonical_label_field=CANONICAL_LABEL_FIELD,
                score_float32_policy="legacy compatibility only; do not mix NumPy/Torch reductions as training labels"))
            put(folder/"index_pointers.json",dict(parts=pointers,no_large_cache_copy=True))
            put(folder/"pair_index.json",dict(pair_coverage=pairs,source_common_support=supports,
                definition="9 matched-draw crossarm primary +6 within-model pairs/source; no per-window score label"))
            technical_summary[split]=dict(speakers=len({r["speaker_id"] for r in rr}),sources=len(supports),units=len(rr),
                windows=sum(r["rows"] for r in rr),pairs=len(pairs),primary_pairs=sum("within" not in p["family"] for p in pairs),
                all_clips_individually_pass_sources=sum(p["all_clips_individually_pass"] for p in supports),
                strict_common_7arm_pass_sources=sum(p["common_7arm_gate"] for p in supports),
                pair_support_pass=sum(p["temporary_measurability_gate"] for p in pairs))
            per_split[split]=dict(path=str(folder/"manifest.json"),sha256=sha(folder/"manifest.json"),units=len(rr))
        assert total_pairs==9000
        put(out/"input_pins.json",dict(producer_protocol_path=str(protocol_path),producer_protocol_sha256=pins.pin(protocol_path),
            original_protocol_path=protocol["original_protocol_path"],original_protocol_sha256=protocol["original_protocol_sha256"],
            reviewed_recovery_path=protocol["reviewed_recovery_path"],reviewed_recovery_sha256=protocol["reviewed_recovery_sha256"],
            material_roots=protocol["material_roots"],parent_audit_root=str(audit_root),files=pins.observed))
        put(out/"batch_acceptance_index.json",dict(batches=batch_receipts,count=57))
        put(out/"dataset_index.json",dict(schema="audio200_dataset4200_index_v1",splits=per_split,
            technical_summary=technical_summary,training_authorized=False,score_comparisons_performed=False,
            canonical_label_field=CANONICAL_LABEL_FIELD,
            legacy_score_float32="preserved; provenance distinguishes NumPy old280 and Torch new3920; not the unified label column",
            confirmation_policy="Numerical labels only in confirmation_sealed/manifest.json; not development/model selection"))
        report=("# 200人数据准备最终验收\n\n## Material Passport\n"
            "academic-research-suite / experiment-agent；机械数据验收，不是解释模型或因果结论。\n\n"
            "600固定来源、200说话人、每人3句；4200完整单元中旧280原缓存复用、新3920完成。"
            "57批全部生产与特征/FA/窗口完毕，并逐批通过父侧独立复核。所有绑定文件在本次finalizer重新读取SHA。\n\n"
            "按正式fit/development/confirmation分别保存2100/840/1260单元及4500/1800/2700固定pairs。"
            "原280所有历史字段保持不变，仅追加本轮来源和分组字段。确认标签只写独立sealed目录；根入口仅技术计数。\n\n"
            "统一权威标签为official_syncnet_score_float32：全部4200条用CPU Torch从原float32距离场按官方mean/median/min逐值派生，"
            "并与replay.original.C严格相同。score_float32保留历史兼容值并标明归约来源，未来学习必须读统一新列；不回写旧试训练或阈值。\n\n"
            "索引指针引用原始大型cache，不复制、不删除历史资料。保留全部低支持、长尾、FA和EOS异常标记，"
            "不因支持门或得分筛掉媒体。每条各自资格与七臂同文字交集分别计数，未冒称窗口是独立样本。\n\n"
            "FA真值、615ms充分性、听觉描述忠实度、偏好预测、因果解释及人类口型质量均未由本次验收证明。"
            "未训练任何模型，也未作确认集排名、分差、显著性或模型选择。\n\n"
            "权威入口：final_status.json、final_manifest.json、dataset_index.json；各split/manifest.json、"
            "pair_index.json和index_pointers.json；input_pins.json和batch_acceptance_index.json保存新读回证据。\n")
        report+="\n恢复仅改batch003 N官方评分及相应媒体清单的输出路由；旧guard退出、0完成/0失败/10待处理及原成功视频均保留。原source、生成次数、参数和划分未改变，未重生成既有视频。\n"
        (out/"report.md").write_text(report,encoding="utf-8")
        files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
        put(out/"final_manifest.json",dict(files=files,file_count=len(files),bytes=sum(f["bytes"] for f in files),
            excludes=["final_manifest.json","final_status.json"],no_large_cache_copy=True))
        for f in files:assert sha(out/f["path"])==f["sha256"]
        put(out/"final_status.json",dict(status="COMPLETE_DATASET4200_TECHNICAL_PREPARATION_ONLY",utc=utc(),
            producer_run=str(run),producer_protocol_sha256=pins.pin(protocol_path),parent_audited_batches=57,
            source_count=600,speaker_count=200,units=4200,reused=280,new=3920,pairs=9000,
            technical_summary=technical_summary,final_manifest_sha256=sha(out/"final_manifest.json"),
            no_training=True,no_scientific_preference_claim=True,FA_truth="UNVERIFIED",elapsed_s=time.time()-started))
        print(json.dumps(dict(status="COMPLETE_DATASET4200_TECHNICAL_PREPARATION_ONLY",units=4200,pairs=9000,output_root=str(out))))
    except Exception as e:
        put(out/"failure.json",dict(status="HOLD_FINALIZATION_FAILURE_NO_OVERWRITE",utc=utc(),error=repr(e),no_training=True))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--producer-run",required=True);p.add_argument("--parent-audit-root",required=True);p.add_argument("--output-root",required=True)
    finalize(p.parse_args())
