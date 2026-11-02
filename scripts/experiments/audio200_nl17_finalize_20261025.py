"""CPU-only final acceptance of source200 N/L17 4-arm /2400 data.

No fitting, scoring, generation, rank/effect summaries, or implicit retries.
All frozen producer batches and their independently reviewed parent audits must
already exist. Only metadata/pointers are materialized; large caches stay put.
"""
from __future__ import annotations
import argparse
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
PHASE="N_L17_source600_v1"
ARMS={("N",0)}|{("L17",d) for d in (1,2,3)}
FOLDERS={"fit":"fit","development":"development","confirmation":"confirmation_sealed"}
EXPECTED_UNITS={"fit":1200,"development":480,"confirmation":720}
EXPECTED_PAIRS={"fit":1800,"development":720,"confirmation":1080}
AUDITOR_SHA="4827934a001043a9796b6cbbd3a8931302434498b978ec64e31cd52402808f6b"
CANONICAL_LABEL_FIELD="official_syncnet_score_float32"


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


def add_source_fields(row,source,reused):
    """Keep every historical key/value, including original_cache_reused."""
    out=dict(row)
    additions=dict(split=source["split"],ref_source_id=source["ref_source_id"],
        natural_source_sha256=source["sha256"],natural_source_pcm_sha256=source["pcm_sha256"],
        reused_in_nl17_phase=reused,status="COMPLETE_DATA_PREPARATION_ONLY")
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
    provenance=row["score_float32_provenance"]
    assert provenance in ("numpy_float32_curve_reduction","torch_float32_curve_reduction")
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
    if row["score_float32_provenance"]=="numpy_float32_curve_reduction":
        legacy_mu=original.mean(0)
        legacy=float(np.median(legacy_mu)-legacy_mu.min())
    else:legacy=official["C"]
    return attach_canonical_score(row,official,legacy)


def validate_full_roster(rows,sources):
    assert len(rows)==len({r["id"] for r in rows})==2400
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
    for group in groups.values():assert len(group)==4 and {(r["arm"],r["draw"]) for r in group}==ARMS
    for split,expected in (("fit",100),("development",40),("confirmation",60)):
        assert sum(split in v for v in speakers.values())==expected
    return groups


def expected_pairs(group):
    ids={(r["arm"],r["draw"]):r["id"] for r in group};pairs=[]
    assert len(group)==len(ids)==4 and set(ids)==ARMS
    for d in (1,2,3):
        pairs.append(("N-L17",ids["N",0],ids["L17",d]))
    for x,y in combinations((1,2,3),2):pairs.append(("L17-within",ids["L17",x],ids["L17",y]))
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
    assert set(actual)==expected,"must use 3 N-L17 primary +3 L17-within pairs"
    pairs=[];support=[]
    for source,g in groups.items():
        den=len("".join(g[0]["text"].split()))
        common=set.intersection(*(valid[r["id"]] for r in g))
        common_robust=set.intersection(*(robust[r["id"]] for r in g))
        individual=[len(valid[r["id"]])>=5 and len(valid[r["id"]])/max(den,1)>=.8 for r in g]
        support.append(dict(source=source,characters=den,all_clips_individually_pass=all(individual),
            common_4arm_anchors=len(common),common_character_indices=sorted(common),
            common_4arm_coverage=len(common)/max(den,1),common_4arm_gate=len(common)>=5 and len(common)/max(den,1)>=.8,
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
    assert audit["status"]=="PASS_PARENT_NL17_BATCH_REPLAY_CLOCK_PATCH",(batch["id"],"parent independent audit not PASS")
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


def phase_index(index,rows):
    ids={r["id"] for r in rows}
    return dict(patches=[p for p in index["patches"] if p["id"] in ids],
        pair_coverage=[p for p in index["pair_coverage"] if p["a_id"] in ids and p["b_id"] in ids and p["family"] in ("N-L17","L17-within")])


def under_material(path,roots):
    q=Path(path)
    assert q.is_absolute() and any(q.resolve().is_relative_to((Path(r)/"batches").resolve()) for r in roots)


def validate_phase_protocol(protocol,run,pins):
    assert protocol["schema"]=="audio200_nl17_producer_v1" and protocol["phase"]==PHASE
    assert protocol["run"]==str(run) and protocol["no_training"] is True
    for key,value in dict(expected_units=2400,cache_reused_units=248,ready_media_units=10,new_media_units=2142,new_feature_units=2152,new_tts_draws=1614).items():assert protocol[key]==value
    pins.pin(protocol["source600_manifest_path"],SOURCE_SHA);pins.pin(protocol["split_freeze_path"],SPLIT_SHA)
    assert protocol["source600_manifest_sha256"]==SOURCE_SHA and protocol["split_freeze_sha256"]==SPLIT_SHA
    pins.pin(protocol["original_producer_protocol_path"],"f2d591056ec61f67ebe6bbf2b26a5260bc0b95eb24257de32e2882e34c61dc5e")
    pins.pin(protocol["recovery_producer_protocol_path"],"5b4cbeca0e971a226521cb9e7a559be00a4978488031b937be84f486f5289d29")
    original=read(protocol["original_producer_protocol_path"])
    recovered=read(protocol["recovery_producer_protocol_path"])
    assert protocol["material_roots"]==[original["run"],recovered["run"],str(run)]
    for k in ("target_rms","seed_bases","generation_kwargs","non_streaming_mode"):assert protocol[k]==original[k]
    assert set(protocol["bundles"])=={"L17"} and protocol["bundles"]["L17"]==original["bundles"]["L17"]
    batches=protocol["work_batches"]
    assert len(batches)==55 and [b["id"] for b in batches]==["bridge003a","batch003_tail"]+[f"batch{i:03d}" for i in range(4,57)]
    ob={b["id"]:b for b in original["batches"]}
    assert batches[0]["source_ids"]==ob["batch003"]["source_ids"][:1]
    assert batches[1]["source_ids"]==ob["batch003"]["source_ids"][1:]
    projections={r["group_path"]:r for r in protocol["projected_group_diffs"]};assert len(projections)==2
    for b in batches:
        assert b["split"]==ob[b["original_batch_id"]]["split"]
        under_material(b["media_manifest_path"],protocol["material_roots"])
        assert Path(b["media_manifest_path"]).is_relative_to(run/"batches")
        if b["id"] not in ("bridge003a","batch003_tail"):assert b["source_ids"]==ob[b["original_batch_id"]]["source_ids"]
        assert {g["arm"] for g in b["groups"]}==({"L17"} if b["id"] in ("bridge003a","batch003_tail") else {"N","L17"})
        for g in b["groups"]:
            under_material(g["path"],protocol["material_roots"])
            gp=Path(g["path"])/"protocol.json";pins.pin(gp,g["protocol_sha256"])
            parent_group=next(x for x in ob[b["original_batch_id"]]["groups"] if x["arm"]==g["arm"])
            if str(g["path"]) not in projections:
                assert g==parent_group
            else:
                d=projections[g["path"]]
                parent_path=Path(parent_group["path"])/"protocol.json"
                assert d["original_group_protocol_path"]==str(parent_path) and d["original_group_protocol_sha256"]==parent_group["protocol_sha256"]
                pins.pin(parent_path,parent_group["protocol_sha256"])
                a,z=read(parent_path),read(gp)
                allowed={"samples","expected_cells","raw_manifest_path","analysis_protocol_path","source_pins"}
                assert set(a)==set(z)
                for k in set(a)-allowed:assert a[k]==z[k]
                assert z["samples"]==[r for r in a["samples"] if r["sample_id"] in b["source_ids"]]
                assert z["expected_cells"]==3*len(b["source_ids"])
                assert z["raw_manifest_path"]==str(Path(g["path"])/"raw_manifest.json")
                assert z["analysis_protocol_path"]==str(Path(g["path"])/"analysis_protocol.json")
                for path,h in a["source_pins"].items():assert z["source_pins"][path]==h
                extra=set(z["source_pins"])-set(a["source_pins"])
                assert extra=={z["analysis_protocol_path"]}
                pins.pin(z["analysis_protocol_path"],z["source_pins"][z["analysis_protocol_path"]])
    expected_stage={f'{b["id"]}_{g["arm"]}_{s}' for b in batches for g in b["groups"] for s in ("synthesize","canonical","render","official")}
    assert len(expected_stage)==432 and set(protocol["stage_receipt_map"])==expected_stage
    for key,v in protocol["stage_receipt_map"].items():assert v["path"]==str(run/"stage_receipts"/(key+".json"))
    assert set(protocol["consumer_receipt_map"])=={b["id"] for b in batches}
    for key,v in protocol["consumer_receipt_map"].items():assert v["path"]==str(run/"consumer_receipts"/(key+".json"))
    return original,recovered


def source_registry(protocol,pins):
    physical=[];parts=[]
    for entry in protocol["reuse_cache_sources"]:
        path=Path(entry["path"]);pins.pin(path,entry["sha256"])
        if path.name=="dataset280_consumer_manifest.json":
            rows=read(path)["rows"];physical.extend(rows)
            selected=[r for r in rows if r["arm"] in ("N","L17")]
            parts.append(dict(batch_id="legacy40",rows=selected,patch_path=str(path.parent/"dataset280_patch_index.json"),clock_path=str(path.parent/"dataset280_window_clock_audit.json"),alignment_paths=sorted({r["FA"]["path"] for r in selected})))
        elif path.name=="manifest.json":
            rows=read(path)["rows"];physical.extend(rows)
            selected=[r for r in rows if r["arm"] in ("N","L17")]
            parts.append(dict(batch_id=path.parents[2].name,rows=selected,patch_path=str(path.parent/"patch_index.json"),clock_path=str(path.parent/"window_clock.json"),alignment_paths=sorted({r["FA"]["path"] for r in selected})))
        elif path.name=="completion.json":
            c=read(path);assert c["status"]=="COMPLETE_FEATURE_FA_WINDOW_DATA_ONLY"
            for f in c["files"]:pins.pin(path.parent/f["path"],f["sha256"])
        else:raise AssertionError("unexpected cache origin schema")
    assert len(physical)==len({r["id"] for r in physical})==434
    selected={r["id"]:r for r in physical if r["arm"] in ("N","L17")};assert len(selected)==248
    assert len({r["sample_id"] for r in selected.values()})==62
    cache_path=protocol["reused_cache_manifest_path"];pins.pin(cache_path,protocol["reused_cache_manifest_sha256"])
    cache=read(cache_path)["rows"];assert len(cache)==len({r["id"] for r in cache})==248
    assert {r["id"] for r in cache}==set(selected)
    for r in cache:
        for k,v in selected[r["id"]].items():assert r[k]==v,(r["id"],k,"cached field changed")
        assert r["split"]=="fit"
    return cache,parts


def finalize(args):
    run,audits,out=Path(args.producer_run),Path(args.parent_audit_root),Path(args.output_root)
    assert not out.exists(),"immutable finalizer output; use reviewed new attempt"
    out.mkdir(parents=True);put(out/"launch.json",dict(utc=utc(),command=sys.argv,source_sha256=sha(__file__),no_training=True))
    pins=FreshPins();started=time.time()
    try:
        assert shutil.disk_usage(out).free>=5*1024**3
        pp=run/"protocol.json";protocol=read(pp);pins.pin(pp)
        original,recovered=validate_phase_protocol(protocol,run,pins)
        for p,h in protocol["source_pins"].items():pins.pin(p,h)
        for bundle in protocol["bundles"].values():
            for key in ("source_pins","model_pins"):
                for p,h in bundle[key].items():pins.pin(p,h)
        source_path=protocol["source600_manifest_path"];sources={r["source_id"]:r for r in read(source_path)["rows"]}
        assert len(sources)==600 and len({r["pcm_sha256"] for r in sources.values()})==600
        for sid,s in sources.items():
            assert sid==s["ref_source_id"]
            pins.pin(s["remote_path"],s["sha256"])
        cache,index_parts=source_registry(protocol,pins)
        old280_path=Path(protocol["project_root"])/"runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json"
        pins.pin(old280_path,OLD_SHA);old_ids={r["id"] for r in read(old280_path)["rows"]}
        all_rows=[]
        for r in cache:
            z=add_source_fields(r,sources[r["sample_id"]],True)
            z.setdefault("score_float32_provenance","numpy_float32_curve_reduction" if r["id"] in old_ids else "torch_float32_curve_reduction")
            z.setdefault("generation_qa_status","legacy_format_not_reaggregated")
            z=attach_fields(z,dict(legacy_input_registry_path=protocol["reused_cache_manifest_path"],legacy_input_registry_sha256=protocol["reused_cache_manifest_sha256"]))
            all_rows.append(z)
        ready_path=protocol["ready_media_manifest_path"];pins.pin(ready_path,protocol["ready_media_manifest_sha256"])
        ready=read(ready_path)["rows"];assert len(ready)==len({r["id"] for r in ready})==10
        assert all(r["arm"]=="N" and r["draw"]==0 and r["status"]=="COMPLETE" for r in ready)
        for receipt in protocol["ready_media_origin_receipts"]:
            pins.pin(receipt["path"],receipt["sha256"]);v=read(receipt["path"])
            assert v["status"]=="COMPLETE" and v["returncode"]==0
            for p,h in v["artifacts"].items():pins.pin(p,h)
        ready_group=next(g for g in recovered["batches"][3]["groups"] if g["arm"]=="N")
        ready_bindings=raw_generation_bindings(ready_group,pins)
        assert set(ready_bindings)=={r["id"] for r in ready}
        ready_byid={r["id"]:r for r in ready}
        batches=protocol["work_batches"];work_sources=[s for b in batches for s in b["source_ids"]]
        cached_sources={r["sample_id"] for r in cache}
        assert len(work_sources)==len(set(work_sources))==538
        assert not cached_sources&set(work_sources) and cached_sources|set(work_sources)==set(sources)
        batch_receipts=[]
        for b in batches:
            media_path=Path(b["media_manifest_path"]);under_material(media_path,protocol["material_roots"])
            media_sha=pins.pin(media_path);media=read(media_path)["rows"]
            assert len(media)==4*len(b["source_ids"]) and {r["sample_id"] for r in media}==set(b["source_ids"])
            assert all(r["arm"] in ("N","L17") and r["split"]==b["split"] and r["status"]=="COMPLETE" for r in media)
            mids={r["id"]:r for r in media};assert len(mids)==len(media)
            gen={uid:ready_bindings[uid] for uid in b["ready_media_ids"]}
            for uid in b["ready_media_ids"]:
                assert mids[uid]==ready_byid[uid],"ready N media must remain exact"
            for g in b["groups"]:
                for stage in ("synthesize","canonical","render","official"):
                    entry=protocol["stage_receipt_map"][f'{b["id"]}_{g["arm"]}_{stage}']
                    pins.pin(entry["path"]);r=read(entry["path"])
                    assert r["status"]=="COMPLETE" and r["returncode"]==0
                    for p,h in r["artifacts"].items():pins.pin(p,h)
                bound=raw_generation_bindings(g,pins);assert not set(gen)&set(bound);gen.update(bound)
            assert set(gen)==set(mids)
            features=media_path.parent/"features";cp=features/"completion.json";csha=pins.pin(cp);c=read(cp)
            assert c["status"]=="COMPLETE_NL17_FEATURE_FA_WINDOW_DATA_ONLY" and c["phase"]==PHASE and c["no_training"] is True
            assert c["media_manifest_sha256"]==media_sha and c["units"]==c["new_forward"]==len(media) and c["reused"]==0
            for f in c["files"]:
                p=features/f["path"];assert p.resolve().is_relative_to(features.resolve())
                pins.pin(p,f["sha256"]);assert p.stat().st_size==f["bytes"]
            crp=protocol["consumer_receipt_map"][b["id"]]["path"];pins.pin(crp);cr=read(crp)
            assert cr["status"]=="COMPLETE"
            for p,h in cr["artifacts"].items():pins.pin(p,h)
            cmd=cr["command"];contract_path=cmd[cmd.index("--contract")+1];contract=read(contract_path)
            pins.pin(contract_path,c["consumer_contract_sha256"]);pins.pin(cmd[1],c["consumer_source_sha256"])
            assert contract["phase"]==PHASE and contract["expected_units"]==2400
            for key in ("engine_source_pins","fa_pins"):
                for p,h in contract[key].items():pins.pin(p,h)
            pins.pin(contract["engine_path"],contract["engine_sha256"])
            pins.pin(contract["legacy_parser_path"],contract["legacy_parser_sha256"])
            assert len(c["split_manifests"])==1;sm=c["split_manifests"][0]
            ap=audits/(b["id"]+".json");pins.pin(ap);audit=read(ap)
            check_parent_audit(audit,b,sm["path"],sm["sha256"],cp,csha)
            for key in ("manifest","clock","patch"):pins.pin(audit[key+"_path"],audit[key+"_sha256"])
            source=Path(audit["auditor_source_path"])
            assert source.resolve().is_relative_to((Path(protocol["project_root"])/"scripts/experiments").resolve())
            pins.pin(source,AUDITOR_SHA)
            pins.pin(sm["path"],sm["sha256"]);rr=read(sm["path"])["rows"]
            assert len(rr)==len(media) and {r["id"] for r in rr}==set(mids)
            for r in rr:
                mr=mids[r["id"]]
                assert r["score_path"]==mr["score_path"] and r["score_sha256"]==mr["score_sha256"]
                z=attach_fields(add_source_fields(r,sources[r["sample_id"]],False),gen[r["id"]])
                all_rows.append(z)
            folder=Path(sm["path"]).parent
            index_parts.append(dict(batch_id=b["id"],rows=rr,patch_path=str(folder/"patch_index.json"),clock_path=str(folder/"window_clock.json"),alignment_paths=[str(folder/"alignment.json")]))
            batch_receipts.append(dict(batch_id=b["id"],sources=len(b["source_ids"]),units=len(media),media_manifest_path=str(media_path),media_manifest_sha256=media_sha,
                consumer_completion_path=str(cp),consumer_completion_sha256=csha,parent_audit_path=str(ap),parent_audit_sha256=pins.pin(ap)))
        validate_full_roster(all_rows,sources)
        assert sum(r["reused_in_nl17_phase"] for r in all_rows)==248
        for r in all_rows:
            for key in ("sequence","scoring_audio","replay","canonical_audio","video","score","activesd"):pins.pin(r[key+"_path"],r[key+"_sha256"])
            pins.pin(r["FA"]["path"],r["FA"]["sha256"])
        all_rows=[canonical_score_row(r) for r in all_rows]
        split_entries={};summary={};total=0
        for split,dirname in FOLDERS.items():
            rr=sorted((r for r in all_rows if r["split"]==split),key=lambda r:r["id"])
            parts=[v for v in index_parts if sources[v["rows"][0]["sample_id"]]["split"]==split]
            pairs=[];support=[];ptr=[]
            for part in parts:
                pins.pin(part["patch_path"]);pins.pin(part["clock_path"])
                phase_part=phase_index(read(part["patch_path"]),part["rows"])
                pp,ss=verify_index_part(phase_part,part["rows"]);pairs.extend(pp);support.extend(ss)
                ptr.append(dict(batch_id=part["batch_id"],unit_ids=[r["id"] for r in part["rows"]],allowed_arms=["N","L17"],
                    allowed_pair_families=["N-L17","L17-within"],physical_artifact_may_include_deferred_L06=part["batch_id"] in ("legacy40","batch000","batch001","batch002"),
                    patch_index_path=part["patch_path"],patch_index_sha256=pins.pin(part["patch_path"]),window_clock_path=part["clock_path"],window_clock_sha256=pins.pin(part["clock_path"]),
                    alignment_files=[dict(path=p,sha256=pins.pin(p)) for p in part["alignment_paths"]]))
            assert len(rr)==EXPECTED_UNITS[split] and len(pairs)==EXPECTED_PAIRS[split]
            assert len(support)==len({r["sample_id"] for r in rr})
            assert len({(p["source"],p["family"],p["a_id"],p["b_id"]) for p in pairs})==len(pairs)
            total+=len(pairs);folder=out/dirname
            put(folder/"manifest.json",dict(schema="audio200_nl17_dataset2400_split_v1",phase=PHASE,split=split,rows=rr,canonical_label_field=CANONICAL_LABEL_FIELD,allow_training=False,FA_truth="UNVERIFIED"))
            put(folder/"index_pointers.json",dict(parts=ptr,no_large_cache_copy=True,phase_selection_required=True))
            put(folder/"pair_index.json",dict(pair_coverage=pairs,source_common_support=support,definition="3 N-L17 primary plus3 L17-within pairs per source"))
            summary[split]=dict(speakers=len({r["speaker_id"] for r in rr}),sources=len(support),units=len(rr),windows=sum(r["rows"] for r in rr),pairs=len(pairs),
                primary_pairs=sum(p["family"]=="N-L17" for p in pairs),all_clips_individually_pass_sources=sum(s["all_clips_individually_pass"] for s in support),
                strict_common_4arm_pass_sources=sum(s["common_4arm_gate"] for s in support),pair_support_pass=sum(p["temporary_measurability_gate"] for p in pairs))
            split_entries[split]=dict(path=str(folder/"manifest.json"),sha256=sha(folder/"manifest.json"),units=len(rr))
        assert total==3600
        put(out/"input_pins.json",dict(producer_protocol_path=str(run/"protocol.json"),producer_protocol_sha256=pins.pin(run/"protocol.json"),files=pins.observed))
        put(out/"batch_acceptance_index.json",dict(batches=batch_receipts,count=55))
        put(out/"dataset_index.json",dict(schema="audio200_nl17_dataset2400_index_v1",phase=PHASE,splits=split_entries,technical_summary=summary,
            canonical_label_field=CANONICAL_LABEL_FIELD,training_authorized=False,score_comparisons_performed=False,excluded_model="L06 deferred, cached physical assets retained",
            confirmation_policy="Numerical labels remain in confirmation_sealed; no rankings/effects/model selection"))
        (out/"report.md").write_text("# N/L17完整200人数据准备验收\n\n## Material Passport\nacademic-research-suite / experiment-agent；机械数据验收，非学习或因果结论。\n\n固定200人×3来源×4臂=2400；N1和L17三draw，0.6B延期，旧七臂物理缓存仅按显式ID引用。原434按arm机械筛248缓存复用；N10原视频/评分复用，其余2152新缓存完整验收。55工作批全部父审通过。\n\n分组100fit/40dev/60confirm，单元1200/480/720；每来源3个N-L17主比较和3个L17within，共3600pairs。每条资格与严格四臂同字交集分开，保留全部异常/缺失。\n\n所有行权威标签official_syncnet_score_float32按原float32距离场Torch exact归约；历史score_float32原值及provenance保留，不能混用旧NumPy兼容列训练。\n\n未训练模型；未对确认集计算效果、排名或模型选择。FA真值、615ms充分性、解释忠实度、人类质量及因果均未由数据验收证明。\n",encoding="utf-8")
        files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
        put(out/"final_manifest.json",dict(files=files,file_count=len(files),bytes=sum(f["bytes"] for f in files),excludes=["final_manifest.json","final_status.json"]))
        for f in files:assert sha(out/f["path"])==f["sha256"]
        put(out/"final_status.json",dict(status="COMPLETE_NL17_DATASET2400_TECHNICAL_PREPARATION_ONLY",phase=PHASE,utc=utc(),units=2400,sources=600,speakers=200,
            cache_reused=248,new_feature=2152,pairs=3600,parent_audited_batches=55,technical_summary=summary,final_manifest_sha256=sha(out/"final_manifest.json"),no_training=True,FA_truth="UNVERIFIED",elapsed_s=time.time()-started))
        print(json.dumps(dict(status="COMPLETE_NL17_DATASET2400_TECHNICAL_PREPARATION_ONLY",units=2400,pairs=3600)))
    except Exception as e:
        put(out/"failure.json",dict(status="HOLD_NL17_FINALIZATION_FAILURE_NO_OVERWRITE",utc=utc(),error=repr(e),no_training=True));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--producer-run",required=True);p.add_argument("--parent-audit-root",required=True);p.add_argument("--output-root",required=True)
    finalize(p.parse_args())
