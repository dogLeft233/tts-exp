"""Frozen SyncNet/CPU FA consumer for the source200 data preparation.

No training, effect analysis, score-based selection, or generation. The producer
owns the exact shared GPU lock while invoking consume. Completed batch receipts
are immutable; interrupted partial batches are never silently replayed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import datetime
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

SOURCE_SHA = "3ef28c8076aba23e64e2bc1a22963e24dba1e9a89ed9e2aeaf7fdaae6821e608"
OLD_SHA = "046b4cbb1a53fc11b2455f90eb943a7cfbac57c80e6ec34039a4f00b0b865acf"
SPLIT_SHA = "3ffb6d516470d5185a71e18d5bd8390b3180de7d6178640a2726c5eaef6fc749"
EXPECTED_ARMS = {("N", 0)} | {(a, d) for a in ("L06", "L17") for d in (1, 2, 3)}
SPLIT_DIR = {"fit": "fit", "development": "development", "confirmation": "confirmation_sealed"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked(path, expected):
    assert sha(path) == expected, (str(path), "SHA mismatch")


def source_norm(text):
    return "".join(text.split())


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm_pairs(rows):
    """The fixed 9 primary + 6 within-model pair set, no all cross-draw."""
    arms = {(r["arm"], int(r["draw"])): r["id"] for r in rows}
    assert len(rows) == len(arms) == 7 and set(arms) == EXPECTED_ARMS
    output = []
    for d in (1, 2, 3):
        for fam, aa, bb in (("N-L06", ("N", 0), ("L06", d)),
                            ("N-L17", ("N", 0), ("L17", d)),
                            ("L17-L06", ("L17", d), ("L06", d))):
            output.append((fam, arms[aa], arms[bb]))
    for a in ("L06", "L17"):
        for x, y in combinations((1, 2, 3), 2):
            output.append((a+"-within", arms[a, x], arms[a, y]))
    assert len(set(output)) == 15
    return output


def validate_manifest(rows, sources):
    assert len(rows) == len({r["id"] for r in rows})
    groups = defaultdict(list)
    for r in rows:
        sid = r["sample_id"]
        assert sid in sources and r["speaker_id"] == sources[sid]["speaker_id"]
        assert r["id"] == f'{sid}_{r["arm"]}_{r["draw"]}'
        assert r["split"] == sources[sid]["split"]
        assert source_norm(r["text"]) == sources[sid]["normalized_text"]
        assert r["status"] == "COMPLETE"
        groups[sid].append(r)
    assert groups, "empty batch"
    for group in groups.values():
        arm_pairs(group)
    return groups


def prepare(args):
    root, out = Path(args.project_root), Path(args.output_root)
    assert not out.exists()
    src = root/"runs/audio_data_inventory200_20261024_attempt05/selected600_natural_manifest.json"
    split = src.parent/"source_split_freeze.json"
    old = root/"runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json"
    checked(src, SOURCE_SHA); checked(split, SPLIT_SHA); checked(old, OLD_SHA)
    sources, old_rows = read(src)["rows"], read(old)["rows"]
    source_by_id = {r["source_id"]: r for r in sources}
    assert len(sources) == len(source_by_id) == 600
    assert Counter(r["split"] for r in sources) == {"fit": 300, "development": 120, "confirmation": 180}
    assert len({r["speaker_id"] for r in sources}) == 200
    for r in old_rows:
        s = source_by_id[r["sample_id"]]
        assert s["split"] == "fit" and s["speaker_id"] == r["speaker_id"]
        assert s["normalized_text"] == source_norm(r["text"])
    replay = root/"runs/syncnet_audio_replay20_20261024_attempt01"
    engine = replay/"executed_source.py"
    seal = {r["relative_path"]: r["sha256"] for r in read(replay/"final_manifest.json")["files"]}
    checked(engine, seal["executed_source.py"])
    p, env = read(replay/"protocol.json"), read(replay/"environment.json")
    # Bind the actual frozen executable used by the 40-source completion, not
    # an optional development copy that need not exist on the execution host.
    legacy_meta = root/"runs/syncnet_audio_dataset40_20261024_attempt02/conditional_execution_contract.json"
    legacy = Path(read(legacy_meta)["consumer_source_path"])
    checked(legacy, read(legacy_meta)["consumer_source_sha256"])
    fa_meta = replay/"fa_started.json"
    fa_pins = read(fa_meta)["model_pins"]
    contract = dict(schema="source200_consumer_contract_v1", utc=utc(), no_training=True,
        project_root=str(root), output_root=str(out), source_manifest_path=str(src),
        source_manifest_sha256=SOURCE_SHA, split_freeze_path=str(split), split_freeze_sha256=SPLIT_SHA,
        old_registry_path=str(old), old_registry_sha256=OLD_SHA, old_units=280,
        expected_sources=600, expected_units=4200, expected_new_units=3920,
        consumer_source_sha256=sha(__file__), legacy_parser_path=str(legacy), legacy_parser_sha256=sha(legacy),
        engine_path=str(engine), engine_sha256=sha(engine), engine_source_pins=p["source_pins"],
        engine_environment={k:env[k] for k in ("executable", "python", "torch", "numpy", "scipy", "opencv")},
        tolerances=p["tolerances"], fa_pins=fa_pins, fa_model_root=os.path.commonpath(list(fa_pins)),
        fa_python="/home/wjj/.venvs/funasr-cpu-20261024/bin/python",
        shared_gpu_lock=str(root/"runs/.audio200_preparation_gpu.lock"),
        old_gpu_locks=[str(root/"runs/.audio_data_supplement_gpu.lock"),str(root/"runs/.syncnet_audio_dataset_cache_gpu.lock")],
        minimum_free_bytes=12*1024**3, next_item_bytes=32*1024**2,
        scientific_limits="FA structure is not timing truth; local615ms sufficiency unverified; no score effects or training",
        confirmation="mechanical preparation only; numerical scores confined to confirmation_sealed manifests")
    out.mkdir(parents=True)
    put(out/"contract.json", contract)
    put(out/"prepare_receipt.json", dict(status="PREPARED_NO_GPU_NO_FA_NO_TRAINING", utc=utc(),
        contract_sha256=sha(out/"contract.json"), units=4200, reuse=280, new=3920,
        command=sys.argv, no_models_loaded=True))
    print(json.dumps(dict(status="PREPARED", contract=str(out/"contract.json"), contract_sha256=sha(out/"contract.json"))))


def require_release(args, contract, source_ids):
    release = read(args.release)
    assert release["allow_training"] is False
    assert release["allow_frozen_syncnet_forward"] is True and release["allow_cpu_fa"] is True
    assert release["consumer_contract_sha256"] == sha(args.contract)
    assert release["consumer_source_sha256"] == sha(__file__) == contract["consumer_source_sha256"]
    allowed = release["allowed_source_ids"]
    assert len(allowed) == len(set(allowed)) and set(source_ids) <= set(allowed)
    source_lookup = {r["source_id"]:r for r in read(contract["source_manifest_path"])["rows"]}
    assert set(allowed) <= set(source_lookup), "release includes unknown sources"
    return release


def parent_gpu_lock(args, contract):
    owner = read(Path(contract["shared_gpu_lock"])/"owner.json")
    assert int(owner["pid"]) == args.parent_lock_owner and owner["pid"] != os.getpid()
    os.kill(int(owner["pid"]), 0)
    assert Path(owner["run"]).resolve().is_relative_to((Path(contract["project_root"])/"runs").resolve())
    for p in contract["old_gpu_locks"]:
        assert not Path(p).exists(), ("other scientific GPU lock", p)
    query = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"], text=True, timeout=15)
    for line in query.strip().splitlines():
        pid, name, memory = [v.strip() for v in line.split(",")]
        assert name == "/usr/libexec/gnome-remote-desktop-daemon" and os.readlink("/proc/"+pid+"/exe") == name and float(memory) <= 1024, ("GPU occupied", line)
    return dict(owner=owner, compute=query)


def rf(i):
    lo, hi = max(0, 640*i-1), 640*i+3439
    return dict(row=i, raw_samples_inclusive=[lo,hi], mfcc_columns_halfopen=[4*i,4*i+20],
                video_frames_halfopen=[i,i+5], score_audio_center_s=(lo+hi)/32000)


def build_indices(rows, alignments, replay_by_id):
    """Index arithmetic only: never alters media/embeddings or optimizes lag."""
    valid, robust, clocks, patches, coverage = {}, {}, [], [], []
    groups = defaultdict(list)
    for r in rows:
        uid = r["id"]; groups[r["sample_id"]].append(r)
        rec, fa = replay_by_id[uid], alignments[uid]
        assert fa["audio_sha256"] == r["canonical_audio_sha256"]
        n, samples = rec["windows"], rec["audio_samples"]
        windows = [rf(i) for i in range(n)]
        assert n > 0 and windows[-1]["raw_samples_inclusive"][1] < samples
        delay = rec["clock"]["lag_samples_scoring_minus_canonical"]/16000
        clocks.append(dict(id=uid, rows=windows, delay_s=delay, audio_samples=samples))
        centers = [w["score_audio_center_s"] for w in windows]
        tokens, ts = fa["tokens"], fa["timestamps_ms"]
        text = source_norm(r["text"])
        same_text = "".join(tokens) == text and len(ts) == len(tokens)
        structural_ok = same_text and not fa.get("issues")
        valid[uid], robust[uid] = set(), set()
        for ci, (char, interval) in enumerate(zip(tokens, ts)):
            center = sum(interval)/2000
            variants = []
            for shift in (-.04, 0., .04):
                anchor = center+delay+shift
                j = min(range(n), key=lambda i: abs(centers[i]-anchor))
                good = structural_ok and j>=5 and j+5<n
                ids = list(range(j-5,j+6)) if good else None
                variants.append(dict(shift_s=shift, mapped_anchor_s=anchor, nearest_row=j,
                    nearest_center_s=centers[j], center_error_ms=(centers[j]-anchor)*1000,
                    support_ok=good, patch_rows=ids,
                    patch_raw_samples_inclusive=[rf(j-5)["raw_samples_inclusive"][0],rf(j+5)["raw_samples_inclusive"][1]] if good else None))
            if variants[1]["support_ok"]: valid[uid].add(ci)
            if all(v["support_ok"] for v in variants): robust[uid].add(ci)
            patches.append(dict(id=uid, source=r["sample_id"], arm=r["arm"], draw=r["draw"],
                character_index=ci, character=char, context="".join(tokens[max(0,ci-2):ci+3]),
                canonical_interval_ms=interval, canonical_center_s=center, score_delay_s=delay,
                long_interval_flag=interval[1]-interval[0]>1000, FA_truth="UNVERIFIED",
                alignment_structural_issues=fa.get("issues", []), variants=variants))
        coverage.append(dict(id=uid, characters=len(text), valid_anchors=len(valid[uid]),
            coverage=len(valid[uid])/len(text) if text else 0, same_text=same_text,
            temporary_measurability_gate=len(valid[uid])>=5 and len(valid[uid])/max(len(text),1)>=.8))
    pairs, source_support = [], []
    cov = {r["id"]:r for r in coverage}
    for source, group in sorted(groups.items()):
        den = len(source_norm(group[0]["text"]))
        common = set.intersection(*(valid[r["id"]] for r in group))
        common_robust = set.intersection(*(robust[r["id"]] for r in group))
        source_support.append(dict(source=source, characters=den,
            all_clips_individually_pass=all(cov[r["id"]]["temporary_measurability_gate"] for r in group),
            common_7arm_anchors=len(common), common_character_indices=sorted(common),
            common_7arm_coverage=len(common)/max(den,1), common_7arm_gate=len(common)>=5 and len(common)/max(den,1)>=.8,
            all_shifts_common_anchors=len(common_robust)))
        for family, a, b in arm_pairs(group):
            shared = sorted(valid[a] & valid[b])
            pairs.append(dict(source=source, family=family, a_id=a, b_id=b,
                common_character_indices=shared, text_characters=den, coverage=len(shared)/max(den,1),
                temporary_measurability_gate=len(shared)>=5 and len(shared)/max(den,1)>=.8))
    return dict(rows=clocks), dict(patches=patches, coverage=coverage, pair_coverage=pairs,
                                  source_common_support=source_support, FA_truth="UNVERIFIED", local615ms_sufficiency="UNVERIFIED")


def fa_inputs(rows):
    """FA needs waveform/text only; never copy score/track payloads to root."""
    return [dict(id=r["id"],audio_path=r["audio_path"],audio_sha256=r["audio_sha256"],text=r["text"]) for r in rows]


def alignment_record(row, raw, duration_ms):
    text=source_norm(row["text"])
    tokens=list(source_norm(raw.get("text", "")))
    stamps=raw.get("timestamp", []);issues=[];last=0
    if "".join(tokens)!=text or len(tokens)!=len(stamps):issues.append("text_or_count")
    for lo,hi in stamps:
        if not 0<=lo<hi<=duration_ms:issues.append("range_or_nonpositive_interval")
        if lo<last:issues.append("nonmonotonic")
        last=hi
    return dict(id=row["id"],audio_sha256=row["audio_sha256"],text=text,tokens=tokens,
        timestamps_ms=stamps,issues=sorted(set(issues)),truth="UNVERIFIED",
        long_intervals=[dict(index=i,interval_ms=t) for i,t in enumerate(stamps) if t[1]-t[0]>1000],
        uncovered_tail_ms=duration_ms-(stamps[-1][1] if stamps else 0))


def fa_worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import torch
    from funasr import AutoModel
    torch.set_num_threads(2)
    contract = read(args.contract)
    for p,h in contract["fa_pins"].items(): checked(p,h)
    model = AutoModel(model=contract["fa_model_root"], device="cpu", disable_update=True)
    rows = read(args.manifest)["rows"]; aligned = []
    for r in rows:
        checked(r["audio_path"], r["audio_sha256"])
        text = source_norm(r["text"])
        raw = model.generate(input=(r["audio_path"],text), data_type=("sound","text"))[0]
        with wave.open(r["audio_path"]) as w: duration_ms=w.getnframes()/16
        aligned.append(alignment_record(r,raw,duration_ms))
    put(Path(args.output_root)/"alignment.json", dict(rows=aligned, status="CPU_ALIGNMENT_STRUCTURE_ONLY", training=False))


def consume(args):
    contract = read(args.contract)
    checked(contract["source_manifest_path"], contract["source_manifest_sha256"])
    checked(contract["split_freeze_path"], contract["split_freeze_sha256"])
    checked(contract["old_registry_path"], contract["old_registry_sha256"])
    source_by_id = {r["source_id"]:r for r in read(contract["source_manifest_path"])["rows"]}
    media = read(args.manifest)["rows"]
    groups = validate_manifest(media, source_by_id)
    require_release(args, contract, groups)
    out = Path(args.output_root)
    if (out/"completion.json").exists():
        completion = read(out/"completion.json")
        assert completion["media_manifest_sha256"] == sha(args.manifest)
        assert completion["consumer_contract_sha256"] == sha(args.contract)
        for item in completion["files"]: checked(out/item["path"],item["sha256"])
        print(json.dumps(dict(status="REUSED_VERIFIED_COMPLETE_BATCH", units=len(media))))
        return
    assert not out.exists(), "partial consumer attempt retained; parent must review, never silently redo"
    out.mkdir(parents=True)
    put(out/"launch.json",dict(utc=utc(),pid=os.getpid(),command=sys.argv,
        media_manifest_sha256=sha(args.manifest),consumer_contract_sha256=sha(args.contract),release_sha256=sha(args.release)))
    started=time.time()
    try:
        checked(contract["legacy_parser_path"],contract["legacy_parser_sha256"])
        parser=import_path("frozen_legacy_parser",contract["legacy_parser_path"])
        old={r["id"]:r for r in read(contract["old_registry_path"])["rows"]}
        all_rows, new_rows, old_selected = [], [], []
        for unit in media:
            bound=parser.parse_score(unit)
            if unit["id"] in old:
                r=old[unit["id"]]
                for k,h in (("sequence_path","sequence_sha256"),("scoring_audio_path","scoring_audio_sha256"),("replay_path","replay_sha256")):
                    checked(r[k],r[h])
                assert r["score_path"]==unit["score_path"] and r["score_sha256"]==unit["score_sha256"]
                assert r["canonical_audio_sha256"]==bound["audio_sha256"]
                checked(r["FA"]["path"],r["FA"]["sha256"])
                old_selected.append(dict(r,split=unit["split"],original_cache_reused=True))
            else:
                new_rows.append(dict(bound,split=unit["split"]))
        if new_rows:
            gate=parent_gpu_lock(args,contract)
            put(out/"gpu_preflight.json",gate)
            assert shutil.disk_usage(out).free>=contract["minimum_free_bytes"]+contract["next_item_bytes"]
            checked(contract["engine_path"],contract["engine_sha256"])
            for p,h in contract["engine_source_pins"].items(): checked(p,h)
            engine=import_path("frozen_source200_engine",contract["engine_path"])
            expected=contract["engine_environment"]
            assert sys.executable==expected["executable"]
            for k,v in (("torch",engine.torch.__version__),("numpy",engine.np.__version__),("scipy",engine.scipy.__version__),("opencv",engine.cv2.__version__)):
                assert v==expected[k], (k,v,expected[k])
            engine.R=Path(contract["project_root"]);engine.OFF=engine.R/"third_party/syncnet_python"
            net=engine.SyncNetInstance(device="cuda")
            net.loadParameters(str(engine.OFF/"data/syncnet_v2.model"));net.__S__.eval();engine.torch.set_num_threads(2)
            for split in ("fit","development","confirmation"):
                selected=[r for r in new_rows if r["split"]==split]
                if not selected:continue
                feature_root=out/SPLIT_DIR[split]/"feature_engine";feature_root.mkdir(parents=True)
                put(feature_root/"protocol.json",dict(tolerances=contract["tolerances"],training=False,contract_sha256=sha(args.contract)))
                engine.OUT=feature_root
                for r in selected:
                    assert shutil.disk_usage(out).free>=contract["minimum_free_bytes"]+contract["next_item_bytes"]
                    rec=engine.replay_one(r,net,False)
                    assert rec["pass_replay"] and rec["mfcc_exact"] and rec["distance_replay_exact"] and rec["clock"]["pass_clock"], r["id"]
            del net
            import gc
            gc.collect();engine.torch.cuda.empty_cache()
            fa_input=out/"fa_input.json";put(fa_input,dict(rows=fa_inputs(new_rows)))
            cmd=[contract["fa_python"],str(Path(__file__).resolve()),"fa-worker","--contract",args.contract,"--manifest",str(fa_input),"--output-root",str(out)]
            with (out/"fa_stdout.log").open("x",encoding="utf-8") as so,(out/"fa_stderr.log").open("x",encoding="utf-8") as se:
                run=subprocess.run(cmd,stdout=so,stderr=se,timeout=1800,env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",HF_HUB_OFFLINE="1"))
            put(out/"fa_exit.json",dict(returncode=run.returncode,command=cmd,utc=utc()))
            assert run.returncode==0,"CPU FA failed; retained partial batch"
            fa_rows={r["id"]:r for r in read(out/"alignment.json")["rows"]}
            for r in new_rows:
                item=out/SPLIT_DIR[r["split"]]/"feature_engine/items"/r["id"]
                rec=read(item/"replay.json")
                all_rows.append(dict(id=r["id"],sample_id=r["sample_id"],speaker_id=r["speaker_id"],arm=r["arm"],draw=r["draw"],text=r["text"],split=r["split"],
                    sequence_path=str(item/"sequence.npz"),sequence_sha256=sha(item/"sequence.npz"),sequence_bytes=(item/"sequence.npz").stat().st_size,
                    scoring_audio_path=str(item/"scoring_audio.wav"),scoring_audio_sha256=sha(item/"scoring_audio.wav"),
                    replay_path=str(item/"replay.json"),replay_sha256=sha(item/"replay.json"),original_cache_reused=False,
                    FA=dict(path=str(out/"alignment.json"),sha256=sha(out/"alignment.json"),id=r["id"]),
                    canonical_audio_path=r["audio_path"],canonical_audio_sha256=r["audio_sha256"],video_path=r["video_path"],video_sha256=r["video_sha256"],
                    score_path=r["score_path"],score_sha256=r["score_sha256"],activesd_path=r["activesd_path"],activesd_sha256=r["activesd_sha256"],
                    rows=rec["windows"],score_float32=rec["original"]["C"]))
        all_rows.extend(old_selected)
        assert len(all_rows)==len(media)
        fa_cache={};aligned={};replays={}
        for r in all_rows:
            fa=r["FA"]
            if fa["path"] not in fa_cache:
                f=read(fa["path"]);fa_cache[fa["path"]]={x["id"]:x for x in f.get("rows",f.get("audio",[]))}
            aligned[r["id"]]=fa_cache[fa["path"]][fa["id"]]
            replays[r["id"]]=read(r["replay_path"])
        split_files=[];summary={}
        for split in ("fit","development","confirmation"):
            rr=sorted((r for r in all_rows if r["split"]==split),key=lambda r:r["id"])
            if not rr:continue
            folder=out/SPLIT_DIR[split]
            clock,patch=build_indices(rr,aligned,replays)
            put(folder/"manifest.json",dict(schema="audio200_ready_consumer_v1",rows=rr,split=split,no_training=True))
            put(folder/"window_clock.json",clock);put(folder/"patch_index.json",patch)
            put(folder/"alignment.json",dict(rows=[aligned[r["id"]] for r in rr],FA_truth="UNVERIFIED"))
            split_files.append(dict(split=split,path=str(folder/"manifest.json"),sha256=sha(folder/"manifest.json"),units=len(rr)))
            summary[split]=dict(units=len(rr),sources=len({r["sample_id"] for r in rr}),windows=sum(r["rows"] for r in rr),pairs=len(patch["pair_coverage"]),
                common_7arm_support_sources=sum(x["common_7arm_gate"] for x in patch["source_common_support"]))
        # Root receipt exposes technical counts and paths, no confirmation scores/rankings.
        files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob("*")) if p.is_file()]
        put(out/"completion.json",dict(status="COMPLETE_FEATURE_FA_WINDOW_DATA_ONLY",utc=utc(),elapsed_s=time.time()-started,
            media_manifest_sha256=sha(args.manifest),consumer_contract_sha256=sha(args.contract),consumer_source_sha256=sha(__file__),
            units=len(all_rows),new_forward=len(new_rows),reused=len(old_selected),summary=summary,split_manifests=split_files,files=files,no_training=True))
        print(json.dumps(dict(status="COMPLETE_FEATURE_FA_WINDOW_DATA_ONLY",units=len(all_rows),new_forward=len(new_rows),reused=len(old_selected),summary=summary)))
    except Exception as e:
        put(out/"failure.json",dict(status="HOLD_FAILURE_NO_AUTOMATIC_RETRY",utc=utc(),error=repr(e),no_training=True))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","consume","fa-worker"])
    p.add_argument("--project-root",default="/home/wjj/tts-audio/tts-exp")
    p.add_argument("--output-root",required=True)
    p.add_argument("--manifest");p.add_argument("--contract");p.add_argument("--release")
    p.add_argument("--parent-lock-owner",type=int)
    args=p.parse_args()
    globals()[args.stage.replace("-","_")](args)


if __name__=="__main__":main()
