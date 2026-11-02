import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("finalizer",Path(__file__).resolve().parents[1]/"scripts/experiments/audio200_nl17_finalize_20261025.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows():
    return [dict(id=f"S_{a}_{d}",sample_id="S",arm=a,draw=d,text="abcdefghij",rows=30) for a,d in sorted(m.ARMS)]


class FinalizerContracts(unittest.TestCase):
    def test_preserves_historical_original_cache_flag(self):
        row=dict(id="x",original_cache_reused=False,score_float32=1.234,FA={"id":"x","path":"old","sha256":"oldsha"})
        source=dict(split="fit",source_id="x",ref_source_id="x",sha256="raw",pcm_sha256="pcm")
        out=m.add_source_fields(row,source,True)
        for k,v in row.items():self.assertEqual(out[k],v)
        self.assertFalse(out["original_cache_reused"])
        self.assertTrue(out["reused_in_nl17_phase"])

    def test_parent_pass_requires_exact_artifact_identity(self):
        a=dict(status="PASS_PARENT_NL17_BATCH_REPLAY_CLOCK_PATCH",batch_id="batch000",source_ids=["S"],
            manifest_path="fit/manifest.json",manifest_sha256="m",consumer_completion_path="completion.json",consumer_completion_sha256="c",
            clock_path="fit/window_clock.json",patch_path="fit/patch_index.json",auditor_source_sha256=m.AUDITOR_SHA,
            auditor_source_path=str(Path(__file__).resolve()))
        batch={"id":"batch000","source_ids":["S"]}
        m.check_parent_audit(a,batch,"fit/manifest.json","m","completion.json","c")
        with self.assertRaises(AssertionError):m.check_parent_audit(a,{"id":"batch001","source_ids":["S"]},"fit/manifest.json","m","completion.json","c")
        with self.assertRaises(AssertionError):m.check_parent_audit(a,batch,"fit/manifest.json","new","completion.json","c")
        with self.assertRaises(AssertionError):m.check_parent_audit(a,{"id":"batch000","source_ids":["X"]},"fit/manifest.json","m","completion.json","c")

    def test_individual_pass_is_not_strict_common_support(self):
        rr=rows();patches=[];valid={}
        for i,r in enumerate(rr):
            # All7 have9/10 coverage individually, different missing characters.
            valid[r["id"]]=set(range(10))-{i}
            for ci in range(10):
                patches.append(dict(id=r["id"],character_index=ci,character=r["text"][ci],
                    variants=[dict(shift_s=s,support_ok=ci in valid[r["id"]],nearest_row=10,patch_rows=list(range(5,16))) for s in (-.04,0.,.04)]))
        pairs=[]
        for family,a,b in m.expected_pairs(rr):
            cc=sorted(valid[a]&valid[b]);pairs.append(dict(family=family,a_id=a,b_id=b,common_character_indices=cc,temporary_measurability_gate=len(cc)>=8))
        pp,ss=m.verify_index_part(dict(patches=patches,pair_coverage=pairs),rr)
        self.assertEqual(len(pp),6)
        self.assertTrue(ss[0]["all_clips_individually_pass"])
        self.assertEqual(ss[0]["common_4arm_anchors"],6)
        self.assertFalse(ss[0]["common_4arm_gate"])
        self.assertEqual({p[0] for p in m.expected_pairs(rr)},{"N-L17","L17-within"})

    def test_changed_file_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.write_bytes(b"new")
            with self.assertRaises(AssertionError):m.FreshPins().pin(p,"old")

    def test_exact2400_roster_rejects_split_or_identity_drift(self):
        sources={};rr=[]
        for speaker in range(200):
            split="fit" if speaker<100 else "development" if speaker<140 else "confirmation"
            for utterance in range(3):
                sid=f"S{speaker:04d}U{utterance}"
                source=dict(source_id=sid,ref_source_id=sid,speaker_id=f"S{speaker:04d}",split=split,normalized_text=sid,sha256=sid,pcm_sha256=sid)
                sources[sid]=source
                for a,d in m.ARMS:
                    rr.append(dict(id=f"{sid}_{a}_{d}",sample_id=sid,speaker_id=source["speaker_id"],split=split,arm=a,draw=d,text=sid,ref_source_id=sid,natural_source_sha256=sid))
        self.assertEqual(len(m.validate_full_roster(rr,sources)),600)
        altered=[dict(r) for r in rr];altered[-1]["split"]="fit"
        with self.assertRaises(AssertionError):m.validate_full_roster(altered,sources)
        duplicated=rr[:-1]+[rr[0]]
        with self.assertRaises(AssertionError):m.validate_full_roster(duplicated,sources)

    def test_actual_producer_reuse_registry_binding(self):
        root=Path(__file__).resolve().parents[1]
        reuse=m.read(root/"runs/audio200_production_20261025_attempt01/reuse280.json")
        old_path=root/"runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json"
        self.assertEqual(m.sha(old_path),m.OLD_SHA)
        old=m.read(old_path)["rows"]
        m.validate_reuse_binding(reuse,old)
        wrong=dict(reuse,source_manifest_sha256=m.SOURCE_SHA)
        with self.assertRaises(AssertionError):m.validate_reuse_binding(wrong,old)
        protocol=m.read(root/"runs/audio200_production_20261025_attempt01/protocol.json")
        self.assertEqual(protocol["source_pins"][protocol["source_manifest_path"]],m.SOURCE_SHA)
        self.assertEqual(protocol["source_pins"][protocol["reuse_manifest_path"]],m.OLD_SHA)

    def test_raw_canonical_binding_preserves_unknown_eos(self):
        import json
        for eos in (None,False,True):
            with tempfile.TemporaryDirectory() as d:
                folder=Path(d);rawfile=folder/"raw.wav";rawfile.write_bytes(b"raw")
                pcm=folder/"pcm.wav";pcm.write_bytes(b"pcm")
                trace=folder/"trace.npy";trace.write_bytes(b"trace")
                sample=dict(sample_id="SRC",speaker_id="S0002",transcript="abc",input24_sha256="ref")
                gp=dict(samples=[sample],roles=["L17_norm1"],expected_cells=1)
                m.put(folder/"protocol.json",gp);digest=m.sha(folder/"protocol.json")
                desc=dict(path=str(trace),sha256=m.sha(trace))
                r=dict(sample_id="SRC",speaker_id="S0002",role="L17_norm1",model="L17",draw_index=1,
                    seed=2026102401,path=str(rawfile),sha256=m.sha(rawfile),target_text="abc",reference_text="abc",reference_audio_sha256="ref",
                    eos_in_talker_sequences=eos,duration_ratio=1.0,**{k:desc for k in ("reference_codes","new_codes","decode_concat_codes","full_decoded","talker_sequences")})
                m.put(folder/"raw_manifest.json",dict(count=1,records=[r],protocol_sha256=digest))
                m.put(folder/"generation/SRC/draw1.json",r)
                c=dict(sample_id="SRC",speaker_id="S0002",role="L17_norm1",path=str(pcm),sha256=m.sha(pcm),source_path=str(rawfile),source_sha256=m.sha(rawfile))
                m.put(folder/"audio_manifest.json",dict(count=1,records=[c],protocol_sha256=digest))
                bound=m.raw_generation_bindings(dict(path=str(folder),arm="L17",protocol_sha256=digest),m.FreshPins())["SRC_L17_1"]
                self.assertIs(bound["eos_in_talker_sequences"],eos)
                self.assertEqual(bound["eos_status"],"unknown" if eos is None else "present" if eos else "absent")
                c["source_sha256"]="different"
                (folder/"audio_manifest.json").write_text(json.dumps(dict(count=1,records=[c],protocol_sha256=digest)),encoding="utf-8")
                with self.assertRaises(AssertionError):m.raw_generation_bindings(dict(path=str(folder),arm="L17",protocol_sha256=digest),m.FreshPins())

    def test_authoritative_score_is_new_field_not_legacy_rewrite(self):
        old=dict(id="x",score_float32=1.000002,score_float32_provenance="numpy_float32_curve_reduction")
        result=m.attach_canonical_score(old,dict(C=1.0),1.000002)
        self.assertEqual(result["score_float32"],1.000002)
        self.assertEqual(result["official_syncnet_score_float32"],1.0)
        self.assertEqual(result["score_float32_provenance"],"numpy_float32_curve_reduction")
        with self.assertRaises(AssertionError):m.attach_canonical_score(old,dict(C=1.0),1.0)
        new=m.attach_canonical_score(dict(id="y",score_float32=1.0,score_float32_provenance="torch_float32_curve_reduction"),dict(C=1.0),1.0)
        self.assertEqual(new["score_float32_provenance"],"torch_float32_curve_reduction")

    def test_physical_seven_arm_index_is_explicitly_filtered(self):
        rr=rows();ids={r["id"] for r in rr}
        physical=dict(patches=[dict(id=r["id"]) for r in rr]+[dict(id="S_L06_1")],
            pair_coverage=[dict(family=f,a_id=a,b_id=b) for f,a,b in m.expected_pairs(rr)]+[
                dict(family="N-L06",a_id="S_N_0",b_id="S_L06_1"),dict(family="L17-L06",a_id="S_L17_1",b_id="S_L06_1")])
        selected=m.phase_index(physical,rr)
        self.assertEqual({p["id"] for p in selected["patches"]},ids)
        self.assertEqual(len(selected["pair_coverage"]),6)
        self.assertEqual({p["family"] for p in selected["pair_coverage"]},{"N-L17","L17-within"})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,"frozen Torch available on execution host only")
    def test_actual_old_curve_official_torch_scalar(self):
        root=Path(__file__).resolve().parents[1]
        old=m.read(root/"runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json")["rows"][0]
        updated=m.canonical_score_row(dict(old,score_float32_provenance="numpy_float32_curve_reduction"))
        self.assertEqual(updated["score_float32"],old["score_float32"])
        self.assertEqual(updated[m.CANONICAL_LABEL_FIELD],m.read(old["replay_path"])["original"]["C"])


if __name__=="__main__":unittest.main()
