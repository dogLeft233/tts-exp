import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("consumer",Path(__file__).resolve().parents[1]/"scripts/experiments/audio200_nl17_feature_consumer_20261025.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows():
    return [dict(id=f"S_{a}_{d}",sample_id="S",speaker_id="S0001",arm=a,draw=d,text="abcde",canonical_audio_sha256=a+str(d)) for a,d in sorted(m.EXPECTED_ARMS)]


class Contracts(unittest.TestCase):
    def test_fixed_pair_family_set(self):
        pairs=m.arm_pairs(rows())
        self.assertEqual(len(pairs),6)
        self.assertEqual(sum("within" not in f for f,a,b in pairs),3)
        self.assertEqual({f for f,a,b in pairs},{"N-L17","L17-within"})

    def test_shared_support_is_intersection_not_individual_gate(self):
        rr=rows();align={};replay={}
        for i,r in enumerate(rr):
            align[r["id"]]=dict(audio_sha256=r["canonical_audio_sha256"],tokens=list("abcde"),timestamps_ms=[[300,340],[400,440],[500,540],[600,640],[700,740]],issues=[])
            replay[r["id"]]=dict(windows=30,audio_samples=24000,clock=dict(lag_samples_scoring_minus_canonical=0))
        clock,index=m.build_indices(rr,align,replay)
        self.assertEqual(index["source_common_support"][0]["common_4arm_anchors"],5)
        # One arm has a bad timestamp structure: exclude anchors, retain all4 rows.
        align[rr[0]["id"]]["issues"]=["range"]
        _,index=m.build_indices(rr,align,replay)
        self.assertEqual(index["source_common_support"][0]["common_4arm_anchors"],0)
        self.assertEqual(len(index["coverage"]),4)
        self.assertEqual(len(index["pair_coverage"]),6)

    def test_center_not_window_start_and_no_edge_padding(self):
        self.assertAlmostEqual(m.rf(10)["score_audio_center_s"],.5074375)
        rr=rows();align={r["id"]:dict(audio_sha256=r["canonical_audio_sha256"],tokens=["a"],timestamps_ms=[[0,10]],issues=[]) for r in rr}
        replay={r["id"]:dict(windows=20,audio_samples=18000,clock=dict(lag_samples_scoring_minus_canonical=0)) for r in rr}
        _,index=m.build_indices(rr,align,replay)
        self.assertTrue(all(not x["variants"][1]["support_ok"] and x["variants"][1]["patch_rows"] is None for x in index["patches"]))

    def test_fa_inputs_cannot_copy_confirmation_scores(self):
        row=dict(id="x",audio_path="a.wav",audio_sha256="sha",text="abc",track={"C":9},score_float32=9,split="confirmation")
        self.assertEqual(set(m.fa_inputs([row])[0]),{"id","audio_path","audio_sha256","text"})

    def test_zero_length_alignment_is_flagged_and_preserved(self):
        row=dict(id="x",audio_sha256="sha",text="a")
        rec=m.alignment_record(row,dict(text="a",timestamp=[[10,10]]),100)
        self.assertIn("range_or_nonpositive_interval",rec["issues"])
        self.assertEqual(rec["timestamps_ms"],[[10,10]])
        good=m.alignment_record(row,dict(text="a",timestamp=[[10,20]]),100)
        self.assertEqual(good["issues"],[])

    def test_arm_only_reuse_selection_keeps_all_fixed_sources_and_draws(self):
        seven={("N",0)}|{(a,d) for a in ("L06","L17") for d in (1,2,3)}
        all_rows=[dict(id=f"S{i}_{a}_{d}",sample_id=f"S{i}",arm=a,draw=d,score_float32=(-1)**i*999) for i in range(62) for a,d in seven]
        selected=m.select_reuse_rows(all_rows)
        self.assertEqual(len(selected),248)
        self.assertEqual(len({r["sample_id"] for r in selected}),62)
        self.assertTrue(all(r["arm"] in ("N","L17") for r in selected))
        # Outcome values are irrelevant to the deterministic arm-only selection.
        changed=[dict(r,score_float32=12345) for r in all_rows]
        self.assertEqual([r["id"] for r in selected],[r["id"] for r in m.select_reuse_rows(changed)])
        with self.assertRaises(AssertionError):m.arm_pairs([dict(r) for r in selected[:3]])


if __name__=="__main__":unittest.main()
