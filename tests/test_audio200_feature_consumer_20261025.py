import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location("consumer",Path(__file__).resolve().parents[1]/"scripts/experiments/audio200_feature_consumer_20261025.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows():
    return [dict(id=f"S_{a}_{d}",sample_id="S",speaker_id="S0001",arm=a,draw=d,text="abcde",canonical_audio_sha256=a+str(d)) for a,d in sorted(m.EXPECTED_ARMS)]


class Contracts(unittest.TestCase):
    def test_fixed_pair_family_set(self):
        pairs=m.arm_pairs(rows())
        self.assertEqual(len(pairs),15)
        self.assertEqual(sum("within" not in f for f,a,b in pairs),9)
        self.assertNotIn(("L17-L06","S_L17_1","S_L06_2"),pairs)

    def test_shared_support_is_intersection_not_individual_gate(self):
        rr=rows();align={};replay={}
        for i,r in enumerate(rr):
            align[r["id"]]=dict(audio_sha256=r["canonical_audio_sha256"],tokens=list("abcde"),timestamps_ms=[[300,340],[400,440],[500,540],[600,640],[700,740]],issues=[])
            replay[r["id"]]=dict(windows=30,audio_samples=24000,clock=dict(lag_samples_scoring_minus_canonical=0))
        clock,index=m.build_indices(rr,align,replay)
        self.assertEqual(index["source_common_support"][0]["common_7arm_anchors"],5)
        # One arm has a bad timestamp structure: exclude its anchors, retain all7 rows/pairs.
        align[rr[0]["id"]]["issues"]=["range"]
        _,index=m.build_indices(rr,align,replay)
        self.assertEqual(index["source_common_support"][0]["common_7arm_anchors"],0)
        self.assertEqual(len(index["coverage"]),7)
        self.assertEqual(len(index["pair_coverage"]),15)

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


if __name__=="__main__":unittest.main()
