import importlib.util,unittest
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/select_replacement_identity.py'
spec=importlib.util.spec_from_file_location('replacement',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReplacementOrder(unittest.TestCase):
    def test_metadata_order_is_preserved_and_disjoint_short_segments_are_not_summed(self):
        rows=module.parse_annotation('Zed0.mp4 00:00-03:00 04:00-07:00\nZed1.mp4 00:00-02:00\nBeta0.mp4 00:00-05:00\nBeta1.mp4 00:00-02:00\nAlpha0.mp4 00:00-05:00\nAlpha1.mp4 00:00-02:00','WDA')
        urls={(r['family'],r['key']):f"https://youtube.com/watch?v={r['key']}" for r in rows}
        candidates,audit=module.choose(rows,urls)
        self.assertEqual([x['identity'] for x in candidates],['Beta','Alpha']);self.assertFalse(audit[0]['eligible'])
        self.assertEqual(candidates[0]['main']['seconds'],300);self.assertEqual(candidates[0]['test']['seconds'],120)

    def test_same_recording_alias_and_previously_failed_identity_cannot_replace(self):
        rows=module.parse_annotation('DebbieDingell0.mp4 00:00-05:00\nDebbieDingell1.mp4 00:00-02:00\nNext0.mp4 00:00-05:00\nNext1.mp4 00:00-02:00','WDA')
        urls={(r['family'],r['key']):'https://youtube.com/watch?v=same-source' for r in rows}
        candidates,audit=module.choose(rows,urls);self.assertEqual(candidates,[])
        self.assertEqual(audit[0]['reason'],'retained/rejected/pretrained identity')

    def test_exact_budget_and_first_eligible_orientation_are_frozen(self):
        rows=module.parse_annotation('Only0.mp4 00:00-04:55\nOnly1.mp4 00:00-01:50\nOnly2.mp4 00:00-06:00','WDA')
        urls={(r['family'],r['key']):f"https://youtube.com/watch?v={r['key']}" for r in rows}
        candidates,_=module.choose(rows,urls);first=candidates[0]
        self.assertEqual(first['main']['key'],'Only0');self.assertEqual(first['test']['key'],'Only1')
        self.assertEqual(first['main']['seconds']+first['test']['seconds'],405)


if __name__=='__main__':unittest.main()
