import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('nl17',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_nl17_production_20261025.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class NL17Projection(unittest.TestCase):
    def test_four_arm_projection_preserves_every_original_field(self):
        rows=[dict(id=f's_{a}_{d}',sample_id='s',arm=a,draw=d,score_float32=float(d),FA={'path':'old'})
              for a,d in [('N',0),('L06',1),('L06',2),('L06',3),('L17',1),('L17',2),('L17',3)]]
        got=m.select_four(rows)
        self.assertEqual(len(got),4)
        self.assertFalse(any(r['arm']=='L06' for r in got))
        for r in got:self.assertEqual(r,next(x for x in rows if x['id']==r['id']))
        got[0]['FA']['path']='changed'
        self.assertEqual(rows[0]['FA']['path'],'old')

    def test_missing_draw_cannot_be_silently_dropped(self):
        with self.assertRaises(AssertionError):
            m.select_four([dict(id='s_N_0',sample_id='s',arm='N',draw=0)])

    def test_subset_only_changes_declared_group_routes_and_count(self):
        old=dict(arm='L17',samples=[{'sample_id':'a'},{'sample_id':'b'}],roles=['L17_norm1','L17_norm2','L17_norm3'],
            expected_cells=6,raw_manifest_path='oldraw',analysis_protocol_path='oldanalysis',source_pins={'old':'sha'},
            generation_kwargs={'temperature':.9},seed_bases=[1,2,3],target_rms=.037)
        snapshot=copy.deepcopy(old);cell=Path('/new/bridge/L17')
        new=m.projected_config(old,old['samples'][:1],cell)
        self.assertEqual(old,snapshot)
        self.assertEqual(new['expected_cells'],3)
        for k in set(old)-{'samples','expected_cells','raw_manifest_path','analysis_protocol_path'}:
            self.assertEqual(new[k],old[k])

    def test_backend_is_frozen_original_source(self):
        h=m.backend()
        self.assertEqual(h.sha(m.BACKEND),m.BACKEND_SHA)
        self.assertEqual(h.TARGET,.0376838172675746)
        self.assertEqual(h.BASES,[2026102201,2026102202,2026102203])


if __name__=='__main__':unittest.main()
