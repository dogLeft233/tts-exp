import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('audio200prod',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_production_20261025.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProductionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.sources=m.read(root/'runs/audio_data_inventory200_20261024_attempt05/selected600_natural_manifest.json')
        cls.old=m.read(root/'runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json')
        cls.original=m.read(root/'runs/qwen_nrt_expansion40_20261022_attempt01/protocol.json')

    def test_roster_has_exact_reuse_and_frozen_pilot(self):
        new=m.make_roster(self.sources,self.old,self.original)
        self.assertEqual(len(new),560)
        self.assertEqual([r['source_id'] for r in new[:2]],['BAC009S0002W0322','BAC009S0002W0363'])
        self.assertEqual(7*len(new)+len(self.old['rows']),4200)

    def test_original_source_sha_mutation_fails(self):
        changed=copy.deepcopy(self.sources)
        sid=self.old['rows'][0]['sample_id']
        next(r for r in changed['rows'] if r['source_id']==sid)['sha256']='bad'
        with self.assertRaises(AssertionError):
            m.make_roster(changed,self.old,self.original)

    def test_original_text_mutation_fails(self):
        changed=copy.deepcopy(self.sources)
        sid=self.old['rows'][0]['sample_id']
        next(r for r in changed['rows'] if r['source_id']==sid)['text']='changed'
        with self.assertRaises(AssertionError):
            m.make_roster(changed,self.old,self.original)

    def test_no_extra_or_missing_original_arm(self):
        changed=copy.deepcopy(self.old)
        changed['rows'][0]['arm']='R'
        with self.assertRaises(AssertionError):
            m.make_roster(self.sources,changed,self.original)

    def test_seed_is_historical_not_utterance_dependent(self):
        self.assertEqual(m.generation_seed('S0002',1),2026102401)
        self.assertEqual(m.generation_seed('S0002',3),2026102403)
        with self.assertRaises(ValueError):
            m.generation_seed('S0002',4)

    def test_append_only_receipt_and_sha_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json'
            m.put(p,{'x':1})
            m.verify({str(p):m.sha(p)})
            with self.assertRaises(FileExistsError):
                m.put(p,{'x':2})
            p.write_text('changed')
            with self.assertRaises(ValueError):
                m.verify({str(p):'wrong'})


if __name__=='__main__':
    unittest.main()
