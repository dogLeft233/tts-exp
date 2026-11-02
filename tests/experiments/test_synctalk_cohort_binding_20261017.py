import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017'
spec=importlib.util.spec_from_file_location('cohort_binding',HERE/'cohort_binding.py')
binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)


class CohortBinding(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.contract=dict(version=binding.VERSION,identities=binding.IDS,members={})
        for identity in binding.IDS:
            self.contract['members'][identity]=dict(revision=binding.REVISIONS[identity],
                remote_root=binding.expected_root(binding.REMOTE,identity).as_posix(),
                persistent_root=binding.expected_root(binding.FROOT,identity).as_posix(),remote_pins=[])

    def write_selection(self):
        (self.root/'cohort_selection.json').write_text(json.dumps(self.contract))

    def admit(self):
        self.write_selection()
        (self.root/'data_qualification.json').write_text(json.dumps(dict(passed=True,training_ready=True,
            cohort_version=binding.VERSION,identities=binding.IDS,
            cohort_selection_sha256=binding.sha(self.root/'cohort_selection.json'))))

    def test_rejected_original_cannot_be_admitted_without_selection(self):
        self.assertEqual(binding.identity_root('ChrisVanHollen',self.root),self.root)
        with self.assertRaises(AssertionError):binding.identity_root('ChrisVanHollen',self.root,True)

    def test_changed_test_reference_namespace_and_duplicate_identity_rejected(self):
        for change in ['root','identity']:
            contract=copy.deepcopy(self.contract)
            if change=='root':self.contract['members']['CathyMcMorrisRodgers']['remote_root']=binding.REMOTE.as_posix()
            else:self.contract['identities']=binding.IDS[:3]+[binding.IDS[0]]
            self.write_selection()
            with self.assertRaises(AssertionError):binding.selection(self.root)
            self.contract=contract

    def test_revision_selection_sha_is_mandatory_at_every_execution_entry(self):
        self.admit()
        self.assertEqual(binding.identity_root('CathyMcMorrisRodgers',self.root,True),
            self.root/'dataset_revisions/CathyMcMorrisRodgers_recording2_v2')
        self.contract['members']['CathyMcMorrisRodgers']['note']='same paths but unadmitted changed provenance'
        self.write_selection()
        with self.assertRaises(AssertionError):binding.identity_root('CathyMcMorrisRodgers',self.root,True)

    def test_same_length_mutated_original_pcm_and_traversal_are_rejected(self):
        identity='ChrisVanHollen';selected=binding.expected_root(self.root,identity);selected.mkdir(parents=True)
        file=selected/'test_original.wav';file.write_bytes(b'originalPCM16')
        row=dict(path=file.name,bytes=file.stat().st_size,sha256=binding.sha(file))
        self.contract['members'][identity]['remote_pins']=[row];self.write_selection()
        self.assertTrue(binding.verify_member_pins(self.root,identity)['passed'])
        file.write_bytes(b'changed_PCM16')
        with self.assertRaises(AssertionError):binding.verify_member_pins(self.root,identity)
        row['path']='../test_original.wav';self.write_selection()
        with self.assertRaises(AssertionError):binding.verify_member_pins(self.root,identity)

    def test_restoration_uses_selected_archive_and_receiver_and_restores_globals_on_failure(self):
        from unittest.mock import patch
        sys.path.insert(0,str(HERE))
        import restore_cohort_supervision as restoration
        import cohort_binding as actual
        self.admit()
        # The actual production resolver must parse the fixture registry too.
        prior=restoration.original.ROOT,restoration.original.REMOTE
        rejected=self.root/'prepared/ChrisVanHollen/rejected_original.txt'
        rejected.parent.mkdir(parents=True);rejected.write_text('retain')
        def failure(identity,mode):
            self.assertEqual(restoration.original.ROOT,actual.expected_root(self.root,identity))
            self.assertEqual(restoration.original.REMOTE,actual.expected_root(actual.REMOTE,identity).as_posix())
            self.assertEqual(mode,'full')
            raise RuntimeError('synthetic receiver disconnection')
        with patch.object(restoration,'FROOT',self.root),patch.object(restoration.original,'run',failure):
            with self.assertRaisesRegex(RuntimeError,'receiver disconnection'):restoration.run('ChrisVanHollen','full')
        self.assertEqual((restoration.original.ROOT,restoration.original.REMOTE),prior)
        self.assertEqual(rejected.read_text(),'retain')


if __name__=='__main__':unittest.main()
