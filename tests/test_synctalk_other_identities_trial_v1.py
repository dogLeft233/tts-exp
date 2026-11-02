"""Control-flow tests: terminal identity failures continue, lost jobs never replay."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SOURCE=Path(__file__).resolve().parents[1]/'scripts/experiments/synctalk_multispeaker_full_20261017'
sys.path.insert(0,str(SOURCE))
spec=importlib.util.spec_from_file_location('other_trial_control',SOURCE/'other_identities_trial_v1.py')
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)

class TrialLifecycleTests(unittest.TestCase):
    def test_remote_checkpoint_paths_stay_posix_on_windows(self):
        spec=importlib.util.spec_from_file_location('handoff',SOURCE/'other_identities_controller_handoff_v2.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        prefix=trial.RUN+'/training/PatrickLeahy/seed42/gain_+0/coarse/'
        previous={'epoch':1,'checkpoint_path':prefix+'ep0001/attempt01/workspace/checkpoints/ngp_ep0001.pth'}
        current={'epoch':2,'checkpoint_path':prefix+'ep0002/attempt01/workspace/checkpoints/ngp_ep0002.pth'}
        old,new,local_old,local_new,relative=module.checkpoint_locations(previous,current)
        self.assertEqual(old.as_posix(),previous['checkpoint_path']);self.assertNotIn('\\',str(old))
        self.assertEqual(relative,'training/PatrickLeahy/seed42/gain_+0/coarse/ep0001/attempt01')
        self.assertTrue(local_old.is_relative_to(trial.OUT))

    def prepare(self,root):
        (root/'qualification').mkdir()
        (root/'qualification/result.json').write_text(json.dumps({'passed':True}))
        (root/'contract.json').write_text(json.dumps({'sources':{},'identities':['PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']}))

    def test_terminal_failure_is_preserved_before_each_identity_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.prepare(root);events=[]
            def action(verb,identity,**kwargs):
                events.append(('launch',identity));return {'state':'completed','returncode':1,'passed':False}
            def persist(relative):events.append(('preserve',relative));return {'passed':True}
            with patch.object(trial,'OUT',root),patch.object(trial,'retire',side_effect=lambda i:events.append(('retire',i))),patch.object(trial,'restore',side_effect=lambda i:events.append(('restore',i))),patch.object(trial,'remote_action',side_effect=action),patch.object(trial,'remote_python',return_value=b'false'),patch.object(trial,'persist',side_effect=persist):
                trial.run()
            result=json.loads((root/'receipt.json').read_text())
            self.assertEqual(result['status'],'all_three_trials_finished');self.assertFalse(result['passed'])
            self.assertEqual([f['identity'] for f in result['failures']],['PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'])
            for identity in result['identities']:
                launch=events.index(('launch',identity));retire=events.index(('retire',identity))
                self.assertEqual(events[launch+1][0],'preserve');self.assertLess(launch+1,retire)
            self.assertEqual(sum(e[0]=='launch' for e in events),3)

    def test_lost_job_stops_without_retry_or_next_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.prepare(root)
            with patch.object(trial,'OUT',root),patch.object(trial,'retire'),patch.object(trial,'restore') as restore,patch.object(trial,'remote_action',return_value={'state':'lost','passed':False}) as action,patch.object(trial,'persist') as persist:
                with self.assertRaisesRegex(AssertionError,'lost job'):trial.run()
            self.assertEqual(action.call_count,1);self.assertEqual(restore.call_count,1);persist.assert_not_called()

if __name__=='__main__':unittest.main()
