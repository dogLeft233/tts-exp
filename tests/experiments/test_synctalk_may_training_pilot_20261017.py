"""Counterfeit-control rejection for the training pilot, without a GPU import."""
import ast
import unittest
from pathlib import Path

source=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_may_training_pilot_20261017/validate_results.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
names={'ContractError','verify_training','verify_pcm'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)


class PilotControls(unittest.TestCase):
    def rows(self):
        return [dict(gain_db=g,successful_optimizer_updates=60000,initial_state_sha256='same',
                     actual_ray_schedule_sha256='rays',frame_order_sha256='frames') for g in [-6,0,6]]

    def test_rejects_unequal_initialization(self):
        rows=self.rows();rows[1]['initial_state_sha256']='different'
        with self.assertRaisesRegex(ValueError,'initialization'):scope['verify_training'](rows,60000,'same')

    def test_rejects_unequal_update_budget(self):
        rows=self.rows();rows[2]['successful_optimizer_updates']=59999
        with self.assertRaisesRegex(ValueError,'budget'):scope['verify_training'](rows,60000,'same')

    def test_rejects_different_actual_rays(self):
        rows=self.rows();rows[0]['actual_ray_schedule_sha256']='different'
        with self.assertRaisesRegex(ValueError,'rays'):scope['verify_training'](rows,60000,'same')

    def test_rejects_missing_arm(self):
        with self.assertRaisesRegex(ValueError,'missing'):scope['verify_training'](self.rows()[:2],60000,'same')

    def test_rejects_changed_score_pcm(self):
        with self.assertRaisesRegex(ValueError,'PCM'):scope['verify_pcm'](b'gain_audio',b'original_audio')


if __name__=='__main__':unittest.main()
