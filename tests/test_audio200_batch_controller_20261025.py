import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('controller',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_batch_controller_20261025.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ControllerContracts(unittest.TestCase):
    def test_child_failure_or_unresolved_launch_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            m.put(p/'producer_launch.json',{'command':['not-run']})
            with patch.object(m.subprocess,'Popen') as called:
                with self.assertRaisesRegex(RuntimeError,'unresolved'):
                    m.checked(p,p,'producer',['not-run'],float('inf'),1)
                called.assert_not_called()
            m.put(p/'producer_exit.json',{'returncode':1,'error':'actual failure'})
            with patch.object(m.subprocess,'Popen') as called:
                with self.assertRaisesRegex(RuntimeError,'previous failure'):
                    m.checked(p,p,'producer',['not-run'],float('inf'),1)
                called.assert_not_called()

    def test_completed_receipt_is_verified_before_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);log=p/'producer.log';log.write_text('done')
            m.put(p/'producer_exit.json',dict(command=['same'],returncode=0,error=None,log=str(log),log_sha256=m.sha(log)))
            with patch.object(m.subprocess,'Popen') as called:
                self.assertEqual(m.checked(p,p,'producer',['same'],float('inf'),1)['returncode'],0)
                called.assert_not_called()
            log.write_text('changed')
            with self.assertRaises(ValueError):
                m.checked(p,p,'producer',['same'],float('inf'),1)

    def test_authorization_rejects_training_before_io(self):
        with self.assertRaises(AssertionError):
            m.validate_authorization(dict(allow_media_generation=True,allow_training=True))

    def test_immutable_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';m.put(p,{'a':1})
            with self.assertRaises(FileExistsError):
                m.put(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})

    def test_real_finalizer_terminal_contract_and_content_drift(self):
        # Read the actual writer's terminal call, not a duplicated assumed filename.
        source=Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_finalize_20261025.py'
        terminal_calls=[]
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='put' or len(node.args)<2:
                continue
            path,value=node.args[:2]
            if isinstance(path,ast.BinOp) and isinstance(path.right,ast.Constant) and isinstance(value,ast.Call):
                status=next((k.value.value for k in value.keywords if k.arg=='status' and isinstance(k.value,ast.Constant)),None)
                if status==m.FINAL_PASS:
                    terminal_calls.append(path.right.value)
        self.assertEqual(terminal_calls,['final_status.json'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);payload=p/'dataset_index.json';payload.write_text('verified')
            seal=p/'final_manifest.json'
            m.put(seal,{'files':[{'path':payload.name,'sha256':m.sha(payload),'bytes':payload.stat().st_size}]})
            terminal=p/terminal_calls[0]
            m.put(terminal,dict(status=m.FINAL_PASS,no_training=True,units=4200,source_count=600,speaker_count=200,
                reused=280,new=3920,pairs=9000,final_manifest_sha256=m.sha(seal)))
            self.assertEqual(m.accept_final_output(p),terminal)
            payload.write_text('tampered')
            with self.assertRaises(ValueError):
                m.accept_final_output(p)


if __name__=='__main__':
    unittest.main()
