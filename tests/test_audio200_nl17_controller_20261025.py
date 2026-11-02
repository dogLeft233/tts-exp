import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('nl17ctrl',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_nl17_controller_20261025.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class NL17Controller(unittest.TestCase):
    def test_training_is_rejected(self):
        with self.assertRaises(AssertionError):
            m.validate(dict(phase=m.PHASE,allow_training=True),None)

    def test_reuses_frozen_safe_controller_utilities(self):
        u=m.utility();self.assertEqual(u.sha(Path(u.__file__)),m.UTILITY_SHA)
        self.assertEqual(u.MATERIAL_CAP,60*1024**3)

    def test_final2400_terminal_not_old4200(self):
        u=m.utility()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=p/'index.json';data.write_text('data')
            manifest=p/'final_manifest.json';u.put(manifest,dict(files=[dict(path=data.name,sha256=u.sha(data),bytes=data.stat().st_size)]))
            u.put(p/'final_status.json',dict(status=m.FINAL_PASS,no_training=True,units=2400,source_count=600,speaker_count=200,pairs=3600,final_manifest_sha256=u.sha(manifest)))
            self.assertEqual(m.final_output(p,u),p/'final_status.json')
            data.write_text('drift')
            with self.assertRaises(ValueError):m.final_output(p,u)


if __name__=='__main__':unittest.main()
