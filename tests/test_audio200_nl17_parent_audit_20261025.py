import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_nl17_parent_audit_20261025.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Contract(unittest.TestCase):
    def test_exact4arm_and6pairs(self):
        rows=[dict(arm='N',draw=0,id='N')]+[dict(arm='L17',draw=i,id='L17'+str(i)) for i in (1,2,3)]
        p=m.expected_pairs(rows)
        self.assertEqual(len(p),6)
        self.assertEqual(sum(f=='N-L17' for f,a,b in p),3)
        with self.assertRaises(AssertionError):m.expected_pairs(rows+[dict(arm='L06',draw=1,id='L061')])
    def test_window_centers_and_edges(self):
        c=[(max(0,640*i-1)+640*i+3439)/32000 for i in range(30)]
        self.assertFalse(m.nearest_window([0,10],0.,0.,c)[2])
        self.assertEqual(m.nearest_window([400,440],.069,0.,c)[1],10)
if __name__=='__main__':unittest.main()
