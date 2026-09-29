"""Synthetic nonmeasurable-path audit; no production imports or real result access."""
from pathlib import Path
import ast,hashlib,json,math,time
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'runs/grid_geometry_protocol_audit_20260927';S=R/'scripts/experiments/grid_geometry_calibration_20260927.py'
tree=ast.parse(S.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metric','choose','diagnostics']]
class P:
 def __init__(self,s=''):self.s=s
 def __truediv__(self,q):return P(self.s+'/'+q)
def read(p):
 if p.s.endswith('lag_lock.json'):return {'lag':0}
 if p.s.endswith('input_gate.json'):return {'rows':[{'id':'synthetic','qc':{'passed':True}}]}
 if p.s.endswith('BASE.json'):return {'qc':{'passed':True}}
 raise AssertionError(p.s)
def trajectory(s,d,v='BASE'):
 x=np.ones(75)*.05 if d=='RAW' else .05+.01*np.sin(np.arange(75))
 return x,np.ones(75,dtype=bool)
views=['BASE','REPEAT','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4','FROZEN','REVERSE','WARP_0.8','WARP_1.2','TX_-8','TX_8','TY_-8','TY_8','ROT_-5','ROT_5','SCALE_0.9','SCALE_1.1'];written={}
ns={'np':np,'math':math,'OUT':P(),'INTERIOR':np.arange(14,55),'LAGS':np.arange(-5,6),'VIEWS':views,'protocol':lambda:{},'read':read,'trajectory':trajectory,'write':lambda p,x:written.update({p.s:x}),'sha':lambda p:'synthetic'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(S),'exec'),ns)
try:ns['diagnostics']();outcome={'completed':True,'nonmeasurable_source_failed':not written['/calibration_diagnostics.json']['rows'][0]['passed']}
except TypeError as e:outcome={'completed':False,'exception_type':'TypeError','message':str(e)}
data={'time':time.time(),'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'synthetic constant RAW trajectory only; zero media or model reads','outcome':outcome}
dest=O/('zero_dynamic_'+data['source_sha256'][:12]+'.json');assert not dest.exists();dest.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
