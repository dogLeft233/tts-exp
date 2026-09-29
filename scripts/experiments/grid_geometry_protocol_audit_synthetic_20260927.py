"""Independent synthetic audit only; AST-extracted pure functions, no media/model imports."""
from pathlib import Path
import ast, hashlib, json, math, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/grid_geometry_protocol_audit_20260927'
SOURCE=ROOT/'scripts/experiments/grid_geometry_calibration_20260927.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tree=ast.parse(SOURCE.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metric','choose','lag_lock']]
class MockPath:
 def __init__(self,s=''):self.s=s
 def __truediv__(self,q):return MockPath(self.s+'/'+q)
 def exists(self):return True
reads=[];calls=[];written={};T=np.arange(14,55);L=np.arange(-5,6)
rng=np.random.default_rng(20260927);fixtures={f's{i}':rng.normal(size=75) for i in range(2,10)}
def read(p):
 reads.append(p.s)
 assert p.s.endswith('input_gate.json')
 return {'rows':[{'id':s,'qc':{'passed':True}} for s in fixtures]}
def trajectory(s,d):
 calls.append([s,d]);assert d in ['REAL','RAW'],'FIXED is forbidden during lag selection'
 y=fixtures[s];x=y if d=='REAL' else y[np.clip(np.arange(75)-3,0,74)]
 return x,np.ones(75,dtype=bool)
ns={'np':np,'math':math,'time':time,'OUT':MockPath(),'INTERIOR':T,'LAGS':L,'protocol':lambda:None,'read':read,'trajectory':trajectory,'write':lambda p,x:written.update({p.s:x}),'sha':lambda p:'synthetic_no_file_hash'}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
ns['lag_lock']();lock=written['/lag_lock.json'];assert lock['lag']==3 and lock['source_count']==8
# A physically nonnegative quasiperiodic aperture example: magnitude monotonicity fails
# while the implementation's four shift recoveries are unique and exact.
ix=np.arange(75);x=.05+.01*np.sin(2*np.pi*ix/5.3)+.00001*ix;cases=[]
for d in [-4,-2,2,4]:
 y=x[np.clip(ix-d,0,74)];m=ns['metric'](x[T],y[T]);rr=[ns['metric'](x[T],y[T+l])['r'] for l in L];found=ns['choose'](rr)
 assert found==d and sorted(rr)[-2]<.941
 cases.append({'delay':d,'E':m['E'],'r_by_lag':rr,'recovered':found,'runner_up_r':sorted(rr)[-2]})
assert cases[0]['E']>cases[1]['E'] and cases[3]['E']>cases[2]['E']
result={'status':'PASS','time':time.time(),'source_sha256':sha(SOURCE),'audit_code_sha256':sha(__file__),'scope':'only synthetic arrays; no production module import, media, results or model forward','actual_lag_lock_AST':{'lag':lock['lag'],'source_count':lock['source_count'],'read_paths':reads,'trajectory_calls':calls,'FIXED_accesses':0},'nonmonotonic_shift_counterexample':{'formula':'.05+.01*sin(2*pi*t/5.3)+.00001*t','interior':T.tolist(),'cases':cases,'conclusion':'E(|4|)<=E(|2|) is not a necessary condition for exact temporal recovery; no real-data threshold decision was made by this audit'}}
dest=OUT/'synthetic_actual_implementation.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','lag':lock['lag'],'shift_E':[c['E'] for c in cases],'source_sha256':sha(SOURCE)}))
