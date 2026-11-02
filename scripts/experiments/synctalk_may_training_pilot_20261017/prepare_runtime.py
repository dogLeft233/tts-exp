"""Patch only loader split handling and seal-qualified AVE cache use."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

R = Path('/dev/shm/synctalk-may-gain-inference-20261016')
T = Path('/dev/shm/synctalk-may-training-pilot-20261017')
p = R/'official/nerf_triplane/provider.py'
original = p.read_text()
assert original.count("_split = 'val' if type == 'test' else type") == 1
assert original.count('frames = frames[:100]') == 1
tree = ast.parse(original)
node = next(n for n in ast.walk(tree) if isinstance(n,ast.If) and n.lineno == 171)
assert ast.unparse(node.test) == "self.opt.asr_model == 'ave'"
lines = original.splitlines(keepends=True)
first, last = node.body[0].lineno - 1, node.body[-1].end_lineno
lines[first:last] = ['                    aud_features = np.load(self.opt.verified_ave_cache, allow_pickle=False)\n']
modified = ''.join(lines).replace("_split = 'val' if type == 'test' else type", '_split = type')
modified = modified.replace('frames = frames[:100]', 'frames = frames')
assert modified != original
p.write_text(modified)
diff = ''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
                                  fromfile='provider.py.original',tofile='provider.py.pilot'))
(T/'provider_compatibility.patch').write_text(diff)
report = {'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
          'modified_sha256':hashlib.sha256(modified.encode()).hexdigest(),
          'changes':['test reads independent test JSON','validation reads all rows',
                     'self AVE uses exact sealed padded cache verified against WAV and official encoder'],
          'network_changes':False,'loss_changes':False}
(T/'provider_compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
