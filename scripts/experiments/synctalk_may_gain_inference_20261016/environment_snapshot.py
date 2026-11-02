import hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path

r=Path('/dev/shm/synctalk-may-gain-inference-20261016')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
out={'python':sys.version,'executable':sys.executable,'packages':{},'files':[]}
for name in ['torch','torchvision','numpy','scipy','librosa','soundfile','opencv-python','python_speech_features']:
 try:out['packages'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:pass
for base in [r/'build',r/'resources',r/'official/nerf_triplane/checkpoints']:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix in ['.so','.bz2','.deb','.whl','.pth']:
   out['files'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
detector=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python/detectors/s3fd/weights/sfd_face.pth')
out['detector_sha256']=sha(detector)
out['official_worktree_patch']=subprocess.check_output(['git','-C',str(r/'official'),'diff'],text=True)
(r/'environment_snapshot.json').write_text(json.dumps(out,indent=2))
print(out['packages'])
