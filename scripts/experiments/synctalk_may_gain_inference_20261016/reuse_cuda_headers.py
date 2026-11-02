from pathlib import Path
r=Path('/dev/shm/synctalk-may-gain-inference-20261016')
n=Path('/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/lib/python3.10/site-packages/nvidia')
count=0
for inc in n.glob('*/include'):
 for src in inc.rglob('*'):
  if not src.is_file():continue
  dst=r/'cuda/include'/src.relative_to(inc)
  dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():dst.symlink_to(src);count+=1
print('Reused read-only CUDA header files',count)
