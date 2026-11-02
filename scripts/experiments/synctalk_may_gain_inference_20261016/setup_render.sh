set -eu
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
mkdir -p "$R/resources/wheels" "$R/cuda" "$R/deps" "$R/build" "$R/logs"
if [ ! -d "$R/official" ]; then cp -a "$A/official" "$R/official"; fi
for p in "$R"/resources/cuda-*.tar.bz2; do tar -xjf "$p" -C "$R/cuda"; done
"$A/env/bin/python" - <<'PY'
from pathlib import Path
import zipfile,difflib,json,hashlib
r=Path('/dev/shm/synctalk-may-gain-inference-20261016')
for p in (r/'resources/wheels').glob('*.whl'):
 with zipfile.ZipFile(p) as z:z.extractall(r/'deps')
patches=[]
for name in ['raymarching','freqencoder','shencoder','gridencoder']:
 p=r/'official'/name/'backend.py'
 old=p.read_text()
 new=old.replace('-std=c++14','-std=c++17').replace("'-O3', '-std=c++17',", "'-O3', '-std=c++17', '-allow-unsupported-compiler',",1)
 p.write_text(new)
 patches.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name+'/backend.py.original',tofile=name+'/backend.py'))
(r/'compatibility.patch').write_text(''.join(patches))
cuda=r/'cuda'
if not (cuda/'lib64').exists():(cuda/'lib64').symlink_to('lib')
runtime=Path('/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12')
(cuda/'lib/libcudart.so').unlink(missing_ok=True)
(cuda/'lib/libcudart.so').symlink_to(runtime)
PY
"$R/cuda/bin/nvcc" --version
export CUDA_HOME="$R/cuda"
export PATH="$R/cuda/bin:$R/deps/bin:$PATH"
export PYTHONPATH="$R/deps:$R/official"
export TORCH_EXTENSIONS_DIR="$R/build"
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=2
cd "$R/official"
