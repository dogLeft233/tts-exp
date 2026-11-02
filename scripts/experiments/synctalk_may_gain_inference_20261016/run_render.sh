set -eu
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
export CC="$R/gcc12/usr/bin/gcc-12"
export CXX="$R/gcc12/usr/bin/g++-12"
export CUDA_HOME="$R/cuda"
export PATH="$R/cuda/bin:$R/deps/bin:$PATH"
export PYTHONPATH="$R/deps:$R/official"
export TORCH_EXTENSIONS_DIR="$R/build"
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=2
export OMP_NUM_THREADS=2
cd "$R/official"
OUT="${1:-$R/render_attempt01}"
if [ -e "$OUT" ]; then printf 'Refusing to overwrite existing run: %s\n' "$OUT" >&2; exit 1; fi
NAME="$(basename "$OUT")"
"$A/env/bin/python" -u "$R/render.py" --output "$OUT" > "$R/logs/${NAME}.log" 2>&1
