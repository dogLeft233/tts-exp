set -eu
R=/dev/shm/synctalk-may-gain-inference-20261016
T=/dev/shm/synctalk-may-training-pilot-20261017
export CC="$R/gcc12/usr/bin/gcc-12"
export CXX="$R/gcc12/usr/bin/g++-12"
export CUDA_HOME="$R/cuda"
export PATH="$R/cuda/bin:$R/deps/bin:$PATH"
export PYTHONPATH="$T:$R/deps:$R/official"
export TORCH_EXTENSIONS_DIR="$R/build"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$T/matplotlib-cache"
cd "$R/official"
exec /home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -u "$@"
