set -eu
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
mkdir -p "$R/deps/bin"
chmod +x "$R/deps/ninja-1.11.1.3.data/scripts/ninja"
ln -sf "$R/deps/ninja-1.11.1.3.data/scripts/ninja" "$R/deps/bin/ninja"
export CUDA_HOME="$R/cuda"
export PATH="$R/cuda/bin:$R/deps/bin:$PATH"
export PYTHONPATH="$R/deps:$R/official"
export TORCH_EXTENSIONS_DIR="$R/build"
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=2
export OMP_NUM_THREADS=2
cd "$R/official"
"$A/env/bin/python" -u -c 'import nerf_triplane.network; from gridencoder import GridEncoder; from shencoder import SHEncoder; from freqencoder import FreqEncoder; print("ALL_ENCODER_IMPORT_PASS")' > "$R/logs/build_import_attempt03.log" 2>&1
