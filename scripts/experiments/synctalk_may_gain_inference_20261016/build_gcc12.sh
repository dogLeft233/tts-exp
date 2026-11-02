set -eu
R=/dev/shm/synctalk-may-gain-inference-20261016
A=/home/wjj/tts-audio/synctalk-amplitude-audit-20261016
mkdir -p "$R/deps/bin"
chmod +x "$R/deps/ninja-1.11.1.3.data/scripts/ninja"
ln -sf "$R/deps/ninja-1.11.1.3.data/scripts/ninja" "$R/deps/bin/ninja"
for p in "$R"/resources/gcc12/*.deb; do dpkg-deb -x "$p" "$R/gcc12"; done
mkdir -p "$R/gcc12/usr/lib/x86_64-linux-gnu"
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$R/gcc12/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
ln -sf /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 "$R/gcc12/usr/lib/x86_64-linux-gnu/libgcc_s.so.1"
export CC="$R/gcc12/usr/bin/gcc-12"
export CXX="$R/gcc12/usr/bin/g++-12"
"$CXX" --version | head -1
printf '#include <vector>\n' | "$CXX" -std=c++17 -x c++ -fsyntax-only -
export CUDA_HOME="$R/cuda"
export PATH="$R/cuda/bin:$R/deps/bin:$PATH"
export PYTHONPATH="$R/deps:$R/official"
export TORCH_EXTENSIONS_DIR="$R/build"
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=2
export OMP_NUM_THREADS=2
cd "$R/official"
"$A/env/bin/python" -u -c 'import nerf_triplane.network; from gridencoder import GridEncoder; from shencoder import SHEncoder; from freqencoder import FreqEncoder; print("ALL_ENCODER_IMPORT_PASS")' > "$R/logs/build_import_attempt04.log" 2>&1
