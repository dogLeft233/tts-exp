#!/bin/bash
set -eu
ROOT=/dev/shm/synctalk-multispeaker-data-20261017
TOOLCHAIN=/dev/shm/synctalk-may-gain-inference-20261016
export CC="$TOOLCHAIN/gcc12/usr/bin/gcc-12"
export CXX="$TOOLCHAIN/gcc12/usr/bin/g++-12"
export CUDA_HOME="$TOOLCHAIN/cuda"
export PATH="$TOOLCHAIN/cuda/bin:$TOOLCHAIN/deps/bin:$PATH"
export PYTHONPATH="$ROOT/deps:$ROOT:$ROOT/official:$TOOLCHAIN/deps"
export TORCH_EXTENSIONS_DIR="$TOOLCHAIN/build"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export TORCH_HOME="$ROOT/assets/torch"
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT/official"
exec /home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python "$@"
