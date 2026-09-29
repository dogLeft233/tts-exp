"""Evidence audit and activation-patching protocol for the TTS/TFG study.

The package deliberately keeps the CPU protocol and pure mathematical
transforms importable without importing torch or initializing CUDA.  GPU work
is isolated in :mod:`worker` and is only entered through the resource guard in
:mod:`protocol`.
"""

from .protocol import PROTOCOL_ID

__all__ = ["PROTOCOL_ID"]
