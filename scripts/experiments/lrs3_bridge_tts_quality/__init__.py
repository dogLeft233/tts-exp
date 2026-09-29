"""Paired LRS3 bridge comparison for local and cloud TTS sources.

The package deliberately keeps the numerical protocol in small, importable
modules.  GPU work is performed by the stage runner and is never started at
import time.
"""

from .config import PROTOCOL_ID, RunPaths, run_root_for

__all__ = ["PROTOCOL_ID", "RunPaths", "run_root_for"]
