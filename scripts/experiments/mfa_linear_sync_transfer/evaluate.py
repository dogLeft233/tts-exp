"""Public decision seams for the transfer evaluation."""
from __future__ import annotations

from .real_video import decide_real_video, evaluate_real_video_record
from .replacement import decide_replacement, replacement_record_evidence

__all__ = [
    "decide_real_video",
    "evaluate_real_video_record",
    "decide_replacement",
    "replacement_record_evidence",
]
