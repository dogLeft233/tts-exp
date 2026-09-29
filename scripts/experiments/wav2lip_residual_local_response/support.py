from __future__ import annotations

from typing import Iterable

import numpy as np

from . import config
from .common import ProtocolError


def chunk_columns() -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * 80.0 / config.FPS)
        if start + 16 > config.MEL_FRAMES:
            chunks.append(np.arange(config.MEL_FRAMES - 16, config.MEL_FRAMES, dtype=np.int64)); break
        chunks.append(np.arange(start, start + 16, dtype=np.int64)); index += 1
    if len(chunks) != config.FRAME_COUNT: raise ProtocolError(f"chunk count changed: {len(chunks)}")
    return chunks


def row_supports() -> dict[int, np.ndarray]:
    chunks = chunk_columns(); result: dict[int, np.ndarray] = {}
    for row in config.U_ROWS:
        if row + 4 >= len(chunks): raise ProtocolError(f"SyncNet row lacks five frames: {row}")
        result[row] = np.unique(np.concatenate(chunks[row : row + 5])).astype(np.int64)
    return result


def exposure_for(k: Iterable[int]) -> dict[str, object]:
    support = row_supports(); allowed = set(int(value) for value in k)
    if not allowed or min(allowed) < 0 or max(allowed) >= config.MEL_FRAMES: raise ProtocolError("K is outside [0,308)")
    rows = []
    for row in config.U_ROWS:
        columns = support[row]; overlap = np.asarray(sorted(set(columns.tolist()) & allowed), dtype=np.int64)
        rows.append({"row": int(row), "support": columns.tolist(), "overlap": overlap.tolist(), "exposure": float(len(overlap) / len(columns))})
    return {"u_rows": list(config.U_ROWS), "chunk_count": len(chunk_columns()), "rows": rows}
