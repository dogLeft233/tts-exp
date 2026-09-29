"""Static-only Wav2Lip worker entrypoint for this experiment.

The forward implementation is shared with the audited static-image bridge,
but this package owns the subprocess entrypoint so the protocol binds the
exact worker used by this experiment.  Its CLI accepts only the PNG+PCM
contract exposed by that worker; no LRS3/video input is introduced here.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.static_image_bridge.render_worker import (  # noqa: E402
    chunk_mels,
    encode_ffv1_stream,
    main,
)


__all__ = ["chunk_mels", "encode_ffv1_stream", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
