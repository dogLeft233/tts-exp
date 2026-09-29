"""Staged TTS phone-separability mechanism experiment.

The package intentionally keeps the protocol runner thin.  Inventory,
feature extraction, metrics, audio construction, timing checks, and the
independent checker are separate modules so a failed historical asset does
not invalidate unrelated contrasts.
"""

PROTOCOL_ID = "phone_separability_mechanism_v1"

__all__ = ["PROTOCOL_ID"]
