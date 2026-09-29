"""LRS3 phoneme-gain to static-portrait TFG experiment.

The package is deliberately isolated from the historical enhancement runners.
Its public boundary is a small, auditable protocol runner; worker requests use
whitelists so an LRS3 video path cannot accidentally reach inference.
"""

PROTOCOL_ID = "phone_gain_static_tfg_mfa_v1"
MEASUREMENT_VERSION = "phone_gain_static_tfg_mfa_v1"

__all__ = ["MEASUREMENT_VERSION", "PROTOCOL_ID"]
