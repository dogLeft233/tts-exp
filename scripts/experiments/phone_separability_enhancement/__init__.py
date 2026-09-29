"""V2 natural-clock phone-separability enhancement experiment.

The package is deliberately separate from ``phone_separability_mechanism``.
The latter remains the frozen v1 reference implementation; this package owns
the corrected signed-margin/support protocol and the constrained enhancer.
"""

PROTOCOL_ID = "phone_separability_enhancement_v2"
MEASUREMENT_VERSION = "signed_margin_fixed_support_v2"

__all__ = ["MEASUREMENT_VERSION", "PROTOCOL_ID"]
