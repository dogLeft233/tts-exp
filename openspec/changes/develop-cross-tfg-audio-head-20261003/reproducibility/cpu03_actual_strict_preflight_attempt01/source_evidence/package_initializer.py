"""Clock-preserving, model-agnostic audio enhancement components.

The package keeps file I/O and simple baselines usable without importing
PyTorch.  Neural components are loaded lazily from :mod:`heads`.
"""

from .audio_io import (
    CANONICAL_SAMPLE_RATE,
    AudioReadInfo,
    AudioWriteInfo,
    read_canonical_wav,
    write_float32_wav,
)
from .baselines import (
    FIXED_POSITIVE_CONTROL_RMS,
    FlatGainInfo,
    LocalCompressionInfo,
    ProfileSearchResult,
    SpectralProfile,
    apply_global_rms,
    apply_flat_gain,
    apply_local_envelope_compression,
    apply_spectral_profile,
    search_profiles,
)
from .pipeline import transform
from .overdub import MuxInfo, mux_original_pcm

__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "AudioReadInfo",
    "AudioWriteInfo",
    "FIXED_POSITIVE_CONTROL_RMS",
    "FlatGainInfo",
    "LocalCompressionInfo",
    "ProfileSearchResult",
    "SpectralProfile",
    "read_canonical_wav",
    "write_float32_wav",
    "apply_global_rms",
    "apply_flat_gain",
    "apply_local_envelope_compression",
    "apply_spectral_profile",
    "search_profiles",
    "transform",
    "MuxInfo",
    "mux_original_pcm",
]


def __getattr__(name: str):
    if name in {"SpectralResidualHead", "EnhancementStack"}:
        from .heads import EnhancementStack, SpectralResidualHead

        return {
            "SpectralResidualHead": SpectralResidualHead,
            "EnhancementStack": EnhancementStack,
        }[name]
    raise AttributeError(name)
