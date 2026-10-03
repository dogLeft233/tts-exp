"""Differentiable Torch reproduction of the pinned Wav2Lip audio frontend.

The mel basis and numerical hparams are obtained from the loaded, hash-pinned
Wav2Lip ``audio.py``/``hparams.py``.  The NumPy official frontend remains the
reference: call :func:`check_frontend_parity` before enabling an optimizer.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # training extras are optional for CPU I/O users
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class Wav2LipMelSpec:
    sample_rate_hz: int = 16_000
    n_fft: int = 800
    win_length: int = 800
    hop_length: int = 200
    n_mels: int = 80
    fmin_hz: float = 55.0
    fmax_hz: float = 7_600.0
    preemphasis: float = 0.97
    preemphasize: bool = True
    min_level_db: float = -100.0
    ref_level_db: float = 20.0
    symmetric_mels: bool = True
    max_abs_value: float = 4.0
    # Official Wav2Lip inference calls load_wav (resample only) and does not
    # peak-normalize the waveform before its mel transform.  Keep this field
    # explicit in the serialized contract so a mistaken training-only .9
    # preprocessing assumption fails closed.
    peak_rescale: None = None
    amp_floor: float = 1e-5
    stft_center: bool = True
    # librosa.stft defaults to constant centering padding; torch.stft defaults
    # to reflect, so make the official behavior explicit here.
    stft_pad_mode: str = "constant"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("the cross-TFG audio contract is fixed at 16 kHz")
        if self.n_fft <= 0 or self.n_fft % 2 or self.win_length != self.n_fft:
            raise ValueError("Wav2Lip STFT must use an even n_fft and win_length=n_fft")
        if self.hop_length <= 0 or self.hop_length > self.win_length:
            raise ValueError("invalid Wav2Lip STFT hop length")
        if self.n_mels != 80 or not (0 <= self.fmin_hz < self.fmax_hz <= self.sample_rate_hz / 2):
            raise ValueError("unexpected Wav2Lip mel dimensions or frequency range")
        if not (0 <= self.preemphasis < 1):
            raise ValueError("preemphasis must be in [0,1)")
        if not (self.min_level_db < 0 and math.isfinite(self.min_level_db)):
            raise ValueError("min_level_db must be finite and negative")
        if not math.isfinite(self.ref_level_db):
            raise ValueError("ref_level_db must be finite")
        if not math.isfinite(self.max_abs_value) or self.max_abs_value <= 0:
            raise ValueError("max_abs_value must be finite and positive")
        if self.peak_rescale is not None:
            raise ValueError("official Wav2Lip inference has no pre-mel peak rescale")
        if not math.isfinite(self.amp_floor) or self.amp_floor <= 0:
            raise ValueError("amp_floor must be finite and positive")
        if self.stft_pad_mode not in {"reflect", "constant", "replicate"}:
            raise ValueError("unsupported STFT pad mode")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_librosa_positional_compatibility_shim() -> dict[str, str]:
    """Apply the frozen Wav2Lip positional ``filters.mel`` compatibility shim.

    Wav2Lip's historical call passes ``sr`` and ``n_fft`` positionally.  This
    maps only those two arguments to their librosa 0.10 keyword names and
    leaves all remaining librosa defaults unchanged.
    """

    import librosa

    original = librosa.filters.mel
    original_signature = str(inspect.signature(original))
    if getattr(original, "_cross_tfg_positional_compat", False):
        return {
            "librosa_version": str(getattr(librosa, "__version__", "unknown")),
            "original_signature": original_signature,
            "shim": "positional sr,n_fft mapped to keyword arguments; remaining defaults unchanged",
        }

    def compatible_mel(*args: Any, **kwargs: Any):
        positional = list(args)
        if positional and "sr" not in kwargs:
            kwargs["sr"] = positional.pop(0)
        if positional and "n_fft" not in kwargs:
            kwargs["n_fft"] = positional.pop(0)
        if positional:
            raise TypeError("unexpected positional arguments in librosa.filters.mel compatibility shim")
        return original(**kwargs)

    setattr(compatible_mel, "_cross_tfg_positional_compat", True)
    librosa.filters.mel = compatible_mel
    return {
        "librosa_version": str(getattr(librosa, "__version__", "unknown")),
        "original_signature": original_signature,
        "shim": "positional sr,n_fft mapped to keyword arguments; remaining defaults unchanged",
    }


def _hparams_mapping(audio_module: Any, hparams_module: Any) -> dict[str, Any]:
    source = getattr(audio_module, "hp", getattr(hparams_module, "hparams", hparams_module))
    data = getattr(source, "data", source)
    if isinstance(data, dict):
        mapping = dict(data)
    elif hasattr(data, "__dict__"):
        mapping = {key: value for key, value in vars(data).items() if not key.startswith("_")}
    else:
        mapping = {}
    aliases = {
        "sample_rate_hz": ("sample_rate", "sample_rate_hz"),
        "n_fft": ("n_fft",),
        "win_length": ("win_size", "win_length"),
        "hop_length": ("hop_size", "hop_length"),
        "n_mels": ("num_mels", "n_mels"),
        "fmin_hz": ("fmin", "fmin_hz"),
        "fmax_hz": ("fmax", "fmax_hz"),
        "preemphasis": ("preemphasis",),
        "preemphasize": ("preemphasize",),
        "min_level_db": ("min_level_db",),
        "ref_level_db": ("ref_level_db",),
        "symmetric_mels": ("symmetric_mels",),
        "max_abs_value": ("max_abs_value",),
    }
    result: dict[str, Any] = {}
    for target, names in aliases.items():
        for name in names:
            if name in mapping:
                result[target] = mapping[name]
                break
            if hasattr(source, name):
                result[target] = getattr(source, name)
                break
    return result


def import_official_wav2lip_audio(wav2lip_root: str | Path) -> tuple[Any, Wav2LipMelSpec, dict[str, Any]]:
    """Import the pinned Wav2Lip audio frontend without editing its checkout."""

    root = Path(wav2lip_root).resolve(strict=True)
    audio_path = (root / "audio.py").resolve(strict=True)
    hparams_path = (root / "hparams.py").resolve(strict=True)
    shim = install_librosa_positional_compatibility_shim()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    for module_name, expected_path in (("audio", audio_path), ("hparams", hparams_path)):
        existing = sys.modules.get(module_name)
        if existing is not None:
            actual = Path(getattr(existing, "__file__", "")).resolve()
            if actual != expected_path:
                raise RuntimeError(f"{module_name} is already imported from another Wav2Lip checkout")
    audio_module = importlib.import_module("audio")
    hparams_module = importlib.import_module("hparams")
    hp = _hparams_mapping(audio_module, hparams_module)
    defaults = asdict(Wav2LipMelSpec())
    # These are frozen Wav2Lip hparams; fail closed if this model checkout uses
    # another frontend instead of silently training against a different input.
    for key in (
        "sample_rate_hz", "n_fft", "win_length", "hop_length", "n_mels",
        "fmin_hz", "fmax_hz", "preemphasis", "preemphasize", "min_level_db", "ref_level_db",
        "symmetric_mels", "max_abs_value",
    ):
        if key not in hp:
            raise RuntimeError(f"Wav2Lip hparams do not expose required frontend setting {key}")
        expected = defaults[key]
        actual = hp[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(f"Wav2Lip frontend {key} differs from the frozen profile")
        elif actual != expected:
            raise RuntimeError(f"Wav2Lip frontend {key} differs from the frozen profile")
    spec = Wav2LipMelSpec(**{key: hp[key] for key in hp if key in defaults})
    metadata = {
        "wav2lip_root": str(root),
        "audio_path": str(audio_path),
        "audio_sha256": _sha256_file(audio_path),
        "hparams_path": str(hparams_path),
        "hparams_sha256": _sha256_file(hparams_path),
        "hparams": hp,
        "librosa_compatibility": shim,
        "mel_basis_source": "official audio._linear_to_mel(identity frequency basis)",
    }
    return audio_module, spec, metadata


def official_mel_basis(audio_module: Any, spec: Wav2LipMelSpec) -> np.ndarray:
    """Recover the exact filter matrix held/constructed by ``audio.py``."""

    frequency_bins = spec.n_fft // 2 + 1
    for name in ("mel_basis", "_mel_basis"):
        candidate = getattr(audio_module, name, None)
        if candidate is not None:
            value = np.asarray(candidate)
            if value.shape == (spec.n_mels, frequency_bins):
                if not np.isfinite(value).all():
                    raise ValueError("official Wav2Lip mel basis is non-finite")
                return np.ascontiguousarray(value, dtype=np.float32)
    converter = getattr(audio_module, "_linear_to_mel", None)
    if converter is not None:
        basis = np.asarray(converter(np.eye(frequency_bins, dtype=np.float32)))
        if basis.shape == (frequency_bins, spec.n_mels):
            basis = basis.T
        if basis.shape == (spec.n_mels, frequency_bins) and np.isfinite(basis).all():
            return np.ascontiguousarray(basis, dtype=np.float32)
    builder = getattr(audio_module, "_build_mel_basis", None)
    if builder is not None:
        basis = np.asarray(builder())
        if basis.shape == (spec.n_mels, frequency_bins) and np.isfinite(basis).all():
            return np.ascontiguousarray(basis, dtype=np.float32)
    raise RuntimeError("could not recover the filter matrix from pinned Wav2Lip audio.py")


if nn is not None:

    class Wav2LipTorchMel(nn.Module):
        """Torch mel transform with fixed official basis and live audio gradients."""

        def __init__(self, mel_basis: np.ndarray | Tensor, spec: Wav2LipMelSpec) -> None:
            super().__init__()
            matrix = torch.as_tensor(mel_basis, dtype=torch.float32)
            expected = (spec.n_mels, spec.n_fft // 2 + 1)
            if tuple(matrix.shape) != expected or not bool(torch.isfinite(matrix).all().item()):
                raise ValueError(f"mel basis must be finite {expected}, got {tuple(matrix.shape)}")
            self.spec = spec
            self.register_buffer("mel_basis", matrix.contiguous())
            self.register_buffer("window", torch.hann_window(spec.win_length, periodic=True))

        @staticmethod
        def _as_batch(waveform: Tensor) -> tuple[Tensor, str]:
            if waveform.ndim == 1:
                return waveform.unsqueeze(0), "mono"
            if waveform.ndim == 2:
                return waveform, "batch"
            if waveform.ndim == 3 and waveform.shape[1] == 1:
                return waveform[:, 0, :], "channel"
            raise ValueError("waveform must be [N], [B,N], or [B,1,N]")

        def forward(self, waveform: Tensor) -> Tensor:
            if torch.is_complex(waveform):
                raise TypeError("Wav2Lip mel input must be real")
            x, layout = self._as_batch(waveform)
            if x.shape[-1] <= self.spec.n_fft // 2:
                raise ValueError("waveform is too short for centered official Wav2Lip STFT")
            # scipy.signal.lfilter in the pinned NumPy frontend promotes the
            # float32 waveform to float64 because its coefficients are Python
            # floats.  Mirror that arithmetic for the official mel parity,
            # then cast back to float32 as Wav2Lip inference does.
            x = x.to(dtype=torch.float64).contiguous()
            if not bool(torch.isfinite(x).all().item()):
                raise ValueError("Wav2Lip mel input contains NaN or infinity")
            if self.spec.preemphasize:
                emphasized = torch.cat(
                    (x[..., :1], x[..., 1:] - self.spec.preemphasis * x[..., :-1]), dim=-1
                )
            else:
                emphasized = x
            spectrum = torch.stft(
                emphasized,
                n_fft=self.spec.n_fft,
                hop_length=self.spec.hop_length,
                win_length=self.spec.win_length,
                window=self.window.to(device=x.device, dtype=x.dtype),
                center=self.spec.stft_center,
                pad_mode=self.spec.stft_pad_mode,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            magnitude = spectrum.abs()
            basis = self.mel_basis.to(device=x.device, dtype=x.dtype)
            mel = torch.einsum("mf,bft->bmt", basis, magnitude)
            mel_db = 20.0 * torch.log10(mel.clamp_min(self.spec.amp_floor)) - self.spec.ref_level_db
            if self.spec.symmetric_mels:
                normalized = (
                    2.0 * self.spec.max_abs_value
                    * ((mel_db - self.spec.min_level_db) / -self.spec.min_level_db)
                    - self.spec.max_abs_value
                ).clamp(-self.spec.max_abs_value, self.spec.max_abs_value)
            else:
                normalized = (
                    self.spec.max_abs_value
                    * ((mel_db - self.spec.min_level_db) / -self.spec.min_level_db)
                ).clamp(0.0, self.spec.max_abs_value)
            if not bool(torch.isfinite(normalized).all().item()):
                raise FloatingPointError("Wav2Lip Torch mel contains NaN or infinity")
            normalized = normalized.to(torch.float32)
            if layout == "mono":
                return normalized[0]
            return normalized

else:

    class Wav2LipTorchMel:  # pragma: no cover - optional PyTorch dependency
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Wav2LipTorchMel requires PyTorch") from _TORCH_IMPORT_ERROR


def official_pre_mel_waveform(waveform: np.ndarray, spec: Wav2LipMelSpec) -> np.ndarray:
    """Return input unchanged, as pinned Wav2Lip inference does before mel."""

    x = np.asarray(waveform)
    if x.dtype != np.float32 or x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("official frontend parity input must be nonempty finite mono float32")
    if spec.peak_rescale is not None:
        raise ValueError("official Wav2Lip inference does not peak-rescale before mel")
    return np.ascontiguousarray(x.copy())


def check_frontend_parity(
    frontend: Any,
    audio_module: Any,
    waveform: np.ndarray,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare Torch output with official ``audio.melspectrogram`` numerically.

    This function is intended for deterministic synthetic probes or a
    predeclared fit-only calibration clip. It raises instead of authorizing
    training when parity exceeds the caller's frozen numerical tolerance.
    """

    if torch is None:
        raise RuntimeError("frontend parity check requires PyTorch") from _TORCH_IMPORT_ERROR
    if not math.isfinite(atol) or atol < 0 or not math.isfinite(rtol) or rtol < 0:
        raise ValueError("parity tolerances must be finite and non-negative")
    values = official_pre_mel_waveform(waveform, frontend.spec)
    device = frontend.mel_basis.device
    with torch.no_grad():
        actual = frontend(torch.from_numpy(values).to(device=device)).detach().cpu().numpy()
    expected = np.asarray(audio_module.melspectrogram(values), dtype=np.float32)
    if actual.ndim == 3 and actual.shape[0] == 1:
        actual = actual[0]
    if expected.shape != actual.shape or not np.isfinite(expected).all():
        raise RuntimeError(f"official/Torch mel shape or finite mismatch: {expected.shape} vs {actual.shape}")
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    maximum_absolute = float(difference.max(initial=0.0))
    mean_absolute = float(difference.mean())
    close = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
    result = {
        "status": "PASS" if close else "FAIL",
        "shape": [int(value) for value in expected.shape],
        "official_dtype": str(expected.dtype),
        "torch_dtype": str(actual.dtype),
        "maximum_absolute_difference": maximum_absolute,
        "mean_absolute_difference": mean_absolute,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "official_float32_sha256": hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest(),
        "torch_float32_sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
        "same_bits": bool(np.array_equal(expected, actual)),
    }
    if not close:
        raise RuntimeError(f"Torch Wav2Lip frontend failed frozen official mel parity: {result}")
    return result


def official_chunk_starts(mel_frame_count: int, *, fps_num: int = 25, fps_den: int = 1, width: int = 16) -> list[int]:
    """Reproduce the official ``int(i*80/fps)`` loop and repeated final tail."""

    if fps_num <= 0 or fps_den <= 0 or width <= 0 or mel_frame_count < width:
        raise ValueError("invalid FPS or mel dimensions for Wav2Lip chunking")
    starts: list[int] = []
    index = 0
    while True:
        start = int(index * 80.0 * fps_den / fps_num)
        if start + width > mel_frame_count:
            starts.append(mel_frame_count - width)
            break
        starts.append(start)
        index += 1
    return starts


__all__ = [
    "Wav2LipMelSpec",
    "Wav2LipTorchMel",
    "check_frontend_parity",
    "import_official_wav2lip_audio",
    "official_chunk_starts",
    "official_mel_basis",
    "official_pre_mel_waveform",
    "install_librosa_positional_compatibility_shim",
]
