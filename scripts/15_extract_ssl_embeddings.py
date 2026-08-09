#!/usr/bin/env python3
"""Extract per-layer embeddings from HuBERT and XLS-R pre-trained models.

Wav2Sem-style feature separability analysis following Ditto (Wang et al.,
2024). Extracts frozen SSL features at multiple Transformer layers for paired
natural/TTS samples, supporting both frame-level and token-pooled embeddings.

Layer selection
---------------
Default layers [0, 6, 11, 12] were chosen to span early (layer 0 — raw
acoustic), middle (layer 6 — phonetic), and late (layers 11-12 — lexical/
semantic) representations of HuBERT Base's 13-layer stack. This fixed
selection enables cross-study comparisons without overfitting to SyncNet
correlation scores. The same layer indices are applied to XLS-R.

Usage
-----
    python scripts/15_extract_ssl_embeddings.py                          \\
        [--manifest FILE] [--models hubert,xlsr] [--layers 0,6,11,12]   \\
        [--output-dir DIR] [--device cuda] [--batch-size N] [--smoke]

Dependencies
------------
    pip install torch transformers librosa numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path set-up
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import TARGET_SR, OUTPUT_BASE, embedding_file_stem, half_open_span_mask
from manifest import load_manifest

# ---------------------------------------------------------------------------
# Third-party imports (heavy)
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "hubert": "facebook/hubert-base-ls960",
    "xlsr": "facebook/wav2vec2-large-xlsr-53",
}
"""Known SSL model names and their HuggingFace identifiers."""

DEFAULT_LAYERS: list[int] = [0, 6, 11, 12]
"""Default layer selection spanning early, middle, and late representations."""

DEFAULT_MODELS: list[str] = ["hubert", "xlsr"]

DEFAULT_OUTPUT_DIR: Path = OUTPUT_BASE / "embeddings"

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _load_audio_mono(
    filepath: str | Path, target_sr: int = TARGET_SR
) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono at the requested sample rate."""
    if librosa is None:
        raise ImportError("librosa is required; install with: pip install librosa")
    y, sr_val = librosa.load(str(filepath), sr=target_sr, mono=True)
    return y, int(sr_val)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _compute_frame_stride(model_config: Any) -> int:
    """Compute the CNN encoder downsampling factor from the model config.

    Multiplies all ``conv_stride`` values to get the effective number of
    input audio samples per output Transformer frame.
    """
    strides = getattr(model_config, "conv_stride", None)
    if strides is None:
        # Fallback: most HuBERT/wav2vec2 base models use 320-sample stride
        strides = [5, 2, 2, 2, 2, 2, 2]
    stride = 1
    for value in strides:
        stride *= int(value)
    return stride


def _get_embedding_dim(model_config: Any) -> int:
    return int(model_config.hidden_size)


def load_model(
    model_name: str, device: str = "cpu"
) -> tuple[Any, int, int, int]:
    """Load a frozen SSL model from HuggingFace.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"facebook/hubert-base-ls960"``.
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``).

    Returns
    -------
    model : PreTrainedModel
        Loaded model with ``output_hidden_states=True``.
    embedding_dim : int
        Dimensionality of hidden states.
    frame_stride : int
        Number of audio samples per output frame.
    num_layers : int
        Number of Transformer layers. Valid saved-layer indices are
        ``0`` (input representation) through ``num_layers`` (final output).
    """
    if torch is None:
        raise ImportError("torch is required; install with: pip install torch")

    from transformers import AutoModel  # heavy import, deferred

    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, trust_remote_code=False,
        local_files_only=True,
    )
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    embedding_dim = _get_embedding_dim(model.config)
    frame_stride = _compute_frame_stride(model.config)
    num_layers = int(model.config.num_hidden_layers)

    return model, embedding_dim, frame_stride, num_layers


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_frame_embeddings(
    model: Any,
    audio: np.ndarray,
    sample_rate: int,
    layers: list[int],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frame-level embeddings for the requested layers.

    Parameters
    ----------
    model : PreTrainedModel
        Frozen HuBERT/XLS-R model.
    audio : np.ndarray
        Mono waveform, shape (n_samples,), float32.
    sample_rate : int
        Audio sample rate in Hz (must be 16000).
    layers : list[int]
        Hidden-state indices to extract. Index 0 is the input representation;
        indices 1 through ``num_hidden_layers`` are Transformer outputs.
    device : str
        Torch device string.

    Returns
    -------
    embeddings : np.ndarray
        Shape (n_layers, n_frames, embedding_dim), float32.
    frame_times : np.ndarray
        Shape (n_frames,), centre time in seconds for each frame.
    """
    if torch is None:
        raise ImportError("torch is required")
    if sample_rate != TARGET_SR:
        logger.warning(
            "Expected sample rate %d, got %d; results may be misaligned",
            TARGET_SR, sample_rate,
        )

    waveform = torch.from_numpy(audio.copy()).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(waveform, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (batch, time, dim)

    frame_stride = _compute_frame_stride(model.config)
    n_frames = hidden_states[0].shape[1]

    frame_times = (
        np.arange(n_frames, dtype=np.float32) * (frame_stride / TARGET_SR)
        + (frame_stride / TARGET_SR) / 2.0
    )

    embedding_dim = _get_embedding_dim(model.config)
    valid_layers = [l for l in layers if 0 <= l <= model.config.num_hidden_layers]
    if len(valid_layers) != len(layers):
        logger.warning(
            "Requested layers %s exceed model depth (%d); using subset %s",
            layers, model.config.num_hidden_layers, valid_layers,
        )
        layers = valid_layers
    emb = np.zeros((len(layers), n_frames, embedding_dim), dtype=np.float32)
    for i, layer_idx in enumerate(layers):
        emb[i] = hidden_states[layer_idx].squeeze(0).cpu().numpy()

    return emb, frame_times


# ---------------------------------------------------------------------------
# Token pooling
# ---------------------------------------------------------------------------


def _frame_to_token_pooling(
    layer_embeddings: np.ndarray,
    frame_times: np.ndarray,
    token_spans: list[dict],
) -> list[dict]:
    """Mean-pool frame embeddings into per-token vectors.

    For each token span, collects all frames whose centre time falls within
    ``[start_s, end_s]`` and computes the element-wise mean of their
    embeddings.

    Parameters
    ----------
    layer_embeddings : np.ndarray
        Frame-level embeddings for a single layer, shape (n_frames, dim).
    frame_times : np.ndarray
        Centre time in seconds for each frame, shape (n_frames,).
    token_spans : list[dict]
        Token alignment entries, each containing ``start_s`` and ``end_s``.

    Returns
    -------
    pooled : list[dict]
        Copy of *token_spans* with an ``"embedding"`` key added, holding
        the pooled vector (shape (dim,)) or ``None`` if no frames matched.
    """
    result: list[dict] = []
    for token in token_spans:
        start = float(token["start_s"])
        end = float(token["end_s"])
        mask = half_open_span_mask(frame_times, start, end)
        if not np.any(mask):
            pooled = None
        else:
            pooled = layer_embeddings[mask].mean(axis=0)
        entry = dict(token)
        entry["embedding"] = pooled.tolist() if pooled is not None else None
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _make_output_stem(
    sample_id: int | str, condition: str, variant: str, model_key: str
) -> str:
    """Return the filename stem for a given sample/model combination."""
    return f"{sample_id}_{condition}_{variant}_{model_key}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_layers(
    layers: list[int], num_model_layers: int, model_key: str
) -> list[int]:
    """Validate layer indices and return a deduplicated, sorted list.

    Raises ``ValueError`` if any layer index is out of range or negative.
    """
    if not layers:
        raise ValueError("At least one layer must be specified")
    for layer in layers:
        if layer < 0 or layer > num_model_layers:
            raise ValueError(
                f"Layer {layer} out of range [0, {num_model_layers}] "
                f"for model '{model_key}'"
            )
    seen = set()
    deduped: list[int] = []
    for layer in sorted(layers):
        if layer not in seen:
            seen.add(layer)
            deduped.append(layer)
    return deduped


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load the alignment manifest JSON."""
    if not manifest_path.exists():
        logger.warning("Manifest file not found: %s", manifest_path)
        return []
    try:
        return load_manifest(manifest_path)
    except ValueError as exc:
        logger.warning("Manifest validation failed for %s: %s", manifest_path, exc)
        return []


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_all(
    manifest: list[dict],
    models: list[str],
    layers: list[int],
    output_dir: Path,
    device: str = "cpu",
    smoke: bool = False,
) -> dict[str, int]:
    """Run embedding extraction for all samples, models, and layers.

    Parameters
    ----------
    manifest : list[dict]
        Manifest entries with ``sample_id``, ``condition``, ``variant``,
        ``filepath``, and optionally ``tokens``.
    models : list[str]
        Model keys from ``MODEL_REGISTRY``.
    layers : list[int]
        Layer indices to extract.
    output_dir : Path
        Directory for output files.
    device : str
        Torch device string.
    smoke : bool
        If True, process only the first manifest entry.

    Returns
    -------
    counts : dict
        Keys ``"processed"``, ``"skipped"``, ``"failed"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

    entries = manifest[:1] if smoke else manifest

    loaded_models: dict[str, tuple[Any, int, int, int]] = {}

    for model_key in models:
        model_id = MODEL_REGISTRY[model_key]
        logger.info("Loading model %s from %s ...", model_key, model_id)
        try:
            model, embedding_dim, frame_stride, num_layers = load_model(
                model_id, device=device
            )
            loaded_models[model_key] = (
                model, embedding_dim, frame_stride, num_layers
            )
            logger.info(
                "  %s: dim=%d stride=%d layers=%d",
                model_key, embedding_dim, frame_stride, num_layers,
            )
            _validate_layers(layers, num_layers, model_key)
        except Exception as exc:
            logger.error(
                "Failed to load model %s (%s): %s", model_key, model_id, exc
            )
            logger.warning("Skipping model %s", model_key)

    for entry in entries:
        sample_id = entry.get("sample_id", entry.get("utterance_id", "unknown"))
        condition = entry["condition"]
        variant = entry.get("variant", "raw")
        filepath = Path(entry.get("audio_path", entry.get("filepath", "")))
        tokens = entry.get("tokens", [])

        if not filepath.exists():
            logger.warning(
                "Audio file not found for sample %s (%s/%s): %s",
                sample_id, condition, variant, filepath,
            )
            counts["failed"] += 1
            continue

        try:
            audio, sr = _load_audio_mono(filepath)
            duration_s = len(audio) / sr
        except Exception as exc:
            logger.error(
                "Failed to load audio sample %s (%s/%s): %s",
                sample_id, condition, variant, exc,
            )
            counts["failed"] += 1
            continue

        for model_key, (model, embedding_dim, frame_stride, num_layers) in loaded_models.items():
            stem = embedding_file_stem(entry, model_key, variant)
            npy_path = output_dir / f"{stem}.npy"
            json_path = output_dir / f"{stem}.json"
            npy_path.parent.mkdir(parents=True, exist_ok=True)

            if npy_path.exists() and json_path.exists():
                logger.debug("Skipping existing: %s", stem)
                counts["skipped"] += 1
                continue

            t0 = time.perf_counter()
            try:
                emb, frame_times = extract_frame_embeddings(
                    model, audio, sr, layers, device=device,
                )
            except Exception as exc:
                logger.error(
                    "Embedding extraction failed for %s: %s", stem, exc,
                )
                counts["failed"] += 1
                continue

            np.save(npy_path, emb)
            n_frames = emb.shape[1]

            token_entries: list[dict] | None = None
            if tokens:
                # Pool from last requested layer (most semantic)
                best_layer = emb[-1]
                token_entries = _frame_to_token_pooling(
                    best_layer, frame_times, tokens,
                )

            # Record the layers ACTUALLY stored, not the requested list:
            # ``extract_frame_embeddings`` drops any layer above the final
            # hidden-state index, so the NPY rows may be a strict subset.
            stored_layers = [l for l in layers if 0 <= l <= num_layers]
            metadata: dict = {
                "sample_id": sample_id,
                "utterance_id": entry.get("utterance_id", str(sample_id)),
                "dataset": entry.get("dataset", "legacy"),
                "speaker_id": entry.get("speaker_id", str(sample_id)),
                "paired_key": entry.get("paired_key"),
                "split": entry.get("split"),
                "representation_only": bool(entry.get("representation_only", False)),
                "alignment_source": entry.get("alignment_source", "missing"),
                "alignment_manifest": entry.get("alignment_manifest"),
                "alignment_manifest_sha256": entry.get("alignment_manifest_sha256"),
                "condition": condition,
                "tts_provider": entry.get("tts_provider"),
                "variant": variant,
                "embedding_stem": entry.get("embedding_stem", entry.get("embedding_file_stem")),
                "model": model_key,
                "model_id": MODEL_REGISTRY[model_key],
                "sample_rate": sr,
                "duration_s": round(duration_s, 4),
                "audio_path": str(filepath),
                "audio_sha256": entry.get("source_sha256", entry.get("natural_sha256")),
                "preprocessing": {
                    "mono": True,
                    "target_sample_rate": TARGET_SR,
                    "resampling": "librosa.load",
                    "normalization": "librosa.load_default_float32",
                },
                "frame_stride_samples": int(frame_stride),
                "frame_times_s": [round(float(t), 6) for t in frame_times],
                "layers": stored_layers,
                "num_frames": int(n_frames),
                "embedding_dim": embedding_dim,
                "shape": [len(stored_layers), int(n_frames), int(embedding_dim)],
                "filepath": str(npy_path.relative_to(output_dir.parent.parent)),
                "tokens": token_entries,
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            elapsed = time.perf_counter() - t0
            logger.info(
                "  %s  frames=%d  duration=%.1fs  elapsed=%.1fs",
                stem, n_frames, duration_s, elapsed,
            )
            counts["processed"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-layer HuBERT/XLS-R embeddings for Wav2Sem analysis.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to alignment manifest JSON (produced by 14_prepare_mandarin_alignment.py).",
    )
    parser.add_argument(
        "--models", type=str, default="hubert,xlsr",
        help="Comma-separated model keys (default: hubert,xlsr).",
    )
    parser.add_argument(
        "--layers", type=str, default="0,6,11,12",
        help="Comma-separated 0-indexed layer indices (default: 0,6,11,12).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for embeddings (default: data/wav2sem_analysis/embeddings/).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Process only the first manifest entry.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Processing batch size (reserved for future use).",
    )
    return parser.parse_args(argv)


def _auto_detect_device() -> str:
    """Detect CUDA availability; warn when falling back to CPU."""
    if torch is None:
        logger.warning("torch not available; using cpu")
        return "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available; using %s", device)
        return device
    logger.warning("CUDA not available; using cpu (may be slow)")
    return "cpu"


def _parse_comma_list(raw: str) -> list[int]:
    """Parse a comma-separated string of integers."""
    result: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        result.append(int(x))
    return result


def _parse_model_keys(raw: str) -> list[str]:
    """Parse a comma-separated list of model names, validating against registry."""
    keys: list[str] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        x = x.lower()
        if x not in MODEL_REGISTRY:
            logger.warning(
                "Unknown model key '%s'; valid keys: %s",
                x, ", ".join(MODEL_REGISTRY),
            )
            continue
        keys.append(x)
    return keys


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # --- Validate dependencies ---
    if torch is None:
        print("Error: torch is required. Install with: pip install torch", file=sys.stderr)
        return 1
    if librosa is None:
        print("Error: librosa is required. Install with: pip install librosa", file=sys.stderr)
        return 1

    # --- Parse arguments ---
    models = _parse_model_keys(args.models)
    if not models:
        print("Error: no valid model keys provided", file=sys.stderr)
        return 1

    layers = _parse_comma_list(args.layers)
    if not layers:
        print("Error: no layers specified", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    device = args.device if args.device else _auto_detect_device()

    # --- Load manifest ---
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = OUTPUT_BASE / "manifest" / "alignment.json"
    logger.info("Loading manifest from %s", manifest_path)
    manifest = _load_manifest(manifest_path)
    if not manifest:
        print(f"Error: no entries found in manifest: {manifest_path}", file=sys.stderr)
        return 1
    logger.info("Loaded %d manifest entries", len(manifest))

    # --- Process ---
    try:
        counts = process_all(
            manifest=manifest,
            models=models,
            layers=layers,
            output_dir=output_dir,
            device=device,
            smoke=args.smoke,
        )
    except (IOError, OSError) as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "Done — processed: %d  skipped: %d  failed: %d",
        counts["processed"], counts["skipped"], counts["failed"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
