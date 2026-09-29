from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .intervention import (
    build_frame_map,
    build_patch,
    decompose_dynamic,
    mel_frame_clock,
    patch_metadata,
    shift_frame_map,
)
from .protocol import (
    MODELS,
    NATURAL,
    ProtocolError,
    SCIENCE_CONDITIONS,
    bytes_sha256,
    canonical_hash,
    file_sha256,
    resolve_path,
    write_json,
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CPU-only test environments
        raise ProtocolError("torch is required for the Wav2Lip worker") from exc
    return torch


def _prepare_inputs(frames: Sequence[np.ndarray], boxes: Sequence[Sequence[int]], chunks: Sequence[np.ndarray]) -> tuple[list[np.ndarray], list[list[int]]]:
    import cv2

    if len(boxes) < len(chunks) or len(frames) < len(chunks):
        raise ProtocolError("frames/boxes are shorter than mel chunks")
    images: list[np.ndarray] = []
    selected_boxes: list[list[int]] = []
    for index, chunk in enumerate(chunks):
        _ = chunk
        top, bottom, left, right = (int(value) for value in boxes[index])
        frame = np.asarray(frames[index]).copy()
        face = frame[top:bottom, left:right]
        if face.size == 0:
            raise ProtocolError(f"empty ROI at frame {index}")
        face = cv2.resize(face, (96, 96))
        masked = face.copy()
        masked[48:] = 0
        images.append(np.concatenate((masked, face), axis=2).astype(np.float32) / 255.0)
        selected_boxes.append([top, bottom, left, right])
    return images, selected_boxes


def _tensor_batch(torch: Any, values: Sequence[np.ndarray], *, device: str, mel: bool) -> Any:
    array = np.asarray(values, dtype=np.float32)
    if mel:
        tensor = torch.from_numpy(array[:, None, :, :])
    else:
        tensor = torch.from_numpy(np.transpose(array, (0, 3, 1, 2)))
    return tensor.to(device=device, dtype=torch.float32)


def capture_bottleneck(
    model: Any,
    mel_chunks: Sequence[np.ndarray],
    images: Sequence[np.ndarray],
    *,
    device: str,
    batch_size: int = 4,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Capture ``audio_encoder`` output with absolute indices and guaranteed hook cleanup."""

    torch = _torch()
    if len(mel_chunks) != len(images):
        raise ProtocolError("mel/image count mismatch during bottleneck capture")
    indices = list(range(len(mel_chunks))) if frame_indices is None else [int(value) for value in frame_indices]
    if len(indices) != len(mel_chunks):
        raise ProtocolError("frame index count mismatch during bottleneck capture")
    captured: list[Any] = []
    hook = None

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(value, "shape") or len(value.shape) != 4 or int(value.shape[1]) != 512 or tuple(value.shape[2:]) != (1, 1):
            raise ProtocolError(f"audio_encoder output is not [B,512,1,1]: {getattr(value, 'shape', None)}")
        captured.append(value.detach().cpu().to(torch.float32))

    try:
        if not hasattr(model, "audio_encoder"):
            raise ProtocolError("Wav2Lip model has no audio_encoder")
        hook = model.audio_encoder.register_forward_hook(capture)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(mel_chunks), int(batch_size)):
                stop = min(len(mel_chunks), start + int(batch_size))
                mel_batch = _tensor_batch(torch, mel_chunks[start:stop], device=device, mel=True)
                image_batch = _tensor_batch(torch, images[start:stop], device=device, mel=False)
                model(mel_batch, image_batch)
    finally:
        if hook is not None:
            hook.remove()
    if len(captured) != (len(mel_chunks) + int(batch_size) - 1) // int(batch_size):
        raise ProtocolError("audio_encoder hook did not observe every batch")
    value = torch.cat(captured, dim=0).numpy().reshape(len(mel_chunks), 512).astype(np.float32)
    return {"values": value, "frame_indices": indices, "shape": list(value.shape), "hook_removed": True, "batch_size": int(batch_size)}


class PatchedForward:
    """Runtime-only output replacement for one Wav2Lip audio bottleneck."""

    def __init__(self, model: Any, patches: np.ndarray, *, device: str) -> None:
        self.model = model
        self.patches = np.asarray(patches, dtype=np.float32)
        self.device = device
        self._hook: Any | None = None
        self._indices: list[int] | None = None
        self.calls = 0

    def __enter__(self) -> "PatchedForward":
        if self.patches.ndim != 2 or self.patches.shape[1] != 512:
            raise ProtocolError("patch table must have shape (frames, 512)")
        if not hasattr(self.model, "audio_encoder"):
            raise ProtocolError("Wav2Lip model has no audio_encoder")
        self._hook = self.model.audio_encoder.register_forward_hook(self._replace)
        return self

    def set_frame_indices(self, indices: Sequence[int]) -> None:
        self._indices = [int(value) for value in indices]

    def _replace(self, _module: Any, _inputs: Any, output: Any) -> Any:
        torch = _torch()
        value = output[0] if isinstance(output, (tuple, list)) else output
        if self._indices is None or len(self._indices) != int(value.shape[0]):
            raise ProtocolError("patched forward has no matching absolute frame indices")
        patched = value.clone()
        for position, index in enumerate(self._indices):
            if index < 0 or index >= len(self.patches):
                raise ProtocolError(f"patched frame index out of range: {index}")
            replacement = torch.as_tensor(self.patches[index], dtype=value.dtype, device=value.device).reshape(512, 1, 1)
            patched[position] = replacement
        self.calls += 1
        if isinstance(output, tuple):
            return (patched, *output[1:])
        if isinstance(output, list):
            return [patched, *output[1:]]
        return patched

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._indices = None


def _decode_prediction(prediction: Any) -> list[np.ndarray]:
    array = prediction.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
    # Match the registered generation_worker path exactly: its decoder output
    # is copied into the BGR ROI without a second colour-space conversion.
    return [np.clip(item, 0, 255).astype(np.uint8) for item in array]


def _forward_frames(
    model: Any,
    images: Sequence[np.ndarray],
    chunks: Sequence[np.ndarray],
    boxes: Sequence[Sequence[int]],
    *,
    device: str,
    batch_size: int,
    patch: np.ndarray | None = None,
) -> list[np.ndarray]:
    torch = _torch()
    import cv2

    prepared, selected_boxes = _prepare_inputs(images, boxes, chunks)
    rendered: list[np.ndarray] = []
    context = PatchedForward(model, patch, device=device) if patch is not None else contextlib.nullcontext()
    with context as patched_forward:
        with torch.no_grad():
            for start in range(0, len(chunks), int(batch_size)):
                stop = min(len(chunks), start + int(batch_size))
                if patch is not None:
                    assert isinstance(patched_forward, PatchedForward)
                    patched_forward.set_frame_indices(range(start, stop))
                mel_batch = _tensor_batch(torch, chunks[start:stop], device=device, mel=True)
                image_batch = _tensor_batch(torch, prepared[start:stop], device=device, mel=False)
                prediction = model(mel_batch, image_batch)
                for offset, predicted in enumerate(_decode_prediction(prediction)):
                    top, bottom, left, right = selected_boxes[start + offset]
                    frame = np.asarray(images[start + offset]).copy()
                    patch_image = cv2.resize(predicted, (right - left, bottom - top))
                    frame[top:bottom, left:right] = patch_image
                    rendered.append(frame)
    if patch is not None and not isinstance(context, contextlib.nullcontext):
        pass
    return rendered


def render_patches(
    model: Any,
    frames: Sequence[np.ndarray],
    boxes: Sequence[Sequence[int]],
    chunks: Sequence[np.ndarray],
    patches: Mapping[str, np.ndarray],
    *,
    device: str,
    batch_size: int = 4,
    condition_names: Sequence[str] = SCIENCE_CONDITIONS,
) -> dict[str, list[np.ndarray]]:
    """Render registered conditions with one model and explicit batch indices."""

    output: dict[str, list[np.ndarray]] = {}
    for condition in condition_names:
        patch = None if condition == "BASE" else np.asarray(patches[condition], dtype=np.float32)
        output[condition] = _forward_frames(model, frames, chunks, boxes, device=device, batch_size=batch_size, patch=patch)
    return output


def _load_wav2lip_worker() -> Any:
    from scripts.experiments.wav2lip_face_roi_replacement import generation_worker

    return generation_worker


def _load_model(checkpoint: Path, device: str) -> Any:
    worker = _load_wav2lip_worker()
    return worker.load_model(checkpoint, device)


def _mux_pcm(video_only: Path, audio: Path, output: Path, ffmpeg: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    command = [str(ffmpeg), "-y", "-v", "error", "-i", str(video_only), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "matroska", str(temporary)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"PCM mux failed: {(result.stderr or '')[-1000:]}")
    temporary.replace(output)
    return {"command": command, "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(), "pcm_sha256": _muxed_pcm_hash(output, ffmpeg)}


def _muxed_pcm_hash(path: Path, ffmpeg: Path) -> str:
    result = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"], capture_output=True, check=False, timeout=300)
    if result.returncode != 0:
        raise ProtocolError(f"cannot decode patch PCM: {path}")
    return bytes_sha256(result.stdout)


def render_patch_cell(
    recipient: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    direction: str,
    output_dir: Path,
    checkpoint: Path,
    ffmpeg: Path,
    device: str,
    batch_size: int = 4,
    lam: float = 0.5,
    smoothing_frames: int = 5,
    mapping: Mapping[str, Any],
    condition_names: Sequence[str] = SCIENCE_CONDITIONS,
) -> dict[str, Any]:
    """Capture two arms and render BASE/COHERENT/SCRAMBLED/ERASE for one pair."""

    if device != "cuda":
        raise ProtocolError("patch rendering must use the guarded CUDA worker")
    import cv2

    started = time.monotonic()
    reference = recipient.get("reference") or {}
    reference_image = cv2.imread(str(resolve_path(str(reference.get("reference", "")))), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise ProtocolError("cannot read frozen external reference frame")
    box = [int(value) for value in reference.get("box_top_bottom_left_right", [])]
    if len(box) != 4:
        raise ProtocolError("reference box is malformed")
    worker = _load_wav2lip_worker()
    recipient_chunks, recipient_info = worker.mel_chunks(Path(str(recipient["audio"])))
    donor_chunks, donor_info = worker.mel_chunks(Path(str(donor["audio"])))
    recipient_clock = mel_frame_clock(int(recipient_info["mel_shape"][1]), len(recipient_chunks))
    donor_clock = mel_frame_clock(int(donor_info["mel_shape"][1]), len(donor_chunks))
    frame_count = len(recipient_chunks)
    frames = [reference_image.copy() for _ in range(frame_count)]
    boxes = [box[:] for _ in range(frame_count)]
    recipient_images, _ = _prepare_inputs(frames, boxes, recipient_chunks)
    donor_images, _ = _prepare_inputs([reference_image.copy() for _ in donor_chunks], [box[:] for _ in donor_chunks], donor_chunks)
    model = _load_model(checkpoint, device)
    recipient_latent = capture_bottleneck(model, recipient_chunks, recipient_images, device=device, batch_size=batch_size)["values"]
    donor_latent = capture_bottleneck(model, donor_chunks, donor_images, device=device, batch_size=batch_size)["values"]
    frame_map = build_frame_map(
        [item.center_s for item in recipient_clock],
        [item.center_s for item in donor_clock],
        list(mapping.get("segments", [])),
        direction="n_to_t" if direction == "natural_from_tts" else "t_to_n",
        support_blocks=list(mapping.get("support_blocks", [])),
        recipient_tail=[item.tail_reused for item in recipient_clock],
        donor_tail=[item.tail_reused for item in donor_clock],
    )
    blocks = decompose_dynamic(recipient_latent, donor_latent, frame_map, smoothing_frames=smoothing_frames, minimum_core_frames=25)
    if tuple(condition_names) == ("BASE", "IDENTITY", "SHIFT_PLUS3", "SHIFT_MINUS3"):
        identity = recipient_latent.copy()
        shifted_plus = decompose_dynamic(
            recipient_latent,
            donor_latent,
            shift_frame_map(frame_map, 3, donor_count=len(donor_latent)),
            smoothing_frames=smoothing_frames,
            minimum_core_frames=25,
        )
        shifted_minus = decompose_dynamic(
            recipient_latent,
            donor_latent,
            shift_frame_map(frame_map, -3, donor_count=len(donor_latent)),
            smoothing_frames=smoothing_frames,
            minimum_core_frames=25,
        )
        patches = {
            "BASE": recipient_latent,
            "IDENTITY": identity,
            "SHIFT_PLUS3": build_patch(recipient_latent, shifted_plus, "COHERENT", lam=lam),
            "SHIFT_MINUS3": build_patch(recipient_latent, shifted_minus, "COHERENT", lam=lam),
        }
        blocks_for_metadata = blocks
    else:
        patches = {condition: build_patch(recipient_latent, blocks, condition, lam=lam) for condition in condition_names}
        blocks_for_metadata = blocks
    rendered = render_patches(model, frames, boxes, recipient_chunks, patches, device=device, batch_size=batch_size, condition_names=condition_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    np.save(latent_dir / "recipient.npy", recipient_latent.astype(np.float32), allow_pickle=False)
    np.save(latent_dir / "donor.npy", donor_latent.astype(np.float32), allow_pickle=False)
    outputs: dict[str, Any] = {}
    for condition, frames_out in rendered.items():
        video_only = output_dir / f"{condition}.video_only.mkv"
        output = output_dir / f"{condition}.mkv"
        worker.encode_video(frames_out, video_only, ffmpeg)
        mux = _mux_pcm(video_only, Path(str(recipient["audio"])), output, ffmpeg)
        video_only.unlink(missing_ok=True)
        outputs[condition] = {"video": str(output.resolve()), "video_sha256": file_sha256(output), "audio_pcm_sha256": mux["pcm_sha256"], "frame_count": len(frames_out), "patch": patch_metadata(blocks_for_metadata, condition=condition, lam=lam)}
    result = {
        "schema_version": 1,
        "protocol_id": "tts_evidence_temporal_patch_v1",
        "status": "complete",
        "direction": direction,
        "source_group": recipient["source_group"],
        "sample_id": recipient["sample_id"],
        "recipient_arm": recipient["arm"],
        "donor_arm": donor["arm"],
        "recipient_audio": recipient["audio"],
        "recipient_audio_sha256": recipient["audio_sha256"],
        "recipient_audio_pcm_sha256": recipient.get("audio_pcm_sha256"),
        "donor_audio": donor["audio"],
        "donor_audio_sha256": donor["audio_sha256"],
        "mapping": dict(mapping),
        "mapping_hash": canonical_hash(mapping),
        "support_hash": canonical_hash(mapping.get("support_blocks", [])),
        "recipient_latent": str((latent_dir / "recipient.npy").resolve()),
        "recipient_latent_sha256": file_sha256(latent_dir / "recipient.npy"),
        "donor_latent": str((latent_dir / "donor.npy").resolve()),
        "donor_latent_sha256": file_sha256(latent_dir / "donor.npy"),
        "latent_layer": "audio_encoder.output",
        "latent_shape": list(recipient_latent.shape),
        "lambda": float(lam),
        "smoothing_frames": int(smoothing_frames),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device": device,
        "batch_size": int(batch_size),
        "condition_names": list(condition_names),
        "mode": "technical_controls" if tuple(condition_names) == ("BASE", "IDENTITY", "SHIFT_PLUS3", "SHIFT_MINUS3") else "science_conditions",
        "hook_removed": True,
        "outputs": outputs,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    write_json(output_dir / "receipt.json", result)
    return result
