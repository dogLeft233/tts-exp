"""Frozen Wav2Lip/SyncNet adapter for waveform-head fitting.

The context contains the exact per-frame face crops and the corresponding
SyncNet face crops from one pinned source-video timeline.  Audio is transformed
to official Wav2Lip mel chunks with a Torch frontend; output face patches are
then pasted into the same SyncNet crops before fixed-clock five-frame scoring.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as F
except ImportError as exc:  # optional training dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None

from .wav2lip_frontend import Wav2LipTorchMel, official_chunk_starts


@dataclass(frozen=True)
class Wav2LipSyncContext:
    """Prepared images/crop geometry from one frozen source-video timeline.

    ``face_crops_bgr`` contains the raw Wav2Lip generation boxes resized to
    96x96, as used by its inference path. ``syncnet_crops_bgr`` contains the
    corresponding fixed SyncNet scoring crops (224x224, BGR, 0..255). Each
    patch box is the exact destination of the Wav2Lip 96x96 prediction inside
    that scoring crop. These arrays are read-only conditioning data; the new
    waveform affects only the Wav2Lip mel input.
    """

    face_crops_bgr: Any
    syncnet_crops_bgr: Any
    patch_boxes_xyxy: Any
    mel_starts: Sequence[int]
    frame_timestamps_seconds: Sequence[float]
    # Engineering probes may deliberately render a strict prefix of the
    # frozen source clock to bound autograd memory. When true, mel_starts must
    # equal the corresponding prefix of the official full-clip chunk grid.
    mel_starts_are_prefix: bool = False
    fps_num: int = 25
    fps_den: int = 1
    # First frame of the selected SyncNet track in the full source video.
    # The official runner crops audio to [first_track_frame,
    # last_track_frame + 1) before computing MFCCs.
    track_start_frame: int | None = None
    # Optional exact PCM16 crop produced by the pinned ffmpeg/readback path.
    # When absent, training uses a clearly-labelled saturating np.rint proxy
    # from canonical float32 PCM; it is not claimed codec-equivalent.
    syncnet_pcm16_crop: Any | None = None
    syncnet_pcm16_crop_sha256: str | None = None
    expected_sample_count: int | None = None
    source_video_sha256: str | None = None
    face_crop_sha256: str | None = None
    syncnet_crop_sha256: str | None = None

    @property
    def frame_count(self) -> int:
        return int(self.face_crops_bgr.shape[0])

    def validate(self) -> None:
        if torch is None:
            raise RuntimeError("Wav2LipSyncContext validation requires PyTorch") from _TORCH_IMPORT_ERROR
        faces = torch.as_tensor(self.face_crops_bgr)
        crops = torch.as_tensor(self.syncnet_crops_bgr)
        boxes = torch.as_tensor(self.patch_boxes_xyxy)
        timestamps = np.asarray(self.frame_timestamps_seconds, dtype=np.float64)
        mel_starts = np.asarray(self.mel_starts, dtype=np.int64)
        if self.fps_num <= 0 or self.fps_den <= 0:
            raise ValueError("video frame rate must be a positive rational")
        if faces.ndim != 4 or tuple(faces.shape[1:]) != (3, 96, 96) or faces.shape[0] == 0:
            raise ValueError("Wav2Lip face crops must have shape [T,3,96,96]")
        if crops.ndim != 4 or tuple(crops.shape[1:]) != (3, 224, 224) or crops.shape[0] != faces.shape[0]:
            raise ValueError("SyncNet crops must have shape [T,3,224,224] and match face frames")
        if boxes.shape != (faces.shape[0], 4) or boxes.is_floating_point() or boxes.is_complex():
            raise ValueError("patch_boxes_xyxy must be integer [T,4]")
        if mel_starts.shape != (faces.shape[0],) or np.any(mel_starts < 0):
            raise ValueError("mel start indices must contain one non-negative integer per frame")
        if timestamps.shape != (faces.shape[0],) or not np.isfinite(timestamps).all():
            raise ValueError("frame timestamps must be finite and match the context frame count")
        if timestamps.size > 1:
            delta = np.diff(timestamps)
            expected = self.fps_den / self.fps_num
            if np.any(delta <= 0) or not np.allclose(delta, expected, atol=1e-6, rtol=0.0):
                raise ValueError("source video timeline is not the frozen constant-FPS frame clock")
        if faces.is_floating_point():
            if not bool(torch.isfinite(faces).all().item()):
                raise ValueError("Wav2Lip face crops contain non-finite pixels")
            if float(faces.min().item()) < 0 or float(faces.max().item()) > 255.0:
                raise ValueError("Wav2Lip face crop pixels must be BGR in [0,255]")
        if crops.is_floating_point():
            if not bool(torch.isfinite(crops).all().item()):
                raise ValueError("SyncNet crops contain non-finite pixels")
            if float(crops.min().item()) < 0 or float(crops.max().item()) > 255.0:
                raise ValueError("SyncNet crop pixels must be BGR in [0,255]")
        if self.expected_sample_count is not None:
            if self.expected_sample_count <= 0:
                raise ValueError("expected_sample_count must be positive")
        if self.track_start_frame is not None and self.track_start_frame < 0:
            raise ValueError("track_start_frame must be non-negative")
        if self.syncnet_pcm16_crop is not None:
            pcm16 = np.asarray(self.syncnet_pcm16_crop)
            if pcm16.dtype != np.int16 or pcm16.ndim != 1 or pcm16.size == 0:
                raise ValueError("SyncNet's exact score crop must be nonempty mono int16 PCM")
            if self.syncnet_pcm16_crop_sha256 is not None:
                actual = hashlib.sha256(np.ascontiguousarray(pcm16).tobytes()).hexdigest()
                if actual != self.syncnet_pcm16_crop_sha256:
                    raise ValueError("SyncNet PCM16 crop array SHA does not match its binding")
        for index, box in enumerate(boxes.detach().cpu().tolist()):
            x1, y1, x2, y2 = [int(value) for value in box]
            if not (0 <= x1 < x2 <= 224 and 0 <= y1 < y2 <= 224):
                raise ValueError(f"invalid SyncNet patch paste box at frame {index}: {box}")


def prepare_wav2lip_face_input(face_crops_bgr: Tensor) -> Tensor:
    """Build the official six-channel masked/full Wav2Lip face tensor."""

    if torch is None:
        raise RuntimeError("face tensor preparation requires PyTorch") from _TORCH_IMPORT_ERROR
    faces = torch.as_tensor(face_crops_bgr)
    if faces.ndim != 4 or tuple(faces.shape[1:]) != (3, 96, 96):
        raise ValueError("face crops must have shape [T,3,96,96]")
    faces = faces.to(dtype=torch.float32)
    if not bool(torch.isfinite(faces).all().item()) or float(faces.min().item()) < 0 or float(faces.max().item()) > 255.0:
        raise ValueError("face crops must contain finite BGR pixels in [0,255]")
    full = faces / 255.0
    masked = full.clone()
    masked[:, :, 48:, :] = 0.0
    return torch.cat((masked, full), dim=1).contiguous()


def _straight_through_quantize(value: Tensor, *, mode: str) -> Tensor:
    """Approximate official uint8 I/O while retaining an optimizer gradient.

    This is a training proxy only.  The real AVI/JPEG/SyncNet path must still
    be checked on fit/dev data; this function does not claim codec equivalence.
    """

    bounded = value.clamp(0.0, 255.0)
    if mode == "truncate":
        integer = torch.floor(bounded)
    elif mode == "round":
        integer = torch.round(bounded)
    else:
        raise ValueError("quantization mode must be truncate or round")
    return bounded + (integer - bounded).detach()


def _single_clip_audio(waveform: Tensor) -> Tensor:
    x = torch.as_tensor(waveform)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError("training adapter currently processes one clip per step")
        x = x[0]
    if x.ndim != 1 or x.numel() == 0 or not x.is_floating_point():
        raise ValueError("audio must be one non-empty float waveform [N]")
    x = x.to(dtype=torch.float32)
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("audio contains NaN or infinity")
    return x


def _syncnet_pcm16_mfcc(
    waveform: Tensor | None,
    *,
    pcm16_crop: np.ndarray | None = None,
    psf_mfcc: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute official SyncNet MFCCs from the scoring PCM16 derivative.

    Natural float32 PCM remains the scoring source. Prefer the exact int16
    crop extracted by the pinned ffmpeg pipeline; without it, make an explicit
    saturating round-to-nearest int16 proxy from float32. The fallback is only
    for optimization plumbing and is not a codec-parity claim.
    """

    if pcm16_crop is None:
        if waveform is None:
            raise ValueError("float32 waveform is required when no exact PCM16 crop is supplied")
        natural = _single_clip_audio(waveform).detach().cpu().numpy().astype(np.float64)
        pcm16 = np.rint(np.clip(natural * 32768.0, -32768.0, 32767.0)).astype(np.int16)
    else:
        pcm16 = np.asarray(pcm16_crop)
        if pcm16.dtype != np.int16 or pcm16.ndim != 1 or pcm16.size == 0:
            raise ValueError("exact SyncNet PCM16 crop must be nonempty mono int16")
        pcm16 = np.ascontiguousarray(pcm16)
    if psf_mfcc is None:
        try:
            from python_speech_features import mfcc as psf_mfcc
        except ImportError as exc:  # pragma: no cover - remote model environment
            raise RuntimeError("SyncNet training requires python_speech_features") from exc
    # SyncNetInstance.evaluate calls python_speech_features.mfcc(audio,
    # sample_rate) with its frozen defaults (25 ms window, 10 ms hop,
    # 13 coefficients, 26 filters, preemphasis 0.97, lifter 22 and energy in
    # coefficient zero). Do not substitute the Wav2Lip 80-bin mel frontend.
    features = np.asarray(psf_mfcc(pcm16, samplerate=16_000), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 13 or not np.isfinite(features).all():
        raise RuntimeError(f"SyncNet MFCC frontend returned unexpected shape {features.shape}")
    return np.ascontiguousarray(features), np.ascontiguousarray(pcm16)


def _official_track_audio_crop(waveform: Tensor, context: Wav2LipSyncContext) -> Tensor:
    """Crop natural PCM to the selected track's official 25 fps interval."""

    audio = _single_clip_audio(waveform)
    if context.fps_num != 25 or context.fps_den != 1:
        raise ValueError("the pinned SyncNet evaluator is fixed at 25 fps")
    if context.track_start_frame is None:
        frame_position = float(context.frame_timestamps_seconds[0]) * 25.0
        start_frame = int(round(frame_position))
        if not math.isclose(frame_position, start_frame, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "timestamps must expose the selected track's absolute 25 fps start; "
                "otherwise pass track_start_frame explicitly"
            )
    else:
        start_frame = int(context.track_start_frame)
        if not math.isclose(
            float(context.frame_timestamps_seconds[0]), start_frame / 25.0,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise ValueError("track_start_frame disagrees with the first absolute timestamp")
    start_sample = start_frame * 640
    stop_sample = (start_frame + context.frame_count) * 640
    if stop_sample > int(audio.numel()):
        raise ValueError("natural audio does not cover the selected SyncNet track interval")
    cropped = audio[start_sample:stop_sample]
    if cropped.numel() != context.frame_count * 640:
        raise RuntimeError("selected-track audio crop changed the frozen source clock")
    return cropped


class Wav2LipSyncAdapter:
    """Differentiable frozen Wav2Lip video plus natural-audio SyncNet scorer.

    The mel frontend must carry a numerical ``check_frontend_parity`` receipt
    with status PASS. Wav2Lip and SyncNet weights are frozen, but forward
    autograd remains enabled from fixed-anchor SyncNet distance through pasted
    video crops, Wav2Lip and the waveform head.
    """

    def __init__(
        self,
        *,
        wav2lip_model: Any,
        syncnet_model: Any,
        mel_frontend: Wav2LipTorchMel,
        frontend_parity: dict[str, Any],
        frame_batch_size: int = 8,
        syncnet_batch_size: int = 32,
        psf_mfcc: Callable[..., Any] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("Wav2LipSyncAdapter requires PyTorch") from _TORCH_IMPORT_ERROR
        if frontend_parity.get("status") != "PASS":
            raise ValueError("official Wav2Lip mel parity must pass before training")
        if frame_batch_size <= 0 or syncnet_batch_size <= 0:
            raise ValueError("frame batch sizes must be positive")
        self.wav2lip_model = wav2lip_model
        self.syncnet_model = syncnet_model
        self.mel_frontend = mel_frontend
        self.frontend_parity = dict(frontend_parity)
        self.frame_batch_size = int(frame_batch_size)
        self.syncnet_batch_size = int(syncnet_batch_size)
        self.psf_mfcc = psf_mfcc
        self.wav2lip_model.eval()
        self.syncnet_model.eval()
        for model in (self.wav2lip_model, self.syncnet_model):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def render_video(self, context: Wav2LipSyncContext, enhanced_audio: Tensor) -> Tensor:
        """Render/paste T frames without changing their frozen source clock."""

        context.validate()
        audio = _single_clip_audio(enhanced_audio)
        if context.expected_sample_count is not None and audio.numel() != context.expected_sample_count:
            raise ValueError("enhancement changed audio sample count")
        device = next(self.wav2lip_model.parameters()).device
        mel = self.mel_frontend(audio.to(device=device))
        official_starts = official_chunk_starts(
            int(mel.shape[-1]), fps_num=context.fps_num, fps_den=context.fps_den
        )
        frozen_starts = [int(value) for value in context.mel_starts]
        if context.mel_starts_are_prefix:
            if len(frozen_starts) >= len(official_starts) or official_starts[: len(frozen_starts)] != frozen_starts:
                raise ValueError("engineering mel starts are not a strict prefix of the official full clock")
        elif official_starts != frozen_starts:
            raise ValueError("enhanced audio mel-chunk clock differs from frozen source context")
        if len(frozen_starts) != context.frame_count:
            raise ValueError("Wav2Lip produced mel chunks that do not cover every frozen video frame")
        starts = frozen_starts
        chunks = torch.stack([mel[:, start : start + 16] for start in starts], dim=0)
        if tuple(chunks.shape[1:]) != (80, 16):
            raise RuntimeError("Wav2Lip audio chunk has unexpected shape")
        face_input = prepare_wav2lip_face_input(
            torch.as_tensor(context.face_crops_bgr, device=device)
        )
        predicted = []
        for first in range(0, context.frame_count, self.frame_batch_size):
            stop = min(context.frame_count, first + self.frame_batch_size)
            mel_batch = chunks[first:stop, None, :, :].contiguous()
            image_batch = face_input[first:stop].contiguous()
            prediction = self.wav2lip_model(mel_batch, image_batch)
            if not torch.is_tensor(prediction) or tuple(prediction.shape) != (stop - first, 3, 96, 96):
                actual_shape = tuple(prediction.shape) if torch.is_tensor(prediction) else type(prediction).__name__
                raise RuntimeError(f"Wav2Lip forward returned unexpected patch shape {actual_shape}")
            if not bool(torch.isfinite(prediction).all().item()):
                raise FloatingPointError("Wav2Lip output patch contains NaN or infinity")
            predicted.append(prediction)
        # Official inference multiplies sigmoid output by 255 and casts to
        # uint8 before OpenCV resizes/pastes it.  Use a straight-through
        # truncation for the frozen differentiable proxy.
        patches_0_255 = _straight_through_quantize(
            torch.cat(predicted, dim=0) * 255.0, mode="truncate"
        )
        scoring_crops = torch.as_tensor(context.syncnet_crops_bgr, device=device, dtype=torch.float32)
        boxes = torch.as_tensor(context.patch_boxes_xyxy, device=device, dtype=torch.int64)
        pasted = []
        for index in range(context.frame_count):
            x1, y1, x2, y2 = [int(value) for value in boxes[index].tolist()]
            patch = F.interpolate(
                patches_0_255[index : index + 1],
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0]
            patch = _straight_through_quantize(patch, mode="round")
            frame = scoring_crops[index].clone()
            frame[:, y1:y2, x1:x2] = patch
            pasted.append(frame)
        result = torch.stack(pasted, dim=0)
        if tuple(result.shape) != (context.frame_count, 3, 224, 224):
            raise RuntimeError("patch paste changed the SyncNet scoring crop clock/shape")
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("pasted SyncNet frames contain NaN or infinity")
        return result

    def sync_distance_curve(
        self,
        generated_video_crops: Tensor,
        natural_audio: Tensor,
        *,
        context: Wav2LipSyncContext,
        vshift: int = 15,
    ) -> Tensor:
        """Return SyncNetInstance's mean distance over the 31 lag columns.

        This reproduces the pinned evaluator's PCM16 MFCC input, selected-track
        audio crop, five-frame visual windows, common-length gate and
        ``calc_pdist`` reduction. Generated video remains a differentiable
        Wav2Lip/crop proxy; it is not equivalent to the actual AVI/JPEG/codec
        path. Fit/dev efficacy still requires official rendered evaluation.
        """

        if torch is None:
            raise RuntimeError("SyncNet scoring requires PyTorch") from _TORCH_IMPORT_ERROR
        context.validate()
        if generated_video_crops.ndim != 4 or tuple(generated_video_crops.shape[1:]) != (3, 224, 224):
            raise ValueError("generated video crops must have shape [T,3,224,224]")
        if int(generated_video_crops.shape[0]) != context.frame_count:
            raise ValueError("generated crop count differs from the frozen source context")
        if not isinstance(vshift, int) or vshift < 1:
            raise ValueError("vshift must be a positive integer")
        frame_count = int(generated_video_crops.shape[0])
        if frame_count <= 5:
            raise ValueError("SyncNet requires at least six source-clock frames")

        audio_crop = _official_track_audio_crop(natural_audio, context)
        if context.syncnet_pcm16_crop is not None:
            pcm16_audio = np.asarray(context.syncnet_pcm16_crop)
            if pcm16_audio.dtype != np.int16 or pcm16_audio.ndim != 1 or pcm16_audio.size == 0:
                raise ValueError("exact SyncNet PCM16 crop is not a nonempty mono int16 vector")
            if abs(int(pcm16_audio.size) - int(audio_crop.numel())) > 640:
                raise ValueError("exact SyncNet PCM16 crop length differs from the frozen track interval")
            mfcc_frames_by_coeff, _ = _syncnet_pcm16_mfcc(
                None, pcm16_crop=pcm16_audio, psf_mfcc=self.psf_mfcc
            )
            audio_sample_count = int(pcm16_audio.size)
        else:
            mfcc_frames_by_coeff, pcm16_audio = _syncnet_pcm16_mfcc(
                audio_crop, psf_mfcc=self.psf_mfcc
            )
            audio_sample_count = int(pcm16_audio.size)
        # SyncNetInstance uses min(len(images), floor(len(decoded_audio)/640))
        # rather than deriving its common clock from MFCC frame count.
        common_length = min(frame_count, math.floor(audio_sample_count / 640))
        window_count = common_length - 5
        if window_count <= 0:
            raise ValueError("official SyncNet common support has fewer than six frames")

        audio_windows = []
        frame_windows = []
        for start in range(window_count):
            mfcc_start = start * 4
            mfcc_stop = mfcc_start + 20
            if mfcc_stop > mfcc_frames_by_coeff.shape[0]:
                raise ValueError("SyncNet MFCC does not cover the official fixed-clock windows")
            audio_windows.append(mfcc_frames_by_coeff[mfcc_start:mfcc_stop].T)
            frame_windows.append(
                generated_video_crops[start : start + 5].permute(1, 0, 2, 3)
            )
        audio_np = np.stack(audio_windows, axis=0).astype(np.float32, copy=False)
        device = generated_video_crops.device
        audio_tensor = torch.from_numpy(audio_np[:, None, :, :]).to(device=device)
        video_tensor = torch.stack(frame_windows, dim=0).contiguous()
        if tuple(audio_tensor.shape[1:]) != (1, 13, 20):
            raise RuntimeError(f"SyncNet audio window shape mismatch: {tuple(audio_tensor.shape)}")
        if tuple(video_tensor.shape[1:]) != (3, 5, 224, 224):
            raise RuntimeError(f"SyncNet video window shape mismatch: {tuple(video_tensor.shape)}")

        audio_embeddings = []
        visual_embeddings = []
        for first in range(0, window_count, self.syncnet_batch_size):
            stop = min(window_count, first + self.syncnet_batch_size)
            with torch.no_grad():
                audio_embedding = self.syncnet_model.forward_aud(audio_tensor[first:stop])
            visual_embedding = self.syncnet_model.forward_lip(video_tensor[first:stop])
            if audio_embedding.shape != visual_embedding.shape:
                raise RuntimeError("SyncNet audio and visual embedding shapes differ")
            if not bool(torch.isfinite(audio_embedding).all().item()) or not bool(torch.isfinite(visual_embedding).all().item()):
                raise FloatingPointError("SyncNet encoder returned non-finite embeddings")
            audio_embeddings.append(audio_embedding)
            visual_embeddings.append(visual_embedding)
        audio_embedding = torch.cat(audio_embeddings, dim=0)
        visual_embedding = torch.cat(visual_embeddings, dim=0)

        # Official calc_pdist pads the *audio embedding sequence* on both
        # sides and computes a distance row for each video query. Columns are
        # ordered from -vshift to +vshift.
        padded_audio = F.pad(audio_embedding, (0, 0, vshift, vshift))
        distance_rows = []
        for index in range(window_count):
            candidates = padded_audio[index : index + 2 * vshift + 1]
            query = visual_embedding[index : index + 1].expand(2 * vshift + 1, -1)
            distance_rows.append(F.pairwise_distance(query, candidates))
        distance_by_lag = torch.stack(distance_rows, dim=1)
        curve = distance_by_lag.mean(dim=1)
        if tuple(curve.shape) != (2 * vshift + 1,) or not bool(torch.isfinite(curve).all().item()):
            raise FloatingPointError("SyncNet lag-distance curve is invalid")
        return curve

    def sync_score(
        self,
        generated_video_crops: Tensor,
        natural_audio: Tensor,
        *,
        context: Wav2LipSyncContext,
        vshift: int = 15,
        anchor_lag_frames: int = 0,
    ) -> Tensor:
        """Return median-minus-zero-lag distance as the differentiable proxy.

        This is the source-clock ``Canchor`` contrast on proxy frames, not an
        official codec-qualified score. It explicitly rewards lower aligned
        distance relative to off-lag distances, rather than uniform distance
        reduction. Final method selection uses the official rendered 31-lag
        curve on development clips.
        """

        if not isinstance(anchor_lag_frames, int) or not -vshift <= anchor_lag_frames <= vshift:
            raise ValueError("anchor_lag_frames must lie within the frozen SyncNet lag range")
        curve = self.sync_distance_curve(
            generated_video_crops,
            natural_audio,
            context=context,
            vshift=vshift,
        )
        return curve.median() - curve[vshift + anchor_lag_frames]


__all__ = ["Wav2LipSyncAdapter", "Wav2LipSyncContext", "prepare_wav2lip_face_input"]
