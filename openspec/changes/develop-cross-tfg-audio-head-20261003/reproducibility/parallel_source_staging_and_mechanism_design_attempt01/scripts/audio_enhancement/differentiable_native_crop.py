"""Differentiable approximation of Wav2Lip's full-frame/native SyncNet crop path.

The hard forward path is defined by the pinned inference/tracker operations:
truncate the Wav2Lip sigmoid patch to uint8, resize with OpenCV into the
static full-resolution face rectangle, paste into the unmodified portrait,
constant-pad each frame with BGR value 110, crop with the saved SyncNet
processed-track rectangle, and resize to 224x224. The tensor path uses
bilinear interpolation with straight-through hard values so gradients pass
from SyncNet crops back through the Wav2Lip patch. It does not emulate DIVX,
XVID, or JPEG encoding; those remain part of official rendered fit/dev scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as F
except ImportError as exc:  # optional fit dependency
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class NativeCropGeometry:
    """Frozen image/track geometry for one center-frame window.

    `source_portrait_bgr_u8` is the original 512x512 OpenCV BGR portrait.
    `face_box_y1_y2_x1_x2` is the pinned Wav2Lip static-image crop rectangle.
    Each `padded_crop_xyxy` is indexed in the image after adding `bsi`
    constant-110 pixels on all sides, matching SyncNet `run_pipeline.py`.
    """

    source_portrait_bgr_u8: np.ndarray
    face_box_y1_y2_x1_x2: tuple[int, int, int, int]
    padded_crop_xyxy: np.ndarray
    padding_bsi: np.ndarray
    frame_indices: tuple[int, ...]
    fps_num: int = 25
    fps_den: int = 1
    border_value_bgr: tuple[int, int, int] = (110, 110, 110)
    crop_size: int = 224

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    def validate(self) -> None:
        if self.source_portrait_bgr_u8.dtype != np.uint8 or self.source_portrait_bgr_u8.shape != (512, 512, 3):
            raise ValueError("native renderer requires the pinned 512x512 uint8 BGR portrait")
        if self.fps_num != 25 or self.fps_den != 1:
            raise ValueError("native renderer source clock must remain 25/1 fps")
        if self.crop_size != 224 or not self.frame_count:
            raise ValueError("native renderer requires nonempty 224x224 crop geometry")
        if self.padded_crop_xyxy.shape != (self.frame_count, 4) or self.padding_bsi.shape != (self.frame_count,):
            raise ValueError("native crop boxes/padding do not match the frame map")
        if any(int(frame) < 0 for frame in self.frame_indices):
            raise ValueError("source-clock frame indices must be nonnegative")
        if self.frame_count > 1 and not np.array_equal(
            np.diff(np.asarray(self.frame_indices, dtype=np.int64)),
            np.ones(self.frame_count - 1, dtype=np.int64),
        ):
            raise ValueError("native crop frame map must preserve a contiguous source clock")
        if not np.issubdtype(self.padded_crop_xyxy.dtype, np.integer) or not np.issubdtype(self.padding_bsi.dtype, np.integer):
            raise ValueError("native crop boxes and padding values must be integer arrays")
        if tuple(self.border_value_bgr) != (110, 110, 110):
            raise ValueError("official SyncNet crop padding is frozen to constant BGR 110")
        y1, y2, x1, x2 = self.face_box_y1_y2_x1_x2
        if not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512):
            raise ValueError("Wav2Lip static face box leaves the 512x512 source portrait")
        for index, (box, bsi) in enumerate(zip(self.padded_crop_xyxy, self.padding_bsi)):
            x0, cy0, x1p, y1p = [int(v) for v in box]
            bsi = int(bsi)
            padded_size = 512 + 2 * bsi
            if bsi < 0 or not (0 <= x0 < x1p <= padded_size and 0 <= cy0 < y1p <= padded_size):
                raise ValueError(f"native padded crop rectangle is invalid at frame {index}: {box.tolist()}, bsi={bsi}")
            if (x1p - x0) <= 0 or (y1p - cy0) <= 0:
                raise ValueError(f"native crop rectangle is empty at frame {index}")


def official_track_crop_geometry(proc_track: dict[str, Sequence[float]], frame_index: int,
                                 *, crop_scale: float = 0.40) -> tuple[np.ndarray, int]:
    """Return the exact run_pipeline padded-image crop rectangle and `bsi`.

    `proc_track` stores `x` as the face center x and `y` as face center y,
    as assigned in the pinned `crop_video`; `s` is the half-size after median
    smoothing. The crop bounds are half-open indices in the padded image.
    """
    bs = float(proc_track["s"][frame_index])
    cx = float(proc_track["x"][frame_index])
    cy = float(proc_track["y"][frame_index])
    if not np.isfinite([bs, cx, cy]).all() or bs <= 0 or crop_scale != 0.40:
        raise ValueError("invalid/floating crop scale or processed-track geometry")
    bsi = int(bs * (1.0 + 2.0 * crop_scale))
    x0 = int(cx + bsi - bs * (1.0 + crop_scale))
    x1 = int(cx + bsi + bs * (1.0 + crop_scale))
    y0 = int(cy + bsi - bs)
    y1 = int(cy + bsi + bs * (1.0 + 2.0 * crop_scale))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("processed track produced an empty native crop")
    return np.asarray([x0, y0, x1, y1], dtype=np.int32), bsi


def build_native_crop_geometry(
    source_portrait_bgr_u8: np.ndarray,
    face_box_y1_y2_x1_x2: Sequence[int],
    proc_track: dict[str, Sequence[float]],
    track_local_indices: Sequence[int],
    source_clock_frames: Sequence[int],
    *,
    crop_scale: float = 0.40,
) -> NativeCropGeometry:
    """Bind per-frame tracked crops to an explicit, contiguous 25-fps map.

    `track_local_indices` address the frozen selected track's processed
    geometry arrays; `source_clock_frames` are their original rendered
    25-fps frame indices. Both mappings are caller-supplied and checked, so
    evaluator-local crop indices cannot silently replace source-clock time.
    """
    local = np.asarray(track_local_indices, dtype=np.int64)
    source = np.asarray(source_clock_frames, dtype=np.int64)
    if local.ndim != 1 or source.ndim != 1 or local.size == 0 or local.shape != source.shape:
        raise ValueError("track-local/source-clock frame maps must be equally sized nonempty vectors")
    if np.any(local < 0) or np.any(source < 0):
        raise ValueError("track-local and source-clock frame indices must be nonnegative")
    if local.size > 1 and (not np.array_equal(np.diff(local), np.ones(local.size - 1, dtype=np.int64))
                           or not np.array_equal(np.diff(source), np.ones(source.size - 1, dtype=np.int64))):
        raise ValueError("fit crop window must retain contiguous track-local and source-clock frames")
    crop_boxes: list[np.ndarray] = []
    padding: list[int] = []
    for index in local.tolist():
        box, bsi = official_track_crop_geometry(proc_track, index, crop_scale=crop_scale)
        crop_boxes.append(box)
        padding.append(bsi)
    face_box = tuple(int(value) for value in face_box_y1_y2_x1_x2)
    if len(face_box) != 4:
        raise ValueError("static Wav2Lip face box must contain y1,y2,x1,x2")
    geometry = NativeCropGeometry(
        source_portrait_bgr_u8=np.asarray(source_portrait_bgr_u8),
        face_box_y1_y2_x1_x2=face_box,
        padded_crop_xyxy=np.stack(crop_boxes).astype(np.int32, copy=False),
        padding_bsi=np.asarray(padding, dtype=np.int32),
        frame_indices=tuple(int(value) for value in source.tolist()),
    )
    geometry.validate()
    return geometry


def crop_official_numpy(frame_bgr_u8: np.ndarray, padded_crop_xyxy: Sequence[int], bsi: int,
                        *, crop_size: int = 224) -> np.ndarray:
    """Run exact official constant-pad/crop/OpenCV-resize hard operations."""
    frame = np.asarray(frame_bgr_u8)
    if frame.dtype != np.uint8 or frame.shape != (512, 512, 3):
        raise ValueError("official crop source must be a 512x512 uint8 BGR frame")
    bsi = int(bsi)
    if bsi < 0:
        raise ValueError("padding size cannot be negative")
    padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=110)
    x0, y0, x1, y1 = [int(v) for v in padded_crop_xyxy]
    ph, pw = padded.shape[:2]
    if not (0 <= x0 < x1 <= pw and 0 <= y0 < y1 <= ph):
        raise ValueError(f"official crop rectangle outside padded frame: {(x0, y0, x1, y1)} vs {(pw, ph)}")
    crop = np.ascontiguousarray(padded[y0:y1, x0:x1])
    resized = cv2.resize(crop, (int(crop_size), int(crop_size)))
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _as_torch_bgr(frame_chw: Tensor, device) -> Tensor:
    source = torch.as_tensor(frame_chw, device=device, dtype=torch.float32)
    if tuple(source.shape) != (3, 512, 512) or not torch.isfinite(source).all().item():
        raise ValueError("each differentiable full frame must be finite [3,512,512] BGR")
    if source.min().item() < 0 or source.max().item() > 255:
        raise ValueError("full-frame BGR pixels must lie in [0,255]")
    return source


def _straight_through(hard: Tensor, soft: Tensor) -> Tensor:
    return soft + (hard - soft).detach()


def render_native_syncnet_crops(
    wav2lip_predictions_01: Tensor,
    geometry: NativeCropGeometry,
    *,
    source_frames_bgr_u8: np.ndarray | None = None,
    return_full_frames: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Paste Wav2Lip predictions on 512px frames, then emulate tracked crops.

    Forward values use the exact official uint8/OpenCV/padding/crop operators
    (before video/JPEG codecs). Gradients use bilinear Torch interpolation
    through both the Wav2Lip-to-full-frame resize and full-frame-to-224 crop.
    The only spatial inputs that vary by frame are the frozen official face
    rectangle and selected SyncNet tracker geometry; frame clocks are never
    shifted or resampled.
    """
    if torch is None:
        raise RuntimeError("native differentiable crop rendering requires PyTorch") from _TORCH_IMPORT_ERROR
    geometry.validate()
    if wav2lip_predictions_01.ndim != 4 or tuple(wav2lip_predictions_01.shape[1:]) != (3, 96, 96):
        raise ValueError("Wav2Lip predictions must have shape [T,3,96,96]")
    if int(wav2lip_predictions_01.shape[0]) != geometry.frame_count:
        raise ValueError("prediction count differs from the frozen native frame map")
    if not torch.isfinite(wav2lip_predictions_01).all().item():
        raise FloatingPointError("Wav2Lip predictions contain NaN or infinity")
    if wav2lip_predictions_01.min().item() < 0 or wav2lip_predictions_01.max().item() > 1:
        raise ValueError("Wav2Lip sigmoid predictions must lie in [0,1]")

    device = wav2lip_predictions_01.device
    source_frames = (np.repeat(geometry.source_portrait_bgr_u8[None, ...], geometry.frame_count, axis=0)
                     if source_frames_bgr_u8 is None else np.asarray(source_frames_bgr_u8))
    if source_frames.dtype != np.uint8 or source_frames.shape != (geometry.frame_count, 512, 512, 3):
        raise ValueError("source BGR frame array must have shape [T,512,512,3] uint8")
    y1, y2, x1, x2 = [int(v) for v in geometry.face_box_y1_y2_x1_x2]
    target_size = (y2 - y1, x2 - x1)

    output_crops: list[Tensor] = []
    output_full: list[Tensor] = []
    for index in range(geometry.frame_count):
        prediction = wav2lip_predictions_01[index]
        prediction_255 = prediction * 255.0
        # Official inference casts the 96px prediction to uint8 before the
        # OpenCV resize. Use this exact hard quantization in the forward pass.
        patch_u8_chw = torch.floor(prediction_255.detach().clamp(0.0, 255.0)).to(torch.uint8)
        patch_u8_hwc = np.ascontiguousarray(patch_u8_chw.permute(1, 2, 0).cpu().numpy())
        hard_resized_hwc = cv2.resize(patch_u8_hwc, (target_size[1], target_size[0]))
        hard_resized = torch.as_tensor(
            np.ascontiguousarray(hard_resized_hwc.transpose(2, 0, 1)), device=device, dtype=torch.float32
        )
        soft_resized = F.interpolate(
            prediction_255[None], size=target_size, mode="bilinear", align_corners=False
        )[0]
        patch_resized = _straight_through(hard_resized, soft_resized)

        base = _as_torch_bgr(source_frames[index].transpose(2, 0, 1), device)
        full_soft = base.clone()
        full_soft[:, y1:y2, x1:x2] = patch_resized
        hard_full_hwc = np.array(source_frames[index], copy=True)
        hard_full_hwc[y1:y2, x1:x2] = hard_resized_hwc
        hard_full = torch.as_tensor(hard_full_hwc.transpose(2, 0, 1).copy(), device=device, dtype=torch.float32)
        full = _straight_through(hard_full, full_soft)

        bsi = int(geometry.padding_bsi[index])
        x0p, y0p, x1p, y1p = [int(v) for v in geometry.padded_crop_xyxy[index]]
        padded = F.pad(full[None], (bsi, bsi, bsi, bsi), mode="constant", value=110.0)[0]
        soft_source_crop = padded[:, y0p:y1p, x0p:x1p]
        if soft_source_crop.shape[1] <= 0 or soft_source_crop.shape[2] <= 0:
            raise RuntimeError(f"native source crop is empty at frame {index}")
        soft_224 = F.interpolate(
            soft_source_crop[None], size=(geometry.crop_size, geometry.crop_size),
            mode="bilinear", align_corners=False,
        )[0]
        hard_224_np = crop_official_numpy(hard_full_hwc, geometry.padded_crop_xyxy[index], bsi,
                                         crop_size=geometry.crop_size)
        hard_224 = torch.as_tensor(hard_224_np.transpose(2, 0, 1).copy(), device=device, dtype=torch.float32)
        output_crops.append(_straight_through(hard_224, soft_224))
        if return_full_frames:
            output_full.append(full)

    crops = torch.stack(output_crops, dim=0)
    expected = (geometry.frame_count, 3, geometry.crop_size, geometry.crop_size)
    if tuple(crops.shape) != expected or not torch.isfinite(crops).all().item():
        raise RuntimeError("native SyncNet crop output shape/finite check failed")
    if return_full_frames:
        return crops, torch.stack(output_full, dim=0)
    return crops


__all__ = [
    "NativeCropGeometry",
    "official_track_crop_geometry",
    "build_native_crop_geometry",
    "crop_official_numpy",
    "render_native_syncnet_crops",
]
