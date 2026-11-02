#!/usr/bin/env python3
"""Capture official SyncNet forward vectors without changing its score path.

This wrapper temporarily observes ``forward_lip`` and ``forward_aud`` during
``SyncNetInstance.evaluate``. It returns the evaluator's original result
unchanged and copies the unnormalized float32 head outputs to CPU arrays.
The caller supplies the source-clock query-frame map produced by the same
frozen video/crop mapping used for the distance matrix.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class EmbeddingCaptureError(RuntimeError):
    pass


def evaluate_with_forward_capture(
    syncnet_instance: Any,
    opt: Any,
    videofile: str,
    query_frame_index: np.ndarray,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Run the pinned official evaluator and capture its pre-distance vectors.

    ``query_frame_index`` must map the evaluator's crop-local row order onto
    the frozen source-clock frame indices. This function checks its length
    against the evaluator's returned ``N x 31`` matrix and does not infer or
    shift the map.
    """
    model = getattr(syncnet_instance, "__S__", None)
    if model is None:
        raise EmbeddingCaptureError("SyncNetInstance has no __S__ model")

    original_lip = getattr(model, "forward_lip", None)
    original_aud = getattr(model, "forward_aud", None)
    if not callable(original_lip) or not callable(original_aud):
        raise EmbeddingCaptureError("SyncNet model lacks official forward_lip/forward_aud methods")
    had_instance_lip = "forward_lip" in getattr(model, "__dict__", {})
    had_instance_aud = "forward_aud" in getattr(model, "__dict__", {})
    previous_instance_lip = getattr(model, "__dict__", {}).get("forward_lip")
    previous_instance_aud = getattr(model, "__dict__", {}).get("forward_aud")

    captured_lip: list[np.ndarray] = []
    captured_aud: list[np.ndarray] = []

    def observing_forward(
        original: Callable[..., Any], destination: list[np.ndarray], label: str
    ) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            if not hasattr(output, "detach") or not hasattr(output, "dtype"):
                raise EmbeddingCaptureError(f"{label} returned a non-tensor output")
            if str(output.dtype) != "torch.float32":
                raise EmbeddingCaptureError(f"{label} output dtype is {output.dtype}, expected torch.float32")
            if output.ndim != 2 or output.shape[1] != 1024:
                raise EmbeddingCaptureError(f"{label} output shape is {tuple(output.shape)}, expected (N, 1024)")
            array = output.detach().to(device="cpu").contiguous().numpy()
            if array.dtype != np.dtype("float32") or not np.isfinite(array).all():
                raise EmbeddingCaptureError(f"{label} output is not finite float32")
            destination.append(np.array(array, dtype="<f4", order="C", copy=True))
            return output

        return call

    setattr(model, "forward_lip", observing_forward(original_lip, captured_lip, "forward_lip"))
    setattr(model, "forward_aud", observing_forward(original_aud, captured_aud, "forward_aud"))
    try:
        official_result = syncnet_instance.evaluate(opt, videofile)
    finally:
        if had_instance_lip:
            setattr(model, "forward_lip", previous_instance_lip)
        else:
            delattr(model, "forward_lip")
        if had_instance_aud:
            setattr(model, "forward_aud", previous_instance_aud)
        else:
            delattr(model, "forward_aud")

    if not isinstance(official_result, tuple) or len(official_result) != 3:
        raise EmbeddingCaptureError("official evaluate result must be (offset, confidence, distance_matrix)")
    distance_matrix = np.asarray(official_result[2])
    if distance_matrix.ndim != 2 or distance_matrix.shape[1] != 31:
        raise EmbeddingCaptureError(f"official distance matrix shape is {distance_matrix.shape}, expected (N, 31)")
    if not captured_lip or not captured_aud:
        raise EmbeddingCaptureError("official evaluate did not call both forward methods")

    video = np.concatenate(captured_lip, axis=0)
    audio = np.concatenate(captured_aud, axis=0)
    query_frames = np.asarray(query_frame_index)
    if query_frames.dtype != np.dtype("int32") or query_frames.shape != (distance_matrix.shape[0],):
        raise EmbeddingCaptureError(
            "query_frame_index must be int32 with one source-clock frame per official distance row"
        )
    if video.shape != audio.shape or video.shape[0] != distance_matrix.shape[0]:
        raise EmbeddingCaptureError(
            "captured forward-vector row counts do not match each other and the official distance matrix"
        )

    arrays = {
        "video_embedding_forward_float32": np.ascontiguousarray(video, dtype="<f4"),
        "audio_embedding_forward_float32": np.ascontiguousarray(audio, dtype="<f4"),
        "query_frame_index_int32": np.ascontiguousarray(query_frames, dtype="<i4"),
    }
    return official_result, arrays
