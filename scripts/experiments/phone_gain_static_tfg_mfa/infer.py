"""Pure-forward inference for A/B/C; no teacher, optimizer, or backward path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.experiments.phone_separability_enhancement.audio import BandGainRenderer

from .conditioning import Conditioning, PhoneVocabulary
from .config import state_dict_hash
from .audio_contract import export_safe_pcm
from .model import build_model
from .train import _normalize_features, load_checkpoint


def infer_pcm(source_pcm: np.ndarray, protected_mask: np.ndarray, conditioning: Conditioning, checkpoint: str | Path, *, vocabulary: PhoneVocabulary, device: str = "cpu", audio: Mapping[str, Any] | None = None, network: Mapping[str, Any] | None = None, model: Any | None = None, feature_stats: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    audio_cfg = dict(audio or {})
    network_cfg = dict(network or {})
    source = np.asarray(source_pcm, dtype=np.int16).reshape(-1)
    protected = np.asarray(protected_mask, dtype=bool).reshape(-1)
    if source.size != protected.size or not np.isfinite(source.astype(np.float64)).all():
        raise ValueError("source PCM/protected mask mismatch")
    conditioning.validate(samples=source.size)
    if not np.array_equal(conditioning.protected_mask.astype(bool), protected):
        raise ValueError("conditioning protected mask differs from inference protected mask")
    if str(conditioning.vocabulary_hash) != str(vocabulary.hash):
        raise ValueError("conditioning vocabulary hash differs from inference vocabulary")
    renderer = BandGainRenderer(sample_rate=int(audio_cfg.get("sample_rate", 16000)), n_fft=int(audio_cfg.get("n_fft", 512)), win_length=int(audio_cfg.get("win_length", 512)), hop_length=int(audio_cfg.get("hop_length", 128)), n_bands=int(audio_cfg.get("n_bands", 24)), max_gain_db=float(audio_cfg.get("max_gain_db", 6.0)))
    if model is None:
        model = build_model(vocab_size=len(vocabulary.labels), embedding_dim=int(network_cfg.get("embedding_dim", 16)), audio_channels=int(audio_cfg.get("n_bands", 24)), hidden_channels=int(network_cfg.get("hidden_channels", 64)), max_gain_db=float(audio_cfg.get("max_gain_db", 6.0)), dilations=tuple(int(value) for value in network_cfg.get("residual_dilations", [1, 2, 4]))).to(device)
        with torch.no_grad():
            model.embedding.weight[vocabulary.sil_id].zero_()
            model.embedding.weight[vocabulary.unk_id].zero_()
        payload = load_checkpoint(model, checkpoint, device=device)
        if int(payload.get("schema_version", 0)) < 3 or str(payload.get("mode")) != str(conditioning.mode):
            raise ValueError("checkpoint does not satisfy the repaired training contract")
        with torch.no_grad():
            model.embedding.weight[vocabulary.sil_id].zero_()
            model.embedding.weight[vocabulary.unk_id].zero_()
    model.eval()
    before = state_dict_hash(model.state_dict())
    x = torch.as_tensor(source.astype(np.float32) / 32768.0, device=device)
    with torch.inference_mode():
        features, _ = renderer.band_features(x)
        features = _normalize_features(features, feature_stats)
        if conditioning.phone_ids.size != int(features.shape[-1]):
            raise ValueError("conditioning frame count does not match STFT")
        ids = torch.as_tensor(conditioning.phone_ids, dtype=torch.long, device=device).unsqueeze(0)
        timing = torch.as_tensor(conditioning.timing, dtype=features.dtype, device=device).unsqueeze(0)
        gain = model(features, ids, timing, mode=conditioning.mode)
        edit_mask = conditioning.edit_mask if conditioning.edit_mask is not None else (~protected).astype(np.float32)
        rendered, render_meta = renderer.render(x, gain, torch.as_tensor(edit_mask, dtype=features.dtype, device=device))
        output, pcm_meta = export_safe_pcm(
            rendered,
            source,
            protected,
            max_residual_ratio=float(audio_cfg.get("max_residual_energy_ratio", 0.01)),
            min_snr_db=float(audio_cfg.get("min_snr_db", 20.0)),
            max_rms_change_db=float(audio_cfg.get("max_rms_change_db", 1.0)),
        )
    after = state_dict_hash(model.state_dict())
    if before != after:
        raise RuntimeError("model state changed during inference")
    metadata = {"schema_version": 1, "mode": conditioning.mode, "checkpoint": str(Path(checkpoint).resolve()), "model_hash_before": before, "model_hash_after": after, "pcm_meta": pcm_meta, "render": render_meta, "pure_forward": True, "teacher_used": False, "optimizer_used": False, "backward_used": False}
    return output, metadata


__all__ = ["infer_pcm"]
