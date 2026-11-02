#!/usr/bin/env python3
"""Run a frozen V3 waveform candidate through the qualified Stage-A scorer.

This is a source-only method adapter. It loads the exact V3 CLI configuration
and candidate records once, installs the CLI's nine frozen theta values in the
pinned V3 head, and applies that head to each original canonical waveform.
The pinned Stage-A core remains responsible for original-PCM mux-back and
SyncNet scoring; this adapter only extends its audio transform dispatch and
routes MuseTalk/Ditto through the already-qualified silent-video adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
ADAPTER_RUNNER_SHA256 = "545dab985196c3b089e41c30dc1b8e069e7d07cbe0f3c3c648f1fef53325532a"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"
W2L_MODEL_ID = "wav2lip_gan_bac9a81e"
TRANSFORM_METHOD = "v3_parameter_checkpoint"
CLI_CONFIG_SCHEMA = "cross_tfg_v3_enhancement_cli_config_v1"
CLI_CANDIDATE_ROLE = "METHOD_CANDIDATE"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path, expected_sha256: str):
    resolved = path.resolve(strict=True)
    actual = _sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"pinned source SHA mismatch for {resolved}: {actual}")
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned source module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, resolved


class _CandidateTransform:
    """One initialized V3 candidate with the CLI's validation and forward path."""

    def __init__(self, *, cli_path: Path, cli_sha256: str, config_path: Path,
                 config_sha256: str, repo_root: Path):
        self.cli, self.cli_path = _load_module(
            "cross_tfg_v3_candidate_cli_attempt01", cli_path, cli_sha256
        )
        config_path = config_path.resolve(strict=True)
        config_raw = config_path.read_bytes()
        if _sha256_bytes(config_raw) != config_sha256:
            raise RuntimeError("V3 candidate CLI configuration SHA differs from the frozen plan")
        config, validated_raw, pinned = self.cli._validate_config(config_path)
        if validated_raw != config_raw:
            raise RuntimeError("V3 candidate configuration changed during validation")
        if config.get("schema") != CLI_CONFIG_SCHEMA or config.get("profile_role") != CLI_CANDIDATE_ROLE:
            raise RuntimeError("V3 config is not the frozen METHOD_CANDIDATE CLI contract")

        import torch

        self.np = np
        self.torch = torch
        self.config = config
        self.config_path = config_path
        self.config_sha256 = config_sha256
        self.pinned_sources = pinned
        self.imported_sources = self.cli._verify_imported_sources(pinned)
        self.device = self.cli._validate_runtime(config["runtime"], torch, np)
        self.candidate_binding = self.cli._candidate_parameter_values(config, np)
        if self.candidate_binding is None or self.candidate_binding["completed_optimizer_steps"] != 120:
            raise RuntimeError("V3 Dev candidate must be the fixed terminal step-120 checkpoint")

        head_module = __import__(
            "scripts.audio_enhancement.shared_recipient_transition_head_v3",
            fromlist=["SharedRecipientTransitionHeadV3"],
        )
        head_class = getattr(head_module, "SharedRecipientTransitionHeadV3", None)
        if head_class is None:
            raise RuntimeError("pinned V3 implementation has no SharedRecipientTransitionHeadV3")
        self.cli._constructor_fields(head_class, config["constructor"])
        self.head = head_class(**config["constructor"]).to(self.device)
        theta = np.asarray(config["parameter"]["values"], dtype=np.float32)
        if theta.shape != (3, 3) or not np.isfinite(theta).all():
            raise RuntimeError("V3 parameter is not nine finite float32 values")
        with torch.no_grad():
            self.head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device=self.device))
        self.head.eval()

        from scripts.audio_enhancement.baselines import apply_global_rms
        from scripts.audio_enhancement.audio_io import pcm_float32_sha256

        self.apply_global_rms = apply_global_rms
        self.pcm_float32_sha256 = pcm_float32_sha256
        self.theta_values = theta.tolist()
        self.head_sha256 = _sha256_file(Path(head_module.__file__).resolve())
        self.cli_sha256 = _sha256_file(self.cli_path)
        self.repo_root = repo_root.resolve(strict=True)

    def transform(self, samples: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        source = np.ascontiguousarray(samples, dtype=np.float32)
        if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
            raise ValueError("V3 candidate input must be a nonempty finite canonical float32 waveform")
        constructor = self.config["constructor"]
        baseline, baseline_info = self.apply_global_rms(
            source,
            target_rms=float(constructor["target_rms"]),
            peak_ceiling=constructor["peak_ceiling"],
            silence_epsilon=float(constructor["rms_epsilon"]),
        )
        source_tensor = self.torch.from_numpy(source).to(device=self.device, dtype=self.torch.float32)
        baseline_tensor = self.torch.from_numpy(np.ascontiguousarray(baseline)).to(
            device=self.device, dtype=self.torch.float32
        )
        with self.torch.inference_mode():
            result = self.head(source_tensor, baseline_tensor)
        if result.get("schema") != self.config["head_schema"]:
            raise RuntimeError("V3 candidate head result schema changed")
        output_tensor = result.get("output_pcm_float32")
        if not isinstance(output_tensor, self.torch.Tensor) or output_tensor.dtype != self.torch.float32:
            raise RuntimeError("V3 candidate output must be a float32 tensor")
        if output_tensor.ndim != 1 or output_tensor.numel() != source.size:
            raise RuntimeError("V3 candidate changed waveform length")
        if not bool(self.torch.isfinite(output_tensor).all().item()):
            raise RuntimeError("V3 candidate output is non-finite")
        output = np.ascontiguousarray(output_tensor.detach().cpu().numpy(), dtype=np.float32)
        if output.shape != source.shape or not np.isfinite(output).all():
            raise RuntimeError("V3 candidate output violates the canonical F32 contract")
        peak = float(np.max(np.abs(output.astype(np.float64))))
        ceiling = constructor["peak_ceiling"]
        if ceiling is not None and peak > float(ceiling):
            raise RuntimeError("V3 candidate exceeds its declared peak ceiling; output is never clipped")
        return output, {
            "operation": "pinned_v3_shared_recipient_transition_head",
            "v3_cli_path": str(self.cli_path),
            "v3_cli_sha256": self.cli_sha256,
            "v3_cli_config_path": str(self.config_path),
            "v3_cli_config_sha256": self.config_sha256,
            "head_schema": self.config["head_schema"],
            "head_implementation_sha256": self.head_sha256,
            "candidate_checkpoint_sha256": self.candidate_binding["checkpoint_sha256"],
            "candidate_training_record_sha256": self.candidate_binding["training_record_sha256"],
            "candidate_selection_record_sha256": self.candidate_binding["selection_record_sha256"],
            "completed_optimizer_steps": self.candidate_binding["completed_optimizer_steps"],
            "theta_band_contrasts_db": self.theta_values,
            "input_pcm_float32_sha256": self.pcm_float32_sha256(source),
            "output_pcm_float32_sha256": self.pcm_float32_sha256(output),
            "canonical_rms_baseline": baseline_info.__dict__,
            "output_rms": float(np.sqrt(np.mean(output.astype(np.float64) ** 2))),
            "output_peak_abs": peak,
            "peak_ceiling": ceiling,
            "peak_limiter_applied": False,
            "head_normalization": self.cli._json_diagnostic_value(
                {
                    key: value
                    for key, value in result["normalization"].items()
                    if key != "normalization_residual_added_to_canonical_baseline"
                },
                "head_normalization",
            ),
            "positive_flux_valid_transition_count": int(result["positive_flux_valid_transition_count"]),
            "negative_flux_valid_transition_count": int(result["negative_flux_valid_transition_count"]),
            "source_pcm_clock_changed": False,
            "development_or_confirmation_data_opened_by_transform": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", required=True)
    parser.add_argument("--cross-frontend-adapter-runner", required=True)
    parser.add_argument("--v3-cli", required=True)
    parser.add_argument("--expected-v3-cli-sha256", required=True)
    own, remaining = parser.parse_known_args(argv)

    core, core_path = _load_module("cross_tfg_v3_candidate_stagea_core", Path(own.stagea_core), CORE_SHA256)
    adapter_runner, adapter_path = _load_module(
        "cross_tfg_v3_candidate_cross_frontend_adapter",
        Path(own.cross_frontend_adapter_runner),
        ADAPTER_RUNNER_SHA256,
    )
    cli_path = Path(own.v3_cli).resolve(strict=True)
    cli_sha = own.expected_v3_cli_sha256.lower()
    if len(cli_sha) != 64 or _sha256_file(cli_path) != cli_sha:
        raise RuntimeError("V3 CLI entrypoint does not match its frozen SHA")
    stagea_args = core.build_parser().parse_args(remaining)
    project_root = Path(stagea_args.project_root).resolve(strict=True)

    original_transform = core._transform
    original_generator = core._run_generator
    candidate: _CandidateTransform | None = None
    candidate_config_key: tuple[str, str] | None = None

    def transform(samples: np.ndarray, method: Mapping[str, Any]):
        nonlocal candidate, candidate_config_key
        config = method.get("config")
        if not isinstance(config, Mapping) or config.get("method") != TRANSFORM_METHOD:
            return original_transform(samples, method)
        cli_config_value = config.get("v3_cli_config_path")
        cli_config_sha = config.get("v3_cli_config_sha256")
        if not isinstance(cli_config_value, str) or not isinstance(cli_config_sha, str):
            raise RuntimeError("V3 candidate plan method config lacks its CLI config path/SHA")
        cli_config_path = Path(cli_config_value)
        if not cli_config_path.is_absolute():
            cli_config_path = project_root / cli_config_path
        key = (str(cli_config_path.resolve(strict=True)), cli_config_sha)
        try:
            Path(key[0]).relative_to(project_root)
        except ValueError as exc:
            raise RuntimeError("V3 candidate CLI config must remain inside the frozen project tree") from exc
        if candidate_config_key is not None and key != candidate_config_key:
            raise RuntimeError("one Dev run cannot silently mix V3 candidate configurations")
        if candidate is None:
            candidate = _CandidateTransform(
                cli_path=cli_path,
                cli_sha256=cli_sha,
                config_path=Path(key[0]),
                config_sha256=cli_config_sha,
                repo_root=project_root,
            )
            candidate_config_key = key
        return candidate.transform(samples)

    def generator(**kwargs):
        row = kwargs.get("row", {})
        if isinstance(row, Mapping) and row.get("model_id") == W2L_MODEL_ID:
            return adapter_runner._wav2lip_native_generator(core, **kwargs)
        model_generator = kwargs["model"].get("generator", {})
        if isinstance(model_generator, Mapping) and model_generator.get("contract") == ADAPTER_CONTRACT:
            return adapter_runner._adapter_generator(core, **kwargs)
        return original_generator(**kwargs)

    core._transform = transform
    core._run_generator = generator
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(main())
