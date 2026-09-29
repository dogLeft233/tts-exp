from __future__ import annotations

from pathlib import Path

import yaml

from scripts.experiments.mfa_linear_video_retiming.config import load_config


def test_load_config_preserves_model_venv_python_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config_path = repo / "scripts" / "configs" / "mfa.yaml"
    config_path.parent.mkdir(parents=True)
    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base_python.chmod(0o755)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    paths = {
        key: str(tmp_path / key)
        for key in (
            "mfa_summary", "clean_manifest", "mini_manifest", "legacy_eval_dir", "portrait_registry",
            "wav2lip_root", "wav2lip_checkpoint", "syncnet_root", "syncnet_model",
            "ffmpeg", "ffprobe", "render_worker", "legacy_score_worker", "official_score_adapter",
        )
    }
    paths.update({"wav2lip_python": str(venv_python), "syncnet_python": str(venv_python)})
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "protocol": "mfa_linear_video_retiming_v1",
        "sample_ids": ["1", "101", "201"],
        "portraits": ["3", "6", "9"],
        "paths": paths,
        "search": {"max_candidates": 256, "max_seconds_per_sample": 1200.0},
        "budget": {"total_gpu_seconds": 10800},
    }), encoding="utf-8")

    config = load_config(config_path)

    assert config["paths"]["wav2lip_python"] == str(venv_python.absolute())
    assert config["paths"]["syncnet_python"] == str(venv_python.absolute())
    assert Path(config["paths"]["wav2lip_python"]).resolve() == base_python.resolve()
