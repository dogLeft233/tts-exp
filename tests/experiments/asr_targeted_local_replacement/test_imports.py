from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("module_name", ("config", "protocol", "patch_audio", "evaluate", "analyze", "run"))
def test_modules_import_without_media_or_model(module_name: str):
    module = importlib.import_module(f"scripts.experiments.asr_targeted_local_replacement.{module_name}")
    assert module is not None
