import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/ops/deploy_qwen3_tts_1p7b_20261023.py'


def test_import_does_not_load_gpu_or_start_deployment():
    code = ('import runpy,sys; runpy.run_path(sys.argv[1],run_name="import_check"); '
            'assert "torch" not in sys.modules; assert "requests" not in sys.modules')
    subprocess.run([sys.executable, '-c', code, str(SCRIPT)], check=True)


def test_corrupt_installed_file_is_rejected_without_downloading(tmp_path):
    spec = importlib.util.spec_from_file_location('deploy_qwen17', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / 'runs/qwen3_tts_1p7b_deployment_20261023_attempt01'
    out.mkdir(parents=True)
    model = tmp_path / 'models/Qwen3-TTS-12Hz-1.7B-Base'
    model.mkdir(parents=True)
    (model / 'config.json').write_bytes(b'corrupted')
    (out / 'modelscope_listing.json').write_text(json.dumps({'Data': {'Files': [
        dict(Type='blob', Path='config.json', Sha256='0' * 64, Size=9, Revision='pinned')
    ]}}))
    with patch.object(sys, 'argv', [str(SCRIPT), 'download', '--root', str(tmp_path)]), \
         patch('requests.get', side_effect=AssertionError('network must not be called')):
        with pytest.raises(RuntimeError, match='installed file mismatch'):
            module.main()
    assert not (out / 'weights_manifest.json').exists()
