"""A partial corpus cannot pass the independent synthesis audit."""
import importlib.util
import json
from pathlib import Path
import tempfile

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/experiments/cloud_av_preference_synthesis_audit_20261027.py'
spec = importlib.util.spec_from_file_location('cloud_audit', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_draw_fails_before_audio_processing():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'protocol.json').write_text(json.dumps(dict(samples=[dict(sample_id=str(i)) for i in range(20)])))
        rows = [dict(sample_id=str(i), draw_index=d) for i in range(20) for d in [1, 2, 3]]
        rows.pop()
        (root / 'audio_manifest.json').write_text(json.dumps(dict(records=rows)))
        try:
            module.audit(root)
        except AssertionError:
            pass
        else:
            raise AssertionError('An incomplete59-draw corpus was accepted')
        assert not (root / 'synthesis_independent_audit.json').exists()
