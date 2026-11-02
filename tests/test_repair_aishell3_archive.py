import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('repair', Path(__file__).parents[1] / 'scripts/repair_aishell3_archive.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_range_requires_status_and_content_range(monkeypatch):
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **kw: Response())
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='exact range'):
        module.fetch_range(0, 3)


def test_range_rejects_short_response(monkeypatch):
    class Response:
        status = 206
        headers = {'Content-Range': f'bytes 0-3/{module.TOTAL}'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return b'ab'
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *a, **kw: Response())
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='size mismatch'):
        module.fetch_range(0, 3)
