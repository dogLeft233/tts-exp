"""Selection and range-read safeguards for the HDTF acquisition utility."""
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "hdtf_subset", Path(__file__).parents[1] / "scripts/acquire_hdtf_diagnostic_subset.py"
)
subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subset)


def test_identity_is_shared_across_categories_and_takes():
    names = ["videos/WDA_Alice0_000.mp4", "videos/WRA_Alice2_001.mp4",
             "videos/WRA_Bob_000.mp4", "videos/RD_Radio1_000.mp4"]
    entries = [zipfile.ZipInfo(name) for name in names]
    selected = subset.select_candidates(entries)
    assert {subset.identity(x.filename) for x in selected} == {"Alice", "Bob"}
    assert len(selected) == 2
    assert [x.filename for x in selected] == [x.filename for x in subset.select_candidates(entries)]


def test_range_file_can_read_zip_and_reject_bad_content_range(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/example.mp4", b"example media" * 100)
    data = buf.getvalue()

    class Response(io.BytesIO):
        status = 206

    def open_range(request, **kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        response = Response(data[start:end + 1])
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        return response

    monkeypatch.setattr(subset.urllib.request, "urlopen", open_range)
    remote = subset.RemoteFile("https://example.org/data.zip", len(data))
    with zipfile.ZipFile(remote) as archive:
        assert archive.read("videos/example.mp4") == b"example media" * 100

    def wrong_range(request, **kwargs):
        response = Response(b"incorrect")
        response.headers = {"Content-Range": "bytes 0-8/9"}
        return response

    monkeypatch.setattr(subset.urllib.request, "urlopen", wrong_range)
    monkeypatch.setattr(subset.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="exact HTTP range"):
        subset.RemoteFile("https://example.org/data.zip", len(data)).read(4)
