"""Tests for scripts/38_prepare_frames.py (quality-filtered master frames)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "_prepare_frames",
    _SCRIPTS / "38_prepare_frames.py",
)
assert _spec is not None
pf = importlib.util.module_from_spec(_spec)
sys.modules["_prepare_frames"] = pf
assert _spec.loader is not None
_spec.loader.exec_module(pf)

_fq_spec = importlib.util.spec_from_file_location(
    "_frame_quality", _SCRIPTS / "frame_quality.py"
)
assert _fq_spec is not None
fq = importlib.util.module_from_spec(_fq_spec)
sys.modules["_frame_quality"] = fq
assert _fq_spec.loader is not None
_fq_spec.loader.exec_module(fq)


def make_video(path: Path, n_frames: int = 30, width: int = 640, height: int = 360):
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (width, height),
    )
    for i in range(n_frames):
        frame = np.full((height, width, 3), 40 + i % 200, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def stub_metrics(frames_table):
    """extract_metrics callable mapping frame index -> FaceMetrics | None."""

    def extract(frame):
        return frames_table.get(frame)

    return extract


def write_video_manifest(tmp_path: Path, video: Path, stem: str = "S1") -> Path:
    manifest = tmp_path / "video_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "stem": stem,
                        "speaker_key": stem,
                        "video_local_path": str(video),
                        "duration_s": 1.2,
                    }
                ]
            }
        )
    )
    return manifest


def test_prepare_frames_writes_original_resolution_and_manifest(tmp_path, monkeypatch):
    from PIL import Image

    video = make_video(tmp_path / "s1.mp4")
    manifest = write_video_manifest(tmp_path, video)

    def extract(frame):
        gray = cv2_cvt(frame)
        return fq.FaceMetrics(
            yaw_deg=0.0,
            face_area_px=gray.size,
            blur_score=fq.compute_blur_score(gray),
        )

    monkeypatch.setattr(pf, "extract_metrics_for_frame", extract)

    result = pf.prepare_frames(
        repo=tmp_path,
        run_id="run1",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    out_png = tmp_path / "runs" / "run1" / "00_frames" / "1.png"
    assert out_png.exists()
    with Image.open(out_png) as img:
        assert img.size == (640, 360)

    frame_manifest = json.loads(
        (tmp_path / "runs" / "run1" / "00_frames" / "frame_manifest.json").read_text()
    )
    record = frame_manifest["records"][0]
    assert record["sample_id"] == 1
    assert record["stem"] == "S1"
    assert record["detector"] == "stub"
    assert record["source_resolution"] == [640, 360]
    assert record["output_resolution"] == [640, 360]
    assert set(record) >= {
        "sha256",
        "frame_idx",
        "timestamp_s",
        "yaw_deg",
        "face_area_px",
        "blur_score",
    }


def cv2_cvt(frame):
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def test_prepare_frames_rejects_sample_when_no_acceptable_frame(tmp_path, monkeypatch):
    video = make_video(tmp_path / "s2.mp4")
    manifest = write_video_manifest(tmp_path, video, stem="S2")

    monkeypatch.setattr(pf, "extract_metrics_for_frame", lambda frame: None)

    result = pf.prepare_frames(
        repo=tmp_path,
        run_id="run2",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    assert result["samples_ok"] == 0
    assert len(result["rejected_candidates"]) == 1
    assert "no acceptable frame" in result["rejected_candidates"][0]["error"]


def test_prepare_frames_is_deterministic(tmp_path, monkeypatch):
    video = make_video(tmp_path / "s3.mp4")
    manifest = write_video_manifest(tmp_path, video, stem="S3")

    def extract(frame):
        gray = cv2_cvt(frame)
        return fq.FaceMetrics(
            yaw_deg=0.0,
            face_area_px=gray.size,
            blur_score=fq.compute_blur_score(gray),
        )

    monkeypatch.setattr(pf, "extract_metrics_for_frame", extract)

    r1 = pf.prepare_frames(
        repo=tmp_path,
        run_id="run3a",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    r2 = pf.prepare_frames(
        repo=tmp_path,
        run_id="run3b",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    assert r1["records"][0]["frame_idx"] == r2["records"][0]["frame_idx"]
    assert r1["records"][0]["sha256"] == r2["records"][0]["sha256"]


def test_prepare_frames_reuses_cached_frame_on_rerun(tmp_path, monkeypatch):
    video = make_video(tmp_path / "s4.mp4")
    manifest = write_video_manifest(tmp_path, video, stem="S4")

    def extract(frame):
        gray = cv2_cvt(frame)
        return fq.FaceMetrics(
            yaw_deg=0.0,
            face_area_px=gray.size,
            blur_score=fq.compute_blur_score(gray),
        )

    monkeypatch.setattr(pf, "extract_metrics_for_frame", extract)

    first = pf.prepare_frames(
        repo=tmp_path,
        run_id="run4",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    second = pf.prepare_frames(
        repo=tmp_path,
        run_id="run4",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    assert first["records"][0]["cached"] is False
    assert second["records"][0]["cached"] is True
    assert second["records"][0]["sha256"] == first["records"][0]["sha256"]


def test_first_frame_strategy_prefers_earliest_acceptable_frame():
    table = {
        0: None,  # frame 0: no face
        1: fq.FaceMetrics(yaw_deg=5.0, face_area_px=60000.0, blur_score=50.0),
        2: fq.FaceMetrics(yaw_deg=0.0, face_area_px=90000.0, blur_score=200.0),
    }
    candidates = [(i, i * 0.04, i) for i in range(3)]
    result = pf.first_acceptable_frame(candidates, lambda f: table[f])
    assert result == (1, table[1])


def test_first_frame_strategy_falls_back_when_first_unacceptable():
    table = {
        0: fq.FaceMetrics(yaw_deg=45.0, face_area_px=60000.0, blur_score=50.0),
        1: fq.FaceMetrics(yaw_deg=2.0, face_area_px=80000.0, blur_score=150.0),
        2: fq.FaceMetrics(yaw_deg=0.0, face_area_px=90000.0, blur_score=200.0),
    }
    candidates = [(i, i * 0.04, i) for i in range(3)]
    result = pf.first_acceptable_frame(candidates, lambda f: table[f])
    assert result == (1, table[1])


def test_prepare_frames_uses_first_strategy_by_default(tmp_path, monkeypatch):
    """CLI default picks frame near video start when it passes the gate."""
    video = make_video(tmp_path / "s5.mp4", n_frames=40)
    manifest = write_video_manifest(tmp_path, video, stem="S5")

    def extract(frame):
        gray = cv2_cvt(frame)
        return fq.FaceMetrics(
            yaw_deg=0.0,
            face_area_px=gray.size,
            blur_score=fq.compute_blur_score(gray),
        )

    monkeypatch.setattr(pf, "extract_metrics_for_frame", extract)

    result = pf.prepare_frames(
        repo=tmp_path,
        run_id="run5",
        video_manifest_path=manifest,
        count=1,
        detector_name="stub",
    )
    assert result["records"][0]["frame_idx"] == 0
