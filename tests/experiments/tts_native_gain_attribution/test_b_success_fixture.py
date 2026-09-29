from __future__ import annotations

import hashlib
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scripts.experiments.tts_native_gain_attribution import config, generation, syncnet
from scripts.experiments.tts_native_gain_attribution.common import (
    file_sha256,
    write_self_hashed_json,
)


def _write_wav(path: Path, phase: int) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (1200 * np.sin(np.linspace(0.0, 20.0 * np.pi, config.SAMPLE_RATE * 3) + phase)).astype("<i2")
    payload = samples.tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(payload)
    return file_sha256(path), hashlib.sha256(payload).hexdigest()


def test_crossed_score_cpu_success_fixture(monkeypatch, tmp_path: Path) -> None:
    """Exercise the complete B scorer without loading CUDA or SyncNet weights."""

    monkeypatch.setattr(config, "SAMPLE_IDS", (151,))
    monkeypatch.setattr(config, "CONTROL_IDS", (151,))
    monkeypatch.setattr(config, "EXPECTED_B_VIDEOS", 12)
    monkeypatch.setattr(config, "EXPECTED_B_REPEATS", 2)
    monkeypatch.setattr(config, "EXPECTED_B_SCIENCE", 28)
    monkeypatch.setattr(config, "EXPECTED_B_CONTROLS", 4)
    monkeypatch.setattr(config, "MIN_INTERIOR_ROWS", 25)

    paths = config.RunPaths(tmp_path / "run")
    audio_records: list[dict[str, object]] = []
    for source in ("N", "T"):
        for condition in ("A0", "NOISE", "DENOISE"):
            container_hash, pcm_hash = _write_wav(paths.root / "audio" / f"{source}_{condition}.wav", len(audio_records))
            audio_records.append(
                {
                    "sample_id": 151,
                    "source": source,
                    "condition": condition,
                    "path": str(paths.root / "audio" / f"{source}_{condition}.wav"),
                    "file_sha256": container_hash,
                    "pcm_sha256": pcm_hash,
                }
            )
    audio_manifest = {"records": audio_records}
    assets = {"records": [{"sample_id": 151, "source_group": "fixture-group-151"}]}

    videos: list[dict[str, object]] = []
    for source in ("N", "T"):
        for driver in config.B_DRIVERS:
            for seed in config.SEEDS:
                audio = next(item for item in audio_records if item["source"] == source and item["condition"] == driver)
                cell_id = f"BGEN:151:{source}:{driver}:seed{seed}:main"
                videos.append(
                    {
                        "stage": "B_GENERATION",
                        "id": 151,
                        "source": source,
                        "source_group": "fixture-group-151",
                        "driver_condition": driver,
                        "seed": seed,
                        "role": "main",
                        "cell_id": cell_id,
                        "path": str(paths.root / "videos" / f"{cell_id}.mp4"),
                        "video_hash": f"video-{cell_id}",
                        "model_hash": "fixture-model",
                        "pcm_path": audio["path"],
                        "pcm_hash": audio["pcm_sha256"],
                        "status": "COMPLETE",
                    }
                )
    for source in ("N", "T"):
        audio = next(item for item in audio_records if item["source"] == source and item["condition"] == "A0")
        cell_id = f"BGEN:151:{source}:A0:seed42:repeat1"
        videos.append(
            {
                "stage": "B_GENERATION_REPEAT",
                "id": 151,
                "source": source,
                "source_group": "fixture-group-151",
                "driver_condition": "A0",
                "seed": 42,
                "role": "repeat1",
                "cell_id": cell_id,
                "path": str(paths.root / "videos" / f"{cell_id}.mp4"),
                "video_hash": f"video-{cell_id}",
                "model_hash": "fixture-model",
                "pcm_path": audio["path"],
                "pcm_hash": audio["pcm_sha256"],
                "status": "COMPLETE",
            }
        )
    generation_manifest = {"status": "COMPLETE", "videos": videos}

    def fake_prepare(_paths: config.RunPaths, rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            cell_id = str(row["cell_id"])
            crop = paths.crossed / "fixture_crops" / f"{cell_id}.avi"
            crop.parent.mkdir(parents=True, exist_ok=True)
            crop.write_bytes(cell_id.encode("utf-8"))
            roi_path = paths.crossed / "fixture_roi" / str(row["id"]) / str(row["source"]) / f"seed{row['seed']}.json"
            if not roi_path.is_file():
                write_self_hashed_json(
                    roi_path,
                    {
                        "status": "COMPLETE",
                        "candidate_tracking_forbidden": True,
                        "padding_rows": [],
                        "frame_indices": list(range(25, 55)),
                        "bbox": [[10.0, 10.0, 110.0, 110.0]] * 30,
                    },
                )
            result[cell_id] = {
                "path": str(crop),
                "video_hash": f"crop-{cell_id}",
                "pixel_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                "pts_sha256": "fixture-pts",
                "roi_hash": f"roi-{row['id']}-{row['source']}-{row['seed']}",
                "roi_path": str(roi_path),
                "frame_count": 30,
                "padding_rows": [],
            }
        return result

    monkeypatch.setattr(generation, "_prepare_b_media", fake_prepare)

    @contextmanager
    def fake_gpu_lease(**_kwargs: object):
        yield {"gate": "PASS", "fixture": True}

    monkeypatch.setattr(generation, "gpu_lease", fake_gpu_lease)

    class FakeSyncNetEngine:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def extract_visual(self, path: Path) -> tuple[np.ndarray, dict[str, object]]:
            parts = path.stem.split(":")
            driver = parts[3]
            code = {"A0": 0.0, "NOISE": 1.0, "DENOISE": 2.0}[driver]
            value = np.zeros((80, config.EMBEDDING_DIM), dtype=np.float32)
            value[:, 0] = code
            return value, {"fixture": True}

        def extract_audio(self, path: Path) -> tuple[np.ndarray, dict[str, object]]:
            name = path.stem.rsplit("_", 1)[-1]
            code = 3.0 if "PLUS_200MS" in path.stem else {"A0": 0.0, "NOISE": 1.0, "DENOISE": 2.0}[name]
            value = np.zeros((80, config.EMBEDDING_DIM), dtype=np.float32)
            value[:, 0] = code
            return value, {"fixture": True}

        @staticmethod
        def distance_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
            visual_code = round(float(visual[0, 0]))
            audio_code = round(float(audio[0, 0]))
            center = 15 + (5 if audio_code == 3 else visual_code - audio_code)
            columns = np.arange(config.LAG_COUNT, dtype=np.float32)
            return np.broadcast_to(np.abs(columns - center), (80, config.LAG_COUNT)).copy()

    monkeypatch.setattr(syncnet, "SyncNetEngine", FakeSyncNetEngine)
    result = generation.crossed_score_stage(paths, assets, audio_manifest, generation_manifest)

    assert result["status"] == "COMPLETE"
    assert result["science_cell_count"] == 28
    assert result["control_cell_count"] == 4
    assert len(result["cells"]) == 28
    assert len(result["four_cell_decomposition"]) == 8
    assert result["control_gate"] == "PASS"
    assert {row["control"]["status"] for row in result["controls"]} == {"IDENTITY_PASS", "DELAY_DETECTED"}
    for source in ("N", "T"):
        supports = {
            tuple(row["support_rows"])
            for row in result["cells"]
            if row["source"] == source
        }
        assert len(supports) == 1
        assert len(next(iter(supports))) == 50
