"""Unit tests for scripts/19_generate_eval_matrix.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "19_generate_eval_matrix.py"
_spec = importlib.util.spec_from_file_location("_gxe_matrix", str(_SCRIPT))
_gxe_matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gxe_matrix)

gxe_key = _gxe_matrix.gxe_key
decompose = _gxe_matrix.decompose
classify_sample = _gxe_matrix.classify_sample
complete_matrix_samples = _gxe_matrix.complete_matrix_samples
build_ffmpeg_command = _gxe_matrix.build_ffmpeg_command
condition_for_axis = _gxe_matrix.condition_for_axis


class TestGxeMatrixStructure:
    def test_gxe_matrix_structure(self):
        cells = {
            "G_natural_E_natural": {"sync_c": 4.73, "sync_d": 8.573},
            "G_natural_E_tts": {"sync_c": 5.12, "sync_d": 7.98},
            "G_tts_E_natural": {"sync_c": 5.34, "sync_d": 7.82},
            "G_tts_E_tts": {"sync_c": 6.49, "sync_d": 7.36},
        }
        expected_keys = {
            "G_natural_E_natural",
            "G_natural_E_tts",
            "G_tts_E_natural",
            "G_tts_E_tts",
        }
        assert set(cells.keys()) == expected_keys
        for v in cells.values():
            assert "sync_c" in v
            assert "sync_d" in v

    def test_complete_matrix_samples_excludes_partial_cells(self):
        results = {
            "1": {
                "G_natural_E_natural": {},
                "G_natural_E_tts": {},
                "G_tts_E_natural": {},
                "G_tts_E_tts": {},
            },
            "9": {"G_natural_E_natural": {}, "G_natural_E_tts": {}},
        }
        assert complete_matrix_samples(results) == ["1"]


class TestDecompositionMath:
    def test_decomposition_math(self):
        G_nat_E_nat = 3.0
        G_tts_E_tts = 5.0
        G_tts_E_nat = 4.0
        G_nat_E_tts = 4.5

        total = G_tts_E_tts - G_nat_E_nat
        gen = G_tts_E_nat - G_nat_E_nat
        scorer = G_nat_E_tts - G_nat_E_nat
        interaction = total - gen - scorer

        assert total == 2.0
        assert gen == 1.0
        assert scorer == 1.5
        assert interaction == -0.5

    def test_decompose_function(self):
        total = {"sync_c": 2.0, "sync_d": -1.0}
        gen_effect = {"sync_c": 1.0, "sync_d": -0.5}
        scorer_effect = {"sync_c": 1.5, "sync_d": -0.3}

        interaction = decompose(total, gen_effect, scorer_effect)
        assert interaction["sync_c"] == round(2.0 - 1.0 - 1.5, 4)
        assert interaction["sync_d"] == round(-1.0 - (-0.5) - (-0.3), 4)

    def test_decompose_accounts_for_total(self):
        total = {"x": 10.0, "y": -3.0}
        gen = {"x": 4.0, "y": -1.0}
        score = {"x": 3.0, "y": -1.5}
        interaction = decompose(total, gen, score)

        for key in total:
            assert round(gen[key] + score[key] + interaction[key], 4) == round(total[key], 4)


class TestClassifyScorerDriven:
    def test_classify_scorer_driven(self):
        gen_effect = {"sync_c": 0.1, "sync_d": -0.1}
        scorer_effect = {"sync_c": 1.5, "sync_d": -1.2}
        assert classify_sample(gen_effect, scorer_effect) == "scorer_driven"

    def test_classify_scorer_driven_strong(self):
        gen_effect = {"sync_c": 0.05, "sync_d": 0.01}
        scorer_effect = {"sync_c": 3.0, "sync_d": -2.5}
        assert classify_sample(gen_effect, scorer_effect) == "scorer_driven"


class TestClassifyGeneratorDriven:
    def test_classify_generator_driven(self):
        gen_effect = {"sync_c": 1.5, "sync_d": -1.2}
        scorer_effect = {"sync_c": 0.1, "sync_d": -0.1}
        assert classify_sample(gen_effect, scorer_effect) == "generator_driven"

    def test_classify_generator_driven_strong(self):
        gen_effect = {"sync_c": 5.0, "sync_d": -3.0}
        scorer_effect = {"sync_c": 0.2, "sync_d": -0.1}
        assert classify_sample(gen_effect, scorer_effect) == "generator_driven"


class TestClassifyMixed:
    def test_classify_mixed(self):
        gen_effect = {"sync_c": 1.0, "sync_d": -0.8}
        scorer_effect = {"sync_c": 1.2, "sync_d": -0.9}
        assert classify_sample(gen_effect, scorer_effect) == "mixed"

    def test_classify_mixed_zero(self):
        gen_effect = {"sync_c": 0.0, "sync_d": 0.0}
        scorer_effect = {"sync_c": 0.0, "sync_d": 0.0}
        assert classify_sample(gen_effect, scorer_effect) == "mixed"


class TestSampleSkipWhenMissingVideo:
    def test_sample_skip_when_missing_video(self):
        find_existing_videos = _gxe_matrix.find_existing_videos
        with patch.object(_gxe_matrix, "find_existing_videos", return_value={}):
            result = find_existing_videos(Path("/nonexistent"), ["natural_raw"])
            assert result == {}


class TestFfmpegCommandBuilding:
    def test_ffmpeg_command_correct_structure(self):
        cmd = build_ffmpeg_command(
            Path("/tmp/video.mp4"),
            Path("/tmp/audio.wav"),
            Path("/tmp/output.mp4"),
        )
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-map" in cmd
        assert "0:v:0" in cmd
        assert "1:a:0" in cmd
        assert "-shortest" in cmd
        assert str(Path("/tmp/output.mp4")) in cmd

        i_indices = [i for i, a in enumerate(cmd) if a == "-i"]
        assert len(i_indices) == 2
        assert cmd[i_indices[0] + 1] == str(Path("/tmp/video.mp4"))
        assert cmd[i_indices[1] + 1] == str(Path("/tmp/audio.wav"))

    def test_ffmpeg_shortest_flag(self):
        cmd = build_ffmpeg_command(
            Path("video.mp4"),
            Path("audio.wav"),
            Path("out.mp4"),
        )
        assert "-shortest" in cmd
        shortest_idx = cmd.index("-shortest")
        assert shortest_idx > 0
        assert cmd[-1] == str(Path("out.mp4"))

    def test_ffmpeg_time_aligns_audio_to_video_duration(self):
        """Cross-cell audio must be time-aligned to the generator video."""
        cmd = build_ffmpeg_command(
            Path("video.mp4"),
            Path("audio.wav"),
            Path("out.mp4"),
            video_duration=5.0,
            audio_duration=5.5,
        )
        assert "-af" in cmd
        assert any("atempo=1.1" in arg for arg in cmd)
        assert "-t" in cmd
        assert "5.000000" in cmd
    def test_ffmpeg_diagonal_remux_skips_time_stretch(self):
        cmd = build_ffmpeg_command(
            Path("video.mp4"),
            Path("audio.wav"),
            Path("out.mp4"),
            video_duration=5.0,
            audio_duration=5.5,
            align_duration=False,
        )
        assert "-af" not in cmd
        assert "-t" not in cmd
        assert "aac" in cmd



class TestSyncnetPipeline:
    def test_pipeline_overwrites_cached_face_tracks(self):
        """A forced matrix rerun must replace SyncNet's cached face tracks."""
        with patch.object(
            _gxe_matrix.subprocess,
            "run",
            side_effect=[None, SimpleNamespace(stdout="", stderr="")],
        ) as run:
            _gxe_matrix.run_syncnet_pipeline(
                Path("syncnet"),
                "python",
                "bin",
                Path("video.mp4"),
                Path("data"),
                "reference",
                Path("model"),
            )

        assert "--overwrite" in run.call_args_list[0].args[0]


class TestGxeKey:
    def test_gxe_key_format(self):
        assert gxe_key("natural", "natural") == "G_natural_E_natural"
        assert gxe_key("tts", "natural") == "G_tts_E_natural"
        assert gxe_key("natural", "tts") == "G_natural_E_tts"

    def test_gxe_key_with_other_labels(self):
        assert gxe_key("fish", "natural") == "G_fish_E_natural"
        assert gxe_key("qwen3", "tts") == "G_qwen3_E_tts"


class TestConditionForAxis:
    def test_returns_mapped_condition(self):
        assert condition_for_axis("natural", ["natural_raw", "tts_raw"]) == "natural_raw"
        assert condition_for_axis("tts", ["natural_raw", "tts_raw"]) == "tts_raw"

    def test_fallback_to_original(self):
        assert condition_for_axis("unknown", ["natural_raw", "tts_raw"]) == "unknown"

    def test_uses_exact_match(self):
        assert condition_for_axis("tts_raw", ["natural_raw", "tts_raw"]) == "tts_raw"
