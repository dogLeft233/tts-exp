from pathlib import Path

from scripts.experiments.phoneme_tfg_association.official_syncnet_eval import parse_score


def test_parse_official_syncnet_score_accepts_negative_offset(tmp_path: Path) -> None:
    log = tmp_path / "syncnet.log"
    log.write_text(
        "Min dist:  7.870\nConfidence:  6.855\nAV offset:  -2\n",
        encoding="utf-8",
    )

    assert parse_score(log) == {
        "sync_c": 6.855,
        "sync_d": 7.87,
        "av_offset": -2,
    }


def test_parse_official_syncnet_score_rejects_incomplete_log(tmp_path: Path) -> None:
    log = tmp_path / "syncnet.log"
    log.write_text("Model loaded\n", encoding="utf-8")

    assert parse_score(log) is None
