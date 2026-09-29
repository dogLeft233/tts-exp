"""Protect the independent cohort and common-lag selection contract."""
import numpy as np

from scripts.experiments.tts_static_lag_calibration import BASE, EVAL, read


def test_frozen_static_calibration_cohort_and_geometry():
    p=read(BASE/"protocol.json")
    e=read(EVAL/"protocol.json")
    assert len(p["rows"])==26
    assert not {r["id"] for r in p["rows"]}&{r["id"] for r in e["rows"]}
    assert p["images"]==e["images"]


def test_median_half_integer_truncates_toward_zero():
    assert int(np.median([2,3]))==2
    assert int(np.median([-3,-2]))==-2
