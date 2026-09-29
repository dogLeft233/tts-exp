from __future__ import annotations

import json

from scripts.experiments.phone_separability_enhancement.check import _walk_finite


def test_checker_finite_walk_is_recursive() -> None:
    assert _walk_finite({"a": [1.0, 2.0]}) == []
    assert _walk_finite({"a": [float("inf")]}) == ["$.a[0]"]


def test_checker_rejects_nonfinite_json_constant(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x": NaN}')
    try:
        from scripts.experiments.phone_separability_enhancement.check import _load_json_strict
        _load_json_strict(path)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite JSON must be rejected")
