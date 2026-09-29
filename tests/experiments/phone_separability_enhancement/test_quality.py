from __future__ import annotations

from scripts.experiments.phone_separability_enhancement.quality import audit_timing, one_to_one_pause_match


def test_pause_matching_is_one_to_one_and_checks_edges() -> None:
    rows = one_to_one_pause_match([(0.2, 0.4), (0.5, 0.7)], [(0.2, 0.4), (0.52, 0.72)], iou_min=0.5, edge_error_ms=20)
    assert len(rows) == 2
    assert all(row["source_index"] is not None for row in rows)
    assert rows[1]["pass"] is False


def test_timing_audit_does_not_accept_only_high_iou() -> None:
    natural = [{"label": "a", "start_s": 0.0, "end_s": 0.2, "speech": True}]
    shifted = [{"label": "a", "start_s": 0.03, "end_s": 0.23, "speech": True}]
    result = audit_timing(natural, shifted)
    assert result["timing_pass"] is False
