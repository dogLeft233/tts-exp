import numpy as np

from scripts.experiments.static_image_bridge.images import (
    crop_zero_padded,
    generation_box,
    score_box,
    select_detection,
)


def test_detection_selection_uses_score_then_xy_tie_break() -> None:
    selected = select_detection(
        [
            {"x1": 20, "y1": 4, "x2": 40, "y2": 24, "score": 0.95},
            {"x1": 5, "y1": 8, "x2": 25, "y2": 28, "score": 0.95},
            {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.89},
        ]
    )
    assert selected["x1"] == 5.0


def test_generation_box_adds_only_the_frozen_bottom_padding() -> None:
    box = generation_box({"x1": 10.2, "y1": 20.8, "x2": 50.1, "y2": 70.2}, width=80, height=75)
    assert box == [10, 20, 51, 75]


def test_score_crop_is_square_and_zero_padded_outside_the_frame() -> None:
    frame = np.full((4, 5, 3), 7, dtype=np.uint8)
    box = score_box({"x1": 0, "y1": 0, "x2": 2, "y2": 2})
    crop = crop_zero_padded(frame, box, output_size=box["side"])
    assert crop.shape == (box["side"], box["side"], 3)
    assert np.all(crop[0, 0] == 0)
    assert np.all(crop[-1, -1] == 7)
