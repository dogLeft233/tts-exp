import numpy as np
import pytest

from scripts.experiments.static_image_bridge.frontal_verify import independent_crop
from scripts.experiments.static_image_bridge.images import crop_zero_padded


@pytest.mark.parametrize("box", [[0, 0, 80, 80], [-20, -30, 100, 90], [10, 10, 210, 210]])
def test_independent_crop_agrees_with_spec_geometry(box):
    frame = np.random.default_rng(42).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    expected = crop_zero_padded(frame, {"box": box, "side": box[2] - box[0]})
    assert np.array_equal(independent_crop(frame, box), expected)
