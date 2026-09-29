import numpy as np

from scripts.experiments.local_swap_minimal_replay.audio import local_swap, quarter_swap


def test_local_swap_preserves_non_equal_tail_and_matches_registered_boundaries() -> None:
    values = np.arange(23, dtype=np.int16)
    result, boundaries = local_swap(values)
    assert boundaries == {"length": 23, "b1": 5, "b2": 11, "b3": 17}
    assert result.tolist() == [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22]
    assert result.dtype == np.int16


def test_joint_frame_and_pcm_swap_is_exact_a_c_b_d() -> None:
    values = np.arange(4 * 3, dtype=np.int16)
    result, mapping = quarter_swap(values, 3)
    assert result.tolist() == [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
    assert result.tolist() == values[mapping["output_to_input"]].tolist()
